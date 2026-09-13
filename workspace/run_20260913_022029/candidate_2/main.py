"""
SchNet with separate task heads for excitation energies vs oscillator strengths,
and angle-aware features for QM8.

Implements:
- Standard SchNet continuous-filter convolution interaction blocks
- Angle module computing per-atom triplet-angle statistics
- Separate output heads: 8 energy targets + 8 oscillator strength targets
- Weighted MSE loss emphasizing oscillator strengths
- Per-property error diagnostics

Stages:
  python main.py --stage baseline   # quick sanity check on subset
  python main.py --stage train      # full training with early stopping
  python main.py --stage evaluate   # metrics + error analysis + results.json
"""

import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from qm8_data import load_qm8

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants -- column names match what pandas produces from qm8.sdf.csv
# (duplicate PBE0 columns get .1 suffix from pd.read_csv auto-disambiguation)
# ---------------------------------------------------------------------------
TARGET_NAMES = [
    "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2",
    "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0",
    "E1-PBE0.1", "E2-PBE0.1", "f1-PBE0.1", "f2-PBE0.1",
    "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM",
]

# Indices into the 16-target vector
ENERGY_INDICES = [0, 1, 4, 5, 8, 9, 12, 13]       # all E1, E2
OSCILLATOR_INDICES = [2, 3, 6, 7, 10, 11, 14, 15]  # all f1, f2


# ===================================================================
# Model components
# ===================================================================

class GaussianSmearing(nn.Module):
    """Expand scalar distances into a Gaussian basis."""

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        self.register_buffer("centers", torch.linspace(start, stop, num_gaussians))
        self.register_buffer("width", torch.tensor((stop - start) / num_gaussians))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = (dist.unsqueeze(-1) - self.centers) / self.width
        return torch.exp(-0.5 * diff * diff)


def radius_graph_batch(pos, r, batch):
    """Build radius graph across a batch, respecting batch boundaries.

    Uses a block-diagonal approach: for each molecule, compute pairwise
    distances and select edges within cutoff. Returns [2, E] edge_index.
    """
    device = pos.device
    # Ensure batch is on the same device as pos so that all index tensors
    # created from batch (via torch.where, boolean masks, etc.) are valid
    # for indexing pos.
    batch = batch.to(device)
    num_nodes = pos.size(0)
    unique_batches = torch.unique(batch)

    row_list, col_list = [], []

    for b in unique_batches:
        mask = (batch == b)
        idx = torch.where(mask)[0]
        n = idx.size(0)
        if n < 2:
            continue
        pos_b = pos[idx]  # [n, 3]
        # squared distances
        d2 = torch.cdist(pos_b, pos_b, p=2)  # [n, n]
        # upper triangle, exclude diagonal
        triu = torch.triu_indices(n, n, offset=1, device=device)
        keep = d2[triu[0], triu[1]] < r
        src_local = triu[0][keep]
        dst_local = triu[1][keep]
        # map back to global indices
        row_list.append(idx[src_local])
        row_list.append(idx[dst_local])  # undirected
        col_list.append(idx[dst_local])
        col_list.append(idx[src_local])

    if len(row_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.stack([torch.cat(row_list), torch.cat(col_list)], dim=0)


class SchNetInteraction(nn.Module):
    """One SchNet continuous-filter convolution interaction block."""

    def __init__(self, hidden_dim: int = 128, num_gaussians: int = 50):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atomwise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_dist_expanded):
        row, col = edge_index
        W = self.filter_net(edge_dist_expanded)
        messages = h[col] * W
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, messages)
        return h + self.atomwise(aggr)


class AngleModule(nn.Module):
    """Compute per-atom angle features from local triplet geometry.

    For each atom with >= 2 neighbours, computes cos/sin statistics
    over all neighbour-pair angles. Vectorized over atoms within a molecule.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, pos, edge_index):
        """h: [N, F], pos: [N, 3], edge_index: [2, E] (single molecule)."""
        row, col = edge_index
        num_nodes = h.size(0)
        num_edges = row.size(0)

        if num_edges == 0:
            return h

        vec = pos[row] - pos[col]                        # [E, 3]
        dist = torch.norm(vec, dim=-1, keepdim=True)
        vec_norm = vec / (dist + 1e-8)

        # degree per node
        degree = torch.zeros(num_nodes, dtype=torch.long, device=h.device)
        degree.index_add_(0, row, torch.ones(num_edges, dtype=torch.long, device=h.device))

        angle_feats = torch.zeros(num_nodes, 4, device=h.device)

        # Process each node -- but only those with degree >= 2
        for j in range(num_nodes):
            deg = degree[j].item()
            if deg < 2:
                continue
            mask = (row == j)
            nbr_vecs = vec_norm[mask]                    # [deg, 3]
            cos_mat = torch.mm(nbr_vecs, nbr_vecs.t())   # [deg, deg]
            # upper triangle without diagonal
            triu = torch.triu_indices(deg, deg, offset=1, device=h.device)
            cos_vals = cos_mat[triu[0], triu[1]]
            sin_vals = torch.sqrt(torch.clamp(1.0 - cos_vals * cos_vals, min=0.0))
            angle_feats[j, 0] = cos_vals.mean()
            angle_feats[j, 1] = cos_vals.max()
            angle_feats[j, 2] = sin_vals.mean()
            angle_feats[j, 3] = sin_vals.max()

        return h + self.mlp(angle_feats)


class SchNetWithSeparateHeads(nn.Module):
    """SchNet with separate output heads for energies vs oscillator strengths
    and angle-aware features."""

    def __init__(self, hidden_dim: int = 128, num_interactions: int = 3,
                 cutoff: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        self.atom_embedding = nn.Embedding(100, hidden_dim, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_gaussians)
            for _ in range(num_interactions)
        ])

        self.angle_module = AngleModule(hidden_dim)

        self.postprocess = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- separate heads ---
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, 8),
        )
        self.oscillator_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, 8),
        )

    def forward(self, data):
        # Move entire input to the model's device to avoid mixed-device errors.
        device = next(self.parameters()).device
        data = data.to(device)
        z, pos, batch = data.z, data.pos, data.batch

        h = self.atom_embedding(z)

        edge_index = radius_graph_batch(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=-1)
        edge_dist_expanded = self.distance_expansion(dist)

        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_dist_expanded)

        # --- angle features (per-molecule) ---
        h_angled = h.clone()
        for b in torch.unique(batch):
            mol_mask = (batch == b)
            local_idx = torch.where(mol_mask)[0]

            edge_mask = mol_mask[row] & mol_mask[col]
            local_row = row[edge_mask]
            local_col = col[edge_mask]

            # Build local index map on the same device as h.
            idx_map = -torch.ones(batch.size(0), dtype=torch.long, device=h.device)
            idx_map[local_idx] = torch.arange(local_idx.size(0), device=h.device)
            local_ei = torch.stack([idx_map[local_row], idx_map[local_col]], dim=0)

            h_local = self.angle_module(h[local_idx], pos[local_idx], local_ei)
            h_angled[local_idx] = h_local

        h = h_angled

        h = self.postprocess(h)
        h_pooled = global_add_pool(h, batch)

        energy_pred = self.energy_head(h_pooled)
        osc_pred = self.oscillator_head(h_pooled)

        out = torch.zeros(h_pooled.size(0), 16, device=h_pooled.device)
        out[:, ENERGY_INDICES] = energy_pred
        out[:, OSCILLATOR_INDICES] = osc_pred
        return out


# ===================================================================
# Loss
# ===================================================================

class WeightedMSELoss(nn.Module):
    """MSE with higher weight on oscillator strengths (harder to predict)."""

    def __init__(self, osc_weight: float = 5.0):
        super().__init__()
        weights = torch.ones(16)
        weights[OSCILLATOR_INDICES] = osc_weight
        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        diff = (pred - target) ** 2
        return (diff * self.weights.unsqueeze(0)).mean()


# ===================================================================
# Data helpers
# ===================================================================

def build_dataloader(molecules, batch_size=32, shuffle=True,
                     scaler=None, fit_scaler=False):
    """Convert QM8Molecule list -> PyG DataLoader with optional scaling.

    y is stored as [1, 16] (2D) to force PyG to treat it as a graph-level
    target regardless of molecule size.
    """
    data_list = []
    targets = []

    for mol in molecules:
        z = torch.tensor(mol.atomic_numbers, dtype=torch.long)
        pos = torch.tensor(mol.positions, dtype=torch.float)
        y = torch.tensor([mol.labels[n] for n in TARGET_NAMES], dtype=torch.float)
        targets.append(y.numpy())
        data_list.append(Data(z=z, pos=pos, y=y.unsqueeze(0)))

    targets = np.array(targets)

    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(targets)

    if scaler is not None:
        for d in data_list:
            d.y = torch.tensor(
                scaler.transform(d.y.numpy().reshape(1, -1)),
                dtype=torch.float,
            )

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    return loader, scaler


# ===================================================================
# Training / evaluation loops
# ===================================================================

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.squeeze(1)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * batch.num_graphs
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return total_loss / len(loader.dataset), preds, targets


# ===================================================================
# Stages
# ===================================================================

def run_baseline():
    """Quick run on a small subset to catch bugs before the real training."""
    print("=== BASELINE ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    molecules = load_qm8()
    print(f"Loaded {len(molecules)} molecules total")

    # Small subset for speed
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(molecules))[:1000]
    subset = [molecules[i] for i in idx]

    n_train = int(0.7 * len(subset))
    n_val = int(0.15 * len(subset))
    train_mols = subset[:n_train]
    val_mols = subset[n_train:n_train + n_val]
    test_mols = subset[n_train + n_val:]

    train_loader, scaler = build_dataloader(
        train_mols, batch_size=16, shuffle=True, fit_scaler=True)
    val_loader, _ = build_dataloader(
        val_mols, batch_size=16, shuffle=False, scaler=scaler)
    test_loader, _ = build_dataloader(
        test_mols, batch_size=16, shuffle=False, scaler=scaler)

    # Smaller model for baseline speed
    model = SchNetWithSeparateHeads(
        hidden_dim=32, num_interactions=1, cutoff=5.0, num_gaussians=20)
    model = model.to(device)

    loss_fn = WeightedMSELoss(osc_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        print(f"  epoch {epoch+1:2d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    test_loss, preds, targets = evaluate_model(model, test_loader, loss_fn, device)
    preds_unscaled = scaler.inverse_transform(preds)
    targets_unscaled = scaler.inverse_transform(targets)
    per_mae = np.mean(np.abs(preds_unscaled - targets_unscaled), axis=0)

    print(f"\nTest loss (scaled): {test_loss:.6f}")
    print("Per-property MAE (unscaled):")
    for i, name in enumerate(TARGET_NAMES):
        tag = "E" if i in ENERGY_INDICES else "f"
        print(f"  {name:20s}  {per_mae[i]:.6f}  [{tag}]")

    energy_mae = np.mean(per_mae[ENERGY_INDICES])
    osc_mae = np.mean(per_mae[OSCILLATOR_INDICES])
    print(f"\nMean energy MAE:     {energy_mae:.6f}")
    print(f"Mean oscillator MAE: {osc_mae:.6f}")
    print("Baseline complete.")


def run_train():
    """Full training run with early stopping on validation loss."""
    print("=== TRAIN ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    molecules = load_qm8()
    print(f"Loaded {len(molecules)} molecules")

    # 80 / 10 / 10 split
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(molecules))
    n_train = int(0.8 * len(molecules))
    n_val = int(0.1 * len(molecules))

    train_mols = [molecules[i] for i in idx[:n_train]]
    val_mols = [molecules[i] for i in idx[n_train:n_train + n_val]]
    test_mols = [molecules[i] for i in idx[n_train + n_val:]]
    print(f"Train: {len(train_mols)}  Val: {len(val_mols)}  Test: {len(test_mols)}")

    train_loader, scaler = build_dataloader(
        train_mols, batch_size=64, shuffle=True, fit_scaler=True)
    val_loader, _ = build_dataloader(
        val_mols, batch_size=64, shuffle=False, scaler=scaler)
    test_loader, _ = build_dataloader(
        test_mols, batch_size=64, shuffle=False, scaler=scaler)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    model = SchNetWithSeparateHeads(
        hidden_dim=128, num_interactions=3, cutoff=5.0, num_gaussians=50)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = WeightedMSELoss(osc_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(1, 201):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  lr={lr:.2e}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  *")
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "scaler": scaler}, "best_model.pt")
    print(f"Best val loss: {best_val_loss:.6f}  -> saved best_model.pt")


def run_evaluate():
    """Load best checkpoint, compute metrics, write results.json + diagnostic plots."""
    print("=== EVALUATE ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    molecules = load_qm8()
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(molecules))
    n_train = int(0.8 * len(molecules))
    n_val = int(0.1 * len(molecules))
    test_mols = [molecules[i] for i in idx[n_train + n_val:]]
    print(f"Test molecules: {len(test_mols)}")

    ckpt = torch.load("best_model.pt", map_location=device)
    scaler = ckpt["scaler"]

    model = SchNetWithSeparateHeads(
        hidden_dim=128, num_interactions=3, cutoff=5.0, num_gaussians=50)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_loader, _ = build_dataloader(
        test_mols, batch_size=64, shuffle=False, scaler=scaler)

    loss_fn = WeightedMSELoss(osc_weight=5.0)
    test_loss, preds, targets = evaluate_model(model, test_loader, loss_fn, device)

    preds_u = scaler.inverse_transform(preds)
    targets_u = scaler.inverse_transform(targets)

    per_mae = np.mean(np.abs(preds_u - targets_u), axis=0)
    per_rmse = np.sqrt(np.mean((preds_u - targets_u) ** 2, axis=0))

    energy_mae = float(np.mean(per_mae[ENERGY_INDICES]))
    osc_mae = float(np.mean(per_mae[OSCILLATOR_INDICES]))
    energy_rmse = float(np.mean(per_rmse[ENERGY_INDICES]))
    osc_rmse = float(np.mean(per_rmse[OSCILLATOR_INDICES]))

    results = {
        "test_loss_scaled": float(test_loss),
        "overall_mae": float(np.mean(per_mae)),
        "overall_rmse": float(np.mean(per_rmse)),
        "energy_mae": energy_mae,
        "oscillator_mae": osc_mae,
        "energy_rmse": energy_rmse,
        "oscillator_rmse": osc_rmse,
        "per_property_mae": {n: float(v) for n, v in zip(TARGET_NAMES, per_mae)},
        "per_property_rmse": {n: float(v) for n, v in zip(TARGET_NAMES, per_rmse)},
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTest loss (scaled): {test_loss:.6f}")
    print(f"Overall MAE:  {results['overall_mae']:.6f}")
    print(f"Overall RMSE: {results['overall_rmse']:.6f}")
    print(f"Energy MAE:   {energy_mae:.6f}   RMSE: {energy_rmse:.6f}")
    print(f"Osc   MAE:    {osc_mae:.6f}   RMSE: {osc_rmse:.6f}")
    print("\nPer-property MAE:")
    for i, name in enumerate(TARGET_NAMES):
        tag = "E" if i in ENERGY_INDICES else "f"
        print(f"  {name:20s}  {per_mae[i]:.6f}  [{tag}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        colors = ["#2166ac" if i in ENERGY_INDICES else "#b2182b" for i in range(16)]
        ax.bar(range(16), per_mae, color=colors)
        ax.set_xticks(range(16))
        ax.set_xticklabels(TARGET_NAMES, rotation=90, fontsize=7)
        ax.set_ylabel("MAE")
        ax.set_title("Per-property MAE (blue = energy, red = oscillator)")

        ax = axes[0, 1]
        f2_idx = TARGET_NAMES.index("f2-CC2")
        ax.scatter(targets_u[:, f2_idx], preds_u[:, f2_idx], alpha=0.4, s=4, color="#b2182b")
        lims = [targets_u[:, f2_idx].min(), targets_u[:, f2_idx].max()]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlabel("True f2-CC2")
        ax.set_ylabel("Predicted f2-CC2")
        ax.set_title("f2-CC2: predicted vs true")

        ax = axes[1, 0]
        e1_idx = TARGET_NAMES.index("E1-CC2")
        ax.scatter(targets_u[:, e1_idx], preds_u[:, e1_idx], alpha=0.4, s=4, color="#2166ac")
        lims = [targets_u[:, e1_idx].min(), targets_u[:, e1_idx].max()]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlabel("True E1-CC2")
        ax.set_ylabel("Predicted E1-CC2")
        ax.set_title("E1-CC2: predicted vs true")

        ax = axes[1, 1]
        e_err = (preds_u[:, ENERGY_INDICES] - targets_u[:, ENERGY_INDICES]).ravel()
        o_err = (preds_u[:, OSCILLATOR_INDICES] - targets_u[:, OSCILLATOR_INDICES]).ravel()
        ax.hist(e_err, bins=60, alpha=0.5, label="Energy errors", color="#2166ac")
        ax.hist(o_err, bins=60, alpha=0.5, label="Oscillator errors", color="#b2182b")
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.set_title("Error distributions: energies vs oscillators")
        ax.legend()

        plt.tight_layout()
        plt.savefig("error_analysis.png", dpi=150)
        print("\nSaved error_analysis.png")
    except ImportError:
        print("\n(matplotlib not available -- skipping diagnostic plots)")

    print("\nEvaluation complete -- results.json written.")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["baseline", "train", "evaluate"])
    args = parser.parse_args()

    if args.stage == "baseline":
        run_baseline()
    elif args.stage == "train":
        run_train()
    elif args.stage == "evaluate":
        run_evaluate()