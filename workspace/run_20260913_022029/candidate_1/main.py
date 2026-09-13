import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qm8_data import load_qm8


# ---------------------------------------------------------------------------
# Pure-PyTorch radius graph (avoids pyg-lib dependency)
# ---------------------------------------------------------------------------

def radius_graph_pytorch(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    max_num_neighbors: int = 100,
) -> torch.Tensor:
    """Build edges for all atom pairs within `cutoff`, respecting batch boundaries.

    Returns edge_index (2, E) with source->target edges.
    """
    num_nodes = pos.size(0)
    device = pos.device

    edge_list = []
    for g in batch.unique():
        mask = batch == g
        idx = mask.nonzero(as_tuple=False).view(-1)
        n = idx.size(0)
        if n < 2:
            continue
        g_pos = pos[idx]  # (n, 3)
        diffs = g_pos.unsqueeze(0) - g_pos.unsqueeze(1)  # (n, n, 3)
        dists = diffs.norm(dim=-1)  # (n, n)
        triu = torch.triu(dists, diagonal=1)
        src_local, dst_local = torch.where((triu > 0) & (triu < cutoff))
        src_global = idx[src_local]
        dst_global = idx[dst_local]
        edge_list.append(torch.stack([src_global, dst_global], dim=0))
        edge_list.append(torch.stack([dst_global, src_global], dim=0))

    if len(edge_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = torch.cat(edge_list, dim=1)

    if max_num_neighbors is not None and edge_index.size(1) > 0:
        src = edge_index[0]
        dst = edge_index[1]
        dists = (pos[src] - pos[dst]).norm(dim=-1)
        keep = []
        for node in range(num_nodes):
            node_mask = src == node
            if node_mask.sum() <= max_num_neighbors:
                keep.append(torch.where(node_mask)[0])
            else:
                node_dists = dists[node_mask]
                _, topk_idx = torch.topk(node_dists, max_num_neighbors, largest=False)
                keep.append(torch.where(node_mask)[0][topk_idx])
        if keep:
            keep = torch.cat(keep)
            edge_index = edge_index[:, keep]

    return edge_index


# ---------------------------------------------------------------------------
# SchNet building blocks
# ---------------------------------------------------------------------------

class CosineCutoff(nn.Module):
    """Cosine cutoff envelope: f(r) = 0.5 * (1 + cos(pi * r / cutoff))."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(np.pi * distances / self.cutoff))


class RBFExpansion(nn.Module):
    """Expand interatomic distances in a radial basis of Gaussians."""

    def __init__(self, cutoff: float, num_gaussians: int):
        super().__init__()
        centers = torch.linspace(0, cutoff, num_gaussians)
        self.register_buffer("centers", centers)
        self.width = cutoff / num_gaussians

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        gamma = 1.0 / (2.0 * self.width ** 2)
        return torch.exp(-gamma * (d - self.centers) ** 2)


class InteractionBlock(nn.Module):
    """One SchNet continuous-filter interaction block."""

    def __init__(self, num_filters: int, num_gaussians: int, cutoff: float):
        super().__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.Softplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.atomwise_mlp = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.Softplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        rbf_features: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        W = self.filter_network(rbf_features)
        W = W * self.cutoff_fn(distances).unsqueeze(-1)
        messages = x[src] * W
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, messages)
        return x + self.atomwise_mlp(aggr)


class SchNet(nn.Module):
    """SchNet: continuous-filter convolutional neural network for 3D molecular data."""

    def __init__(
        self,
        num_atom_types: int = 100,
        embedding_dim: int = 128,
        num_filters: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        num_outputs: int = 16,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.embedding = nn.Embedding(num_atom_types, embedding_dim, padding_idx=0)

        if embedding_dim != num_filters:
            self.proj = nn.Linear(embedding_dim, num_filters)
        else:
            self.proj = nn.Identity()

        self.rbf = RBFExpansion(cutoff, num_gaussians)

        self.interactions = nn.ModuleList(
            [
                InteractionBlock(num_filters, num_gaussians, cutoff)
                for _ in range(num_interactions)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.Softplus(),
            nn.Linear(num_filters // 2, num_outputs),
        )

    def forward(self, data: Data) -> torch.Tensor:
        z = data.z
        pos = data.pos
        batch = data.batch

        edge_index = radius_graph_pytorch(pos, batch, self.cutoff, max_num_neighbors=100)

        if edge_index.size(1) == 0:
            x = self.embedding(z)
            x = self.proj(x)
            num_graphs = int(batch.max().item() + 1)
            graph_features = torch.zeros(
                num_graphs, x.size(1), device=x.device, dtype=x.dtype
            )
            graph_features.index_add_(0, batch, x)
            return self.readout(graph_features)

        src, dst = edge_index
        distances = (pos[src] - pos[dst]).norm(dim=-1)

        rbf_features = self.rbf(distances)

        x = self.embedding(z)
        x = self.proj(x)

        for interaction in self.interactions:
            x = interaction(x, edge_index, distances, rbf_features)

        num_graphs = int(batch.max().item() + 1)
        graph_features = torch.zeros(
            num_graphs, x.size(1), device=x.device, dtype=x.dtype
        )
        graph_features.index_add_(0, batch, x)

        return self.readout(graph_features)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QM8Dataset(torch.utils.data.Dataset):
    def __init__(self, molecules, label_keys=None):
        self.molecules = molecules
        if label_keys is None:
            label_keys = sorted(molecules[0].labels.keys())
        self.label_keys = label_keys

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        mol = self.molecules[idx]
        z = torch.tensor(mol.atomic_numbers, dtype=torch.long)
        pos = torch.tensor(mol.positions, dtype=torch.float32)
        labels = torch.tensor(
            [mol.labels[k] for k in self.label_keys], dtype=torch.float32
        )
        # Shape (1, num_outputs) so batching stacks to (batch_size, num_outputs)
        return Data(z=z, pos=pos, y=labels.unsqueeze(0))


class NormalizedDataset(torch.utils.data.Dataset):
    """Wraps a subset of QM8Dataset and normalises targets."""

    def __init__(self, base_dataset, indices, mean, std):
        self.base = base_dataset
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = self.base[self.indices[idx]]
        data.y = (data.y - self.mean) / self.std
        return data


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
    return total_loss / n_graphs


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_graphs = 0
    all_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mae = mean_absolute_error(all_targets, all_preds)
    return total_loss / n_graphs, mae, all_preds, all_targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=str, required=True, choices=["baseline", "train", "evaluate"]
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- Load data -----------------------------------------------------------
    print("Loading QM8 data ...")
    molecules = load_qm8()
    print(f"Loaded {len(molecules)} molecules")

    label_keys = sorted(molecules[0].labels.keys())
    num_outputs = len(label_keys)
    print(f"Target properties ({num_outputs}): {label_keys}")

    dataset = QM8Dataset(molecules, label_keys)

    # 80 / 10 / 10 split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Normalisation stats from training set
    train_labels = np.stack(
        [dataset[i].y.numpy().squeeze(0) for i in train_ds.indices], axis=0
    )
    mean = torch.tensor(train_labels.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(train_labels.std(axis=0), dtype=torch.float32)
    std = torch.clamp(std, min=1e-8)

    print(f"Target mean  (first 4): {mean[:4].tolist()}")
    print(f"Target std   (first 4): {std[:4].tolist()}")

    train_norm = NormalizedDataset(dataset, train_ds.indices, mean, std)
    val_norm = NormalizedDataset(dataset, val_ds.indices, mean, std)
    test_norm = NormalizedDataset(dataset, test_ds.indices, mean, std)

    # ---- Model ---------------------------------------------------------------
    model = SchNet(
        num_atom_types=100,
        embedding_dim=128,
        num_filters=128,
        num_interactions=3,
        num_gaussians=50,
        cutoff=5.0,
        num_outputs=num_outputs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # =====================================================================
    #  BASELINE
    # =====================================================================
    if args.stage == "baseline":
        n_base = min(200, n_train)
        n_base_val = min(50, n_val)

        base_train = torch.utils.data.Subset(train_norm, range(n_base))
        base_val = torch.utils.data.Subset(val_norm, range(n_base_val))

        train_loader = DataLoader(base_train, batch_size=32, shuffle=True)
        val_loader = DataLoader(base_val, batch_size=32, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        print("\n=== BASELINE (5 epochs on small subset) ===")
        for epoch in range(5):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_mae, _, _ = evaluate_model(
                model, val_loader, criterion, device
            )
            print(
                f"  epoch {epoch+1:2d}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"val_mae={val_mae:.6f}"
            )

        print("Baseline complete.")

    # =====================================================================
    #  TRAIN
    # =====================================================================
    elif args.stage == "train":
        train_loader = DataLoader(train_norm, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_norm, batch_size=32, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        patience = 15
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        os.makedirs("checkpoints", exist_ok=True)

        print("\n=== TRAIN (early stopping, patience=15) ===")
        for epoch in range(1, 501):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_mae, _, _ = evaluate_model(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch:3d}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"val_mae={val_mae:.6f}  "
                f"lr={lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "mean": mean,
                        "std": std,
                        "label_keys": label_keys,
                    },
                    "checkpoints/best_model.pt",
                )
                print(f"    -> best model saved (val_loss={val_loss:.6f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping after {epoch} epochs "
                        f"(best was epoch {best_epoch}, val_loss={best_val_loss:.6f})"
                    )
                    break

        print(
            f"Training finished.  Best epoch {best_epoch}  "
            f"val_loss={best_val_loss:.6f}"
        )

    # =====================================================================
    #  EVALUATE
    # =====================================================================
    elif args.stage == "evaluate":
        ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        mean = ckpt["mean"].to(device)
        std = ckpt["std"].to(device)
        label_keys = ckpt["label_keys"]

        test_loader = DataLoader(test_norm, batch_size=32, shuffle=False)
        criterion = nn.MSELoss()

        test_loss, test_mae_norm, preds_norm, targets_norm = evaluate_model(
            model, test_loader, criterion, device
        )

        # Denormalise
        mean_np = mean.cpu().numpy()
        std_np = std.cpu().numpy()
        preds = preds_norm * std_np + mean_np
        targets = targets_norm * std_np + mean_np

        overall_mae = float(mean_absolute_error(targets, preds))
        per_prop_mae = np.mean(np.abs(preds - targets), axis=0)

        print(f"\n=== EVALUATE ===")
        print(f"Test MSE (normalised): {test_loss:.6f}")
        print(f"Test MAE  (overall)  : {overall_mae:.6f}")
        print(f"Per-property MAE:")
        for i, key in enumerate(label_keys):
            print(f"  {key:20s}  {per_prop_mae[i]:.6f}")

        # ---- results.json -------------------------------------------------
        results = {
            "test_mse_normalised": float(test_loss),
            "test_mae_overall": overall_mae,
            "per_property_mae": {
                key: float(per_prop_mae[i]) for i, key in enumerate(label_keys)
            },
        }

        # ---- Diagnostics ---------------------------------------------------
        method_groups: dict[str, list[int]] = {}
        for i, key in enumerate(label_keys):
            method = key.split("-", 1)[1] if "-" in key else "unknown"
            method_groups.setdefault(method, []).append(i)

        print("\nPer-method MAE:")
        method_mae = {}
        for method, idxs in method_groups.items():
            m = float(np.mean(per_prop_mae[idxs]))
            method_mae[method] = m
            print(f"  {method:10s}  {m:.6f}")
        results["per_method_mae"] = method_mae

        e_idx = [i for i, k in enumerate(label_keys) if k.startswith("E")]
        f_idx = [i for i, k in enumerate(label_keys) if k.startswith("f")]
        e_mae = float(np.mean(per_prop_mae[e_idx])) if e_idx else 0.0
        f_mae = float(np.mean(per_prop_mae[f_idx])) if f_idx else 0.0
        print(f"\nExcitation energies   MAE: {e_mae:.6f}")
        print(f"Oscillator strengths  MAE: {f_mae:.6f}")
        results["excitation_energy_mae"] = e_mae
        results["oscillator_strength_mae"] = f_mae

        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults written to results.json")

        # ---- Plots ---------------------------------------------------------
        test_indices = test_ds.indices
        heavy_counts = np.array(
            [
                sum(1 for an in molecules[i].atomic_numbers if an > 1)
                for i in test_indices
            ]
        )
        per_mol_mae = np.mean(np.abs(preds - targets), axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(per_mol_mae, bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Per-molecule MAE")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Error distribution")

        unique_hc = np.unique(heavy_counts)
        mean_by_size = [per_mol_mae[heavy_counts == c].mean() for c in unique_hc]
        axes[1].bar(unique_hc, mean_by_size, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Heavy atom count")
        axes[1].set_ylabel("Mean MAE")
        axes[1].set_title("MAE vs molecular size")

        plt.tight_layout()
        plt.savefig("error_analysis.png", dpi=150)
        print("Saved error_analysis.png")

        method_colors = {
            "CC2": "tab:blue",
            "PBE0": "tab:green",
            "PBE0.1": "tab:orange",
            "CAM": "tab:red",
        }

        def _method_of(key: str) -> str:
            return key.split("-", 1)[1] if "-" in key else "unknown"

        colors = [method_colors.get(_method_of(k), "gray") for k in label_keys]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(label_keys)), per_prop_mae, color=colors, edgecolor="black")
        ax.set_xticks(range(len(label_keys)))
        ax.set_xticklabels(label_keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("MAE")
        ax.set_title("Per-property MAE")
        plt.tight_layout()
        plt.savefig("per_property_mae.png", dpi=150)
        print("Saved per_property_mae.png")


if __name__ == "__main__":
    main()