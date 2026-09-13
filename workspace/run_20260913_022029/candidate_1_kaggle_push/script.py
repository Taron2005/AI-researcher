import subprocess as _subprocess, sys as _sys
_subprocess.check_call([_sys.executable, '-m', 'pip', 'install', '-q'] + ['matplotlib', 'rdkit', 'torch_geometric'])

"""
QM8 loading -- solved once here, not left for an agent to rediscover later.

CLAUDE.md originally claimed PyTorch Geometric has a native QM8 loader. That
was never actually checked against PyG's real source and turned out to be
wrong -- PyG has QM7b and QM9, not QM8 (see DECISIONS.md for the correction).
DeepChem does have a real QM8 loader, but importing deepchem at all pulls in
TensorFlow as a hard dependency, and its default featurizer (CoulombMatrix)
isn't raw 3D positions anyway -- so this loads the same underlying files
DeepChem itself uses, directly, with no deepchem/TensorFlow dependency.

Verified empirically (not assumed) before writing this:
- The molecule structures (gdb8.tar.gz -> qm8.sdf) and the two label files
  (the standalone qm8.csv and the bundled qm8.sdf.csv) all have exactly
  21,786 records, and SDF molecule N's SMILES matches CSV row N's SMILES --
  a plain positional join is correct, not id-based matching.
- Plain rdkit.Chem.SDMolSupplier reads each record's conformer as embedded
  in the file, with no conformer-generation step -- these are QM8's actual
  original 3D coordinates (the geometry the CC2/PBE0/CAM labels were
  computed on), not a re-embedded/generated substitute. Confirmed by
  checking a real record's conformer directly, not by reading library docs.
- The raw CSV's header has a real duplicate: "E1-PBE0,E2-PBE0,f1-PBE0,
  f2-PBE0" appears twice, verbatim, with genuinely different values each
  time -- not a display artifact. This is PBE0 computed at two different
  basis sets (def2-SVP and def2-TZVP; confirmed against the original QM8
  paper's methodology, Ramakrishnan et al. 2015). A naive dict-based CSV
  read (e.g. csv.DictReader) silently collapses duplicate column names to
  the last occurrence, silently discarding 4 real columns -- caught by
  inspecting the raw header, not assumed safe. pandas.read_csv instead
  auto-suffixes the second occurrence (E1-PBE0.1, etc.), which also
  matches the column naming used in published QM8 benchmark code -- used
  here for that reason, not just because it's convenient.
"""

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

# RDKit logs every sanitization failure to the console by default (one WARNING
# + ERROR pair per bad record) -- with ~39 known-bad records in QM8's raw SDF
# that's pure noise. Suppressed here; load_qm8() reports the real skip count
# explicitly instead, so the information isn't lost, just not spammed.
RDLogger.DisableLog("rdApp.*")

GDB8_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz"
# QM8_DATA_DIR env var wins if set -- this file gets COPIED into each
# candidate's own directory for portability (see roles/software_engineer.py),
# and a __file__-relative path silently resolves to the wrong location once
# copied elsewhere, forcing a redundant ~8.7MB re-download and ~20-30s
# re-parse on every candidate run instead of reusing the one real cache.
# Found by actually running a candidate locally, not by inspection --
# see DECISIONS.md.
DATA_DIR = Path(os.environ.get("QM8_DATA_DIR") or (Path(__file__).parent.parent / "data" / "qm8"))


@dataclass
class QM8Molecule:
    smiles: str
    atomic_numbers: list[int]
    positions: list[tuple[float, float, float]]  # original 3D coordinates, per atom
    labels: dict[str, float]  # the 16 target properties


def _download_and_extract() -> None:
    """
    Uses a completion marker, not "does the file exist", to decide whether
    this already ran -- an interrupted download or extraction (this process
    getting killed mid-run has genuinely happened during this project)
    leaves a partial file that plain `.exists()` would mistake for a
    complete one, silently working with truncated data afterward.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / ".download_complete"
    if marker.exists():
        return

    archive_path = DATA_DIR / "gdb8.tar.gz"
    if not archive_path.exists():
        # Download to a temp path and rename only on success, so a killed-
        # mid-download process can never leave a partial file at the real
        # path for a later run to mistake as complete.
        tmp_path = archive_path.with_suffix(".tar.gz.tmp")
        urllib.request.urlretrieve(GDB8_URL, tmp_path)
        tmp_path.rename(archive_path)

    with tarfile.open(archive_path) as tar:
        tar.extractall(DATA_DIR, filter="data")

    marker.touch()


def load_qm8() -> list[QM8Molecule]:
    """
    Downloads (once, cached under data/qm8/) and parses QM8's molecules with
    their original 3D coordinates and labels. Returns fewer than 21,786 --
    a small number of records (verified: 39 out of 21,786, about 0.18%) fail
    RDKit's sanitization due to invalid valence states in the raw SDF, a
    known issue with GDB-derived structures, not a bug in this loader. The
    skip count is logged so this loss is visible, not silent.
    """
    _download_and_extract()

    labels_df = pd.read_csv(DATA_DIR / "qm8.sdf.csv").drop(columns=["gdb9_index"])
    labels_by_row = labels_df.to_dict(orient="records")

    supplier = Chem.SDMolSupplier(str(DATA_DIR / "qm8.sdf"), removeHs=False)
    molecules = []
    skipped = 0
    for mol, labels in zip(supplier, labels_by_row):
        if mol is None:
            skipped += 1
            continue
        conf = mol.GetConformer()
        molecules.append(QM8Molecule(
            smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)),
            atomic_numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
            positions=[tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            labels=labels,
        ))

    print(f"load_qm8: loaded {len(molecules)} molecules, skipped {skipped} (RDKit sanitization failures)")
    return molecules



import sys as _sys

_main_code = compile('import argparse\nimport json\nimport os\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import random_split\nfrom torch_geometric.loader import DataLoader\nfrom torch_geometric.data import Data\nfrom sklearn.metrics import mean_absolute_error\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\n\n\n\n\n# ---------------------------------------------------------------------------\n# Pure-PyTorch radius graph (avoids pyg-lib dependency)\n# ---------------------------------------------------------------------------\n\ndef radius_graph_pytorch(\n    pos: torch.Tensor,\n    batch: torch.Tensor,\n    cutoff: float,\n    max_num_neighbors: int = 100,\n) -> torch.Tensor:\n    """Build edges for all atom pairs within `cutoff`, respecting batch boundaries.\n\n    Returns edge_index (2, E) with source->target edges.\n    """\n    num_nodes = pos.size(0)\n    device = pos.device\n\n    edge_list = []\n    for g in batch.unique():\n        mask = batch == g\n        idx = mask.nonzero(as_tuple=False).view(-1)\n        n = idx.size(0)\n        if n < 2:\n            continue\n        g_pos = pos[idx]  # (n, 3)\n        diffs = g_pos.unsqueeze(0) - g_pos.unsqueeze(1)  # (n, n, 3)\n        dists = diffs.norm(dim=-1)  # (n, n)\n        triu = torch.triu(dists, diagonal=1)\n        src_local, dst_local = torch.where((triu > 0) & (triu < cutoff))\n        src_global = idx[src_local]\n        dst_global = idx[dst_local]\n        edge_list.append(torch.stack([src_global, dst_global], dim=0))\n        edge_list.append(torch.stack([dst_global, src_global], dim=0))\n\n    if len(edge_list) == 0:\n        return torch.zeros((2, 0), dtype=torch.long, device=device)\n\n    edge_index = torch.cat(edge_list, dim=1)\n\n    if max_num_neighbors is not None and edge_index.size(1) > 0:\n        src = edge_index[0]\n        dst = edge_index[1]\n        dists = (pos[src] - pos[dst]).norm(dim=-1)\n        keep = []\n        for node in range(num_nodes):\n            node_mask = src == node\n            if node_mask.sum() <= max_num_neighbors:\n                keep.append(torch.where(node_mask)[0])\n            else:\n                node_dists = dists[node_mask]\n                _, topk_idx = torch.topk(node_dists, max_num_neighbors, largest=False)\n                keep.append(torch.where(node_mask)[0][topk_idx])\n        if keep:\n            keep = torch.cat(keep)\n            edge_index = edge_index[:, keep]\n\n    return edge_index\n\n\n# ---------------------------------------------------------------------------\n# SchNet building blocks\n# ---------------------------------------------------------------------------\n\nclass CosineCutoff(nn.Module):\n    """Cosine cutoff envelope: f(r) = 0.5 * (1 + cos(pi * r / cutoff))."""\n\n    def __init__(self, cutoff: float):\n        super().__init__()\n        self.cutoff = cutoff\n\n    def forward(self, distances: torch.Tensor) -> torch.Tensor:\n        return 0.5 * (1.0 + torch.cos(np.pi * distances / self.cutoff))\n\n\nclass RBFExpansion(nn.Module):\n    """Expand interatomic distances in a radial basis of Gaussians."""\n\n    def __init__(self, cutoff: float, num_gaussians: int):\n        super().__init__()\n        centers = torch.linspace(0, cutoff, num_gaussians)\n        self.register_buffer("centers", centers)\n        self.width = cutoff / num_gaussians\n\n    def forward(self, distances: torch.Tensor) -> torch.Tensor:\n        d = distances.unsqueeze(-1)\n        gamma = 1.0 / (2.0 * self.width ** 2)\n        return torch.exp(-gamma * (d - self.centers) ** 2)\n\n\nclass InteractionBlock(nn.Module):\n    """One SchNet continuous-filter interaction block."""\n\n    def __init__(self, num_filters: int, num_gaussians: int, cutoff: float):\n        super().__init__()\n        self.filter_network = nn.Sequential(\n            nn.Linear(num_gaussians, num_filters),\n            nn.Softplus(),\n            nn.Linear(num_filters, num_filters),\n        )\n        self.atomwise_mlp = nn.Sequential(\n            nn.Linear(num_filters, num_filters),\n            nn.Softplus(),\n            nn.Linear(num_filters, num_filters),\n        )\n        self.cutoff_fn = CosineCutoff(cutoff)\n\n    def forward(\n        self,\n        x: torch.Tensor,\n        edge_index: torch.Tensor,\n        distances: torch.Tensor,\n        rbf_features: torch.Tensor,\n    ) -> torch.Tensor:\n        src, dst = edge_index\n        W = self.filter_network(rbf_features)\n        W = W * self.cutoff_fn(distances).unsqueeze(-1)\n        messages = x[src] * W\n        aggr = torch.zeros_like(x)\n        aggr.index_add_(0, dst, messages)\n        return x + self.atomwise_mlp(aggr)\n\n\nclass SchNet(nn.Module):\n    """SchNet: continuous-filter convolutional neural network for 3D molecular data."""\n\n    def __init__(\n        self,\n        num_atom_types: int = 100,\n        embedding_dim: int = 128,\n        num_filters: int = 128,\n        num_interactions: int = 3,\n        num_gaussians: int = 50,\n        cutoff: float = 5.0,\n        num_outputs: int = 16,\n    ):\n        super().__init__()\n        self.cutoff = cutoff\n\n        self.embedding = nn.Embedding(num_atom_types, embedding_dim, padding_idx=0)\n\n        if embedding_dim != num_filters:\n            self.proj = nn.Linear(embedding_dim, num_filters)\n        else:\n            self.proj = nn.Identity()\n\n        self.rbf = RBFExpansion(cutoff, num_gaussians)\n\n        self.interactions = nn.ModuleList(\n            [\n                InteractionBlock(num_filters, num_gaussians, cutoff)\n                for _ in range(num_interactions)\n            ]\n        )\n\n        self.readout = nn.Sequential(\n            nn.Linear(num_filters, num_filters // 2),\n            nn.Softplus(),\n            nn.Linear(num_filters // 2, num_outputs),\n        )\n\n    def forward(self, data: Data) -> torch.Tensor:\n        z = data.z\n        pos = data.pos\n        batch = data.batch\n\n        edge_index = radius_graph_pytorch(pos, batch, self.cutoff, max_num_neighbors=100)\n\n        if edge_index.size(1) == 0:\n            x = self.embedding(z)\n            x = self.proj(x)\n            num_graphs = int(batch.max().item() + 1)\n            graph_features = torch.zeros(\n                num_graphs, x.size(1), device=x.device, dtype=x.dtype\n            )\n            graph_features.index_add_(0, batch, x)\n            return self.readout(graph_features)\n\n        src, dst = edge_index\n        distances = (pos[src] - pos[dst]).norm(dim=-1)\n\n        rbf_features = self.rbf(distances)\n\n        x = self.embedding(z)\n        x = self.proj(x)\n\n        for interaction in self.interactions:\n            x = interaction(x, edge_index, distances, rbf_features)\n\n        num_graphs = int(batch.max().item() + 1)\n        graph_features = torch.zeros(\n            num_graphs, x.size(1), device=x.device, dtype=x.dtype\n        )\n        graph_features.index_add_(0, batch, x)\n\n        return self.readout(graph_features)\n\n\n# ---------------------------------------------------------------------------\n# Dataset\n# ---------------------------------------------------------------------------\n\nclass QM8Dataset(torch.utils.data.Dataset):\n    def __init__(self, molecules, label_keys=None):\n        self.molecules = molecules\n        if label_keys is None:\n            label_keys = sorted(molecules[0].labels.keys())\n        self.label_keys = label_keys\n\n    def __len__(self):\n        return len(self.molecules)\n\n    def __getitem__(self, idx):\n        mol = self.molecules[idx]\n        z = torch.tensor(mol.atomic_numbers, dtype=torch.long)\n        pos = torch.tensor(mol.positions, dtype=torch.float32)\n        labels = torch.tensor(\n            [mol.labels[k] for k in self.label_keys], dtype=torch.float32\n        )\n        # Shape (1, num_outputs) so batching stacks to (batch_size, num_outputs)\n        return Data(z=z, pos=pos, y=labels.unsqueeze(0))\n\n\nclass NormalizedDataset(torch.utils.data.Dataset):\n    """Wraps a subset of QM8Dataset and normalises targets."""\n\n    def __init__(self, base_dataset, indices, mean, std):\n        self.base = base_dataset\n        self.indices = indices\n        self.mean = mean\n        self.std = std\n\n    def __len__(self):\n        return len(self.indices)\n\n    def __getitem__(self, idx):\n        data = self.base[self.indices[idx]]\n        data.y = (data.y - self.mean) / self.std\n        return data\n\n\n# ---------------------------------------------------------------------------\n# Training helpers\n# ---------------------------------------------------------------------------\n\ndef train_epoch(model, loader, optimizer, criterion, device):\n    model.train()\n    total_loss = 0.0\n    n_graphs = 0\n    for batch in loader:\n        batch = batch.to(device)\n        optimizer.zero_grad()\n        pred = model(batch)\n        loss = criterion(pred, batch.y)\n        loss.backward()\n        optimizer.step()\n        total_loss += loss.item() * batch.num_graphs\n        n_graphs += batch.num_graphs\n    return total_loss / n_graphs\n\n\n@torch.no_grad()\ndef evaluate_model(model, loader, criterion, device):\n    model.eval()\n    total_loss = 0.0\n    n_graphs = 0\n    all_preds = []\n    all_targets = []\n    for batch in loader:\n        batch = batch.to(device)\n        pred = model(batch)\n        loss = criterion(pred, batch.y)\n        total_loss += loss.item() * batch.num_graphs\n        n_graphs += batch.num_graphs\n        all_preds.append(pred.cpu().numpy())\n        all_targets.append(batch.y.cpu().numpy())\n    all_preds = np.concatenate(all_preds, axis=0)\n    all_targets = np.concatenate(all_targets, axis=0)\n    mae = mean_absolute_error(all_targets, all_preds)\n    return total_loss / n_graphs, mae, all_preds, all_targets\n\n\n# ---------------------------------------------------------------------------\n# Main\n# ---------------------------------------------------------------------------\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\n        "--stage", type=str, required=True, choices=["baseline", "train", "evaluate"]\n    )\n    parser.add_argument(\n        "--device",\n        type=str,\n        default="cuda" if torch.cuda.is_available() else "cpu",\n    )\n    args = parser.parse_args()\n\n    device = torch.device(args.device)\n    print(f"Using device: {device}")\n\n    # ---- Load data -----------------------------------------------------------\n    print("Loading QM8 data ...")\n    molecules = load_qm8()\n    print(f"Loaded {len(molecules)} molecules")\n\n    label_keys = sorted(molecules[0].labels.keys())\n    num_outputs = len(label_keys)\n    print(f"Target properties ({num_outputs}): {label_keys}")\n\n    dataset = QM8Dataset(molecules, label_keys)\n\n    # 80 / 10 / 10 split\n    n = len(dataset)\n    n_train = int(0.8 * n)\n    n_val = int(0.1 * n)\n    n_test = n - n_train - n_val\n\n    train_ds, val_ds, test_ds = random_split(\n        dataset,\n        [n_train, n_val, n_test],\n        generator=torch.Generator().manual_seed(42),\n    )\n\n    # Normalisation stats from training set\n    train_labels = np.stack(\n        [dataset[i].y.numpy().squeeze(0) for i in train_ds.indices], axis=0\n    )\n    mean = torch.tensor(train_labels.mean(axis=0), dtype=torch.float32)\n    std = torch.tensor(train_labels.std(axis=0), dtype=torch.float32)\n    std = torch.clamp(std, min=1e-8)\n\n    print(f"Target mean  (first 4): {mean[:4].tolist()}")\n    print(f"Target std   (first 4): {std[:4].tolist()}")\n\n    train_norm = NormalizedDataset(dataset, train_ds.indices, mean, std)\n    val_norm = NormalizedDataset(dataset, val_ds.indices, mean, std)\n    test_norm = NormalizedDataset(dataset, test_ds.indices, mean, std)\n\n    # ---- Model ---------------------------------------------------------------\n    model = SchNet(\n        num_atom_types=100,\n        embedding_dim=128,\n        num_filters=128,\n        num_interactions=3,\n        num_gaussians=50,\n        cutoff=5.0,\n        num_outputs=num_outputs,\n    ).to(device)\n\n    n_params = sum(p.numel() for p in model.parameters())\n    print(f"Model parameters: {n_params:,}")\n\n    # =====================================================================\n    #  BASELINE\n    # =====================================================================\n    if args.stage == "baseline":\n        n_base = min(200, n_train)\n        n_base_val = min(50, n_val)\n\n        base_train = torch.utils.data.Subset(train_norm, range(n_base))\n        base_val = torch.utils.data.Subset(val_norm, range(n_base_val))\n\n        train_loader = DataLoader(base_train, batch_size=32, shuffle=True)\n        val_loader = DataLoader(base_val, batch_size=32, shuffle=False)\n\n        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n        criterion = nn.MSELoss()\n\n        print("\\n=== BASELINE (5 epochs on small subset) ===")\n        for epoch in range(5):\n            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)\n            val_loss, val_mae, _, _ = evaluate_model(\n                model, val_loader, criterion, device\n            )\n            print(\n                f"  epoch {epoch+1:2d}  "\n                f"train_loss={train_loss:.6f}  "\n                f"val_loss={val_loss:.6f}  "\n                f"val_mae={val_mae:.6f}"\n            )\n\n        print("Baseline complete.")\n\n    # =====================================================================\n    #  TRAIN\n    # =====================================================================\n    elif args.stage == "train":\n        train_loader = DataLoader(train_norm, batch_size=32, shuffle=True)\n        val_loader = DataLoader(val_norm, batch_size=32, shuffle=False)\n\n        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6\n        )\n        criterion = nn.MSELoss()\n\n        patience = 15\n        best_val_loss = float("inf")\n        best_epoch = 0\n        epochs_no_improve = 0\n\n        os.makedirs("checkpoints", exist_ok=True)\n\n        print("\\n=== TRAIN (early stopping, patience=15) ===")\n        for epoch in range(1, 501):\n            train_loss = train_epoch(\n                model, train_loader, optimizer, criterion, device\n            )\n            val_loss, val_mae, _, _ = evaluate_model(\n                model, val_loader, criterion, device\n            )\n            scheduler.step(val_loss)\n\n            lr = optimizer.param_groups[0]["lr"]\n            print(\n                f"  epoch {epoch:3d}  "\n                f"train_loss={train_loss:.6f}  "\n                f"val_loss={val_loss:.6f}  "\n                f"val_mae={val_mae:.6f}  "\n                f"lr={lr:.2e}"\n            )\n\n            if val_loss < best_val_loss:\n                best_val_loss = val_loss\n                best_epoch = epoch\n                epochs_no_improve = 0\n                torch.save(\n                    {\n                        "epoch": epoch,\n                        "model_state_dict": model.state_dict(),\n                        "optimizer_state_dict": optimizer.state_dict(),\n                        "val_loss": val_loss,\n                        "mean": mean,\n                        "std": std,\n                        "label_keys": label_keys,\n                    },\n                    "checkpoints/best_model.pt",\n                )\n                print(f"    -> best model saved (val_loss={val_loss:.6f})")\n            else:\n                epochs_no_improve += 1\n                if epochs_no_improve >= patience:\n                    print(\n                        f"Early stopping after {epoch} epochs "\n                        f"(best was epoch {best_epoch}, val_loss={best_val_loss:.6f})"\n                    )\n                    break\n\n        print(\n            f"Training finished.  Best epoch {best_epoch}  "\n            f"val_loss={best_val_loss:.6f}"\n        )\n\n    # =====================================================================\n    #  EVALUATE\n    # =====================================================================\n    elif args.stage == "evaluate":\n        ckpt = torch.load("checkpoints/best_model.pt", map_location=device)\n        model.load_state_dict(ckpt["model_state_dict"])\n        mean = ckpt["mean"].to(device)\n        std = ckpt["std"].to(device)\n        label_keys = ckpt["label_keys"]\n\n        test_loader = DataLoader(test_norm, batch_size=32, shuffle=False)\n        criterion = nn.MSELoss()\n\n        test_loss, test_mae_norm, preds_norm, targets_norm = evaluate_model(\n            model, test_loader, criterion, device\n        )\n\n        # Denormalise\n        mean_np = mean.cpu().numpy()\n        std_np = std.cpu().numpy()\n        preds = preds_norm * std_np + mean_np\n        targets = targets_norm * std_np + mean_np\n\n        overall_mae = float(mean_absolute_error(targets, preds))\n        per_prop_mae = np.mean(np.abs(preds - targets), axis=0)\n\n        print(f"\\n=== EVALUATE ===")\n        print(f"Test MSE (normalised): {test_loss:.6f}")\n        print(f"Test MAE  (overall)  : {overall_mae:.6f}")\n        print(f"Per-property MAE:")\n        for i, key in enumerate(label_keys):\n            print(f"  {key:20s}  {per_prop_mae[i]:.6f}")\n\n        # ---- results.json -------------------------------------------------\n        results = {\n            "test_mse_normalised": float(test_loss),\n            "test_mae_overall": overall_mae,\n            "per_property_mae": {\n                key: float(per_prop_mae[i]) for i, key in enumerate(label_keys)\n            },\n        }\n\n        # ---- Diagnostics ---------------------------------------------------\n        method_groups: dict[str, list[int]] = {}\n        for i, key in enumerate(label_keys):\n            method = key.split("-", 1)[1] if "-" in key else "unknown"\n            method_groups.setdefault(method, []).append(i)\n\n        print("\\nPer-method MAE:")\n        method_mae = {}\n        for method, idxs in method_groups.items():\n            m = float(np.mean(per_prop_mae[idxs]))\n            method_mae[method] = m\n            print(f"  {method:10s}  {m:.6f}")\n        results["per_method_mae"] = method_mae\n\n        e_idx = [i for i, k in enumerate(label_keys) if k.startswith("E")]\n        f_idx = [i for i, k in enumerate(label_keys) if k.startswith("f")]\n        e_mae = float(np.mean(per_prop_mae[e_idx])) if e_idx else 0.0\n        f_mae = float(np.mean(per_prop_mae[f_idx])) if f_idx else 0.0\n        print(f"\\nExcitation energies   MAE: {e_mae:.6f}")\n        print(f"Oscillator strengths  MAE: {f_mae:.6f}")\n        results["excitation_energy_mae"] = e_mae\n        results["oscillator_strength_mae"] = f_mae\n\n        with open("results.json", "w") as f:\n            json.dump(results, f, indent=2)\n        print("\\nResults written to results.json")\n\n        # ---- Plots ---------------------------------------------------------\n        test_indices = test_ds.indices\n        heavy_counts = np.array(\n            [\n                sum(1 for an in molecules[i].atomic_numbers if an > 1)\n                for i in test_indices\n            ]\n        )\n        per_mol_mae = np.mean(np.abs(preds - targets), axis=1)\n\n        fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n\n        axes[0].hist(per_mol_mae, bins=50, edgecolor="black", alpha=0.7)\n        axes[0].set_xlabel("Per-molecule MAE")\n        axes[0].set_ylabel("Count")\n        axes[0].set_title("Error distribution")\n\n        unique_hc = np.unique(heavy_counts)\n        mean_by_size = [per_mol_mae[heavy_counts == c].mean() for c in unique_hc]\n        axes[1].bar(unique_hc, mean_by_size, edgecolor="black", alpha=0.7)\n        axes[1].set_xlabel("Heavy atom count")\n        axes[1].set_ylabel("Mean MAE")\n        axes[1].set_title("MAE vs molecular size")\n\n        plt.tight_layout()\n        plt.savefig("error_analysis.png", dpi=150)\n        print("Saved error_analysis.png")\n\n        method_colors = {\n            "CC2": "tab:blue",\n            "PBE0": "tab:green",\n            "PBE0.1": "tab:orange",\n            "CAM": "tab:red",\n        }\n\n        def _method_of(key: str) -> str:\n            return key.split("-", 1)[1] if "-" in key else "unknown"\n\n        colors = [method_colors.get(_method_of(k), "gray") for k in label_keys]\n\n        fig, ax = plt.subplots(figsize=(14, 6))\n        ax.bar(range(len(label_keys)), per_prop_mae, color=colors, edgecolor="black")\n        ax.set_xticks(range(len(label_keys)))\n        ax.set_xticklabels(label_keys, rotation=45, ha="right", fontsize=8)\n        ax.set_ylabel("MAE")\n        ax.set_title("Per-property MAE")\n        plt.tight_layout()\n        plt.savefig("per_property_mae.png", dpi=150)\n        print("Saved per_property_mae.png")\n\n\nif __name__ == "__main__":\n    main()', "main.py", "exec")

_sys.argv = ["main.py", "--stage", "train"]
exec(_main_code, globals())

_sys.argv = ["main.py", "--stage", "evaluate"]
exec(_main_code, globals())
