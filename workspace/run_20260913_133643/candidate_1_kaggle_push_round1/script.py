import subprocess as _subprocess, sys as _sys
_subprocess.check_call([_sys.executable, '-m', 'pip', 'install', '-q'] + ['lightgbm', 'rdkit', 'torch_geometric'])

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

_main_code = compile('#!/usr/bin/env python3\n"""ECFP fingerprints + LightGBM baseline for QM8 quantum chemical properties.\n\nStages:\n  baseline  – quick sanity check on 500 molecules\n  train     – full 80/10/10 split, one LightGBM regressor per target, early stopping\n  evaluate  – load best models, compute MAE & diagnostics, write results.json\n"""\n\nimport argparse\nimport json\nimport os\nimport pickle\n\nimport numpy as np\nfrom rdkit import Chem\nfrom rdkit.Chem import AllChem\nfrom rdkit import DataStructs\nfrom sklearn.metrics import mean_absolute_error\nfrom sklearn.model_selection import train_test_split\n\nimport lightgbm as lgb\n\n\n# ---------------------------------------------------------------------------\n# The 16 QM8 target properties (pandas auto-suffixes duplicate column names)\n# Columns 2-5:   RI-CC2/def2TZVP\n# Columns 6-9:   LR-TDPBE0/def2SVP   → E1-PBE0, E2-PBE0, f1-PBE0, f2-PBE0\n# Columns 10-13: LR-TDPBE0/def2TZVP  → E1-PBE0.1, E2-PBE0.1, f1-PBE0.1, f2-PBE0.1\n# Columns 14-17: LR-TDCAM-B3LYP/def2TZVP\n# ---------------------------------------------------------------------------\nTARGETS = [\n    "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2",\n    "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0",          # def2SVP\n    "E1-PBE0.1", "E2-PBE0.1", "f1-PBE0.1", "f2-PBE0.1",  # def2TZVP\n    "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM",\n]\n\n# Fixed hyperparameters – not tuned, one reasonable setting\nFINGERPRINT_RADIUS = 2\nFINGERPRINT_BITS = 2048\nRANDOM_SEED = 42\nBASELINE_N = 500\nEARLY_STOPPING_ROUNDS = 50\n\nLGBM_PARAMS = {\n    "objective": "regression",\n    "metric": "mae",\n    "boosting_type": "gbdt",\n    "num_leaves": 31,\n    "learning_rate": 0.05,\n    "feature_fraction": 0.9,\n    "bagging_fraction": 0.8,\n    "bagging_freq": 5,\n    "verbose": -1,\n    "num_threads": 1,\n    "seed": RANDOM_SEED,\n}\n\nMODEL_DIR = "models"\nMODEL_FILE = os.path.join(MODEL_DIR, "lgbm_models.pkl")\nSPLIT_FILE = os.path.join(MODEL_DIR, "splits.npz")\n\n\n# ---------------------------------------------------------------------------\n# Fingerprint conversion\n# ---------------------------------------------------------------------------\n\ndef smiles_to_fp(smiles: str) -> np.ndarray | None:\n    """Convert a SMILES string to an ECFP/Morgan fingerprint bit vector.\n\n    Returns None if the SMILES fails RDKit sanitisation (handles the\n    edge case where load_qm8 passes a valid molecule that later fails\n    on round-trip, though in practice load_qm8 already filters these).\n    """\n    mol = Chem.MolFromSmiles(smiles)\n    if mol is None:\n        return None\n    fp = AllChem.GetMorganFingerprintAsBitVect(\n        mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS\n    )\n    arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)\n    DataStructs.ConvertToNumpyArray(fp, arr)\n    return arr\n\n\ndef prepare_data(molecules: list) -> tuple:\n    """Extract fingerprint matrix, label dict, and heavy-atom counts."""\n    X_list = []\n    y_dict = {t: [] for t in TARGETS}\n    heavy_atom_counts = []\n\n    for mol in molecules:\n        fp = smiles_to_fp(mol.smiles)\n        if fp is None:\n            continue\n        X_list.append(fp)\n        for t in TARGETS:\n            y_dict[t].append(mol.labels[t])\n        # Heavy atoms = any atom with Z > 1 (i.e. not hydrogen)\n        heavy_atom_counts.append(sum(1 for z in mol.atomic_numbers if z > 1))\n\n    X = np.array(X_list, dtype=np.float32)\n    return X, y_dict, heavy_atom_counts\n\n\n# ---------------------------------------------------------------------------\n# Stage: baseline\n# ---------------------------------------------------------------------------\n\ndef run_baseline() -> None:\n    """Quick end-to-end sanity check on a 500-molecule random subset."""\n    molecules = load_qm8()\n    rng = np.random.RandomState(RANDOM_SEED)\n    idx = rng.choice(len(molecules), min(BASELINE_N, len(molecules)), replace=False)\n    subset = [molecules[i] for i in idx]\n\n    X, y_dict, _ = prepare_data(subset)\n    print(f"Baseline: {len(X)} molecules, {X.shape[1]} fingerprint bits")\n\n    maes = []\n    for target in TARGETS:\n        y = np.array(y_dict[target], dtype=np.float32)\n        model = lgb.LGBMRegressor(\n            n_estimators=100,\n            num_leaves=15,\n            learning_rate=0.1,\n            verbose=-1,\n            random_state=RANDOM_SEED,\n            n_jobs=1,\n        )\n        model.fit(X, y)\n        pred = model.predict(X)\n        mae = float(mean_absolute_error(y, pred))\n        maes.append(mae)\n        print(f"  {target}: train MAE = {mae:.4f}")\n\n    overall = float(np.mean(maes))\n    print(f"Baseline overall MAE (train, {BASELINE_N} mols): {overall:.4f}")\n    print("Baseline completed successfully.")\n\n\n# ---------------------------------------------------------------------------\n# Stage: train\n# ---------------------------------------------------------------------------\n\ndef run_train() -> None:\n    """Full training with 80/10/10 split, early stopping, best model saved."""\n    molecules = load_qm8()\n    X, y_dict, _ = prepare_data(molecules)\n    n = len(X)\n    print(f"Total usable molecules: {n}")\n\n    # 80 / 10 / 10 random split\n    indices = np.arange(n)\n    train_idx, temp_idx = train_test_split(\n        indices, test_size=0.20, random_state=RANDOM_SEED\n    )\n    val_idx, test_idx = train_test_split(\n        temp_idx, test_size=0.50, random_state=RANDOM_SEED\n    )\n    print(\n        f"Split – train: {len(train_idx)}, val: {len(val_idx)}, "\n        f"test: {len(test_idx)}"\n    )\n\n    X_train = X[train_idx]\n    X_val = X[val_idx]\n    X_test = X[test_idx]\n\n    y_train_dict = {t: np.array(y_dict[t])[train_idx].astype(np.float32)\n                    for t in TARGETS}\n    y_val_dict = {t: np.array(y_dict[t])[val_idx].astype(np.float32)\n                  for t in TARGETS}\n\n    os.makedirs(MODEL_DIR, exist_ok=True)\n\n    models = {}\n    for target in TARGETS:\n        print(f"\\n--- Training {target} ---")\n        model = lgb.LGBMRegressor(\n            n_estimators=5000,\n            num_leaves=LGBM_PARAMS["num_leaves"],\n            learning_rate=LGBM_PARAMS["learning_rate"],\n            feature_fraction=LGBM_PARAMS["feature_fraction"],\n            bagging_fraction=LGBM_PARAMS["bagging_fraction"],\n            bagging_freq=LGBM_PARAMS["bagging_freq"],\n            boosting_type=LGBM_PARAMS["boosting_type"],\n            objective=LGBM_PARAMS["objective"],\n            metric=LGBM_PARAMS["metric"],\n            verbose=-1,\n            random_state=RANDOM_SEED,\n            n_jobs=1,\n            num_threads=1,\n        )\n        model.fit(\n            X_train, y_train_dict[target],\n            eval_set=[(X_val, y_val_dict[target])],\n            eval_metric="mae",\n            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS),\n                       lgb.log_evaluation(period=100)],\n        )\n\n        models[target] = model\n        val_pred = model.predict(X_val)\n        val_mae = float(mean_absolute_error(y_val_dict[target], val_pred))\n        print(\n            f"  {target}: best_iteration = {model.best_iteration_}, "\n            f"val MAE = {val_mae:.4f}"\n        )\n\n    # Persist models and split indices for the evaluate stage\n    with open(MODEL_FILE, "wb") as f:\n        pickle.dump(models, f)\n    np.savez(SPLIT_FILE, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)\n\n    print(f"\\nModels saved to {MODEL_FILE}")\n    print("Training completed.")\n\n\n# ---------------------------------------------------------------------------\n# Stage: evaluate\n# ---------------------------------------------------------------------------\n\ndef run_evaluate() -> None:\n    """Load trained models, evaluate on test set, write results.json."""\n    # --- Reload everything needed for a clean evaluation ---\n    with open(MODEL_FILE, "rb") as f:\n        models = pickle.load(f)\n    splits = np.load(SPLIT_FILE)\n    test_idx = splits["test_idx"]\n\n    molecules = load_qm8()\n    X_all, y_dict_all, heavy_atom_counts_all = prepare_data(molecules)\n\n    X_test = X_all[test_idx]\n    y_test_dict = {\n        t: np.array(y_dict_all[t])[test_idx].astype(np.float32) for t in TARGETS\n    }\n    heavy_test = np.array(heavy_atom_counts_all)[test_idx]\n\n    # --- Per-target MAE ---\n    per_target_mae = {}\n    all_errors = []\n\n    for target in TARGETS:\n        y_true = y_test_dict[target]\n        y_pred = models[target].predict(X_test)\n        mae = float(mean_absolute_error(y_true, y_pred))\n        per_target_mae[target] = mae\n        all_errors.extend(np.abs(y_true - y_pred).tolist())\n\n    overall_mae = float(np.mean(list(per_target_mae.values())))\n\n    # --- Error distribution percentiles ---\n    all_errors = np.array(all_errors)\n    percentiles = {\n        "p50": float(np.percentile(all_errors, 50)),\n        "p90": float(np.percentile(all_errors, 90)),\n        "p95": float(np.percentile(all_errors, 95)),\n        "p99": float(np.percentile(all_errors, 99)),\n    }\n\n    # --- Per-molecule mean absolute error vs heavy-atom count ---\n    mol_errors = np.zeros(len(X_test), dtype=np.float32)\n    for i in range(len(X_test)):\n        row_errs = []\n        for target in TARGETS:\n            y_true = y_test_dict[target][i]\n            y_pred = models[target].predict(X_test[i : i + 1])[0]\n            row_errs.append(abs(y_true - y_pred))\n        mol_errors[i] = float(np.mean(row_errs))\n\n    # Pearson correlation\n    corr_matrix = np.corrcoef(heavy_test, mol_errors)\n    corr = float(corr_matrix[0, 1])\n\n    # --- Write results ---\n    results = {\n        "overall_mae": overall_mae,\n        "per_target_mae": per_target_mae,\n        "error_percentiles": percentiles,\n        "error_heavy_atom_correlation": corr,\n    }\n\n    with open("results.json", "w") as f:\n        json.dump(results, f, indent=2)\n\n    print(f"Overall test MAE (avg over 16 targets): {overall_mae:.4f}")\n    print("Per-target MAE:")\n    for t, m in per_target_mae.items():\n        print(f"  {t}: {m:.4f}")\n    print(\n        f"Error percentiles: "\n        f"p50={percentiles[\'p50\']:.4f}, p90={percentiles[\'p90\']:.4f}, "\n        f"p95={percentiles[\'p95\']:.4f}, p99={percentiles[\'p99\']:.4f}"\n    )\n    print(f"Pearson correlation (error vs heavy-atom count): {corr:.4f}")\n    print("Results written to results.json")\n\n\n# ---------------------------------------------------------------------------\n# CLI\n# ---------------------------------------------------------------------------\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(description="ECFP+LightGBM QM8 baseline")\n    parser.add_argument(\n        "--stage",\n        required=True,\n        choices=["baseline", "train", "evaluate"],\n        help="Which stage to run",\n    )\n    args = parser.parse_args()\n\n    if args.stage == "baseline":\n        run_baseline()\n    elif args.stage == "train":\n        run_train()\n    elif args.stage == "evaluate":\n        run_evaluate()\n\n\nif __name__ == "__main__":\n    main()', "main.py", "exec")

_sys.argv = ["main.py", "--stage", "train"]
exec(_main_code, globals())

_sys.argv = ["main.py", "--stage", "evaluate"]
exec(_main_code, globals())
