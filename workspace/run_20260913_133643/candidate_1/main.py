#!/usr/bin/env python3
"""ECFP fingerprints + LightGBM baseline for QM8 quantum chemical properties.

Stages:
  baseline  – quick sanity check on 500 molecules
  train     – full 80/10/10 split, one LightGBM regressor per target, early stopping
  evaluate  – load best models, compute MAE & diagnostics, write results.json
"""

import argparse
import json
import os
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from qm8_data import load_qm8

# ---------------------------------------------------------------------------
# The 16 QM8 target properties (pandas auto-suffixes duplicate column names)
# Columns 2-5:   RI-CC2/def2TZVP
# Columns 6-9:   LR-TDPBE0/def2SVP   → E1-PBE0, E2-PBE0, f1-PBE0, f2-PBE0
# Columns 10-13: LR-TDPBE0/def2TZVP  → E1-PBE0.1, E2-PBE0.1, f1-PBE0.1, f2-PBE0.1
# Columns 14-17: LR-TDCAM-B3LYP/def2TZVP
# ---------------------------------------------------------------------------
TARGETS = [
    "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2",
    "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0",          # def2SVP
    "E1-PBE0.1", "E2-PBE0.1", "f1-PBE0.1", "f2-PBE0.1",  # def2TZVP
    "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM",
]

# Fixed hyperparameters – not tuned, one reasonable setting
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
RANDOM_SEED = 42
BASELINE_N = 500
EARLY_STOPPING_ROUNDS = 50

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "num_threads": 1,
    "seed": RANDOM_SEED,
}

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_models.pkl")
SPLIT_FILE = os.path.join(MODEL_DIR, "splits.npz")


# ---------------------------------------------------------------------------
# Fingerprint conversion
# ---------------------------------------------------------------------------

def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Convert a SMILES string to an ECFP/Morgan fingerprint bit vector.

    Returns None if the SMILES fails RDKit sanitisation (handles the
    edge case where load_qm8 passes a valid molecule that later fails
    on round-trip, though in practice load_qm8 already filters these).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )
    arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def prepare_data(molecules: list) -> tuple:
    """Extract fingerprint matrix, label dict, and heavy-atom counts."""
    X_list = []
    y_dict = {t: [] for t in TARGETS}
    heavy_atom_counts = []

    for mol in molecules:
        fp = smiles_to_fp(mol.smiles)
        if fp is None:
            continue
        X_list.append(fp)
        for t in TARGETS:
            y_dict[t].append(mol.labels[t])
        # Heavy atoms = any atom with Z > 1 (i.e. not hydrogen)
        heavy_atom_counts.append(sum(1 for z in mol.atomic_numbers if z > 1))

    X = np.array(X_list, dtype=np.float32)
    return X, y_dict, heavy_atom_counts


# ---------------------------------------------------------------------------
# Stage: baseline
# ---------------------------------------------------------------------------

def run_baseline() -> None:
    """Quick end-to-end sanity check on a 500-molecule random subset."""
    molecules = load_qm8()
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(molecules), min(BASELINE_N, len(molecules)), replace=False)
    subset = [molecules[i] for i in idx]

    X, y_dict, _ = prepare_data(subset)
    print(f"Baseline: {len(X)} molecules, {X.shape[1]} fingerprint bits")

    maes = []
    for target in TARGETS:
        y = np.array(y_dict[target], dtype=np.float32)
        model = lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=15,
            learning_rate=0.1,
            verbose=-1,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(X, y)
        pred = model.predict(X)
        mae = float(mean_absolute_error(y, pred))
        maes.append(mae)
        print(f"  {target}: train MAE = {mae:.4f}")

    overall = float(np.mean(maes))
    print(f"Baseline overall MAE (train, {BASELINE_N} mols): {overall:.4f}")
    print("Baseline completed successfully.")


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------

def run_train() -> None:
    """Full training with 80/10/10 split, early stopping, best model saved."""
    molecules = load_qm8()
    X, y_dict, _ = prepare_data(molecules)
    n = len(X)
    print(f"Total usable molecules: {n}")

    # 80 / 10 / 10 random split
    indices = np.arange(n)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=RANDOM_SEED
    )
    print(
        f"Split – train: {len(train_idx)}, val: {len(val_idx)}, "
        f"test: {len(test_idx)}"
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train_dict = {t: np.array(y_dict[t])[train_idx].astype(np.float32)
                    for t in TARGETS}
    y_val_dict = {t: np.array(y_dict[t])[val_idx].astype(np.float32)
                  for t in TARGETS}

    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {}
    for target in TARGETS:
        print(f"\n--- Training {target} ---")
        model = lgb.LGBMRegressor(
            n_estimators=5000,
            num_leaves=LGBM_PARAMS["num_leaves"],
            learning_rate=LGBM_PARAMS["learning_rate"],
            feature_fraction=LGBM_PARAMS["feature_fraction"],
            bagging_fraction=LGBM_PARAMS["bagging_fraction"],
            bagging_freq=LGBM_PARAMS["bagging_freq"],
            boosting_type=LGBM_PARAMS["boosting_type"],
            objective=LGBM_PARAMS["objective"],
            metric=LGBM_PARAMS["metric"],
            verbose=-1,
            random_state=RANDOM_SEED,
            n_jobs=1,
            num_threads=1,
        )
        model.fit(
            X_train, y_train_dict[target],
            eval_set=[(X_val, y_val_dict[target])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                       lgb.log_evaluation(period=100)],
        )

        models[target] = model
        val_pred = model.predict(X_val)
        val_mae = float(mean_absolute_error(y_val_dict[target], val_pred))
        print(
            f"  {target}: best_iteration = {model.best_iteration_}, "
            f"val MAE = {val_mae:.4f}"
        )

    # Persist models and split indices for the evaluate stage
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(models, f)
    np.savez(SPLIT_FILE, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    print(f"\nModels saved to {MODEL_FILE}")
    print("Training completed.")


# ---------------------------------------------------------------------------
# Stage: evaluate
# ---------------------------------------------------------------------------

def run_evaluate() -> None:
    """Load trained models, evaluate on test set, write results.json."""
    # --- Reload everything needed for a clean evaluation ---
    with open(MODEL_FILE, "rb") as f:
        models = pickle.load(f)
    splits = np.load(SPLIT_FILE)
    test_idx = splits["test_idx"]

    molecules = load_qm8()
    X_all, y_dict_all, heavy_atom_counts_all = prepare_data(molecules)

    X_test = X_all[test_idx]
    y_test_dict = {
        t: np.array(y_dict_all[t])[test_idx].astype(np.float32) for t in TARGETS
    }
    heavy_test = np.array(heavy_atom_counts_all)[test_idx]

    # --- Per-target MAE ---
    per_target_mae = {}
    all_errors = []

    for target in TARGETS:
        y_true = y_test_dict[target]
        y_pred = models[target].predict(X_test)
        mae = float(mean_absolute_error(y_true, y_pred))
        per_target_mae[target] = mae
        all_errors.extend(np.abs(y_true - y_pred).tolist())

    overall_mae = float(np.mean(list(per_target_mae.values())))

    # --- Error distribution percentiles ---
    all_errors = np.array(all_errors)
    percentiles = {
        "p50": float(np.percentile(all_errors, 50)),
        "p90": float(np.percentile(all_errors, 90)),
        "p95": float(np.percentile(all_errors, 95)),
        "p99": float(np.percentile(all_errors, 99)),
    }

    # --- Per-molecule mean absolute error vs heavy-atom count ---
    mol_errors = np.zeros(len(X_test), dtype=np.float32)
    for i in range(len(X_test)):
        row_errs = []
        for target in TARGETS:
            y_true = y_test_dict[target][i]
            y_pred = models[target].predict(X_test[i : i + 1])[0]
            row_errs.append(abs(y_true - y_pred))
        mol_errors[i] = float(np.mean(row_errs))

    # Pearson correlation
    corr_matrix = np.corrcoef(heavy_test, mol_errors)
    corr = float(corr_matrix[0, 1])

    # --- Write results ---
    results = {
        "overall_mae": overall_mae,
        "per_target_mae": per_target_mae,
        "error_percentiles": percentiles,
        "error_heavy_atom_correlation": corr,
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Overall test MAE (avg over 16 targets): {overall_mae:.4f}")
    print("Per-target MAE:")
    for t, m in per_target_mae.items():
        print(f"  {t}: {m:.4f}")
    print(
        f"Error percentiles: "
        f"p50={percentiles['p50']:.4f}, p90={percentiles['p90']:.4f}, "
        f"p95={percentiles['p95']:.4f}, p99={percentiles['p99']:.4f}"
    )
    print(f"Pearson correlation (error vs heavy-atom count): {corr:.4f}")
    print("Results written to results.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ECFP+LightGBM QM8 baseline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["baseline", "train", "evaluate"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    if args.stage == "baseline":
        run_baseline()
    elif args.stage == "train":
        run_train()
    elif args.stage == "evaluate":
        run_evaluate()


if __name__ == "__main__":
    main()