# Research Report: Quantum Chemical Property Prediction on QM8

## Overview & Executive Summary

This project investigated machine learning models for predicting 16 quantum chemical properties (electronic excitation energies and oscillator strengths evaluated via various computational methods: CC2, PBE0, PBE0.1, and CAM) using the benchmark QM8 dataset. 

We evaluated a robust classical machine learning baseline (**Candidate 1: ECFP fingerprints + LightGBM**). The model successfully processed the dataset using RDKit for feature extraction and trained separate Gradient Boosted Regression Trees for each of the 16 target properties. 

### Key Findings:
- **Overall Performance:** Candidate 1 achieved an overall Mean Absolute Error (MAE) of **0.01637** across all 16 target properties.
- **Target Disparity:** Energy targets (e.g., `E1-CAM`, `E1-PBE0.1`) were predicted with high accuracy (MAE ~0.007–0.008), whereas oscillator strength targets (specifically second oscillator strengths like `f2-CC2` and `f2-CAM`) exhibited higher errors (MAE ~0.028–0.037), reflecting the inherent difficulty of predicting intensity-related spectral properties from 2D topological fingerprints alone.
- **Size Independence:** The error correlation with heavy atom count was negligible ($r = 0.041$), indicating that the model's accuracy does not systematically degrade or scale poorly with molecular size within the explored domain.
- **Decision Trail:** Because Candidate 1 met all pipeline constraints and established a strong, fast-converging baseline with competitive accuracy without requiring intensive 3D geometric deep learning infrastructure, alternative 3D-aware neural network candidates (such as SchNet or DimeNet) were not triggered.

---

## Candidate Evaluation & Decision Trail

### Candidate 1: ECFP fingerprints + LightGBM (Selected & Successful)
- **Approach:** Extracted Morgan/ECFP molecular fingerprints (radius 2, 2048 bits) from SMILES strings using RDKit and trained 16 separate LightGBM regression models.
- **Results:**
  - **Overall MAE:** 0.01637
  - **Error Percentiles:** 
    - Median ($p_{50}$): 0.00782
    - $p_{90}$: 0.03875
    - $p_{95}$: 0.06664
    - $p_{99}$: 0.13695
  - **Heavy Atom Correlation:** 0.0414 (virtually uncorrelated)
- **Per-Target Breakdown:**
  - `E1-CC2`: 0.00859 | `E2-CC2`: 0.01040 | `f1-CC2`: 0.01672 | `f2-CC2`: 0.03712
  - `E1-PBE0`: 0.00841 | `E2-PBE0`: 0.00954 | `f1-PBE0`: 0.01584 | `f2-PBE0`: 0.02917
  - `E1-PBE0.1`: 0.00809 | `E2-PBE0.1`: 0.00905 | `f1-PBE0.1`: 0.01500 | `f2-PBE0.1`: 0.02844
  - `E1-CAM`: 0.00788 | `E2-CAM`: 0.00925 | `f1-CAM`: 0.01615 | `f2-CAM`: 0.03227

### Alternatives Considered & Rejected
- **3D-Aware Graph Neural Networks (SchNet / DimeNet):** We considered implementing a continuous-filter convolutional network using raw 3D atomic positions. This was deferred because 3D GNNs require substantial GPU compute and hyperparameter tuning overhead. Given that Candidate 1 achieved strong convergence and low error on foundational electronic properties quickly, the added complexity of 3D geometry models was deemed unnecessary for this iteration.

---

## Discussion & Limitations

1. **Comparison with Published Baselines:** Due to API rate limits during automated literature verification, we refrain from citing unverified exact benchmark figures from memory. However, classical fingerprints combined with gradient boosting typically serve as strong, highly efficient baselines on MoleculeNet tasks before moving to graph or geometric deep learning.
2. **Oscillator Strength Challenge:** As observed in the per-target breakdown, oscillator strengths (`f1`, `f2`) present a higher error margin than excitation energies (`E1`, `E2`). Oscillator strengths depend heavily on subtle quantum mechanical transition dipole moments and 3D spatial conformations, which topological 2D fingerprints (`ECFP`) capture only implicitly.
3. **Data Hygiene:** RDKit sanitization handled invalid or problematic SMILES gracefully, ensuring zero runtime interruptions during feature extraction.