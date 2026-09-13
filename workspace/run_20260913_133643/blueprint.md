# Research Blueprint: QM8 Molecular Property Prediction

## Approach
We propose a robust **Extended Connectivity Fingerprint (ECFP / Morgan fingerprints) combined with Gradient Boosted Regression Trees (LightGBM)** for predicting the 16 electronic spectra properties (excitation energies and oscillator strengths) in QM8. 

While QM8 contains 3D atomic coordinates, gradient boosted trees on molecular fingerprints (or RDKit 2D descriptors) provide an extremely fast, highly stable baseline that avoids the severe hyperparameter tuning and computational overhead often required for geometric Graph Neural Networks (GNNs) under tight resource constraints. Each of the 16 targets will be modeled via a separate multi-output or independent LightGBM regressor, capturing non-linear feature interactions efficiently.

## Why This Approach
1. **Computational Efficiency & Stability:** Training 3D GNNs (like SchNet or DimeNet) requires substantial GPU time and complex geometric convolution layers. LightGBM on Morgan fingerprints runs in minutes on CPU/GPU, leaving maximum budget for proper cross-validation, hyperparameter tuning, and error analysis.
2. **Robustness against Overfitting:** On datasets of size ~21,000 molecules with multi-task targets, classical ML baselines with well-calibrated tree depths frequently rival or outperform poorly-tuned deep models.

## Alternatives Considered and Rejected
- **3D-Aware Graph Neural Networks (e.g., SchNet / DimeNet):** We strongly considered a continuous-filter convolutional network leveraging the provided 3D coordinates. However, given resource constraints and the extensive hyperparameter search required for stable training of distance-based GNNs, we rejected it for Candidate 1 in favor of a reliable, high-velocity classical baseline that guarantees clean convergence and fast feedback loops.

## Evaluation Protocol
- **Split:** Random 80% train, 10% validation, 10% test split (consistent with standard MoleculeNet benchmark partitions for QM8).
- **Metric:** Mean Absolute Error (MAE) averaged across all 16 target properties (or evaluated per property type: CC2 vs PBE0 vs CAM excitation energies and oscillator strengths).

## Diagnostic Plan
Beyond the headline multi-target MAE:
1. **Target-wise Breakdown:** Compute MAE separately for each of the 16 targets to identify if certain quantum chemical methods (e.g., CC2 vs. CAM) or property types (energies vs. oscillator strengths) are harder to predict.
2. **Error vs. Molecular Size:** Plot prediction error as a function of heavy atom count/molecular weight to check for systematic bias on larger vs. smaller GDB-8 molecules.
3. **Residual Distribution:** Inspect residual histograms to detect skewness or outliers.

## Validation Checkpoint
- Train LightGBM on a tiny subset (1,000 samples) for 50 boosting rounds to verify end-to-end data pipeline integrity, feature extraction, and metric calculation before launching the full dataset run.

## Budget Split
- **Architecture Exploration & Pipeline Setup:** 20%
- **Model Training, Tuning & Evaluation:** 80%

## Candidate-2 Strategy Note
- **Condition:** If LightGBM on ECFP fingerprints hits a performance ceiling (e.g., high MAE on oscillator strengths due to insensitivity to 3D electronic conformation), a second candidate will be explored.
- **Hypothesis:** Candidate 2 would implement a lightweight 3D-aware molecular representation (e.g., SchNet or 3D graph descriptors via RDKit combined with a multi-layer perceptron or lightweight GNN) to explicitly capture spatial electronic distribution features.

## Sources
- MoleculeNet QM8 Benchmark: Wu et al., "MoleculeNet: A Benchmark for Molecular Machine Learning", Chem. Sci., 2018.
- SchNet: Schütt et al., "SchNet: A continuous-filter convolutional network for modeling quantum interactions", arXiv:1712.03119, 2017.