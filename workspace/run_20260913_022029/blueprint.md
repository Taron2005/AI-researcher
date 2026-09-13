# Research Blueprint: QM8 Molecular Property Prediction

## Approach
We propose using a 3D-aware graph neural network (specifically, a distance-geometry or equivariant/invariant message-passing architecture like SchNet or a custom spatial GNN built via PyTorch / PyTorch Geometric / RDKit) that leverages the exact 3D atomic coordinates (`positions`) and atomic numbers provided in the QM8 dataset. QM8 labels consist of 16 electronic spectra properties (excitation energies and oscillator strengths computed at various levels of theory like CC2, PBE0, CAM). Because these quantum-chemical properties depend strongly on spatial conformation, orbital overlap, and interatomic distances, a 3D-aware representation is essential. Candidate 1 will use a SchNet-style continuous-filter convolutional neural network, processing interatomic Euclidean distances alongside atomic number embeddings to predict all 16 target properties simultaneously via multi-task learning.

## Why
QM8 targets are quantum-mechanical electronic excitation properties derived from 3D geometries. Baseline models using 1D/2D molecular graphs or ECFP fingerprints fail to capture subtle stereoelectronic and spatial effects that dictate excitation energies. SchNet provides rotationally and translationally invariant message passing over 3D coordinates, making it a robust and standard baseline for quantum chemical property datasets (such as QM7, QM8, and QM9).

## Sources
- SchNet: A continuous-filter convolutional neural network for modeling quantum interactions (https://arxiv.org/abs/1706.08566)
- MoleculeNet: A benchmark for molecular machine learning (https://arxiv.org/abs/1703.00564)

## Evaluation Protocol
- **Split**: Standard 80/10/10 train/validation/test split (or random split with fixed seed, consistent with MoleculeNet benchmarks).
- **Metric**: Mean Absolute Error (MAE) averaged across all 16 QM8 target properties, mirroring the standard benchmark metric.

## Validation Checkpoint
Before launching full training, we run a sanity check on a small subset (e.g., 500 molecules) for 1 epoch to verify data loader integrity, loss computation across all 16 multi-task outputs, and correct tensor shapes through the 3D GNN layers.

## Budget Split
- **Architecture & Implementation (Candidate 1)**: 50% of effort (setting up the dataset pipeline, parsing `positions` and `atomic_numbers`, building/adapting the GNN).
- **Hyperparameter Tuning & Training Loop**: 50% of effort (learning rate scheduling, batch size optimization, embedding dimensions, number of interaction blocks, and multi-task loss weighting).

## Candidate-2 Strategy Note
- **Condition**: If Candidate 1 underperforms standard published baselines due to optimization difficulty or if multi-task interference across the 16 targets causes high error on specific high-level methods (e.g., CC2 vs PBE0).
- **Hypothesis**: A higher-order 3D GNN (such as DimeNet++ or GemNet for angular/torsional awareness) or decoupled single-task / clustered multi-task training models could better capture angular bond-angle dependencies critical for accurate oscillator strengths and excitation energies.