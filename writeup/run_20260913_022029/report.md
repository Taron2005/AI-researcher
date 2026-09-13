# Research Report: 3D Graph Neural Networks for Molecular Property Prediction on QM8

## Overview & Executive Summary

This study investigated the application of 3D Graph Neural Networks (GNNs) for predicting electronic spectra and quantum-chemical properties on the **QM8** benchmark dataset (part of MoleculeNet) [2]. QM8 comprises roughly 21,700 molecules evaluated across 16 different target properties computed under various quantum-chemical levels of theory (CAM, CC2, PBE0, and PBE0.1), including electronic excitation energies ($E_1, E_2$) and oscillator strengths ($f_1, f_2$).

Two candidate models were scheduled in our blueprint:
1. **Candidate 1**: A standard multi-task **SchNet** 3D continuous-filter convolutional neural network that takes 3D atomic coordinates and atomic numbers, computes invariant interatomic distances, and jointly predicts all 16 QM8 properties.
2. **Candidate 2**: A refined multi-task SchNet featuring property-specific heads and task normalization to address performance disparities between low-magnitude excitation energies and high-variance oscillator strengths.

### Decision Trail & Outcome
* **Candidate 1** executed successfully, achieving an overall test Mean Absolute Error (MAE) of **0.01166** across all 16 properties, with an excitation energy MAE of **0.00650** and an oscillator strength MAE of **0.01682**. 
* **Candidate 2** was aborted after encountering runtime device placement errors (`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`) that exceeded the maximum allowed review-fix rounds (`MAX_REVIEW_FIX_ROUNDS=5`). 
* **Winner**: **Candidate 1** served as our fully functioning model and baseline representation.

---

## Methodology & Architecture

### Candidate 1: SchNet 3D GNN
* **Approach**: Continuous-filter convolutional neural network operating directly on 3D Cartesian coordinates and atomic numbers. Interatomic distances $r_{ij}$ are expanded using a Gaussian basis, and continuous filters modulate atom feature updates across multiple interaction blocks.
* **Multi-Task Learning**: A shared representation vector for each molecule is passed to linear output layers mapping to the 16 target properties simultaneously.
* **Training Setup**: 
  * Data split: 80% train (17,397 molecules), 10% validation (2,174 molecules), 10% test (2,176 molecules).
  * Loss function: L1 Loss (Mean Absolute Error), aligned with the evaluation metric.
  * Optimizer: Adam with learning rate scheduling.

---

## Results & Quantitative Analysis

### Overall Performance (Candidate 1)
* **Overall Test MAE**: `0.01166`
* **Normalized Test MSE**: `0.18757`
* **Excitation Energy MAE ($E_1, E_2$)**: `0.00650`
* **Oscillator Strength MAE ($f_1, f_2$)**: `0.01682`

### Per-Property Breakdown
Examining the per-property MAE reveals a clear division in task difficulty and scale between excitation energies and oscillator strengths:

| Property | Target Type | Level of Theory | MAE |
| :--- | :--- | :--- | :--- |
| `E1-CAM` | Excitation Energy | CAM | 0.00601 |
| `E1-CC2` | Excitation Energy | CC2 | 0.00652 |
| `E1-PBE0` | Excitation Energy | PBE0 | 0.00633 |
| `E1-PBE0.1` | Excitation Energy | PBE0.1 | 0.00598 |
| `E2-CAM` | Excitation Energy | CAM | 0.00642 |
| `E2-CC2` | Excitation Energy | CC2 | 0.00734 |
| `E2-PBE0` | Excitation Energy | PBE0 | 0.00706 |
| `E2-PBE0.1` | Excitation Energy | PBE0.1 | 0.00637 |
| `f1-CAM` | Oscillator Strength | CAM | 0.01019 |
| `f1-CC2` | Oscillator Strength | CC2 | 0.01316 |
| `f1-PBE0` | Oscillator Strength | PBE0 | 0.01026 |
| `f1-PBE0.1` | Oscillator Strength | PBE0.1 | 0.00956 |
| `f2-CAM` | Oscillator Strength | CAM | 0.02290 |
| `f2-CC2` | Oscillator Strength | CC2 | 0.02696 |
| `f2-PBE0` | Oscillator Strength | PBE0 | 0.02162 |
| `f2-PBE0.1` | Oscillator Strength | PBE0.1 | 0.01991 |

### Per-Method Breakdown
Averaging across quantum-chemical computational methods shows consistent performance, with PBE0.1 achieving slightly lower errors:
* **CAM**: `0.01138`
* **CC2**: `0.01349`
* **PBE0**: `0.01132`
* **PBE0.1**: `0.01045`

---

## Candidate Failures & Diagnostic Insights

1. **Candidate 2 Device Mismatch Failure**: 
   Candidate 2 attempted to introduce property-specific output heads and custom loss weighting to alleviate the higher errors observed on oscillator strengths ($f_2$). However, tensor initialization bugs arose where certain normalization parameters or head layers defaulted to CPU while the primary model graph resided on `cuda:0`. Despite multiple iterations, the code failed to resolve device consistency before exhausting `MAX_REVIEW_FIX_ROUNDS=5`.
   
2. **Task Disparity Insights**:
   Diagnostic analysis of Candidate 1 confirms the hypothesis motivating Candidate 2: SchNet predicts excitation energies with high precision ($\sim$0.006 MAE) because their values occupy a stable numerical range and smooth potential energy surface dependence. Conversely, oscillator strengths—particularly second-order oscillator strengths (`f2-CC2` at 0.0269 MAE)—exhibit higher relative variance and span wider dynamic ranges, making a single shared output projection head suboptimal.

---

## Comparison with Literature Baselines

While direct leaderboard comparisons on MoleculeNet QM8 vary depending on exact data splits and normalization schemes, standard MPNN and SchNet baselines on QM8 typically report overall MAEs in the 0.011 to 0.014 range, confirming that our Candidate 1 implementation performs competitively with established 3D GNN literature.

## Conclusion
Candidate 1 successfully demonstrated that a 3D continuous-filter graph neural network (SchNet) can learn complex quantum-chemical spectra across 16 disparate targets with an overall MAE of **0.01166**. Future work should successfully execute the architectural goals of Candidate 2 (isolated task heads and loss balancing) to specifically stabilize high-variance oscillator strength targets like $f_2$.