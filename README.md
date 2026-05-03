# Freezing of Gait Detection for Parkinson's Disease

This repository is a thesis project for automated detection of Freezing of Gait (FoG) episodes using wearable sensor data. It combines data preprocessing, feature engineering, imbalance handling, and model evaluation for both classical machine learning and CNN-based approaches.

## Project Overview

- **Objective:** Detect FoG events in Parkinson’s patients from sensor recordings.
- **Datasets:** Daphnet (`data/raw/DAPHNET/dataset/`) and DeFOG (`data/raw/defog/`).
- **Approaches:**
  - Traditional ML with engineered features
  - Imbalance strategies: class weight, undersampling, SMOTE
  - Models: Logistic Regression, Random Forest, XGBoost, LightGBM
  - CNN experiment in `main_Experiment/CNN/`

## Repository Structure

- `requirements.txt` — Python dependencies.
- `data/` — dataset folders and processed data.
- `main_Experiment/` — main ML pipeline.
  - `main.py` — entry point for traditional model experiments.
  - `preprocess.py` — loads DAPHNET and DeFOG, resamples, windows, and labels data.
  - `features.py` — feature extraction for baseline, extended, and full feature sets.
  - `imbalance.py` — SMOTE and undersampling helpers.
  - `models.py` — model factory for logistic regression, RF, XGBoost, and LightGBM.
  - `train.py` — training loop, cross-validation, evaluation, and artifact saving.
  - `evaluate.py` — metric definitions.
  - `plot.py` — plotting helpers for confusion, ROC, PR, and feature importance.
  - `utils.py` — persistence helpers.
  - `artifacts_3/` — saved results and plots from experiments.
- `main_Experiment/CNN/` — CNN-based experiment folder.
  - `main.py` — CNN training entry point.
  - `train.py` — CNN training and evaluation.
  - `model.py` — CNN architecture.
  - `cnn_utils.py` — data conversion and model evaluation utilities.
- `notebooks/` — exploration and experiment notebooks.

## Installation

1. Create or activate a Python environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Main Experiment

1. Ensure datasets are available:
   - `data/raw/DAPHNET/dataset/` with `.txt` sensor files
   - `data/raw/defog/` with `.csv` files
2. Run the main pipeline:
   ```bash
   cd main_Experiment
   python main.py
   ```

This will:

- load and preprocess DAPHNET and DeFOG data
- build sliding windows
- extract features for baseline, extended, and full sets
- train models using 5-fold cross-validation
- apply imbalance strategies
- save results to `main_Experiment/artifacts_3/results.csv`

## Running the CNN Experiment

1. From `main_Experiment/CNN/` run:
   ```bash
   cd main_Experiment/CNN
   python main.py
   ```
2. The CNN pipeline trains on combined dataset windows and saves model/results in `main_Experiment/CNN/` artifact files.

## Evaluation Output

The main experiment stores metrics such as:

- accuracy
- precision
- recall
- F1 score
- ROC AUC

Results are aggregated in:

- `main_Experiment/artifacts_3/results.csv`

## Notes

- The pipeline uses a window size of 256 samples and stride 128.
- DeFOG data is resampled from 100Hz to 64Hz to match DAPHNET.
- FoG labels are unified across datasets before training.


Use this repository as a reference for FoG detection research and thesis work in Parkinson’s disease assistive sensing.
