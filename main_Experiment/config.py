import os

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

FEATURE_SETS = {
    "baseline": ["std", "energy", "zcr"],
    "extended": [
        "std",
        "energy",
        "zcr",
        "mean",
        "rms",
        "skew",
        "kurtosis",
        "magnitude",
    ],
    "full": [
        "std",
        "energy",
        "zcr",
        "mean",
        "rms",
        "skew",
        "kurtosis",
        "magnitude",
        "fft_power",
    ],
}

IMBALANCE_STRATEGIES = ["none", "class_weight", "undersample", "smote"]

MODELS = ["logreg", "rf", "xgb", "lgbm"]

SAVE_DIR = "artifacts_3"
os.makedirs(SAVE_DIR, exist_ok=True)
