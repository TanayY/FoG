from preprocess import build_datasets
from train import run_pipeline

DAPHNET_PATH = "../data/raw/DAPHNET/dataset"
TLVMC_PATH = "../data/raw/defog"

X_d, y_d, X_t, y_t = build_datasets(DAPHNET_PATH, TLVMC_PATH)

results = run_pipeline(X_d, y_d, X_t, y_t)
print(results.head())
