from preprocess import build_datasets
from train import train_cnn
import numpy as np
from preprocess import reduce_daphnet_to_3_channels

DAPHNET_PATH = "../../data/raw/DAPHNET/dataset"
TLVMC_PATH = "../../data/raw/defog"

X_d, y_d, X_t, y_t = build_datasets(DAPHNET_PATH, TLVMC_PATH)

X_d_reduced = reduce_daphnet_to_3_channels(X_d)

X = np.concatenate([X_d_reduced, X_t], axis=0)
y = np.concatenate([y_d, y_t], axis=0)

print("Final CNN dataset:", X.shape, y.shape)

results = train_cnn(X, y)
print(results)
