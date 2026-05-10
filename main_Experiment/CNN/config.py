import os

RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

SAVE_DIR = "artifacts_cnn_ex2"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "1d_cnn_ex_2"
