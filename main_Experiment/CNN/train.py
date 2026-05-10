import torch
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import *
from model import CNN1D
from cnn_utils import to_tensor, evaluate_model
from utils import save_pickle, save_log


def train_cnn(X, y):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = (X - X.mean()) / (X.std() + 1e-6)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    X_train, y_train = to_tensor(X_train, y_train)
    X_test, y_test = to_tensor(X_test, y_test)

    X_train = X_train.permute(0, 2, 1)
    X_test = X_test.permute(0, 2, 1)

    model = CNN1D(input_channels=X_train.shape[1]).to(device)

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True
    )

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze()

            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    train_time = time.time() - start_time

    X_test = X_test.to(device)
    metrics = evaluate_model(model, X_test, y_test.numpy())

    save_pickle(model, f"{SAVE_DIR}/cnn_model.pkl")

    result = {"model": "1D_CNN", "train_time": train_time, **metrics}

    df = pd.DataFrame([result])
    save_log(df, f"{SAVE_DIR}/cnn_results.csv")

    return df
