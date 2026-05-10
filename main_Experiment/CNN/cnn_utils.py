import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

    preds_bin = (probs > 0.3).astype(int)  # lower threshold

    return {
        "accuracy": accuracy_score(y, preds_bin),
        "precision": precision_score(y, preds_bin, zero_division=0),
        "recall": recall_score(y, preds_bin, zero_division=0),
        "f1": f1_score(y, preds_bin, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
    }
