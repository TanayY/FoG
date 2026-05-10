from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_imbalance(X, y, strategy):
    if strategy == "smote":
        return SMOTE().fit_resample(X, y)
    if strategy == "undersample":
        return RandomUnderSampler().fit_resample(X, y)
    return X, y
