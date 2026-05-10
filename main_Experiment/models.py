from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_model(name, class_weight=None):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight=class_weight)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, class_weight=class_weight, n_jobs=-1
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
    if name == "lgbm":
        return LGBMClassifier(n_estimators=200)
    raise ValueError("Unknown model")
