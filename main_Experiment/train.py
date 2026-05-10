import os
import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from config import *
from features import extract_features, get_feature_names
from imbalance import apply_imbalance

# from main_Experiment.main import X_t
from models import get_model
from evaluate import evaluate
from utils import save_pickle, save_log
from plot import (
    plot_confusion,
    plot_roc,
    plot_pr,
    plot_feature_importance,
    plot_model_comparison,
)


def run_pipeline(X_d, y_d, X_t, y_t):
    results = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for f_name, f_list in FEATURE_SETS.items():
        print(f"\n[Feature Set] {f_name}")

        feature_dir = os.path.join(SAVE_DIR, f_name)
        os.makedirs(feature_dir, exist_ok=True)

        feature_names = get_feature_names(f_list)

        Xd_f = pd.DataFrame(extract_features(X_d, f_list), columns=feature_names)
        Xt_f = pd.DataFrame(extract_features(X_t, f_list), columns=feature_names)

        X_all = pd.concat([Xd_f, Xt_f], axis=0).reset_index(drop=True)
        y_all = np.concatenate([y_d, y_t])

        save_pickle(X_all, os.path.join(feature_dir, "X.pkl"))

        for imb in IMBALANCE_STRATEGIES:
            for m in tqdm(MODELS, desc=f"{f_name}-{imb}"):

                # 🔹 Create model + plots directories
                model_dir = os.path.join(feature_dir, m)
                os.makedirs(model_dir, exist_ok=True)

                plots_dir = os.path.join(model_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                fold_metrics = []

                for tr_idx, te_idx in skf.split(X_all, y_all):
                    # 🔹 IMPORTANT FIX (DataFrame indexing)
                    Xtr, Xte = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                    ytr, yte = y_all[tr_idx], y_all[te_idx]

                    scaler = StandardScaler()
                    Xtr_scaled = scaler.fit_transform(Xtr)
                    Xte_scaled = scaler.transform(Xte)

                    Xtr = pd.DataFrame(Xtr_scaled, columns=feature_names)
                    Xte = pd.DataFrame(Xte_scaled, columns=feature_names)

                    Xbal, ybal = apply_imbalance(Xtr, ytr, imb)

                    cw = "balanced" if imb == "class_weight" else None
                    model = get_model(m, cw)

                    t0 = time.time()
                    model.fit(Xbal, ybal)
                    ttrain = time.time() - t0

                    te_metrics = evaluate(model, Xte, yte)
                    fold_metrics.append(te_metrics)

                avg = pd.DataFrame(fold_metrics).mean().to_dict()

                model_path = os.path.join(model_dir, f"{imb}_model.pkl")
                save_pickle(model, model_path)

                probs = (
                    model.predict_proba(Xte)[:, 1]
                    if hasattr(model, "predict_proba")
                    else model.predict(Xte)
                )
                preds = model.predict(Xte)

                plot_confusion(yte, preds, os.path.join(plots_dir, f"{imb}_cm.png"))
                plot_roc(yte, probs, os.path.join(plots_dir, f"{imb}_roc.png"))
                plot_pr(yte, probs, os.path.join(plots_dir, f"{imb}_pr.png"))
                plot_feature_importance(model, os.path.join(plots_dir, f"{imb}_fi.png"))

                rec = {
                    "model": m,
                    "features": f_name,
                    "imbalance": imb,
                    "train_time": ttrain,
                    **{f"test_{k}": v for k, v in avg.items()},
                }

                results.append(rec)
                df = pd.DataFrame(results)

                save_log(df, os.path.join(SAVE_DIR, "results.csv"))

                plot_model_comparison(df, os.path.join(SAVE_DIR, "model_compare.png"))

                print(f"Saved → {model_dir}")

    return pd.DataFrame(results)
