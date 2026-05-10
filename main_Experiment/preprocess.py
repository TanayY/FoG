import os
import numpy as np
import pandas as pd

daphnet_config = {
    "label_col": 10,
    "fog_value": 2,
    "nonfog_value": 1,
    "sensor_columns": {"ankle": [1, 2, 3], "thigh": [4, 5, 6], "trunk": [7, 8, 9]},
}

tlvmc_config = {
    "label_col": "fog",
    "fog_value": 1,
    "nonfog_value": 0,
    "sensor_columns": {"lower_back": ["AccV", "AccML", "AccAP"]},
}


def load_daphnet(folder_path):
    frames = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            df = pd.read_csv(path, sep=r"\s+", header=None)
            df["file"] = file
            frames.append(df)
    if not frames:
        raise RuntimeError("No DAPHNet files found")
    return pd.concat(frames, ignore_index=True)


def load_tlvmc(folder_path):
    frames = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            # unified FoG label
            df["fog"] = (
                (
                    (df["StartHesitation"] == 1)
                    | (df["Turn"] == 1)
                    | (df["Walking"] == 1)
                )
                & (df["Valid"] == True)
                & (df["Task"] == True)
            )
            df["fog"] = df["fog"].astype(int)
            df["file"] = file
            frames.append(df)
    if not frames:
        raise RuntimeError("No TLVMC files found")
    return pd.concat(frames, ignore_index=True)


def resample_group(group, factor):
    n_old = len(group)
    n_new = max(1, int(n_old * factor))
    idx = np.linspace(0, n_old - 1, n_new).astype(int)
    return group.iloc[idx].copy()


def resample_df(df, sensor_cols, original_freq, target_freq):
    factor = target_freq / original_freq
    out = []
    for file, g in df.groupby("file"):
        g2 = resample_group(g.reset_index(drop=True), factor)
        g2["file"] = file
        out.append(g2)
    return pd.concat(out, ignore_index=True)


def create_windows_indexed(df, config, window=256, step=128):
    X, y = [], []
    sensor_idx = []
    for v in config["sensor_columns"].values():
        sensor_idx.extend(v)
    for file, g in df.groupby("file"):
        g = g.reset_index(drop=True)
        signals = g.iloc[:, sensor_idx].values
        labels = g.iloc[:, config["label_col"]].values
        for i in range(0, len(g) - window, step):
            w = signals[i : i + window]
            l = labels[i : i + window]
            X.append(w)
            y.append(1 if np.any(l == config["fog_value"]) else 0)
    return np.array(X), np.array(y)


def create_windows_named(df, config, window=256, step=128):
    X, y = [], []
    cols = []
    for v in config["sensor_columns"].values():
        cols.extend(v)
    for file, g in df.groupby("file"):
        g = g.reset_index(drop=True)
        signals = g[cols].values
        labels = g[config["label_col"]].values
        for i in range(0, len(g) - window, step):
            w = signals[i : i + window]
            l = labels[i : i + window]
            X.append(w)
            y.append(1 if np.any(l == config["fog_value"]) else 0)
    return np.array(X), np.array(y)


def build_datasets(daphnet_path, tlvmc_path):
    print("Loading DAPHNet...")
    ddf = load_daphnet(daphnet_path)
    print("Windowing DAPHNet...")
    X_d, y_d = create_windows_indexed(ddf, daphnet_config)

    print("Loading TLVMC...")
    tdf = load_tlvmc(tlvmc_path)
    print("Resampling TLVMC to 64Hz...")
    tdf = resample_df(tdf, tlvmc_config["sensor_columns"]["lower_back"], 100, 64)
    print("Windowing TLVMC...")
    X_t, y_t = create_windows_named(tdf, tlvmc_config)

    print("Shapes -> DAPHNet:", X_d.shape, ", TLVMC:", X_t.shape)
    return X_d, y_d, X_t, y_t
