import numpy as np
from scipy.stats import skew, kurtosis


def get_feature_names(feature_list):
    base = []
    for f in feature_list:
        if f != "magnitude":
            base.append(f + "_mean")
            base.append(f + "_std")
    if "magnitude" in feature_list:
        base.extend(["mag_mean", "mag_std"])
    return base


def safe_skew(x):
    if np.std(x) < 1e-6:
        return 0
    return skew(x)


def safe_kurtosis(x):
    if np.std(x) < 1e-6:
        return 0
    return kurtosis(x)


def _per_channel_feats(sig, feats):
    out = []
    if "std" in feats:
        out.append(np.std(sig))
    if "energy" in feats:
        out.append(np.mean(sig**2))
    if "zcr" in feats:
        out.append(np.mean(np.diff(np.sign(sig)) != 0))
    if "mean" in feats:
        out.append(np.mean(sig))
    if "rms" in feats:
        out.append(np.sqrt(np.mean(sig**2)))
    if "skew" in feats:
        out.append(safe_skew(sig))
    if "kurtosis" in feats:
        out.append(safe_kurtosis(sig))
    if "fft_power" in feats:
        out.append(np.mean(np.abs(np.fft.fft(sig))))
    return np.array(out, dtype=float)


def extract_features(X, feature_list):
    # X: (n, T, C) -> produce fixed-size features independent of C
    feats_all = []
    for w in X:
        per_ch = []
        for c in range(w.shape[1]):
            per_ch.append(_per_channel_feats(w[:, c], feature_list))
        per_ch = np.stack(per_ch, axis=0)  # (C, F)
        # aggregate over channels
        agg_mean = np.nanmean(per_ch, axis=0)
        agg_std = np.nanstd(per_ch, axis=0)
        row = np.concatenate([agg_mean, agg_std])
        if "magnitude" in feature_list:
            mag = np.sqrt(np.sum(w**2, axis=1))
            row = np.concatenate([row, [np.mean(mag), np.std(mag)]])
        feats_all.append(row)
    return np.array(feats_all)
