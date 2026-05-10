import pickle
import pandas as pd


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_log(df, path):
    df.to_csv(path, index=False)
