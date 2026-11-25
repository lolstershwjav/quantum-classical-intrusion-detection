"""Matched sampling and train-only angle preparation for UNSW-NB15."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURES=["dur","spkts","dpkts","sbytes","dbytes","rate","sload","dload",
          "sloss","dloss","sinpkt","dinpkt","sjit","djit","smean","dmean"]


def load(directory: Path):
    columns = [*FEATURES,"label"]
    split=[]
    for part in ("training","testing"):
        frame=pd.read_csv(directory/f"UNSW_NB15_{part}-set.csv",usecols=columns)
        filtered=frame.loc[np.isfinite(frame[FEATURES].to_numpy()).all(axis=1)]
        split.append(filtered.drop_duplicates(FEATURES,keep="first"))
    train,test=split
    train_keys=pd.MultiIndex.from_frame(train[FEATURES])
    test=test.loc[train_keys.get_indexer(pd.MultiIndex.from_frame(test[FEATURES]))<0]
    return train,test


def subsets(data,seed,size,validation,testing):
    before,after=data
    training, calibrate, target = [], [], []
    for label in (0, 1):
        train_rows = before.loc[before.label.eq(label), FEATURES].to_numpy()
        test_rows = after.loc[after.label.eq(label), FEATURES].to_numpy()
        ordering = np.random.default_rng([seed, label, 0]).permutation(len(train_rows))
        private = np.random.default_rng([seed, label, 1]).permutation(len(test_rows))
        c, n = validation // 2, size // 2
        calibrate.append(np.log1p(train_rows[ordering[:c]]))
        training.append(np.log1p(train_rows[ordering[c:c+n]]))
        target.append(np.log1p(test_rows[private[:testing//2]]))
    x = np.concatenate(training)
    y = np.repeat([0, 1], size//2)
    val = np.concatenate(calibrate)
    vy = np.repeat([0, 1], validation//2)
    test = np.concatenate(target)
    ty = np.repeat([0, 1], testing//2)
    return x, y, val, vy, test, ty


def process(x,val,test,dimensions):
    raw=StandardScaler().fit(x)
    x,val,test=map(raw.transform,(x,val,test))
    transform=PCA(n_components=dimensions,svd_solver="full").fit(x)
    x,val,test=map(transform.transform,(x,val,test))
    spread=StandardScaler().fit(x)
    x,val,test=(np.clip(spread.transform(v),-2.5,2.5)*.9 for v in (x,val,test))
    return (x,val,test),float(transform.explained_variance_ratio_.sum())
