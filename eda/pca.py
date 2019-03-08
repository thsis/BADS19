"""
Try to reduce dimensionality by a PCA-decomposition.
"""

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from matplotlib import pyplot as plt

from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator

np.random.seed(42)

# 3. Load and clean the data
DATAPATH = os.path.join("data", "BADS_WS1819_known.csv")
UNKNOWNPATH = os.path.join("data", "BADS_WS1819_unknown.csv")
OUTPATH = os.path.join("eda", "pca.png")

KNOWN = clean(DATAPATH)
UNKNOWN = clean(UNKNOWNPATH)
HISTORY = KNOWN.append(UNKNOWN, sort=False)

TRAIN, TEST = train_test_split(KNOWN, test_size=0.2)

with open("variables.txt", "r") as f:
    COLS = f.read().splitlines()

FG = FeatureGenerator(cols=COLS)
FG.fit(HISTORY, 'return')
X_TRAIN, Y_TRAIN = FG.transform(TRAIN,
                                add_ratios=True,
                                add_interactions=True,
                                add_dummies=False)
X_TEST, Y_TEST = FG.transform(TEST,
                              add_ratios=True,
                              add_interactions=True,
                              add_dummies=False)


steps = [('scaler', StandardScaler()),
         ('pca', PCA())]
pipeline = Pipeline(steps)
scaler = pipeline.fit(X_TRAIN, Y_TRAIN)
PCA_DECOMP = pipeline.named_steps["pca"]


FIG, AX = plt.subplots(figsize=(6, 4))
AX.plot(PCA_DECOMP.explained_variance_ratio_)
AX.set_xlabel("Components")
AX.set_ylabel("Explained Variance in %")

FIG.savefig(OUTPATH)
print(PCA_DECOMP.explained_variance_ratio_.cumsum())
