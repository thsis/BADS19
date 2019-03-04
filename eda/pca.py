"""
Try to reduce dimensionality by a PCA-decomposition.
"""

import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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

COLS = ["days_to_delivery",
        "item_price*order_num_items",
        "item_price*order_num_sizes",
        "days_to_delivery*order_seqnum",
        "days_to_delivery*brand_max_price",
        "item_price",
        "days_to_delivery*order_total_value",
        "order_total_value/order_num_colors",
        "order_total_value",
        "days_to_delivery*order_min_price",
        "order_num_items/order_num_colors",
        "item_price*order_min_price",
        "is_item_clothes*order_median_price",
        "is_item_clothes*order_min_price"]

FG = FeatureGenerator()

FG.fit(HISTORY, 'return')
X_TRAIN, Y_TRAIN = FG.transform(TRAIN)
X_TEST, Y_TEST = FG.transform(TEST)


COLSSET = set(COLS)
FGSET = set(FG.outfeatures.columns)

PCA_DECOMP = PCA(n_components=15)
PCA_DECOMP.fit(X_TRAIN)

FIG, AX = plt.subplots(figsize=(6, 4))
AX.plot(PCA_DECOMP.explained_variance_ratio_)
AX.set_xlabel("Components")
AX.set_ylabel("Explained Variance in %")

FIG.savefig(OUTPATH)
print(PCA_DECOMP.explained_variance_ratio_.cumsum())
