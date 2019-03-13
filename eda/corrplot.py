"""
Create corelation plots.
"""

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator


if __name__ == "__main__":
    TRAIN_DATA_PATH = os.path.join("data", "BADS_WS1819_known.csv")
    UNKNOWN_DATA_PATH = os.path.join("data", "BADS_WS1819_unknown.csv")

    KNOWN = clean(TRAIN_DATA_PATH)
    UNKNOWN = clean(UNKNOWN_DATA_PATH)
    HISTORY = KNOWN.append(UNKNOWN, sort=False)

    with open("variables.txt", "r") as f:
        COLS = f.read().splitlines()

    LABELS = [c.replace("_", " ") for c in COLS]

    FG = FeatureGenerator(cols=COLS)
    FG.fit(HISTORY, 'return')
    OUT, _ = FG.transform(KNOWN, ignore_woe=False, add_dummies=True)
    OUT = pd.DataFrame(OUT, columns=COLS)

    CORR = OUT.corr()
    FIG, AX = plt.subplots(figsize=(10, 8))
    sns.set(font_scale=0.75)
    sns.heatmap(CORR, vmin=-1, vmax=1, center=0,
                cmap=plt.cm.seismic,
                xticklabels=[],
                yticklabels=LABELS,
                square=True)

    AX.set_ylabels = LABELS
    plt.tight_layout()

    FIG.savefig("eda/corrplot.png")
    print("Dimensions:", CORR.shape)
