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

    COLS = ["days_to_delivery",
            "days_to_delivery/brand_mean_price",
            "days_to_delivery/order_median_price",
            "item_price*order_num_sizes",
            "item_price",
            "days_to_delivery*order_seqnum",
            "days_to_delivery*brand_max_price",
            "order_max_price",
            "item_price*order_num_items",
            "days_to_delivery*order_total_value",
            "price_off/item_price",
            "order_max_price/brand_max_price",
            "is_item_clothes*order_num_colors",
            "is_item_clothes*order_median_price",
            "order_max_price/brand_mean_price",
            "item_price/brand_mean_price",
            "price_off/days_to_delivery"]

    LABELS = [c.replace("_", " ") for c in COLS]

    FG = FeatureGenerator(cols=COLS)
    FG.fit(HISTORY, 'return')
    OUT, _ = FG.transform(KNOWN, "return")
    OUT = pd.DataFrame(OUT, columns=COLS)

    CORR = OUT.corr()
    FIG, AX = plt.subplots(figsize=(10, 5))
    sns.heatmap(CORR, vmin=-1, vmax=1, center=0,
                cmap=plt.cm.seismic,
                xticklabels=CORR.columns.values,
                yticklabels=CORR.columns.values)

    FIG.suptitle("Feature Correlation Plot", fontsize=22)
    AX.set_ylabels = LABELS
    AX.set_xlables = []

    FIG.savefig("eda/corrplot_t.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    print("Dimensions:", CORR.shape)
