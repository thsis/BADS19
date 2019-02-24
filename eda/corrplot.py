import os
import pandas as pd
import seaborn as sns
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator

from matplotlib import pyplot as plt


if __name__ == "__main__":
    train_data_path = os.path.join("data", "BADS_WS1819_known.csv")
    unknown_data_path = os.path.join("data", "BADS_WS1819_unknown.csv")

    known = clean(train_data_path)
    unknown = clean(unknown_data_path)
    history = known.append(unknown, sort=False)

    cols = ["days_to_delivery",
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

    fg = FeatureGenerator(cols=cols)
    fg.fit(history, 'return')
    out, _ = fg.transform(known, "return")
    out = pd.DataFrame(out, columns=cols)

    corr = out.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                cmap=plt.cm.seismic,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    fig.savefig("eda/corrplot.png")

    print("Dimensions:", corr.shape)
