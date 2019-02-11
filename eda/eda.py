import os
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
    fg = FeatureGenerator()
    fg.fit(history, 'return')
    _, _ = fg.transform(known, "return")
    out = fg.outfeatures.copy()

    corr = out.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                cmap=plt.cm.coolwarm,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.title("Feature Correlation Plot")
    fig.savefig("eda/corrplot.png")

    print("Dimensions:", corr.shape)
