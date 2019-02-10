import os
import seaborn as sns
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator

from matplotlib import pyplot as plt


if __name__ == "__main__":
    traindatapath = os.path.join("data", "BADS_WS1819_known.csv")
    known = clean(traindatapath)
    fg = FeatureGenerator()
    _, _ = fg.fit_transform(known, "return")
    out = fg.outfeatures.copy()

    corr = out.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                cmap=plt.cm.coolwarm,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.suptitle("Feature Correlation Plot")
    fig.savefig("eda/corrplot.png")

    print("Dimensions:", corr.shape)
