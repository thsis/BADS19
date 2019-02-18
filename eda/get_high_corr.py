import os
import pandas as pd
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator


if __name__ == "__main__":
    train_data_path = os.path.join("data", "BADS_WS1819_known.csv")
    unknown_data_path = os.path.join("data", "BADS_WS1819_unknown.csv")

    known = clean(train_data_path)
    unknown = clean(unknown_data_path)
    history = known.append(unknown, sort=False)
    fg = FeatureGenerator()
    fg.fit(history, 'return')
    _, _ = fg.transform(known, ignore_woe=False)
    out = fg.outfeatures.copy()

    corr = out.corr()
    print("Dimensions:", corr.shape)

    table = pd.melt(corr.reset_index(), id_vars="index")
    table.columns = ["var1", "var2", "correlation"]
    not_variance = (table.var1 != table.var2)
    table = table.loc[not_variance & (table.correlation.abs() > 0.8)]

    problematic = table.groupby("var1").var2.nunique()
    print(problematic.sort_values(ascending=False))
    problematic = problematic[problematic > 3].index.tolist()

    update = input("Do you want to update the blacklist? y/n > ")

    if update == "y":
        print("Update blacklist.txt")
        with open(os.path.join("preprocessing", "blacklist.txt"), "w") as f:
            f.writelines("\n".join(problematic), )

    print(table.sort_values("correlation", ascending=False)[::2])
