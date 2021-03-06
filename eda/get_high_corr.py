import os
import sys
import pandas as pd
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator


if __name__ == "__main__":
    train_data_path = os.path.join("data", "BADS_WS1819_known.csv")
    unknown_data_path = os.path.join("data", "BADS_WS1819_unknown.csv")
    blacklist_path = os.path.join("preprocessing", "blacklist.txt")

    known = clean(train_data_path)
    unknown = clean(unknown_data_path)
    history = known.append(unknown, sort=False)
    fg = FeatureGenerator()
    fg.fit(history, 'return')
    _, _ = fg.transform(known, ignore_woe=False, add_dummies=True,
                        add_interactions=True, add_ratios=True)
    out = fg.outfeatures.copy()

    corr = out.corr()
    print("Dimensions:", corr.shape)

    table = pd.melt(corr.reset_index(), id_vars="index")
    table.columns = ["var1", "var2", "correlation"]
    not_variance = (table.var1 != table.var2)
    table = table.loc[not_variance & (table.correlation.abs() > 0.9)]

    problematic = table.groupby("var1").var2.nunique()
    print(problematic.sort_values(ascending=False))
    problematic = problematic[problematic > 3].index.tolist()

    update = input("Do you want to update the blacklist? y/n > ")
    table["correlation"] = table.correlation.abs()
    update_table = table.sort_values("correlation", ascending=False)[::2]

    if update == "y":
        i = 0
        print("Update blacklist.txt")
        for idx, row in update_table.iterrows():
            var1, var2, corr = row
            while True:
                try:
                    i += 1
                    print("=" * 60)
                    print("{0:^60s}".format("Decision Nr: " + str(i)))
                    print("=" * 60)
                    print("1: {0:>40s}".format(var1))
                    print("2: {0:>40s}".format(var2))
                    print("Correlation:", corr)
                    var = input("Which variable do you want to kill? [1, 2]> ")
                    if var == "exit":
                        sys.exit(0)
                    if var == "n":
                        break
                    var_idx = int(var) - 1
                    break
                except Exception:
                    i -= 1
                    print("Entry invalid.")
                    continue
            if var != "n":
                with open(blacklist_path, "a") as f:
                    f.writelines("\n" + row[var_idx])
