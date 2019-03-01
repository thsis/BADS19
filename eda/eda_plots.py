"""
Plots for Exploratory Data Analysis.
"""

import os
import pandas as pd
from matplotlib import pyplot as plt

KNOWNPATH = os.path.join("data", "BADS_WS1819_known.csv")
UNKNOWNPATH = os.path.join("data", "BADS_WS1819_unknown.csv")
OUTPATH = os.path.join("eda", "exploratory.png")

DATES = ["order_date", "delivery_date", "user_dob", "user_reg_date"]
NA_VALUES = ["not_reported", "?", "1994-12-31"]
KNOWN = pd.read_csv(KNOWNPATH, index_col=["order_item_id"],
                    parse_dates=DATES, na_values=NA_VALUES)
UNKNOWN = pd.read_csv(UNKNOWNPATH, index_col=["order_item_id"],
                      parse_dates=DATES, na_values=NA_VALUES)
HISTORY = KNOWN.append(UNKNOWN, sort=False)

KNOWN["days_to_delivery"] = (
    KNOWN.delivery_date - KNOWN.order_date).dt.days


FIG, AXES = plt.subplots(ncols=4, figsize=(10, 3))
PLOT1 = KNOWN.groupby("days_to_delivery").agg({"return": ["mean", "count"]})
PLOT1 = PLOT1.loc[PLOT1["return"]["count"] > 10].reset_index()
PLOT1.columns = ["days_to_delivery", "perc_returned", "total_items"]

PLOT1.plot(kind="scatter", x="days_to_delivery",
           y="perc_returned", ax=AXES[0])
AXES[0].axhline(0.5, c="red", linestyle="--")
AXES[0].set_ylabel("Percentage of Returned Items")
AXES[0].set_xlabel("Days until Item Delivery")

PLOT2 = KNOWN.groupby(KNOWN.delivery_date.dt.dayofweek)["return"].mean()
PLOT2.plot.bar(ax=AXES[1], yerr=PLOT2.std(), capsize=4)
AXES[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], rotation=0)
AXES[1].set_xlabel("Day Item was delivered")
AXES[1].axhline(0.5, c="red", linestyle="--")

PLOT3 = HISTORY.groupby(["user_id", "order_date"]).agg({
    "item_color": "count",
    "item_size": "count",
    "return": "sum"})

PLOT3.loc[PLOT3["return"] > 1, "return"] = 1
PLOT3.boxplot(column=["item_color", "item_size"],
              by="return", ax=(AXES[2], AXES[3]))
AXES[2].set_title("")
AXES[2].set_xticklabels(["Not returned", "Returned"])
AXES[2].set_xlabel("Number of Colors")

AXES[3].set_title("")
AXES[3].set_xticklabels(["Not returned", "Returned"])
AXES[3].set_xlabel("Number of Sizes")


plt.suptitle("")
plt.tight_layout()
FIG.savefig(OUTPATH)
plt.clf()
