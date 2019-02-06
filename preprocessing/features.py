import os
import numpy as np
import pandas as pd
from preprocessing import cleaning


def get_woe(DF, var):
    """Calculate Weight of Evidence for `var`."""
    events = DF.groupby(var)["return"].sum()
    non_events = DF.groupby(var)["return"].count() - events

    events.loc[events == 0] = 0.5
    non_events.loc[non_events == 0] = 0.5

    total_events = DF["return"].sum()
    total_non_events = len(DF) - total_events

    woe = np.log((events/total_events) / (non_events/total_non_events))
    woe.name = var + "_woe"
    return woe


datapath = os.path.join("data", "BADS_WS1819_known.csv")
data = cleaning.clean(datapath)

# ----- Weight of Evidence -----
WOE_item_id = get_woe(data, "item_id")
WOE_item_size = get_woe(data, "item_size")
WOE_brand_id = get_woe(data, "brand_id")

# ----- 'Creative' Engineered Features -----
# Days until item was delivered
data["days_until_delivery"] = (data.delivery_date - data.order_date).dt.days

# Days between order and registration date
data["user_tenure"] = (data.order_date - data.user_reg_date).dt.days

data.head()
# basket size
# Create item identifier, that also reflects sizes
data["item_size_id"] = data["item_id"] + "-" + data["item_size"]
orders = data.groupby(["user_id", "order_date"]).agg({
    "item_size_id": "count",
    "item_price": ["sum", "mean"]})
orders.resetindex

data.merge(orders)
