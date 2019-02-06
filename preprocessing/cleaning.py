import os
import json
import pandas as pd
import numpy as np


# Paths
colorpath = os.path.join("data", "colorconverter.json")

# ----- General Setup -----
# Data types
dtypes = {"item_id": object, "brand_id": object, "user_id": object}

# Colors
with open(colorpath) as f:
    colors = json.load(f)

colorconverter = {"item_color": lambda x: colors.get(x, "other")}


def clean(datapath):
    """Clean BADS data."""
    # Dates
    date_columns = ["order_date", "delivery_date", "user_dob", "user_reg_date"]

    # NA-Values
    na_values = ["not_reported", "?", "1994-12-31"]

    # ----- Data Cleaning -----
    # Read data
    data = pd.read_csv(datapath, parse_dates=date_columns,
                       index_col=["order_item_id"],
                       na_values=na_values,
                       dtype=dtypes,
                       converters=colorconverter)

    # Compute Median Number of Days until Delivery
    na_del = pd.isna(data.delivery_date)
    ndd_med = (data.delivery_date - data.order_date).median()
    # Impute missing values for "delivery_date" by median
    data.loc[na_del, "delivery_date"] = data.loc[na_del, "order_date"]+ndd_med

    # Remove user_dob from 1900
    data.loc[data.user_dob.dt.year <= 1926, "user_dob"] = np.nan

    # Compute Mean Timedelta between user registration and day of birth
    na_dob = pd.isna(data.user_dob)
    dob_diff = (data.user_reg_date - data.user_dob).mean()
    # Impute missing values for "user_dob" by mean of Timedelta
    data.loc[na_dob, "user_dob"] = data.loc[na_dob, "user_reg_date"] - dob_diff

    return data


if __name__ == "__main__":
    datapath = os.path.join("data", "BADS_WS1819_known.csv")
    cleaned = clean(datapath)
    print(cleaned.describe(include="all"))
    print(cleaned.info())
