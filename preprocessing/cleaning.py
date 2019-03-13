"""
Clean the dataset.

Gather all functions which clean the data and compute features that do not
require aggregation.
"""

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


# Sizes
def is_in_group(x, low, high):
    group = np.linspace(low, high, num=6)
    lower = group[:-1]
    upper = group[1:]
    label = ["XS", "S", "M", "L", "XL"]
    for l, u, lab in zip(lower, upper, label):
        if l <= x < u:
            return lab


def clean_size(x):
    groups = [(0, 15), (18, 30), (30, 52), (52, 59), (75, 177), (2930, 4035)]
    try:
        x = float(x)
        for (low, high) in groups:
            out = is_in_group(x, low, high)
            if out is not None:
                return out
    except (TypeError, ValueError):
        return x


def clean(datapath):
    """Clean BADS data.

    Read the data and perform the following tasks:
    * parse dates: `order_date`, `delivery_date`, `user_dob`, `user_reg_date`.
    * replace non-standard coding for missing values.
    * drop esoteric colors.
    * impute missing values for `delivery_date` column by its mean.
    * remove odd reported values for user dates of birth.
    * impute missing `user_dob`: `user_reg_date - mean(user_reg_date-user_dob)`
    * extract dummies according to `size`column:
        + `is_item_pants`: if `size` matches exactly 4 numbers.
        + `is_item_clothes`: if `size` is not `unsized`.
        + `is_item_underwear`: if `size` matches up to 2 numbers.
        + `is_letter_coded`: if `size` not `unsized` but contains characters.
    * extract features according to `delivery_date`:
        + `delivery_thu`: indicator if item was delivered on a Thursday.
        + `delivery_fri`: indicator if item was delivered on a Friday.
        + `days_to_delivery`: number of days between order and delivery.

    Parameters
    ----------
    datapath : str
        Path to dataset.

    Returns
    -------
    data : pd.DataFrame
        Cleaned dataset.
    """
    # Dates
    date_columns = ["order_date", "delivery_date", "user_dob", "user_reg_date"]

    # NA-Values
    na_values = ["not_reported", "1994-12-31"]

    # ----- Data Cleaning -----
    # Read data
    data = pd.read_csv(datapath, parse_dates=date_columns,
                       index_col=["order_item_id"],
                       na_values=na_values,
                       dtype=dtypes)

    data["color"] = data.item_color.apply(lambda x: colors.get(x, "other"))

    # Compute Mean Number of Days until Delivery
    na_del = pd.isna(data.delivery_date)
    ndd_med = (data.delivery_date - data.order_date).median()
    # Impute missing values for "delivery_date" by mean
    data.loc[na_del, "delivery_date"] = data.loc[na_del, "order_date"]+ndd_med
    data["is_na_delivery_date"] = na_del

    # Remove user_dob from 1900
    data.loc[data.user_dob.dt.year <= 1926, "user_dob"] = np.nan

    # Mark missing birth-date
    data["is_dob_missing"] = data.user_dob.isna()

    # Compute Median Timedelta between user registration and day of birth
    na_dob = pd.isna(data.user_dob)
    dob_diff = (data.user_reg_date - data.user_dob).median()
    # Impute missing values for "user_dob" by median of Timedelta
    data.loc[na_dob, "user_dob"] = data.loc[na_dob, "user_reg_date"] - dob_diff

    # Mark pants
    data["is_item_pants"] = data.item_size.str.match("[0-9]{4}")
    # Mark underwear
    data["is_item_underwear"] = data.item_size.str.match("^[3-9]\\+?$")
    # Mark letter denomination
    data["is_letter_coded"] = data.item_size.str.match("^[xlsmXLSM]")
    # Mark non clothing
    data["is_item_clothes"] = ~data.item_size.str.match("unsized")

    # Time related features
    data["days_to_delivery"] = (data.delivery_date-data.order_date).dt.days
    data["user_tenure"] = (data.order_date - data.user_reg_date).dt.days
    data["age"] = (data.order_date - data.user_dob).dt.days
    data["month"] = data.delivery_date.dt.month.astype(object)
    # Constrain user_tenure to be non-negative.
    data.loc[data.user_tenure < 0, "user_tenure"] = 0

    # Additional features
    data["delivery_thu"] = data.delivery_date.dt.dayofweek == 4
    data["delivery_fri"] = data.delivery_date.dt.dayofweek == 5
    data["age2"] = data.age ** 2
    data["user_tenure2"] = data.user_tenure ** 2
    data["item_price2"] = data.item_price ** 2

    # "Cleaned" item size column
    data["item_size_cl"] = data.item_size.str.upper()
    data["item_size_cl"] = data.item_size_cl.str.replace("+", ".5")
    data["item_size_cl"] = data.item_size_cl.str.replace("[xX]{2,}[Ll]", "XL")
    data["item_size_cl"] = data.item_size_cl.str.replace("[xX]{2,}[Ss]", "XS")
    data["item_size_cl"] = data.item_size_cl.apply(clean_size)

    return data


if __name__ == "__main__":
    datapath = os.path.join("data", "BADS_WS1819_known.csv")
    unknownpath = os.path.join("data", "BADS_WS1819_unknown.csv")
    cleaned = clean(datapath)
    print(cleaned.describe(include="all"))
    print(cleaned.info())
