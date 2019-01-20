import copy
import pandas as pd
import numpy as np


def clean_data(X):
    """Deal with crazy values by imputation or setting to NA."""
    DF = copy.deepcopy(X)

    # Set odd delivery dates to NA and replace by median
    weird_date = pd.to_datetime("1994-12-31")
    DF.loc[DF["delivery_date"] == weird_date, "delivery_date"] = np.nan
    na_delivery = DF.delivery_date.isna()
    three_days = pd.to_timedelta(3, 'd')
    DF.loc[na_delivery, "delivery_date"] = DF.delivery_date + three_days

    # Set 1900-birthdays to NA
    DF.loc[DF.user_dob == pd.to_datetime("1900-11-21")] = np.nan
    return DF


def get_woe(DF, var):
    events = DF.groupby(var)["return"].sum()
    non_events = DF.groupby(var)["return"].count() - events

    events.loc[events == 0] = 0.5
    non_events.loc[non_events == 0] = 0.5

    total_events = DF["return"].sum()
    total_non_events = len(DF) - total_events

    woe = np.log((events/total_events) / (non_events/total_non_events))
    return woe.values