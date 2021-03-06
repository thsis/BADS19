
import os
import itertools
import numpy as np
import pandas as pd
from preprocessing import cleaning


class FeatureGenerator:
    """Generate Features for the different algorithms.

    This class interface facilitates the fitting and transforming of different
    datasets and gathers the numerous aggregation levels.

    Parameters
    ----------
    `cols` : `list`
           List of column names to be returned at most. Must be a subset of all
           columns that are defined within the `fit` method.

    Attributes
    ----------
    `cols` : `list`
           Column names of the returned array.

    `dropcols` : `list`
               Column names that should be dropped, because their correlation
               with other variables is too high.

    `target` : `pd.Series`
             Series of class values/labels.

    `features` : `pd.DataFrame`
               Original features, copy of `data`.

    `item_woe` : `pd.DataFrame`
               `item_id` and associated weight of evidence.

    `color_woe` : `pd.DataFrame`
                `item_color` and associated weight of evidence.

    `brand_woe` : `pd.DataFrame`
                `brand_id` and associated weight of evidence.

    `size_woe` : `pd.DataFrame`
               `item_size` and associated weight of evidence.

    `items` : `pd.DataFrame`
            Contains general information on items. Groupby-Keys: `item_id`.

    `orders` : `pd.DataFrame`
            Contains general information on orders. Groupby-Keys: `user_id`
            `order_date`.

    `brands` : `pd.DataFrame`
            Contains general information on brands. Groupby-Keys: `brand_id`.

    `states` : `pd.DataFrame`
            Contains general information on states. Groupby-Keys: `user_state`.

    `outfeatures` : `pd.DataFrame`
                 DataFrame version of the returned `numpy.ndarray`.

    Methods
    -------
    `fit(self, data, target_col)` :
        Fit data.

    `transformself, X, ignore_woe=True, add_dummies=False, \
    add_interactions=True, add_ratios=True)` :
        Tranform data.

    `fit_transform(self, data, target_col, ignore_woe=True, add_dummies=False, \
    add_interactions=True, add_ratios=True)` :
        Fit data, then transform it.
    """

    def __init__(self, cols=None):
        self.cols = cols
        self.target_col, self.features, self.target = None, None, None
        self.items, self.orders, self.brands, self.states = None, None, None, None

        self.color_woe, self.month_woe, self.size_woe = None, None, None
        self.price_qs = None
        self.dummies = None
        self.outfeatures = None
        with open(os.path.join("preprocessing", "blacklist.txt"), "r") as f:
            self.dropcols = f.read().splitlines()

    def fit(self, data, target_col):
        """Compute aggregated data according to different levels.

        The different aggregation levels combine information on:
        * items: group by `item_id`
        * orders: group by `user_id` and `order_date`
        * brands: group by `brand_id`
        * states: group by `states`

        Parameters
        ----------
        `data` : `pd.DataFrame`
            Full set of information to be used. This can be the `known`
            dataset or a concatenation of `known` and `unknown`.
        `target_col` : `str`
            Name of the column that contains the labels
        """
        self.target_col = target_col
        self.features = data.loc[:, data.columns != self.target_col].copy()
        self.target = data.loc[:, self.target_col].copy()

        self.color_woe = self.__fit_woe(data[~data[target_col].isna()],
                                        "item_color")
        self.month_woe = self.__fit_woe(data[~data[target_col].isna()],
                                        "month")
        self.size_woe = self.__fit_woe(data[~data[target_col].isna()],
                                       "item_size")

        self.items = self.__fit_items()
        self.orders = self.__fit_orders()
        self.brands = self.__fit_brands()
        self.states = self.__fit_states()
        self.return_history = self.__fit_return_history()

        self.price_qs = np.quantile(data.item_price.values,
                                    [0.25, 0.50, 0.75])

        return self

    def transform(self, X, ignore_woe=True, add_dummies=False,
                  add_interactions=True, add_ratios=True):
        """Add aggregated information to X and compute additional features.

        Parameters
        ----------
        `X` : `pd.DataFrame`
            DataFrame to be transformed.
        `ignore_woe` : `bool`, optional
            Flag if Weight of Evidence columns should be ignored.
            Default is `True`.

        Returns
        -------
        `out, [y]` : `np.ndarray`
            Note that `y` will only be returned if columns of `X`
            contain the target column.
        """
        # Merge fitted tables for items, orders, brands and states
        out = X.loc[:, X.columns != self.target_col]
        out = self.__merge(out, self.items, impute=self.items.mean())
        out = self.__merge(out, self.orders, impute={
            "order_num_items": 1,
            "order_total_value": self.orders.order_total_value.mean(),
            "order_median_price": self.orders.order_median_price.mean(),
            "order_min_price": self.orders.order_min_price.mean(),
            "order_max_price": self.orders.order_max_price.mean(),
            "order_num_sizes": 1,
            "order_num_colors": 1,
            "order_seqnum": 1})
        out = self.__merge(out, self.brands, impute=self.brands.mean())
        out = self.__merge(out, self.states, impute=self.states.mean())
        out = self.__merge(out, self.return_history, impute=0)

        # Fix broken columns after merge
        out["order_has_gift"] = (out.order_min_price == 0).astype(float)
        out.loc[out.order_seqnum == 0, "order_seqnum"] += 1
        out.loc[out.item_orders.isnull()] = 1
        out.loc[out.order_num_sizes.isnull()] = 1

        woetab = (self.color_woe, self.month_woe, self.size_woe)
        if not ignore_woe:
            for right in woetab:
                out = self.__merge(out, right)

        out = out._get_numeric_data()
        outnames = [col for col in out.columns.tolist() if col[:3] != "woe"]

        # Create special features
        price_off = (out.item_max_price-out.item_price) / out.item_max_price
        out["price_off"] = price_off.fillna(value=0)

        # Create ratios
        if add_ratios:
            out = self.__get_ratios(out, outnames)

        # Create interactions
        if add_interactions:
            out = self.__get_interactions(out, outnames)

        # Add dummies
        if add_dummies:
            dummy_cols = ["user_title", "user_state", "item_size_cl", "month"]
            self.dummies = pd.get_dummies(X[dummy_cols])
            out = pd.concat([out, self.dummies], axis=1)
            q1, q2, q3 = self.price_qs
            out["is_q1_value"] = out.item_price <= q1
            out["is_q2_value"] = (out.item_price > q1) & (out.item_price <= q2)
            out["is_q3_value"] = (out.item_price > q2) & (out.item_price <= q3)
            out["is_q4_value"] = out.item_price > q3
        if not add_dummies:
            dummy_cols = [c for c in out.columns if c[:2] == "is"]
            dummy_cols += ["delivery_thu", "delivery_fri", "order_has_gift"]
            self.dropcols += dummy_cols

        out = out.fillna(out.mean())
        out = out.drop(columns=self.dropcols, errors="ignore")
        self.outfeatures = out

        if self.cols is None:
            self.cols = out.columns.tolist()

        out = out[self.cols].values.astype(np.float64)

        # HACK:
        if self.target_col in X.columns:
            return out, X.loc[:, self.target_col]
        else:
            return out

    def fit_transform(self, data, target_col, ignore_woe=True,
                      add_dummies=False, add_interactions=True,
                      add_ratios=True):
        """Fit data then transform it."""
        self.fit(data, target_col)
        out = self.transform(data,
                             ignore_woe=ignore_woe,
                             add_dummies=add_dummies,
                             add_interactions=add_interactions,
                             add_ratios=add_ratios)
        return out

    def __merge(self, left, right, impute=np.nan):
        merged = left.reset_index().merge(right.reset_index(), how="left")
        merged = merged.set_index("order_item_id")
        merged = merged.fillna(value=impute)
        return merged

    def __fit_woe(self, DF, var):
        events = DF.groupby(var)["return"].sum()
        non_events = DF.groupby(var)["return"].count() - events

        events.loc[events == 0] = 0.5
        non_events.loc[non_events == 0] = 0.5

        total_events = DF["return"].sum()
        total_non_events = len(DF) - total_events

        woe = np.log((events/total_events) / (non_events/total_non_events))
        woe = woe.to_frame()
        woe.columns = ["woe_" + var]
        return woe

    def __fit_items(self):
        items = self.features.groupby("item_id").agg({
            "days_to_delivery": ["max", "mean", "min"],
            "user_id": "count",
            "item_price": "max"})
        itemcols = ["item_max_delivery", "item_mean_delivery",
                    "item_min_delivery", "item_orders", "item_max_price"]
        items.columns = itemcols
        return items

    def __fit_orders(self):
        orders = self.features.groupby(["user_id", "order_date"]).agg({
            "item_id": "count",
            "item_price": ["max", "sum", "median", "min"],
            "item_size": "nunique",
            "item_color": "nunique"})
        ordercols = ["order_num_items", "order_max_price",
                     "order_total_value", "order_median_price",
                     "order_min_price", "order_num_sizes", "order_num_colors"]
        orders.columns = ordercols
        seqnum = orders.reset_index().groupby("user_id").order_date.rank(
            method="dense")
        seqnum.index = orders.index
        orders["order_seqnum"] = seqnum

        brandcounts = self.features.groupby(
            ["user_id", "order_date"]).brand_id.value_counts()
        brandcounts = brandcounts.rename("order_num_brand_id").reset_index()

        orders = orders.reset_index().merge(brandcounts, how="left")
        orders = orders.set_index(["user_id", "order_date"])
        return orders

    def __fit_return_history(self):
        # Create indicator if the order contained any returned items
        returns = self.target.dropna().groupby(
            [self.features.user_id, self.features.order_date]
            ).max().reset_index()

        # Create lag in order to avoid a target leak
        returns["history"] = returns.groupby(
            "user_id")["return"].shift(1).fillna(0)
        returns["seqnum"] = returns.groupby("user_id").order_date.rank(
            method="dense")
        returns["history_cumsum"] = returns.groupby("user_id").history.cumsum()
        returns["history_perc"] = returns.history_cumsum / returns.seqnum
        outcols = ["history", "history_cumsum", "history_perc"]
        returns = returns.set_index(["user_id", "order_date"])

        return returns[outcols]

    def __fit_brands(self):
        brands = self.features.groupby("brand_id").agg({
            "item_price": ["min", "mean", "max"]})
        brandcols = ["brand_min_price", "brand_mean_price", "brand_max_price"]
        brands.columns = brandcols
        return brands

    def __fit_states(self):
        states = self.features.groupby("user_state").agg({
            "days_to_delivery": ["mean", "max"]})
        statecols = ["state_mean_delivery",
                     "state_max_delivery"]
        states.columns = statecols
        return states

    def __get_ratios(self, data, columns):
        # Find all columns that can be put into denominator
        out = data.copy()
        is_dummy = out.dtypes == bool
        cols = out.columns[~is_dummy]
        m = len(cols) // 2
        nominator = cols[:m]
        denominator = cols[m:]

        # Cartesian product of all numerically possible columns
        combinations = itertools.product(nominator, denominator)

        # Inhibit ratios with Weight of Evidence columns.
        for nom, denom in combinations:
            if (nom[:3] == "woe") or (denom[:3] == "woe"):
                continue
            if out[denom].min() == 0:
                continue
            if (nom == "month") or (denom == "month"):
                continue
            # HACK: make sure price-offs are in the nominator.
            if denom == "price_off":
                out["{}/{}".format(denom, nom)] = out[denom] / (out[nom] + 1)
            else:
                out["{}/{}".format(nom, denom)] = out[nom] / (out[denom] + 1)

        return out

    def __get_interactions(self, data, columns):
        m = len(columns) // 2
        factor_a_cols = columns[:m]
        factor_b_cols = columns[m:]
        out = data.copy()
        combinations = itertools.product(factor_a_cols, factor_b_cols)
        # Inhibit interactions with Weight of Evidence columns.
        for a, b in combinations:
            if (a[:3] == "woe") or (b[:3] == "woe"):
                continue
            if out[a].isnull().any() or out[b].isnull().any():
                continue
            if (a == "month") or (b == "month"):
                continue
            if (out[a].dtype == bool) & (out[b].dtype == bool):
                out["{}*{}".format(a, b)] = out[a] & out[b]
            else:
                out["{}*{}".format(a, b)] = out[a] * out[b]

        return out


if __name__ == "__main__":
    DATAPATH = os.path.join("data", "BADS_WS1819_known.csv")
    UNKNOWNPATH = os.path.join("data", "BADS_WS1819_unknown.csv")

    KNOWN = cleaning.clean(DATAPATH)
    UNKNOWN = cleaning.clean(UNKNOWNPATH)
    HISTORY = KNOWN.append(UNKNOWN, sort=False)
    FG = FeatureGenerator()
    FG.fit(HISTORY, "return")

    X_KNOWN, Y_KNOWN = FG.transform(KNOWN, add_dummies=True)
