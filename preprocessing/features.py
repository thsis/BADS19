import os
import itertools
import numpy as np
from preprocessing import cleaning


class FeatureGenerator(object):
    """Generate Features for the different algorithms.

    This class interface facilitates the fitting and transforming of different
    datasets and gathers the numerous aggregation levels.

    Parameters
    ----------
    cols : list
           List of column names to be returned at most. Must be a subset of all
           columns that are defined within the `fit` method.

    Attributes
    ----------
    cols : list
           Column names of the returned array.

    dropcols : list
               Column names that should be dropped, because their correlation
               with other variables is too high.

    target : pd.Series
             Series of class values/labels.

    features : pd.DataFrame
               Original features, copy of `data`.

    item_woe : pd.DataFrame
               `item_id` and associated weight of evidence.

    color_woe : pd.DataFrame
                `item_color` and associated weight of evidence.

    brand_woe : pd.DataFrame
                `brand_id` and associated weight of evidence.

    size_woe : pd.DataFrame
               `item_size` and associated weight of evidence.

    items : pd.DataFrame
            Contains general information on items. Groupby-Keys: `item_id`.

    orders : pd.DataFrame
            Contains general information on orders. Groupby-Keys: `user_id`
            `order_date`.

    brands : pd.DataFrame
            Contains general information on brands. Groupby-Keys: `brand_id`.

    states : pd.DataFrame
            Contains general information on states. Groupby-Keys: `user_state`.

    outfeatures: pd.DataFrame
                 DataFrame version of the returned `numpy.ndarray`.

    Methods
    -------

    """

    def __init__(self, cols=None):
        self.cols = cols
        self.dropcols = [
            "item_max_price", "brand_min_price", "state_min_delivery",
            "order_total_value", "order_median_price",
            "item_price/order_num_sizes", "item_price/order_num_colors",
            "item_price/order_num_items", "days_to_delivery/order_num_colors",
            "days_to_delivery/order_num_sizes",
            "days_to_delivery/order_num_items",
            "user_tenure/order_num_items",
            "user_tenure/order_num_sizes",
            "user_tenure/order_num_colors",
            "item_max_delivery/order_num_items",
            "item_max_delivery/order_num_colors",
            "item_max_delivery/order_num_sizes",
            "item_mean_delivery/order_num_items",
            "item_mean_delivery/order_num_colors",
            "item_mean_delivery/order_num_sizes",
            "item_min_delivery/order_num_items",
            "item_min_delivery/order_num_colors",
            "item_min_delivery/order_num_sizes",
            "item_max_price/order_num_items",
            "item_max_price/order_num_sizes",
            "item_max_price/order_num_colors",
            "order_max_price/order_num_items",
            "order_max_price/order_num_sizes",
            "order_max_price/order_num_colors",
            "order_median_price/order_num_items",
            "order_median_price/order_num_sizes",
            "order_median_price/order_num_colors",
            "brand_min_price/order_num_items",
            "brand_min_price/order_num_sizes",
            "brand_min_price/order_num_colors",
            "state_min_delivery/order_num_items",
            "state_min_delivery/order_num_sizes",
            "state_min_delivery/order_num_colors"]

    def fit(self, data, target_col):
        """Compute aggregated data according to different levels.

        The different aggregation levels combine information on:
        * items: group by `item_id`
        * orders: group by `user_id` and `order_date`
        * brands: group by `brand_id`
        * states: group by `states`

        Parameters
        ----------
        data : pd.DataFrame
               Full set of information to be used. This can be the `known`
               dataset or a concatenation of `known` and `unknown`.
        target_col : str
                     Name of the column that contains the labels
        """
        self.target_col = target_col
        self.features = data.loc[:, data.columns != self.target_col].copy()
        self.target = data.loc[:, self.target_col].copy()

        self.item_woe = self.__fit_woe(data[~data[target_col].isna()],
                                       "order_item_id")
        self.color_woe = self.__fit_woe(data[~data[target_col].isna()],
                                        "item_color")
        self.brand_woe = self.__fit_woe(data[~data[target_col].isna()],
                                        "brand_id")
        self.size_woe = self.__fit_woe(data[~data[target_col].isna()],
                                       "item_size")

        self.items = self.__fit_items()
        self.orders = self.__fit_orders()
        self.brands = self.__fit_brands()
        self.states = self.__fit_states()

        return self

    def transform(self, X, ignore_woe=True):
        """Add aggregated information to X and compute additional features.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be transformed.
        ignore_woe : bool, optional
                     Flag if Weight of Evidence columns should be ignored.
                     Default is `True`.

        Returns
        -------
        out, [y] : np.ndarray
                   Note that `y` will only be returned if columns of `X`
                   contain the target column.
        """
        # Merge fitted tables for items, orders, brands and states
        out = X.loc[:, X.columns != self.target_col]
        out = self.__merge(out, self.items, impute=self.items.mean())
        out = self.__merge(out, self.orders, impute=0)
        out = self.__merge(out, self.brands, impute=self.brands.mean())
        out = self.__merge(out, self.states, impute=self.states.mean())

        woetab = (self.item_woe, self.color_woe, self.brand_woe, self.size_woe)
        if not ignore_woe:
            for right in woetab:
                out = self.__merge(out, right)

        out = out._get_numeric_data()

        # Create special features
        price_off = (out.item_max_price-out.item_price) / out.item_max_price
        out["price_off"] = price_off.fillna(value=0)

        # Create ratios
        out = self.__get_ratios(out)

        out = out.fillna(out.mean())
        self.outfeatures = out
        if self.cols is None:
            self.cols = out.columns
        out = out.loc[:, self.cols].astype(np.float64).values
        # HACK:
        if self.target_col in X.columns:
            return out, X.loc[:, self.target_col]
        else:
            return out

    def fit_transform(self, data, target_col, ignore_woe=True):
        """Fit data then transform it."""
        self.fit(data, target_col)
        out = self.transform(data, ignore_woe)
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
        return woe.rename("woe_" + var)

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
            "item_price": ["max", "sum", "median"],
            "item_size": "nunique",
            "item_color": "nunique"})
        ordercols = ["order_num_items", "order_max_price",
                     "order_total_value", "order_median_price",
                     "order_num_sizes", "order_num_colors"]
        orders.columns = ordercols
        seqnum = orders.reset_index().groupby("user_id").order_date.rank()
        seqnum.index = orders.index
        orders["order_seqnum"] = seqnum
        return orders

    def __fit_brands(self):
        brands = self.features.groupby("brand_id").agg({
            "item_price": ["min", "mean", "max"]})
        brandcols = ["brand_min_price", "brand_mean_price", "brand_max_price"]
        brands.columns = brandcols
        return brands

    def __fit_states(self):
        states = self.features.groupby("user_state").agg({
            "days_to_delivery": ["min", "mean", "max"]})
        statecols = ["state_min_delivery", "state_mean_delivery",
                     "state_max_delivery"]
        states.columns = statecols
        return states

    def __get_ratios(self, data):
        # Find all columns that can be put into denominator
        denominator = ['item_orders', 'order_num_items', 'order_num_sizes',
                       'order_num_colors', 'state_mean_delivery']
        out = data.copy()
        is_dummy = out.dtypes == bool
        is_denominator = out.columns.isin(denominator)
        # Remove features due to ensuing correlation
        protected = []
        is_protected = out.columns.isin(protected)
        nominator_idx = (~is_denominator) & (~is_dummy) & (~is_protected)
        nominator = out.columns[nominator_idx]

        # Cartesian product of all numerically possible columns
        combinations = itertools.product(nominator, denominator)

        for nom, denom in combinations:
            out["{}/{}".format(nom, denom)] = out[nom] / (out[denom] + 1)

        # Remove features that are highly correlated with ratios
        out = out.drop(columns=self.dropcols)
        return out


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns
    traindatapath = os.path.join("data", "BADS_WS1819_known.csv")
    preddatapath = os.path.join("data", "BADS_WS1819_unknown.csv")
    known = cleaning.clean(traindatapath)
    unknown = cleaning.clean(preddatapath)

    subset = ["item_price",
              "order_total_value/state_mean_delivery",
              "days_to_delivery",
              "days_to_delivery/state_mean_delivery",
              "item_price/state_mean_delivery"]
    subset = None

    fg = FeatureGenerator(cols=subset)
    X_known, y_known = fg.fit_transform(known, "return")

    out = fg.outfeatures.copy()
    corr = out.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                cmap=plt.cm.coolwarm,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    ax.set_title("Feature Correlation Plot")

    X_unknown = fg.transform(unknown)
    assert X_known.shape[1] == X_unknown.shape[1]
