import os
import itertools
import numpy as np
import pandas as pd
from preprocessing import cleaning


class FeatureGenerator(object):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, data, target_col):
        self.target_col = target_col
        self.train_idx = data.index.copy()
        self.features = data.loc[:, data.columns != self.target_col].copy()
        self.target = data.loc[:, self.target_col].copy()

        self.item_woe = self.__fit_woe(data, "order_item_id")
        self.color_woe = self.__fit_woe(data, "item_color")
        self.brand_woe = self.__fit_woe(data, "brand_id")
        self.size_woe = self.__fit_woe(data, "item_size")

        self.items = self.__fit_items()
        self.orders = self.__fit_orders()
        self.brands = self.__fit_brands()
        self.states = self.__fit_states()

        return self

    def transform(self, X, ignore_woe=True):
        # Merge fitted tables for items, orders, brands and states
        out = X.loc[:, X.columns != self.target_col]

        for right in self.items, self.orders, self.brands, self.states:
            out = self.__merge(out, right)

        woetab = (self.item_woe, self.color_woe, self.brand_woe, self.size_woe)
        if not ignore_woe:
            for right in woetab:
                out = self.__merge(out, right)

        out = out._get_numeric_data()

        # Create ratios
        # Find all columns that can be put into denominator
        denominator = ['item_orders', 'order_num_items', 'order_num_sizes',
                       'order_num_colors', 'state_mean_delivery']

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
            out["{}/{}".format(nom, denom)] = out[nom] / out[denom]

        # Remove features that are highly correlated with ratios
        out = out.drop(columns=["item_max_price", "brand_min_price",
                                "state_min_delivery",
                                "order_total_value", "order_median_price",
                                # Drop
                                "item_price/order_num_sizes",
                                "item_price/order_num_colors",
                                "item_price/order_num_items",
                                "days_to_delivery/order_num_colors",
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
                                "state_min_delivery/order_num_colors"])

        self.outfeatures = out
        self.column_names = out.columns
        out = out.fillna(out.mean())
        out = out.astype(np.float64).values
        # HACK:
        if self.target_col in X.columns:
            return out, X.loc[:, self.target_col]
        else:
            return out

    def fit_transform(self, data, target_col):
        self.fit(data, target_col)
        out = self.transform(data)
        return out

    def __merge(self, left, right):
        impute = right.mean()
        merged = left.reset_index().merge(right.reset_index(), how="left")
        merged = merged.set_index("order_item_id")
        merged = merged.fillna(impute)
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns
    traindatapath = os.path.join("data", "BADS_WS1819_known.csv")
    preddatapath = os.path.join("data", "BADS_WS1819_unknown.csv")
    known = cleaning.clean(traindatapath)
    unknown = cleaning.clean(preddatapath)

    fg = FeatureGenerator()
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
