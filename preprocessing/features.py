import os
import copy
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

        self.items = self.__fit_items()
        self.orders = self.__fit_orders()
        self.brands = self.__fit_brands()
        self.states = self.__fit_states()

        return self

    def transform(self, X):
        out = X.loc[:, X.columns != self.target_col]
        out = self.__merge(out, self.items)
        out = self.__merge(out, self.orders)
        out = self.__merge(out, self.brands)
        out = self.__merge(out, self.states)

        self.outfeatures = out._get_numeric_data()
        out = out._get_numeric_data().astype(np.float64).values

        # TODO: Create ratios
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

    def __fit_items(self):
        items = self.features.groupby("item_id").agg({
            "days_to_delivery": ["max", "mean", "min"],
            "user_id": "count",
            "item_price": ["min", "mean", "max"]})
        itemcols = ["item_max_delivery", "item_mean_delivery",
                    "item_min_delivery", "item_orders", "item_min_price",
                    "item_mean_price", "item_max_price"]
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
    traindatapath = os.path.join("data", "BADS_WS1819_known.csv")
    preddatapath = os.path.join("data", "BADS_WS1819_unknown.csv")
    known = cleaning.clean(traindatapath)
    unknown = cleaning.clean(preddatapath)

    fg = FeatureGenerator()
    X_known, y_known = fg.fit_transform(known, "return")
    X_unknown = fg.transform(unknown)

    # Playground area:
    numeric = X_known.select_dtypes(['int64', 'float64'])

    out = X_known.copy()
    out.min()
    out["price_off"] = (out.item_max_price-out.item_price) / out.item_max_price
    out["r1"] = out.item_price / out.item_orders
