import os
import copy
import numpy as np
import pandas as pd
from preprocessing import cleaning


class FeatureGenerator(object):
    """Generate Features for BADS1819 known and unknown datasets."""

    def __init__(self):
        pass

    def get_woe(self, DF, var):
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

    def __get_woe_cols(self, DF):
        """Fit WOE and store tables as attributes."""
        self.WOE_item_id = self.get_woe(DF, "item_id")
        self.WOE_item_size = self.get_woe(DF, "item_size")
        self.WOE_item_color = self.get_woe(DF, "item_color")
        self.WOE_brand_id = self.get_woe(DF, "brand_id")

        outdata = DF.reset_index().merge(
            self.WOE_item_id.reset_index(),
            how="left").set_index("order_item_id").fillna(0)
        outdata = outdata.reset_index().merge(
            self.WOE_item_size.reset_index(),
            how="left").set_index("order_item_id").fillna(0)
        outdata = outdata.reset_index().merge(
            self.WOE_item_color.reset_index(),
            how="left").set_index("order_item_id").fillna(0)
        outdata = outdata.reset_index().merge(
            self.WOE_brand_id.reset_index(),
            how="left").set_index("order_item_id").fillna(0)
        return outdata

    def __get_order_cols(self, DF):
        # ----- ORDERS -----
        orders = DF.groupby(["user_id", "order_date"]).agg({
            "item_id": "count",
            "item_price": ["sum", "mean", "median"]})
        orders.reset_index(inplace=True)
        orders.columns = ["user_id", "order_date", "basket_size",
                          "basket_value", "avg_price_basket",
                          "median_price_basket"]

        # Skewness measure: deviation from median value in current order
        skew_basket = orders.avg_price_basket - orders.median_price_basket
        orders["skew_price_basket"] = skew_basket

        # Calculate number of previous orders
        orders["num_prev_orders"] = orders.reset_index().groupby(
            ["user_id"]).order_date.rank() - 1

        # Calculate total money ordered before current order
        orders["amt_prev_orders"] = orders.groupby(
            "user_id").basket_value.cumsum() - orders.basket_value

        # Calculate items ordered before current order
        prev_orders = orders.groupby("user_id").basket_size.cumsum()
        orders["item_prev_orders"] = prev_orders - orders.basket_size

        return orders

    def __get_item_cols(self, DF):
        # ----- ITEMS -----
        items = DF.groupby("item_id").agg({
            "item_price": "max",
            "user_id": "count"})
        items.reset_index(inplace=True)
        items.columns = ["item_id", "max_price", "num_item_orders"]

        return items

    def __get_dummy_cols(self, DF):
        # ----- TIME -----
        dow_order = pd.get_dummies(DF.order_date.dt.dayofweek,
                                   prefix="is_order_dow")
        dow_delivery = pd.get_dummies(DF.delivery_date.dt.dayofweek,
                                      prefix="is_delivery_dow")
        month_order = pd.get_dummies(DF.order_date.dt.month,
                                     prefix="is_order_month")

        # ----- DUMMIES -----
        user_title = pd.get_dummies(DF.user_title,
                                    prefix="is_title")

        region = pd.get_dummies(DF.user_state,
                                prefix="is_state")

        out = pd.concat([DF, dow_order, dow_delivery,
                         month_order, user_title, region],
                        join_axes=[DF.index],
                        axis=1)
        return out

    def fit(self, data):
        self.data = copy.deepcopy(data)
        # ----- Weight of Evidence -----
        self.features = self.__get_woe_cols(data)

        return self

    def transform(self, data):
        # ----- Weight of Evidence -----
        outdata = data.reset_index().merge(
            self.WOE_item_id.reset_index(),
            how="left").set_index("order_item_id").fillna(0)
        outdata = outdata.reset_index().merge(
            self.WOE_item_size.reset_index(),
            how="left").set_index("order_item_id").fillna(0)
        outdata = outdata.reset_index().merge(
            self.WOE_item_color.reset_index(),
            how="left").set_index("order_item_id").fillna(0)
        outdata = outdata.reset_index().merge(
            self.WOE_brand_id.reset_index(),
            how="left").set_index("order_item_id").fillna(0)

        # ----- 'Creative' Engineered Features -----
        # ----- ORDERS -----
        orders = self.__get_order_cols(data)
        outdata = outdata.reset_index().merge(
            orders, how="left").set_index("order_item_id")

        # Deviation from median item price per order
        outdata["item_skew"] = outdata.item_price - outdata.median_price_basket

        # ----- ITEMS -----
        items = self.__get_item_cols(data)
        outdata = outdata.reset_index().merge(
                items, how="left").set_index("order_item_id")

        # ----- DUMMIES -----
        outdata = self.__get_dummy_cols(outdata)

        # ----- CASE-VARIABLES ----- (i.e. no join needed)
        # Days until item was delivered
        days_until_delivery = outdata.delivery_date - outdata.order_date
        outdata["days_until_delivery"] = days_until_delivery.dt.days

        # Days between order and registration date
        tenure = (outdata.order_date - outdata.user_reg_date).dt.days
        outdata["user_tenure"] = tenure

        price_off = (outdata.max_price-outdata.item_price) / outdata.max_price
        outdata["price_off"] = price_off.fillna(0)

        # ----- USERS -----
        outdata["first_timer"] = (outdata.num_prev_orders == 0).astype(int)

        # TODO:
        # ----- 'Dull' Features: i.e. ratios over numerical variables
        outdata["log_price"] = np.log(outdata.item_price.values + 1)

        # TODO:
        # Log(variable)
        outdata = outdata._get_numeric_data()
        X = outdata.loc[:, outdata.columns != "return"].values
        y = outdata["return"].values

        self.features = outdata
        return X, y

    def fit_transform(self, data):
        """Calculate features and return features & labels."""
        self.data = copy.deepcopy(data)
        self.fit(data)
        X, y = self.transform(data)
        return X, y

    def generate(self, newdata):
        """Generate features from unknown data,"""
        self.newdata = copy.deepcopy(newdata)
        data = pd.concat([self.data, self.newdata], sort=True).sort_values(
            ["user_id", "order_date"])
        self.new_features = data
        self.predict_data = data.loc[self.newdata.index]

        _, _ = self.transform(data)
        X = self.features.loc[self.newdata.index,
                              self.features.columns != "return"].values
        return X


if __name__ == "__main__":
    datapath = os.path.join("data", "BADS_WS1819_known.csv")
    unknownpath = os.path.join("data", "BADS_WS1819_unknown.csv")

    known = cleaning.clean(datapath)
    unknown = cleaning.clean(unknownpath)
    fg = FeatureGenerator()
    fg.fit(known)
    X_test = fg.generate(unknown)
    dataset = fg.predict_data
