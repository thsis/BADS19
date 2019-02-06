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

    def fit(self, data):
        """Calculate features for known dataset."""

        self.data = copy.deepcopy(data)
        # ----- Weight of Evidence -----
        self.WOE_item_id = self.get_woe(self.data, "item_id")
        self.WOE_item_size = self.get_woe(self.data, "item_size")
        self.WOE_item_color = self.get_woe(self.data, "item_color")
        self.WOE_brand_id = self.get_woe(self.data, "brand_id")

        # ----- 'Creative' Engineered Features -----
        # Days until item was delivered
        days_until_delivery = (self.data.delivery_date - self.data.order_date)
        self.data["days_until_delivery"] = days_until_delivery.dt.days

        # Days between order and registration date
        tenure = (self.data.order_date - self.data.user_reg_date).dt.days
        self.data["user_tenure"] = tenure

        # basket size
        # Create item identifier, that also reflects sizes
        item_size_id = self.data["item_id"] + "-" + self.data["item_size"]
        self.data["item_size_id"] = item_size_id

        # ----- ORDERS -----
        orders = self.data.groupby(["user_id", "order_date"]).agg({
            "item_size_id": "count",
            "item_id": "count",
            "item_price": ["sum", "mean"]})
        orders.reset_index(inplace=True)
        orders.columns = ["user_id", "order_date", "basket_size",
                          "order_distinct_items",
                          "basket_value", "avg_price_basket"]

        # Calculate number of previous orders
        orders["num_prev_orders"] = orders.reset_index().groupby(
            ["user_id"]).order_date.rank() - 1

        # Calculate total money ordered before current order
        orders["amt_prev_orders"] = orders.groupby(
            "user_id").basket_value.cumsum() - orders.basket_value

        # Calculate items ordered before current order
        prev_orders = orders.groupby("user_id").order_distinct_items.cumsum()
        orders["item_prev_orders"] = prev_orders - orders.order_distinct_items

        # Merge orders into data
        self.data = self.data.merge(orders)

        # ----- USERS -----
        self.data["first_timer"] = self.data.num_prev_orders == 0

        # ----- TIME -----
        dow_order = pd.get_dummies(self.data.order_date.dt.dayofweek,
                                   prefix="is_order_dow")
        dow_delivery = pd.get_dummies(self.data.delivery_date.dt.dayofweek,
                                      prefix="is_delivery_dow")
        month_order = pd.get_dummies(self.data.order_date.dt.month,
                                     prefix="is_order_month")

        # ----- DUMMIES -----
        user_title = pd.get_dummies(self.data.user_title,
                                    prefix="is_title")

        self.data = pd.concat([self.data, dow_order, dow_delivery,
                               month_order, user_title],
                              axis=1)

        # TODO:
        # ----- 'Dull' Features: i.e. ratios over numerical variables
        return self.data


if __name__ == "__main__":
    datapath = os.path.join("data", "BADS_WS1819_known.csv")
    data = cleaning.clean(datapath)
    fg = FeatureGenerator()
    dataset = fg.fit(data)
