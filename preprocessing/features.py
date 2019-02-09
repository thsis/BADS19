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
        self.target = data.loc[:, target_col].copy()

        items = self.features.groupby("item_id")

        return self

    def transform(self, X)




# Playground area:

traindatapath = os.path.join()
