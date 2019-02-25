"""
Fit artificial neural network.

1. Create a logger.
2. Decorate the `models.tuning.minimizer`.
3. Read and clean the data.
4. Define model pipeline.
5. Define hyperparameter space.
6. Perform hyperparameter tuning with train data.
7. Fit pipeline with whole dataset and save predictions.
"""

import os
import logging
import datetime
import pandas as pd
import numpy as np
from preprocessing.features import FeatureGenerator
from preprocessing.cleaning import clean
from models.tuning import minimizer

from hyperopt import hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


# 1. Create a logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join("logs", "ann.log"))
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("{0} Start new run {0}".format("=" * 17))


# 2. Decorate the `models.tuning.minimizer`.
@minimizer
def max_auc(params):
    model = pipeline.set_params(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict_proba(X_test)
    y_pred_train = model.predict_proba(X_train)
    test_score = roc_auc_score(y_true=y_test,
                               y_score=y_pred_test[:, 1])
    train_score = roc_auc_score(y_true=y_train,
                                y_score=y_pred_train[:, 1])
    info = "Train AUC: {0:.{2}f}\tTest AUC: {1:.{2}f}".format(train_score,
                                                              test_score, 3)
    logger.info(info)
    # HACK: Prevent selecting a overfitting solution
    score = test_score - (np.abs(test_score - train_score) > 0.03)

    return {"loss": -score, "status": STATUS_OK}


# 3. Read and clean the data.
datapath = os.path.join("data", "BADS_WS1819_known.csv")
unknownpath = os.path.join("data", "BADS_WS1819_unknown.csv")

timecode = datetime.datetime.now().strftime("%Y%m%d_%H%M")
outpath = os.path.join("predictions",  "ann_predictions" + timecode + ".csv")
logger.info("{}".format(outpath))

known = clean(datapath)
unknown = clean(unknownpath)
history = known.append(unknown, sort=False)
cols = ["days_to_delivery",
        "item_price*order_num_items",
        "item_price*order_num_sizes",
        "days_to_delivery/brand_mean_price", 
        "days_to_delivery*order_seqnum", 
        "days_to_delivery*brand_max_price", 
        "days_to_delivery/order_median_price", 
        "order_total_value", 
        "item_price", 
        "order_total_value/order_num_colors", 
        "is_item_clothes*order_median_price",
        "item_price/brand_mean_price", 
        "order_total_value/brand_mean_price", 
        "days_to_delivery*order_total_value", 
        "item_mean_delivery/brand_min_price", 
        "item_mean_delivery", 
        "item_price/brand_max_price", 
        "order_max_price/order_median_price", 
        "is_letter_coded*brand_min_price"]

train, test = train_test_split(known, test_size=0.2)

fg = FeatureGenerator(cols=cols)
fg.fit(history, 'return')
X_train, y_train = fg.transform(train)
X_test, y_test = fg.transform(test)

# 4. Define model pipeline.
steps = [('scaler', MinMaxScaler()),
         ('ann', MLPClassifier())]
pipeline = Pipeline(steps)

# 5. Define hyperparameter space.
architecture = [(100,), (1000,), (500,), (750,), (100, 100, 100), (100, 100), (30, 30, 30, 30)]
activations = ["identity", "logistic", "tanh", "relu"]
solvers = ["adam", "sgd"]

paramspace = {
    "ann__hidden_layer_sizes": hp.choice("ann__hidden_layer_sizes",
                                         architecture),
    "ann__activation": hp.choice("ann__activation", activations),
    "ann__solver": hp.choice("ann__solver", solvers),
    "ann__alpha": hp.uniform("ann__alpha", 0.00001, 0.0005),
    "ann__momentum": hp.uniform("ann__momentum", 0.1, 0.9),
    "ann__early_stopping": True,
    "ann__batch_size": scope.int(hp.quniform("ann__batch_size", 2, 1000, 1)),
    "ann__max_iter": 1000
}

# 6. Perform hyperparameter tuning with train data.
trials = Trials()
best = max_auc(paramspace=paramspace, trials=trials, max_evals=100)

best["ann__hidden_layer_sizes"] = architecture[best["ann__hidden_layer_sizes"]]
best["ann__activation"] = activations[best["ann__activation"]]
best["ann__solver"] = solvers[best["ann__solver"]]
best["ann__batch_size"] = int(best["ann__batch_size"])

logger.info("{0} Optimal Parameter Space {0}".format("-" * 12))
for param, val in best.items():
    logger.info("{0:<30} {1:>30}".format(param + ":", str(val)))


# 7. Fit pipeline with whole dataset and save predictions.
print("Generate Features")
fg = FeatureGenerator()
fg.fit(history, 'return')
X, y = fg.transform(known, "return")
X_pred = fg.transform(unknown)

print("Calculate Predictions")
clf = pipeline.set_params(**best)
clf.fit(X, y)
preds = clf.predict_proba(X_pred)
y_score = clf.predict_proba(X)

print("Save to file.")
predictions = pd.DataFrame(preds[:, 1]).set_index(unknown.index)
print("Calculate Train Score")
train_score = roc_auc_score(y_true=y,
                            y_score=y_score[:, 1])
predictions.to_csv(outpath, header=["return"])
logger.info("Approximate score: {0:.3}".format(train_score))
