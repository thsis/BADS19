"""
Fit logistic regressions.

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

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


# 1. Create a logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join("logs", "logistic.log"))
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
outpath = os.path.join("predictions",  "logit_predictions" + timecode + ".csv")
logger.info("{}".format(outpath))

known = clean(datapath)
unknown = clean(unknownpath)
history = known.append(unknown, sort=False)


train, test = train_test_split(known, test_size=0.2)
subset = ["item_price",
          "order_total_value/state_mean_delivery",
          "days_to_delivery",
          "days_to_delivery/state_mean_delivery",
          "item_price/state_mean_delivery"]

fg = FeatureGenerator(cols=subset)
fg.fit(history, 'return')
X_train, y_train = fg.transform(train)
X_test, y_test = fg.transform(test)

# 4. Define model pipeline.
steps = [('scaler', StandardScaler()),
         ('poly', PolynomialFeatures()),
         ('logit', LogisticRegression())]
pipeline = Pipeline(steps)

# 5. Define hyperparameter space.
penalties = ["l1", "l2"]
paramspace = {
    "logit__penalty": hp.choice("logit__penalty", penalties),
    "logit__C": hp.uniform("logit__C", 0, 1),
    "logit__solver": 'saga',
    "logit__n_jobs": -1,
    "logit__fit_intercept": False,
    "logit__max_iter": 5000,
    "poly__include_bias": True}

# 6. Perform hyperparameter tuning with train data.
trials = Trials()
best = max_auc(paramspace=paramspace, trials=trials, max_evals=100)

# Change hyperopt output: 0/1 to "l1", "l2"
best["logit__penalty"] = penalties[best["logit__penalty"]]

logger.info("{0} Optimal Parameter Space {0}".format("-" * 12))
for param, val in best.items():
    logger.info("{0:20s}:\t{1:.5}".format(param, val))

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
