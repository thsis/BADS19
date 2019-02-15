"""
Fit random forest.

1. Create a logger.
2. Decorate the `models.tuning.minimizer`.
3. Read and clean the data.
4. Define model pipeline.
5. Define hyperparameter space.
6. Perform hyperparameter tuning with train data.
7. Fit pipeline with whole dataset and save predictions.
8. Log variable importance.
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
from hyperopt.pyll.base import scope

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


# 1. Get logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("random_forest.log")
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("{0} Start new run {0}".format("=" * 17))


# 2. Decorate the `models.tuning.minimizer`.
@minimizer
def max_auc(params):
    """Optimize model hyperparameters with respect to AUC.

    Parameters
    ----------
    param : dictionary
            Dictionary containing named steps of the pipeline as keys and
            `hyperopt.hp` prior distributions as values.
    """
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
outpath = os.path.join("predictions",  "rf_predictions" + timecode + ".csv")
logger.info("{}".format(outpath))

known = clean(datapath)
unknown = clean(unknownpath)
history = known.append(unknown, sort=False)


train, test = train_test_split(known, test_size=0.2)

fg = FeatureGenerator()
fg.fit(history, 'return')
X_train, y_train = fg.transform(train, ignore_woe=False)
X_test, y_test = fg.transform(test, ignore_woe=False)

# 4. Define model pipeline.
steps = [('scaler', StandardScaler()),
         ('rf', RandomForestClassifier())]
pipeline = Pipeline(steps)

# 5. Define hyperparameter space.
paramspace = {
    "rf__n_estimators": scope.int(hp.quniform("rf__n_estimators",
                                              10, 50, 1)),
    "rf__max_features": hp.uniform("rf__max_features", 0.2, 0.5),
    "rf__max_depth": scope.int(hp.quniform("rf__max_depth",
                                           1, 100, 1)),
    "rf__min_samples_split": hp.uniform("rf__min_samples_split", 0.0001, 0.05),
    "rf__min_samples_leaf": hp.uniform("rf__min_samples_leaf", 0.001, 0.05),
    "rf__n_jobs": -1}

# 6. Perform hyperparameter tuning with train data.
trials = Trials()
best = max_auc(paramspace=paramspace, trials=trials, max_evals=10)

logger.info("{0} Optimal Parameter Space {0}".format("-" * 12))
for param, val in best.items():
    logger.info("{0:20s}:\t{1:.5}".format(param, val))

best["rf__n_estimators"] = int(best["rf__n_estimators"])
best["rf__max_depth"] = int(best["rf__max_depth"])

# 7. Fit pipeline with whole dataset and save predictions.
print("Generate Features")
fg = FeatureGenerator()
fg.fit(history, 'return')
X, y = fg.transform(known, ignore_woe=False)
X_pred = fg.transform(unknown, ignore_woe=False)

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

# 8. Log variable importance.
forest = pipeline.named_steps["rf"]
importances = forest.feature_importances_.round(3)
indices = np.argsort(importances)[::-1]

logger.info("{0} Variable Importance {0}".format("-" * 14))
for f in range(len(importances)):
    varname = fg.cols[indices[f]]
    importance = importances[indices[f]]
    msg = "{0:2s}. {1:40s}({2:.4})".format(str(f + 1), varname, importance)
    logger.info(msg)
