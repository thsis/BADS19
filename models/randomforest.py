import os
import logging
import datetime
import pandas as pd

from preprocessing.features import FeatureGenerator
from preprocessing.cleaning import clean

from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm


# Get logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("random_forest.log")
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("{0} Start new run {0}".format("=" * 10))


def minimizer(objective):
    def outer(paramspace, trials, max_evals=100):
        """Generate an inner objective-function and optimize it."""
        pbar = tqdm(total=max_evals)

        def inner(*args, **kwargs):
            """Update the progress bar and call the objective function."""
            pbar.update()
            return objective(*args, **kwargs)

        best = fmin(fn=inner,
                    space=paramspace,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
        pbar.close()
        return best
    return outer


@minimizer
def max_auc(params):
    model = pipeline.set_params(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict_proba(X_test)
    test_score = roc_auc_score(y_true=y_test,
                               y_score=y_pred_test[:, 1])
    y_pred_train = model.predict_proba(X_train)
    train_score = roc_auc_score(y_true=y_train,
                                y_score=y_pred_train[:, 1])
    info = "Train AUC: {0:.{2}f}\tTest AUC: {1:.{2}f}".format(train_score,
                                                              test_score, 3)
    logger.info(info)

    return {"loss": -test_score, "status": STATUS_OK}


datapath = os.path.join("data", "BADS_WS1819_known.csv")
unknownpath = os.path.join("data", "BADS_WS1819_unknown.csv")

timecode = datetime.datetime.now().strftime("%Y%m%d_%H%M")
outpath = os.path.join("predictions",  "rf_predictions" + timecode + ".csv")

known = clean(datapath)
unknown = clean(unknownpath)

train, test = train_test_split(known, test_size=0.2)


fg = FeatureGenerator()
X_train, y_train = fg.fit_transform(train)
X_test, y_test = fg.transform(test)

steps = [('scaler', StandardScaler()),
         ('rf', RandomForestClassifier())]
pipeline = Pipeline(steps)

paramspace = {
    "rf__n_estimators": scope.int(hp.quniform("n_estimators", 10, 10, 1)),
    "rf__max_features": hp.uniform("max_features", 0.5, 1),
    "rf__n_jobs": -1}

trials = Trials()
best = max_auc(paramspace=paramspace, trials=trials, max_evals=10)
best["n_estimators"] = int(best["n_estimators"])

# Predictions:
print("Generate Features")
fg = FeatureGenerator()
X, y = fg.fit_transform(known)
X_pred = fg.generate(unknown)
print("Calculate Predictions")
clf = pipeline.set_params(**best)
clf.fit(X, y)
preds = clf.predict_proba(X_pred)
print("Save to file.")

predictions = pd.DataFrame(preds[:, 1]).set_index(fg.predict_data.index)
predictions.to_csv(outpath)
