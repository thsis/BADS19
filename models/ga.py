"""
Minimize costs directly using a Linear Classifier and a Genetic Algorithm.

1. Create a logger.
2. Decorate the `models.tuning.minimizer`.
3. Read and clean the data.
4. Define model pipeline.
5. Define hyperparameter space.
6. Perform hyperparameter tuning with train data.
7. Fit pipeline with whole dataset and save predictions.
"""

import os
import datetime
from models.tuning import GeneticAlgorithm
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator

from sklearn.model_selection import train_test_split

# 1. Create a logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join("logs", "ga.log"))
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("{0} Start new run {0}".format("=" * 17))


datapath = os.path.join("data", "BADS_WS1819_known.csv")
unknownpath = os.path.join("data", "BADS_WS1819_unknown.csv")

timecode = datetime.datetime.now().strftime("%Y%m%d_%H%M")
outpath = os.path.join("predictions",  "rf_predictions" + timecode + ".csv")

known = clean(datapath)
unknown = clean(unknownpath)
history = known.append(unknown, sort=False)

train, test = train_test_split(known, test_size=0.2)

cols = ["days_to_delivery",
        "days_to_delivery/order_median_price",
        "item_price",
        "days_to_delivery*order_seqnum",
        "order_max_price",
        "item_price*order_num_items"]

fg = FeatureGenerator(cols=cols)
fg.fit(history, 'return')
X_train, y_train = fg.transform(train)
X_test, y_test = fg.transform(test)

ga = GeneticAlgorithm()
ga.fit(X=X_train, y=y_train, price=train.item_price.values)
ga.run(1)
