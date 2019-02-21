import os
import datetime
from models.tuning import GeneticAlgorithm
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator

from sklearn.model_selection import train_test_split

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
