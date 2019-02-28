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
import logging
import argparse

from sklearn.model_selection import train_test_split

from models.tuning import GeneticAlgorithm
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator

# 1. Create a logger.
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FH = logging.FileHandler(os.path.join("logs", "genetic.log"))
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
FORMATTER = logging.Formatter(FORMAT)
FH.setFormatter(FORMATTER)
LOGGER.addHandler(FH)
LOGGER.info("================= Start new run =================")

# 2. Create an argument parser
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--elitism", type=float, default=0.02,
                    help="percentage of population kept in next iteration")
PARSER.add_argument("--population-size", type=int, default=100,
                    help="total number of solutions in pool")
PARSER.add_argument("--jobs", type=int, default=-1,
                    help="number of processors used")
PARSER.add_argument("--crossover-strategy", type=str, default="arithmetic",
                    help="in {'arithmetic', 'point', 'heuristic'}")
PARSER.add_argument("--prob-mutation", type=float, default=0.5,
                    help="probability of mutating a solution")

PARSER.add_argument("--init-loc", type=float, default=0.0,
                    help="location parameter during initialization")
PARSER.add_argument("--init-scale", type=float, default=1.0,
                    help="scale parameter during initialization")

PARSER.add_argument("--maxiter", type=int, default=20,
                    help="maximum number of iterations")
PARSER.add_argument("--subsample", type=float, default=0.5,
                    help="size of subsample")
PARSER.add_argument("--bootstrap", action="store_true", default=False,
                    help="if set, sample with replacement")
PARSER.add_argument("--reset-prob", type=float, default=0.25,
                    help="probability of redrawing the subsample")
ARGS = PARSER.parse_args()
for arg, val in vars(ARGS).items():
    LOGGER.info("%s: %s", arg, val)


DATAPATH = os.path.join("data", "BADS_WS1819_known.csv")
UNKNOWNPATH = os.path.join("data", "BADS_WS1819_unknown.csv")

TIMECODE = datetime.datetime.now().strftime("%Y%m%d_%H%M")
OUTPATH = os.path.join("predictions",  "rf_predictions" + TIMECODE + ".csv")

KNOWN = clean(DATAPATH)
UNKNOWN = clean(UNKNOWNPATH)
HISTORY = KNOWN.append(UNKNOWN, sort=False)

TRAIN, TEST = train_test_split(KNOWN, test_size=0.2)

COLS = ["days_to_delivery",
        "days_to_delivery/order_median_price",
        "item_price",
        "days_to_delivery*order_seqnum",
        "order_max_price",
        "item_price*order_num_items"]

FG = FeatureGenerator(cols=COLS)
FG.fit(HISTORY, 'return')
X_TRAIN, Y_TRAIN = FG.transform(TRAIN)
X_TEST, Y_TEST = FG.transform(TEST)

GA = GeneticAlgorithm()
GA.fit(X=X_TRAIN, y=Y_TRAIN, price=TRAIN.item_price.values)
GA.run(1)
GA = GeneticAlgorithm(elitism=ARGS.elitism,
                      population_size=ARGS.population_size,
                      n_jobs=ARGS.jobs,
                      crossover_strategy=ARGS.crossover_strategy,
                      prob_mutation=ARGS.prob_mutation)
GA.fit(X_TRAIN, Y_TRAIN, X_TRAIN[:, 1],
       fit_intercept=True,
       loc=ARGS.init_loc,
       scale=ARGS.init_scale)
RES = GA.run(maxiter=ARGS.maxiter,
             subsample=ARGS.subsample,
             bootstrap=ARGS.bootstrap,
             reset_prob=ARGS.reset_prob)
TEST_PRED = GA.predict(X_TEST)
TEST_SCORE = GA.get_utility(TEST_PRED, Y_TEST, X_TEST[:, 1], GA.optimal_cutoff)
