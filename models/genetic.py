"""
Minimize costs directly using a Linear Classifier and a Genetic Algorithm.

1. Create a logger.
2. Create an argument parser
3. Read and clean the data.
4.
5. Define hyperparameter space.
6. Perform hyperparameter tuning with train data.
7. Fit pipeline with whole dataset and save predictions.
"""

import os
import datetime
import logging
import argparse

import numpy as np
import pandas as pd
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
PARSER.add_argument("-e", "--elitism", type=float, default=0.02,
                    help="percentage of population kept in next iteration")
PARSER.add_argument("-p", "--population-size", type=int, default=100,
                    help="total number of solutions in pool")
PARSER.add_argument("--jobs", type=int, default=-1,
                    help="number of processors used")
PARSER.add_argument("-c", "--crossover-strategy", type=str, default="arithmetic",
                    help="in {'arithmetic', 'point', 'heuristic'}")
PARSER.add_argument("-m", "--prob-mutation", type=float, default=0.5,
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

# 3. Load and clean the data
DATAPATH = os.path.join("data", "BADS_WS1819_known.csv")
UNKNOWNPATH = os.path.join("data", "BADS_WS1819_unknown.csv")

TIMECODE = datetime.datetime.now().strftime("%Y%m%d_%H%M")
OUTPATH = os.path.join(
    "predictions", "genetic_predictions" + TIMECODE + ".csv")

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

# 4. Run Genetic Algorithm
GA = GeneticAlgorithm(elitism=ARGS.elitism,
                      population_size=ARGS.population_size,
                      n_jobs=ARGS.jobs,
                      crossover_strategy=ARGS.crossover_strategy,
                      prob_mutation=ARGS.prob_mutation)
GA.fit(X=X_TRAIN,
       y=Y_TRAIN,
       price=TRAIN.item_price.values,
       fit_intercept=True,
       loc=ARGS.init_loc,
       scale=ARGS.init_scale)
RES = GA.run(maxiter=ARGS.maxiter,
             subsample=ARGS.subsample,
             bootstrap=ARGS.bootstrap,
             reset_prob=ARGS.reset_prob)
TEST_PRED = GA.predict(X_TEST)
TEST_SCORE = GA.get_utility(TEST_PRED, Y_TEST, X_TEST[:, 1], GA.optimal_cutoff)

# Baseline: everybody gets the message
BASELINE = GA.get_utility(np.zeros(len(Y_TEST)), Y_TEST, X_TEST[:, 1], 0.5)

# Log events
LOGGER.info("------------------- Results -------------------")
LOGGER.info("Baseline score: % -6.2f", BASELINE / len(Y_TEST) * len(UNKNOWN))
LOGGER.info("Test score: % -6.2f", TEST_SCORE / len(TEST_PRED) * len(UNKNOWN))
LOGGER.info("Coefficients: %s", RES.round(2))

LOGGER.info("------------------- History -------------------")
LOGGER.info("| Best Fitness | Mean Fitness | OOB Fitness")
LOGGER.info("|--------------|--------------|-------------")
for best, avg, oob in zip(GA.history["best_fitness"],
                          GA.history["mean_pop_fitness"],
                          GA.history["oob_fitness"]):
    LOGGER.info("| % 12.7f | % 12.7f | % 12.7f", best, avg, oob)

print("\nSave Predictions")
X_PRED = FG.transform(UNKNOWN)
PREDICTIONS = GA.predict(X_PRED)
PREDICTIONS = pd.DataFrame(PREDICTIONS, index=UNKNOWN.index,
                           columns=["return"], dtype=float)
PREDICTIONS.to_csv(OUTPATH)
