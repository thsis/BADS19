"""
Test suite for genetic algorithm
"""
import os
import argparse
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from models.tuning import GeneticAlgorithm


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FH = logging.FileHandler(os.path.join("logs", "genetic_test.log"))
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
FORMATTER = logging.Formatter(FORMAT)
FH.setFormatter(FORMATTER)
LOGGER.addHandler(FH)
LOGGER.info("================= Start new run =================")

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

np.random.seed(42)

X = np.random.uniform(size=(1000, 4))
X_ = np.append(np.ones(1000).reshape((-1, 1)), X, axis=1)
BETA = np.array([-0.5, 1.0, -2.0, 3.0, -4.0])

PROB_Y = 1 / (1 + np.exp(-X_.dot(BETA)))
Y = PROB_Y > 0.7

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y)

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
TEST_SCORE = GA.get_utility(
    TEST_PRED, Y_TEST, X_TEST[:, 1], GA.optimal_cutoff)

# Calculate Baseline
X_BASELINE = (X_TEST - GA.means) / GA.stds
X_BASELINE = np.append(np.ones(len(X_BASELINE)).reshape((-1, 1)),
                       X_BASELINE, axis=1)
BASELINE_PRED = GA.predict_proba(X_BASELINE, BETA)
BASELINE = GA.get_utility(BASELINE_PRED, Y_TEST, X_TEST[:, 1], 0.7)
# Calculate total predicted cost
COST_BASELINE = GA.predict_cost(X_TEST, Y_TEST, X_TEST[:, 1], BETA, 0.7)
COST_PREDICT = GA.predict_cost()

GA.plot(os.path.join("eda", "genetic_test.png"), title="Simulated Data", figsize=(7, 5))
# Print to console
print("\nBaseline score: % 6.4f" % (BASELINE / len(BASELINE_PRED)))
print("Test score: % 11.4f" % (TEST_SCORE / len(TEST_PRED)))

print("Predicted Costs:")
print("Baseline: % 10.2f | Prediction: % 10.2f" %
      (COST_BASELINE, COST_PREDICT))

# Log events
LOGGER.info("------------------- Results -------------------")
LOGGER.info("Baseline score: % 6.10f", BASELINE / len(BASELINE_PRED))
LOGGER.info("Test score: % 6.10f", TEST_SCORE / len(TEST_PRED))
LOGGER.info("Coefficients: %s", RES)
LOGGER.info("True coefficient: %s", BETA)
LOGGER.info("Total Cost: %f", GA.get_utility())

LOGGER.info("--------------- Predicted Costs ---------------")
LOGGER.info("Baseline: % 10.2f | Prediction: % 10.2f",
            COST_BASELINE, COST_PREDICT)


LOGGER.info("------------------- History -------------------")
LOGGER.info("| Best Fitness | Mean Fitness | OOB Fitness")
LOGGER.info("|--------------|--------------|-------------")
for best, avg, oob in zip(GA.history["best_fitness"],
                          GA.history["mean_pop_fitness"],
                          GA.history["oob_fitness"]):
    LOGGER.info("| % 12.7f | % 12.7f | % 12.7f", best, avg, oob)
