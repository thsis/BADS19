import argparse
import logging
import numpy as np
from models.tuning import GeneticAlgorithm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("GA_test.log")
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("{0} Start new run {0}".format("=" * 17))

parser = argparse.ArgumentParser()
parser.add_argument("--elitism", type=float, default=0.02)
parser.add_argument("--population-size", type=int, default=100)
parser.add_argument("--jobs", type=int, default=-1)
parser.add_argument("--crossover-strategy", type=str, default="arithmetic")
parser.add_argument("--prob-mutation", type=float, default=0.5)

parser.add_argument("--ignore-intercept", action="store_true", default=False)
parser.add_argument("--init-loc", type=float, default=0.0)
parser.add_argument("--init-scale", type=float, default=1.0)

parser.add_argument("--iter", type=int, default=20)
parser.add_argument("--subsample", type=float, default=0.5)
parser.add_argument("--bootstrap", action="store_true", default=False)
parser.add_argument("--reset-prob", type=float, default=0.25)

args = parser.parse_args()
print(args.ignore_intercept)
np.random.seed(42)

X = np.random.uniform(size=(1000, 4))
X_ = np.append(np.ones(1000).reshape((-1, 1)), X, axis=1)
p = X[:, 0]
beta = np.array([-0.5, 1.0, -2.0, 3.0, -4.0])

prob_y = 1 / (1 + np.exp(-X_.dot(beta)))
y = prob_y > 0.7

ga = GeneticAlgorithm(elitism=args.elitism,
                      population_size=args.population_size,
                      n_jobs=args.jobs,
                      crossover_strategy=args.crossover_strategy,
                      prob_mutation=args.prob_mutation)
ga.fit(X, y, p,
       fit_intercept=not args.ignore_intercept,
       loc=args.init_loc,
       scale=args.init_scale)
ga.run(iter=args.iter,
       subsample=args.subsample,
       bootstrap=args.bootstrap,
       reset_prob=args.reset_prob)
