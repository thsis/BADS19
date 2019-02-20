"""
Algorithms for parameter tuning.
"""

import copy
import numpy as np
from hyperopt import fmin, tpe
from tqdm import tqdm, trange
from scipy.stats import rankdata


def minimizer(objective):
    """Create a function to minimize an objective function.

    This function is intended to be used as a decorator.

    Parameters
    ----------
    objective
                The objective function that will be optimized. This function
                should return a dictionary of the form
                `{"loss": loss, "status": hyperopt.STATUS_OK}`.
    Returns
    -------
    function
                This decorated function will be optimized over a provided
                parameter space.
    """
    def outer(paramspace, trials, max_evals=100):
        pbar = tqdm(total=max_evals)

        def inner(*args, **kwargs):
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


class GeneticAlgorithm(object):
    def __init__(self, elitism=0.05, population_size=1000,
                 crossover_strategy="arithmetic",
                 prob_mutation=0.05):
        self.ELITISM = elitism
        self.POPULATION_SIZE = population_size
        self.PROB_MUTATION = prob_mutation
        self.NUM_PARENTS = int(self.ELITISM * self.POPULATION_SIZE)
        crossovers = {
            "arithmetic": self.__arithmetic_crossover,
            "point": self.__exchange_point_crossover,
            "heuristic": self.__heuristic_crossover}

        self.CROSSOVER = crossovers[crossover_strategy]

    def fit(self, X, y, price, fit_intercept=True, loc=0, scale=10):
        self.n, self.m = X.shape
        self.X = X
        self.X_means = self.X.mean(axis=0)
        self.X_stds = self.X.std()
        self.X_ = (self.X - self.X_means) / self.X_stds

        if fit_intercept:
            self.m += 1
            # Add column of ones
            self.X_ = np.append(self.X_, np.ones(self.n).reshape((-1, 1)),
                                axis=1)

        self.price = price
        self.y_true = y

        self.pool = np.random.normal(loc=loc, scale=scale,
                                     size=(self.POPULATION_SIZE, self.m))

    def run(self, iter=10):
        for _ in trange(iter, desc='Generation', leave=True, position=0):
            # Initialize
            fitness = np.full(self.POPULATION_SIZE, -np.inf)
            cutoffs = np.full(self.POPULATION_SIZE, -np.inf)

            # Determine fitness
            for i, beta in enumerate(tqdm(self.pool, leave=True, position=1)):
                y_pred = self.__predict_proba(beta)
                fit, cut = self.__get_fitness(y_pred)
                fitness[i] = fit
                cutoffs[i] = cut

            # Create new pool
            fitness_idx = np.argsort(-fitness)
            parents_idx = fitness_idx[:self.NUM_PARENTS]
            self.pool = self.pool[parents_idx, :]

        # Calculate mating chances
        parents_fit = fitness[fitness_idx][:self.NUM_PARENTS]
        r = rankdata(parents_fit)
        p = r / r.sum()

        # Crossover
        while self.pool.shape[0] < self.POPULATION_SIZE:
            parent_a_idx = np.random.choice(range(self.NUM_PARENTS), p=p)
            parent_b_idx = np.random.choice(range(self.NUM_PARENTS), p=p)

            candidate = self.CROSSOVER(parent_a_idx, parent_b_idx)
            if np.random.random() < self.PROB_MUTATION:
                candidate = self.__mutate(candidate)

            self.pool = np.append(self.pool, [candidate], axis=0)

    def __predict_proba(self, b):
        proba = 1 / (1 + np.exp(-self.X_.dot(b)))
        return proba

    def __get_fitness(self, y_prob):
        cutoffs = np.linspace(0, 1, num=100)
        utilities = [self.__get_utility(y_prob, c) for c in cutoffs]
        best = np.argmax(utilities)
        return utilities[best], cutoffs[best]

    def __get_utility(self, y_prob, cutoff):
        utility = 0
        y_pred = y_prob > cutoff
        stack = np.stack([y_pred, self.y_true, self.price], axis=1)
        for y_p, y_t, p in stack:
            if y_p == y_t:
                continue
            elif y_p and not y_t:
                utility -= 0.5 * p
            elif not y_p and y_t:
                utility -= 2.5 * (3 + 0.1*p)
        return utility

    def __arithmetic_crossover(self, idx_a, idx_b):
        lam = np.random.random()
        out = lam * self.pool[idx_a, :] + (1-lam) * self.pool[idx_b, :]
        return out

    def __exchange_point_crossover(self, idx_a, idx_b):
        keep_idx_a = np.random.randint(low=0, high=self.m // 2)
        out = copy.copy(self.pool[idx_b, :])
        out[keep_idx_a] = self.pool[idx_b, keep_idx_a]
        return out

    def __heuristic_crossover(self):
        raise NotImplementedError

    def __mutate(self, candidate):
        pos = np.random.randint(low=0, high=self.m)
        candidate[pos] = np.random.uniform(-5, 5)
        return candidate




if __name__ == "__main__":
    import os
    from preprocessing.cleaning import clean
    from preprocessing.features import FeatureGenerator

    train_data_path = os.path.join("data", "BADS_WS1819_known.csv")
    unknown_data_path = os.path.join("data", "BADS_WS1819_unknown.csv")

    known = clean(train_data_path)
    unknown = clean(unknown_data_path)
    history = known.append(unknown, sort=False)

    cols = ["days_to_delivery",
            "days_to_delivery/order_median_price",
            "item_price*order_num_sizes",
            "item_price",
            "days_to_delivery*order_seqnum",
            "order_max_price",
            "days_to_delivery*order_total_value",
            "price_off/item_price",
            "is_item_clothes*order_num_colors",
            "price_off/days_to_delivery"]

    fg = FeatureGenerator(cols=cols)
    fg.fit(history, 'return')
    X, y = fg.transform(known, "return")
    print(X.shape)

    ga = GeneticAlgorithm(population_size=100)
    ga.fit(X, y, known.item_price.values)
    ga.run()
