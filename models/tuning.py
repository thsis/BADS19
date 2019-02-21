"""
Algorithms for parameter tuning.
"""

import copy
import numpy as np
from hyperopt import fmin, tpe
from tqdm import tqdm, trange
from joblib import Parallel, delayed


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
    def __init__(self, elitism=0.05, population_size=1000, n_jobs=-1,
                 crossover_strategy="arithmetic",
                 prob_mutation=0.05):
        self.ELITISM = elitism
        self.POPULATION_SIZE = population_size
        self.PROB_MUTATION = prob_mutation
        self.NUM_PARENTS = int(self.ELITISM * self.POPULATION_SIZE)
        self.N_JOBS = n_jobs
        crossovers = {
            "arithmetic": self.__arithmetic_crossover,
            "point": self.__exchange_point_crossover,
            "heuristic": self.__heuristic_crossover}

        self.CROSSOVER = crossovers[crossover_strategy]
        self.HISTORY = []

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
            for i, beta in enumerate(tqdm(self.pool, leave=True, position=1,
                                          desc="Iteration Progress")):
                y_pred = self.__predict_proba(self.X_, beta)
                fit, cut = self.__get_fitness(y_pred)
                fitness[i] = fit
                cutoffs[i] = cut

            # Save best candidate
            fitness_idx = np.argsort(-fitness)
            parents_idx = fitness_idx[:self.NUM_PARENTS]
            self.optimal_candidate = self.pool[fitness_idx[0], :]
            self.HISTORY.append(fitness[fitness_idx[0]])

            # Create new pool
            self.pool = self.pool[parents_idx, :]

            # Crossover
            while self.pool.shape[0] < self.POPULATION_SIZE:
                parent_a_idx = np.random.choice(range(self.NUM_PARENTS))
                parent_b_idx = np.random.choice(range(self.NUM_PARENTS))

                candidate = self.CROSSOVER(parent_a_idx, parent_b_idx)
                if np.random.random() < self.PROB_MUTATION:
                    candidate = self.__mutate(candidate)

                self.pool = np.append(self.pool, [candidate], axis=0)

    def transform(self, X):
        out = self.__predict_proba(X, self.optimal_candidate)
        return out

    def __predict_proba(self, X, b):
        proba = 1 / (1 + np.exp(-X.dot(b)))
        return proba

    def __get_fitness(self, y_prob):
        cut = np.linspace(0, 1, num=100)
        u = Parallel(n_jobs=self.N_JOBS)(
            delayed(self.__get_utility)(y_prob, c) for c in cut)
        best = np.argmax(u)
        return u[best], cut[best]

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
    np.random.seed(42)
    X = np.random.uniform(size=(100, 4))
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    true_beta = np.array([1, -2, 3, -4])
    noise = np.random.normal(size=100)
    y_true = X.dot(true_beta) + noise > 0.5
    item_price = np.random.uniform(low=10, high=100, size=100)

    optimum_pred = 1 / (1 + np.exp(-X_scaled.dot(true_beta)))

    ga = GeneticAlgorithm(elitism=0.2, prob_mutation=0.5,
                          crossover_strategy="point")
    ga.fit(X, y_true, item_price)
    opt = ga._GeneticAlgorithm__get_utility(optimum_pred, 0.5)
    ga.run(iter=20)
    print("Optimum based on true beta: ", opt)
    print(ga.HISTORY)
    print(ga.optimal_candidate)
