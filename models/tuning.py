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
    def __init__(self, elitism=0.02, population_size=100, n_jobs=-1,
                 crossover_strategy="arithmetic",
                 prob_mutation=0.5):
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
        self.HISTORY = {"mean_pop_fitness": [],
                        "best_fitness": [],
                        "best_candidate": [],
                        "oob_fitness": []}

    def fit(self, X, y, price, fit_intercept=True, loc=0, scale=1):
        self.fit_intercept = fit_intercept
        self.n, self.m = X.shape
        self.X = X
        self.X_means = self.X.mean(axis=0)
        self.X_stds = self.X.std()
        self.X_ = (self.X - self.X_means) / self.X_stds

        if self.fit_intercept:
            self.m += 1
            # Add column of ones
            self.X_ = np.append(np.ones(self.n).reshape((-1, 1)),
                                self.X_,
                                axis=1)

        self.price = price
        self.y_true = y

        self.pool = np.random.normal(loc=loc, scale=scale,
                                     size=(self.POPULATION_SIZE, self.m))

    def run(self, iter=10, subsample=None, bootstrap=False, reset_prob=1):
        if subsample is None:
            self.sample_size = self.n
            self.BOOTSTRAP = False
            self.oob_size = 1
        else:
            assert 0 < subsample < 1
            self.sample_size = int(subsample * self.n)
            self.BOOTSTRAP = bootstrap
            self.oob_size = self.n - self.sample_size
        for i in trange(iter, desc='Generation', leave=True, position=0):
            # Initialize
            self.fitness = np.full(self.POPULATION_SIZE, -np.inf)
            self.cutoffs = np.full(self.POPULATION_SIZE, -np.inf)
            # Draw random subsample to prevent overfitting
            if i == 0:
                samples = self.__get_sample()
                sample, sample_y, sample_p, oob, oob_y, oob_p = samples
            if np.random.random() < reset_prob:
                samples = self.__get_sample()
                sample, sample_y, sample_p, oob, oob_y, oob_p = samples

            # Determine fitness
            for i, beta in enumerate(tqdm(self.pool, leave=True, position=1,
                                          desc="Iteration Progress")):
                y_pred = self.__predict_proba(sample, beta)
                fit, cut = self.get_fitness(y_pred, sample_y, sample_p)
                self.fitness[i] = fit
                self.cutoffs[i] = cut

            # Save best candidate
            fitness_idx = np.argsort(-self.fitness)
            parents_idx = fitness_idx[:self.NUM_PARENTS]
            self.__update_history(fitness_idx, parents_idx, oob, oob_y, oob_p)

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

        # Determine Solution with best OOB-fitness
        opt_idx = np.argmax(self.HISTORY["oob_fitness"])
        self.optimal_candidate = self.HISTORY["best_candidate"][opt_idx]
        return self.optimal_candidate

    def transform(self, X):
        n, m = X.shape
        X_ = (X - self.X_means) / self.X_stds
        if self.fit_intercept:
            X_ = np.append(np.ones(n), X_, axis=1)
        out = self.__predict_proba(X_, self.optimal_candidate)
        return out

    def __get_sample(self):
        sample_idx = np.random.choice(range(self.n),
                                      size=self.sample_size,
                                      replace=self.BOOTSTRAP)
        oob_idx = [i for i in range(self.n) if i not in sample_idx]

        sample = self.X_[sample_idx, :]
        sample_y = self.y_true[sample_idx]
        sample_price = self.price[sample_idx]

        oob_sample = self.X_[oob_idx]
        oob_y = self.y_true[oob_idx]
        oob_price = self.price[oob_idx]

        return sample, sample_y, sample_price, oob_sample, oob_y, oob_price

    def __predict_proba(self, X, b):
        proba = 1 / (1 + np.exp(-X.dot(b)))
        return proba

    def get_fitness(self, y_prob, y_true, price):
        cut = np.linspace(0, 1, num=100)
        u = Parallel(n_jobs=self.N_JOBS)(
            delayed(self.__get_utility)(y_prob, y_true, price, c) for c in cut)
        best = np.argmax(u)
        return u[best], cut[best]

    def __get_utility(self, y_prob, y_true, price, cutoff):
        utility = 0
        y_pred = y_prob > cutoff
        stack = np.stack([y_pred, y_true, price], axis=1)
        for y_p, y_t, p in stack:
            if y_p == y_t:
                continue
            elif y_p and not y_t:
                utility -= 0.5 * p
            elif not y_p and y_t:
                utility -= 2.5 * (3 + 0.1*p)
        return utility

    def __update_history(self, fitness_idx, parents_idx, oob, y, p):
        self.optimal_candidate = self.pool[fitness_idx[0], :]
        prob = self.__predict_proba(oob, self.optimal_candidate)
        self.oob_fitness, _ = self.get_fitness(prob, y, p)
        # Rescale fitness to compare with OOB-fitness
        self.HISTORY["best_fitness"].append(self.fitness[fitness_idx[0]] / self.sample_size)
        self.HISTORY["best_candidate"].append(self.pool[fitness_idx[0], :])
        self.HISTORY["mean_pop_fitness"].append(self.fitness.mean() / self.sample_size)
        self.HISTORY["oob_fitness"].append(self.oob_fitness / self.oob_size)

    def __arithmetic_crossover(self, idx_a, idx_b):
        lam = np.random.random()
        out = lam * self.pool[idx_a, :] + (1-lam) * self.pool[idx_b, :]
        return out

    def __exchange_point_crossover(self, idx_a, idx_b):
        keep_idx_a = np.random.randint(low=0, high=self.m // 2)
        out = copy.copy(self.pool[idx_b, :])
        out[keep_idx_a] = self.pool[idx_b, keep_idx_a]
        return out

    def __heuristic_crossover(self, idx_a, idx_b):
        lam = np.random.random()
        parents = (self.pool[idx_a], self.pool[idx_b])
        fit = (self.fitness[idx_a], self.fitness[idx_b])
        max_idx, min_idx = np.argmax(fit), np.argmin(fit)
        out = parents[max_idx] + lam * (parents[max_idx]-parents[min_idx])
        return out

    def __mutate(self, candidate):
        pos = np.random.randint(low=0, high=self.m)
        candidate[pos] = np.random.uniform(-5, 5)
        return candidate


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.uniform(size=(1000, 4))
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    true_beta = np.array([1, -2, 3, -4])
    y_true = X.dot(true_beta) > 0.5
    item_price = np.random.uniform(low=10, high=100, size=1000)

    optimum_pred = 1 / (1 + np.exp(-X_scaled.dot(true_beta)))

    ga = GeneticAlgorithm(elitism=0.2, prob_mutation=0.3,
                          crossover_strategy="point")
    ga.fit(X, y_true, item_price)

    opt, cut = ga.get_fitness(optimum_pred,
                              y_true,
                              item_price)
    print("Optimum based on true beta: ", opt)
    print("Best solution:")
    res = ga.run(iter=30, subsample=0.5, bootstrap=False, reset_prob=0.33)
    print(res)

    print("\n| Best Fitness | Mean Fitness | OOB Fitness")
    print("|" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14)
    for best, avg, oob in zip(ga.HISTORY["best_fitness"],
                              ga.HISTORY["mean_pop_fitness"],
                              ga.HISTORY["oob_fitness"]):
        print("|{0:10.3f}{3:<4}|{1:10.5f}{3:<4}|{2:10.5f}".format(best,
                                                                  avg,
                                                                  oob, " "))
