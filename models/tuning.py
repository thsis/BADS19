"""
Algorithms for parameter tuning.
"""

import copy
import numpy as np
from hyperopt import fmin, tpe
from tqdm import tqdm, trange
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


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
        """Call fmin and manage the progress bar."""
        pbar = tqdm(total=max_evals)

        def inner(*args, **kwargs):
            """Update progress bar and call the objective function."""
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


def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


class GeneticAlgorithm:
    """Minimize non-continuous functions through a Genetic Algorithm.

    Attributes
    ----------
    elitism : float
              Percentage of individuals to be kept in next iteration
              and used to form offspring for the next iteration.
    population_size : int
                      Number of individuals in pool.
    prob_mutation : float
                    Probability of mutating a particular individual.
    num_parents : int
                  Number of parents.
    n_jobs : int
             Number of processors used. When -1, all processors are
             used, when -2 all but one are used.
    history: dict
             Dictionary containing information of the current run.
             Keeps records on the average and optimal fitness of the
             population, the optimal candidate and threshold and the
             out of bag fitness in each iteration.
    optimal_candidate : numpy.array
                        Optimal parameter vector.
    optimal_cutoff : float
                     Optimal threshold for transforming probablistic
                     predictions into class labels.


    Methods
    -------
    fit

    run

    predict_proba

    predict

    get_fitness

    get_utility
    """

    def __init__(self, elitism=0.02, population_size=100, n_jobs=-1,
                 crossover_strategy="arithmetic",
                 prob_mutation=0.5):
        self.elitism = elitism
        self.population_size = population_size
        self.prob_mutation = prob_mutation
        self.num_parents = int(self.elitism * self.population_size)
        self.n_jobs = n_jobs
        crossovers = {
            "arithmetic": self.__arithmetic_crossover,
            "point": self.__exchange_point_crossover,
            "heuristic": self.__heuristic_crossover}

        self.crossover = crossovers[crossover_strategy]
        self.history = {"mean_pop_fitness": [],
                        "best_fitness": [],
                        "best_candidate": [],
                        "best_cutoff": [],
                        "oob_fitness": []}
        self.pool = None
        self.X_, self.X, self.means, self.stds, self.X = [None] * 5
        self.n, self.m = None, None
        self.oob_cutoff = None
        self.oob_fitness = None
        self.fit_intercept = None
        self.optimal_candidate = None
        self.optimal_cutoff = None
        self.price, self.y_true = None, None
        self.sample_size = None
        self.bootstrap = None
        self.oob_size, self.fitness, self.cutoffs = None, None, None
        self.maxiter, self.reset_prob = None, None

    def fit(self, X, y, price, fit_intercept=True, loc=0, scale=1):
        """Stage Genetic Algorithm with regards to data.

        Parameters
        ----------
        X : numpy.ndarray
            Train-data to calculate fitness with.
        y : array-like
            Vector of true labels.
        price : array-like
            Vector of item prices.
        fit_intercept : bool
            Flag if intercept should be fitted.
        loc : float
            Location parameter during initialization.
        scale : float
            Scale parameter during initialization.

        Returns
        -------
        self
        """
        self.fit_intercept = fit_intercept
        self.n, self.m = X.shape
        self.X = X
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std()
        self.X_ = (self.X - self.means) / self.stds

        if self.fit_intercept:
            self.m += 1
            # Add column of ones
            self.X_ = np.append(np.ones((self.n, 1)), self.X_, axis=1)

        self.price = price
        self.y_true = y

        self.pool = np.random.normal(loc=loc, scale=scale,
                                     size=(self.population_size, self.m))

    def run(self, maxiter=10, subsample=None, bootstrap=False, reset_prob=1):
        """Train the algorithm.

        Parameters
        ----------
        maxiter : int
            Maximum number of training iterations.
        subsample : float
            Percentage of training data to be used for creating a subsample.
        bootstrap : float
            If true, draw with replacement.
        reset_prob : float
            Chance of redrawing the sample.

        Returns
        -------
        optimal_candidate : numpy.array
        """
        self.maxiter = maxiter
        self.reset_prob = reset_prob

        if subsample is None:
            self.sample_size = self.n
            self.bootstrap = False
            self.oob_size = 1
        else:
            assert 0 < subsample < 1
            self.sample_size = int(subsample * self.n)
            self.bootstrap = bootstrap
            self.oob_size = self.n - self.sample_size

        for j in trange(maxiter, desc='Generation', leave=True, position=0):
            # Draw random subsample to prevent overfitting
            if j == 0:
                samples = self.__get_sample()
                sample, sample_y, sample_p, oob, oob_y, oob_p = samples
            if np.random.random() < reset_prob:
                samples = self.__get_sample()
                sample, sample_y, sample_p, oob, oob_y, oob_p = samples

            # Determine fitness in a parallelized pool of workers
            with Parallel(n_jobs=self.n_jobs) as parallel:
                results = parallel(delayed(self.__get_fitness)(
                    sample, sample_y, sample_p, beta)
                    for beta in tqdm(self.pool))

            results = np.array(results)
            self.fitness = results[:, 0]
            self.cutoffs = results[:, 1]

            # Save best candidate
            fitness_idx = np.argsort(-self.fitness)
            parents_idx = fitness_idx[:self.num_parents]
            self.__update_history(fitness_idx, oob, oob_y, oob_p)

            # Create new pool
            self.pool = self.pool[parents_idx, :]

            # Crossover
            while self.pool.shape[0] < self.population_size:
                parent_a_idx = np.random.choice(range(self.num_parents))
                parent_b_idx = np.random.choice(range(self.num_parents))

                candidate = self.crossover(parent_a_idx, parent_b_idx)
                if np.random.random() < self.prob_mutation:
                    candidate = self.__mutate(candidate)

                self.pool = np.append(self.pool, [candidate], axis=0)

        # Determine Solution with best OOB-fitness
        opt_idx = np.argmax(self.history["oob_fitness"])
        self.optimal_candidate = self.history["best_candidate"][opt_idx]
        self.optimal_cutoff = self.history["best_cutoff"][opt_idx]
        return self.optimal_candidate

    def __get_fitness(self, sample, sample_y, sample_p, beta):
        y_pred = self.predict_proba(sample, beta)
        fit, cut = self.get_fitness(y_pred, sample_y, sample_p)
        return fit, cut

    def plot(self, savepath=None, title=None, **kwargs):
        """Create diagnostic plots

        Parameters
        ----------

        savepath : str
            If set, save plot to `savepath`.
        title : str
            Title of figure. If `None` (default) omit title.
        kwargs : dict
            Additional arguments to `pyplot.subplots`.
        """
        fig, ax = plt.subplots(**kwargs)
        ax.plot(self.history["best_fitness"], label="Training Fitness")
        ax.plot(self.history["oob_fitness"], label="Test Fitness")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_xticks(range(0, self.maxiter, 5))
        ax.set_title(title)

        plt.legend()
        if savepath is not None:
            fig.savefig(savepath)

        return fig, ax

    def __get_sample(self):
        sample_idx = np.random.choice(range(self.n),
                                      size=self.sample_size,
                                      replace=self.bootstrap)
        oob_idx = [i for i in range(self.n) if i not in sample_idx]

        sample = self.X_[sample_idx, :]
        sample_y = self.y_true[sample_idx]
        sample_price = self.price[sample_idx]

        oob_sample = self.X_[oob_idx]
        oob_y = self.y_true[oob_idx]
        oob_price = self.price[oob_idx]

        return sample, sample_y, sample_price, oob_sample, oob_y, oob_price

    def predict_proba(self, X, b=None):
        """Calculate predicted probabilities of positive class.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        b : numpy.array
            Parameter vector. If `None` (default) use optimal parameter
            of the last iteration.

        Returns
        -------
        out : numpy.array
              Probabilities of the positive class computed by a logit-model.
        """
        if b is None:
            b = self.optimal_candidate

        index = X.dot(b)
        out = np.array([sigmoid(i) for i in index])
        return out

    def predict(self, X, cut=None):
        """Calculate predicted class labels.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        cut : float
            Threshold for prediction. If probablity is larger than thres-
            hold predict positive class. If `None` (default) use the opt-
            imal parameter of the last iteration.

        Returns
        -------
        out : numpy.array
            Predicted class labels.
        """
        if cut is None:
            cut = self.optimal_cutoff
        X_ = (X - self.means) / self.stds
        if self.fit_intercept:
            X_ = np.append(np.ones((len(X_), 1)), X_, axis=1)
        proba = self.predict_proba(X_, self.optimal_candidate)
        out = proba > cut
        return out

    def predict_cost(self, X=None, y_true=None, price=None, b=None, cutoff=None):
        """Return predicted cost.

        Predict cost for a given feature matrix, labels, price and cutoff
        or calculate costs on the training set based on the optimal solution.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix. If `None` (default) use the whole training data.
        y_true : array-like
                 Vector of true labels. If `None` (default) use the whole training data.
        price : array-like
                Vector of prices. If `None` (default) use the whole training data.
        b : array-like
            Vector of coefficients. If `None` use the optimal solution of the last iteration.
        cutoff : float
                 Threshold for prediction. If `None` (default) use the optimal threshold of
                 the last iteration.

        Returns
        -------
        out : float
              Predicted cost.
        """
        if X is None:
            X_ = self.X_
        else:
            X_ = (X - self.means) / self.stds
            if self.fit_intercept:
                X_ = np.append(np.ones((len(X_), 1)), X_, axis=1)
        if y_true is None:
            y_true = self.y_true
        if price is None:
            price = self.price
        if b is None:
            b = self.optimal_candidate
        if cutoff is None:
            cutoff = self.optimal_cutoff

        y_prob = self.predict_proba(X_, b=b)
        out = self.get_utility(y_prob=y_prob,
                               y_true=y_true,
                               price=price,
                               cutoff=cutoff)
        return out

    def get_fitness(self, y_prob, y_true, price):
        """Calculate optimal fitness and threshold.

        Evaluate cost/utility function on a grid of thresholds and
        return the optimal values.

        Parameters
        ----------
        y_prob : array-like
                 Vector of predicted probablities of positive class.
        y_true : array-like
                 Vector of true labels.
        price : array-like
                Vector of item prices.

        Returns
        -------
        best_util : float
                    Best achievable utility.
        best_cut : float
                   Threshold that leads to optimal utility.
        """
        cut = np.linspace(0, 1, num=50)
        utility = [self.get_utility(y_prob, y_true, price, c) for c in cut]
        best = np.argmax(utility)
        best_util, best_cut = utility[best], cut[best]
        return best_util, best_cut

    def get_utility(self, y_prob=None, y_true=None, price=None, cutoff=None):
        """Calculate cost/utility.

        Evaluate based on predicted probabilities and a threshold.

        Parameters
        ----------
        y_prob : array-like
                 Vector of predicted probabilities of positive class.
        y_true : array-like
                 Vector of true labels.
        price : array-like
                Vector of item prices.
        cutoff : float
                 Threshold for creating predictions.

        Returns
        -------
        utility : float
                  Cost/utility.
        """
        if all((y_prob is None, y_true is None, price is None, cutoff is None)):
            y_prob = self.predict_proba(self.X_)
            y_true = self.y_true
            price = self.price
            cutoff = self.optimal_cutoff

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

    def __update_history(self, fitness_idx, oob, y, p):
        self.optimal_candidate = self.pool[fitness_idx[0], :]
        prob = self.predict_proba(oob, self.optimal_candidate)
        self.oob_fitness, self.oob_cutoff = self.get_fitness(prob, y, p)
        # Rescale fitness to compare with OOB-fitness
        self.history["best_fitness"].append(
            self.fitness[fitness_idx[0]] / self.sample_size)
        self.history["best_candidate"].append(self.pool[fitness_idx[0], :])
        self.history["mean_pop_fitness"].append(
            self.fitness.mean() / self.sample_size)
        self.history["oob_fitness"].append(self.oob_fitness / self.oob_size)
        self.history["best_cutoff"].append(self.oob_cutoff)

    def __arithmetic_crossover(self, idx_a, idx_b):
        lam = np.random.random()
        out = lam * self.pool[idx_a, :] + (1-lam) * self.pool[idx_b, :]
        return out

    def __exchange_point_crossover(self, idx_a, idx_b):
        keep_idx_a = np.random.randint(low=0, high=self.m // 2)
        out = copy.copy(self.pool[idx_b, :])
        out[keep_idx_a] = self.pool[idx_a, keep_idx_a]
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
    X_RAW = np.random.uniform(size=(1000, 4))
    X_SCALED = (X_RAW - X_RAW.mean(axis=0)) / X_RAW.std(axis=0)

    BETA = np.array([1, -2, 3, -4])
    Y_TRUE = X_SCALED.dot(BETA) > 0.5
    PRICE = np.random.uniform(low=10, high=100, size=1000)

    LOGIT_PRED = 1 / (1 + np.exp(-X_SCALED.dot(BETA)))

    GA = GeneticAlgorithm(elitism=0.2, prob_mutation=0.3,
                          crossover_strategy="point")
    GA.fit(X_RAW, Y_TRUE, PRICE)

    OPT, _ = GA.get_fitness(LOGIT_PRED,
                            Y_TRUE,
                            PRICE)
    print("Optimum based on true beta: ", OPT)
    print("Best solution:")
    RES = GA.run(maxiter=30, subsample=0.5, bootstrap=False, reset_prob=0.33)
    print(RES)

    print("\n| Best Fitness | Mean Fitness | OOB Fitness")
    print("|" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14)
    for opt, avg, score in zip(GA.history["best_fitness"],
                               GA.history["mean_pop_fitness"],
                               GA.history["oob_fitness"]):
        print("|{0:10.3f}{3:<4}|{1:10.5f}{3:<4}|{2:10.5f}".format(opt,
                                                                  avg,
                                                                  score, " "))
