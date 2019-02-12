"""
Algorithms for parameter tuning.
"""

from hyperopt import fmin, tpe
from tqdm import tqdm


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
