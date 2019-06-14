"""
Created on Sat Jun 8 11:35:46 2019
@author: Daniel Schneider
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class Scores:
    """
    Calculate and store specified scores for multiple train-test-splits
    """
    def __init__(self, score_funcs, func_kwargs=None):
        """
        :param score_funcs: sequence of functionals, which take an array of true labels
            as first argument, an array of predicted labels as second argument, and return
            a score float, e.g. many of the scoring functions provided in sklearn.metrics
        :param func_kwargs: sequence of kwargs dicts to pass to the given score_funcs,
            optional
        """

        assert(all([callable(f) for f in score_funcs])
               if hasattr(score_funcs, '__iter__') else False)
        assert(all([isinstance(d, dict) for d in func_kwargs]) & (len(score_funcs) == len(func_kwargs))
               if hasattr(func_kwargs, '__iter__') else True if func_kwargs is None else False)


        self.score_funcs = score_funcs
        if func_kwargs is None:
            self.func_kwargs = [{}] * len(score_funcs)
        else:
            self.func_kwargs = func_kwargs
        self.df = pd.DataFrame(columns=[f.__name__ for f in score_funcs])
        self.it = itertools.count()

    def add_targets(self, yy_true, yy_pred):
        """
        Calculate and store performance scores for given targets.
        :param yy_true: 1d numpy.ndarray of shape (n_samples), optional, "true labels"
        :param yy_pred: 1d numpy.ndarray of shape (n_samples), optional, "predicted labels"
        """

        assert(yy_true.ndim == 1 if isinstance(yy_true, np.ndarray) else False)
        assert(yy_pred.ndim == 1 if isinstance(yy_pred, np.ndarray) else False)
        assert(yy_true.size == yy_pred.size)

        self.df.loc[next(self.it)] = [f(yy_true, yy_pred, **kwargs) for f, kwargs
                                      in zip(self.score_funcs, self.func_kwargs)]

    def get_mean(self):
        """
         Calculate mean scores.
         :return: pandas.DataFrame, "mean scores"
         """

        assert (0 < len(self.df.index))

        return self.df.mean()


class ConfusionMatrix:
    """
    Confusion matrix added up for multiple train-test-splits.
    """
    def __init__(self, labels):
        """
        :param labels: sequence of labels
        """
        assert(hasattr(labels, '__iter__'))

        self.labels = labels
        self.cm = np.zeros((labels.size, labels.size), dtype=np.float)

    def add_targets(self, yy_true, yy_pred):
        """
        Add targets to confusion matrix.
        :param yy_true: 1d numpy.ndarray of shape (n_samples), optional, "true labels"
        :param yy_pred: 1d numpy.ndarray of shape (n_samples), optional, "predicted labels"
        """

        assert(yy_true.ndim == 1 if isinstance(yy_true, np.ndarray) else False)
        assert(yy_pred.ndim == 1 if isinstance(yy_pred, np.ndarray) else False)
        assert(yy_true.size == yy_pred.size)

        self.cm += confusion_matrix(yy_true, yy_pred, self.labels)

    def get_normed_cm(self):
        """
        Obtain normalized confusion matrix.
        :return: 2d np.ndarray, "normalized confusion matrix"
        """
        norm = self.cm.sum(axis=1)[:, np.newaxis]
        return self.cm / np.where(0 < norm, norm, np.inf)
