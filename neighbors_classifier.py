"""
Created on Fri Jun 7 14:45:28 2019
@author: Daniel Schneider
"""

import numpy as np


class NeighborsClassifier:
    def __init__(self, dist_weights_func, weight_label_counts=False):
        """
        Fast (NumPy-based) and simple neighbor classifier with customizable weighting
        function of distance between data points and optional label weighting according
        to the frequency of their occurrence.
        :param dist_weights_func: functional, "user-defined function which accepts
            a numpy.ndarray of distances, and returns an array of the same shape
            containing the weights"
        :param weight_label_counts: bool, optional, "weight neighbors according to
        the frequency of their labels"
        """

        assert(callable(dist_weights_func))
        assert(isinstance(weight_label_counts, bool))

        self.dist_weights_func = dist_weights_func
        self.weight_label_counts = weight_label_counts

        self.xx0, self.yy0 = None, None
        self.labels, self.weights = None, None

    def fit(self, xx, yy):
        """
        Fit model using xx as training data and yy as training labels.
        :param xx: numpy.ndarray of shape (n_samples, n_features), "training data"
        :param yy: numpy.ndarray of shape (n_samples), "training labels"
        """

        assert(xx.ndim == 2 if isinstance(xx, np.ndarray) else False)
        assert((yy.ndim == 1) & (yy.size == xx.shape[0]) if isinstance(yy, np.ndarray)
               else False)

        self.xx0, self.yy0 = xx, yy

        self.labels, counts = np.unique(yy, return_counts=True)
        if self.weight_label_counts:
            weights = counts[0] / counts
            weights /= weights.sum()
        else:
            weights = np.ones_like(self.labels) / self.labels.size
        self.weights = (self.yy0[np.newaxis, :, np.newaxis] == self.labels) * weights

    def predict(self, xx, return_confidence=False):
        """
        Predict the class labels for the provided data.
        :param xx: numpy.ndarray of shape (n_samples, n_features), "data to classify"
        :param return_confidence: bool, optional, "return measure of confidence with
            predicted labels"
        :return: numpy.ndarray of shape (n_samples), "predicted class labels",
            (numpy.ndarray of shape (n_samples), "confidence of predicted labels"
             if return_confidence)
        """

        assert((xx.ndim == 2) & (xx.shape[1] == self.xx0.shape[1])
               if isinstance(xx, np.ndarray) else False)
        assert(isinstance(return_confidence, bool))
        assert(all([x is not None for x
                    in [self.xx0, self.yy0, self.labels, self.weights]]))

        dist = (((self.xx0 - xx[:, np.newaxis]) ** 2).sum(axis=2) ** 0.5)[:, :, np.newaxis]

        prob = (self.dist_weights_func(dist) * self.weights).sum(axis=1)
        prob /= prob.sum(axis=1, keepdims=True)

        if return_confidence:
            return self.labels[np.argmax(prob, axis=1)], np.amax(prob, axis=1)
        else:
            return self.labels[np.argmax(prob, axis=1)]
