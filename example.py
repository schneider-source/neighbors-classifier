"""
Created on Sat Jun 8 11:12:56 2019
@author: Daniel Schneider

Simple example presenting the performance of my neighbors classifier in comparison to
sklearn.neighbors.RadiusNeighborsClassifier using the sklearn iris dataset
(sklearn.datasets.load_iris).
"""

import os
import time
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score

from plot_confusion_matrix import plot_confusion_matrix

from scoring import Scores, ConfusionMatrix
from util import aprint, mkdir
from neighbors_classifier import NeighborsClassifier


class Params:
    """
    Parameters
    """
    def __init__(self):
        aprint('Neighbors classifier example and comparison with '
               'sklearn.neighbors.RadiusNeighborsClassifier', fmt='bh')

        # create directory to dump figures
        self.odir = mkdir('out_neighbors_classifier', replace=True)

        # load iris data, further info at
        # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
        self.dataset = datasets.load_iris()

        # target labels
        self.yy = self.dataset.target
        self.labels = np.unique(self.yy)
        self.label_names = self.dataset.target_names
        self.label_colors = ['C0', 'C1', 'C2']

        # obtain features for classification
        # first rescale to range [0, 1], then get first 2 principle components
        xx = MinMaxScaler().fit_transform(self.dataset.data)
        pca = PCA(n_components=2)
        self.xx = pca.fit_transform(xx, self.yy)
        aprint('\nPCA: EVR = {}, SUM = {}'.format(pca.explained_variance_ratio_,
                                                  pca.explained_variance_ratio_.sum()), fmt='i')

        # create mesh digitizing the feature space
        self.num = 401
        self.zz = np.linspace(-1, 1, self.num)
        self.mesh = np.dstack(np.meshgrid(self.zz, self.zz)).reshape(-1, 2)

        # init classifiers
        # Note that NeighborsClassifier(dist_weights_func=lambda x: np.where(x < 0.5, 1, 0))
        # will yield the same results as RadiusNeighborsClassifier(radius=0.5)
        self.onc = NeighborsClassifier(dist_weights_func=lambda x: np.exp(-5 * x),
                                          weight_label_counts=True)
        self.rnc = RadiusNeighborsClassifier(radius=0.5, outlier_label=-1)


_par_ = Params()


def test_performance(xx=_par_.xx, yy=_par_.yy, odir=_par_.odir):
    """
    Test the performance of both classifiers
    :param xx: numpy.ndarray of shape (n_samples, n_features), optional, "training data"
    :param yy: numpy.ndarray of shape (n_samples), optional, "training labels"
    :param odir: str, optional, "directory to store performance measures"
    """

    # init Scores classes to store some performance scores
    acc_onc, acc_rnc = Scores([accuracy_score]), Scores([accuracy_score])
    cm_onc, cm_rnc = ConfusionMatrix(_par_.labels), ConfusionMatrix(_par_.labels)

    # cross-validate with stratified randomized folds
    splits = ((xx[i], xx[j], yy[i], yy[j]) for i, j in
              StratifiedShuffleSplit(n_splits=100, test_size=0.33).split(xx, yy))

    for xx_train, xx_test, yy_train, yy_test in splits:

        # test own neighbor classifier
        _par_.onc.fit(xx_train, yy_train)
        yy_pred = _par_.onc.predict(xx_test)
        acc_onc.add_targets(yy_test, yy_pred), cm_onc.add_targets(yy_test, yy_pred)

        # test RadiusNeighborsClassifier
        _par_.rnc.fit(xx_train, yy_train)
        yy_pred = _par_.rnc.predict(xx_test)
        acc_rnc.add_targets(yy_test, yy_pred), cm_rnc.add_targets(yy_test, yy_pred)

    # print performance measures for both classifiers
    aprint('\nNeighborsClassifier:', fmt='i')
    aprint(acc_onc.get_mean(), fmt='i')
    aprint('confusion matrix', fmt='i')
    aprint(cm_onc.get_normed_cm(), fmt='i')

    aprint('\nsklearn.neighbors.RadiusNeighborsClassifier:', fmt='i')
    aprint(acc_rnc.get_mean(), fmt='i')
    aprint('confusion matrix', fmt='i')
    aprint(cm_rnc.get_normed_cm(), fmt='i')

    # plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle('Confusion Matrix', fontsize=18)
    plot_confusion_matrix(ax1, cm_onc.get_normed_cm(), _par_.label_names,
                          title='NeighborsClassifier')
    plot_confusion_matrix(ax2, cm_rnc.get_normed_cm(), _par_.label_names,
                          title='sklearn.neighbors.RadiusNeighborsClassifier')
    fig.savefig(os.path.join(odir, 'cm.png'), dpi=200)


def classify_feature_space(xx=_par_.xx, yy=_par_.yy, odir=_par_.odir):
    """
    Classify the feature space and make heatmap figures.
    :param xx: numpy.ndarray of shape (n_samples, n_features), optional, "training data"
    :param yy: numpy.ndarray of shape (n_samples), optional, "training labels"
    :param odir: str, optional, "directory to store heatmap figures of the feature space
        classification"
    """

    _par_.onc.fit(xx, yy)
    onc_pred, onc_conf = _par_.onc.predict(_par_.mesh, return_confidence=True)

    _par_.rnc.fit(xx, yy)
    rnc_pred = _par_.rnc.predict(_par_.mesh)

    # plot figure showing classification of the feature space for classifiers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle('Classification', fontsize=18)

    cmap = colors.LinearSegmentedColormap.from_list('cmap', _par_.label_colors)
    ax1.pcolormesh(_par_.zz, _par_.zz, onc_pred.reshape(_par_.num, -1), cmap=cmap, vmin=0, vmax=2)
    for l, ln, lc in zip(_par_.labels, _par_.label_names, _par_.label_colors):
        ax1.scatter(*xx[yy == l].T, color=lc, edgecolors='k', label=ln)
    ax1.set(title='NeighborsClassifier', aspect='equal',
           xlabel='Principal Axis 1', ylabel='Principal Axis 2')
    ax1.legend()

    cmap = colors.LinearSegmentedColormap.from_list('cmap', ['w', *_par_.label_colors])
    ax2.pcolormesh(_par_.zz, _par_.zz, rnc_pred.reshape(_par_.num, -1), cmap=cmap, vmin=-1, vmax=2)
    for l, ln, lc in zip(_par_.labels, _par_.label_names, _par_.label_colors):
        ax2.scatter(*xx[yy == l].T, color=lc, edgecolors='k', label=ln)
    ax2.set(title='sklearn.neighbors.RadiusNeighborsClassifier', aspect='equal',
           xlabel='Principal Axis 1', ylabel='Principal Axis 2')
    ax2.legend()

    fig.savefig(os.path.join(odir, 'classification.png'), dpi=200)

    # plot figure showing confidence measure of OwnNeighborsClassifier
    fig, ax = plt.subplots()
    fig.colorbar(ax.pcolormesh(_par_.zz, _par_.zz, onc_conf.reshape(_par_.num, -1)))
    for l, ln, lc in zip(_par_.labels, _par_.label_names, _par_.label_colors):
        ax.scatter(*xx[yy == l].T, color=lc, edgecolors='k', label=ln)
    ax.set(title='NeighborsClassifier: Confidence measure',
           xlabel='Principal Axis 1', ylabel='Principal Axis 2')
    ax.legend()
    fig.savefig(os.path.join(odir, 'onc_confidence.png'), dpi=200)


def compare_runtime():
    """
    Compares runtime of classifiers
    """

    t0 = time.time()
    _par_.onc.fit(_par_.xx, _par_.yy)
    _par_.onc.predict(_par_.mesh)
    t_onc = time.time() - t0
    aprint('\nRuntime NeighborsClassifier = {0:4.3f} sec'.format(t_onc), fmt='i')

    t0 = time.time()
    _par_.rnc.fit(_par_.xx, _par_.yy)
    _par_.rnc.predict(_par_.mesh)
    t_rnc = time.time() - t0
    aprint('\nRuntime sklearn.neighbors.RadiusNeighborsClassifier = {0:4.3f} sec'.format(t_rnc),
           fmt='i')

    aprint('\nMy classifier is {0:4.2f} x faster than sklearn classifier.'
           .format(t_rnc / t_onc), fmt='i')


def example():
    """
    Comparison of my own neighbor classifier with sklearn.neighbors.RadiusNeighborsClassifier
    """

    # measure classification performance of both classifiers for the full dataset
    aprint('\nTest performance of classifiers for full dataset', fmt='bi')
    odir = mkdir(os.path.join(_par_.odir, 'full_dataset'))
    test_performance(odir=odir)
    aprint('\nMy NeighborsClassifier should have slightly better performance'
           '\nthan sklearn.neighbors.RadiusNeighborsClassifier.', fmt='i')

    # classify the feature space using the full dataset and create figures
    classify_feature_space(odir=odir)

    # measure classification performance of selected data containing
    # 50 samples of versicolor and 20 of virginica (asymmetric label frequency)
    aprint('\nTest performance of classifiers for asymmetric data', fmt='bi')
    odir = mkdir(os.path.join(_par_.odir, 'asymmetric'))
    test_performance(_par_.xx[50:120], _par_.yy[50:120], odir=odir)
    aprint('\nMy NeighborsClassifier should have significantly better performance'
           '\nthan sklearn.neighbors.RadiusNeighborsClassifier.', fmt='i')

    # classify the feature space using asymmetric data and create figures
    classify_feature_space(_par_.xx[50:120], _par_.yy[50:120], odir=odir)

    # compare runtime of classifiers
    aprint('\nCompare execution time of classifiers', fmt='bi')
    compare_runtime()


if __name__ == '__main__':
    example()
