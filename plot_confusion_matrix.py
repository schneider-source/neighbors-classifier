"""
Created on Mon Jun 10 08:23:21 2019
@author: Daniel Schneider
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(ax, cm, label_names, title='', cmap=plt.cm.Blues):
    """
    Taken and modified from sklearn example:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots given confusion matrix.
    """

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=label_names, yticklabels=label_names,
           ylabel='True label', xlabel='Predicted label',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
