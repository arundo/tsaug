"""
Reverse module
"""

from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def reverse(X, Y=None):
    """Reverse time series.

    Parameters
    ----------
    X : numpy.ndarray
        Time series to be augmented. Matrix with shape (n,), (N, n) or (N, n,
        c), where n is the length of each series, N is the number of series,
        and c is the number of channels.

    Y : numpy.ndarray, optional
        Binary labels of time series, where 0 represents a normal point and 1
        represents an anomalous points. Matrix with shape (n,), (N, n) or (N,
        n, cl), where n is the length of each series, N is the number of
        series, and cl is the number of classes (i.e. types of anomaly).
        Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    X_aug = X[:, ::-1, :].copy()

    if Y is None:
        Y_aug = None
    else:
        Y_aug = Y[:, ::-1, :].copy()

    return X_aug, Y_aug


class Reverse(_Augmentor):
    """Augmentor that reverses time series."""

    def __init__(self):
        super().__init__(augmentor_func=reverse, is_random=False)
