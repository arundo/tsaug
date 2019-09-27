"""
Cross sum
"""

import numpy as np
from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def cross_sum(X, Y=None, inds=None):
    """Sum cross given time series.

    Time series will be summed with others based on the given indices. Time
    points at which at least one time series of summation is anomalous will be
    marked as anomalous.

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

    inds : numpy.array, optional
        Indices of time series to sum with. Matrix with shape (N, m), where N
        is the number of series, and m is the maximal number of series to sum
        with each series. The i-th output series is the sum of the i-th input
        series and the ind[i][j]-th time series for all j. Values of ind[i][j]
        can be NaN for series to be summed with less than m series.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    N, n, c = X.shape

    if inds is None:
        inds = np.zeros((N, 0))

    if inds.ndim == 1:
        inds = inds.reshape((N, 1))

    if inds.shape[0] != N:
        raise ValueError("Wrong shape of inds")

    m = inds.shape[1]

    X_aug = X.copy()
    if Y is None:
        Y_aug = None
    else:
        Y_aug = Y.copy()

    for k in range(m):
        X_aug[np.isnan(inds[:, k]) >= 0] = (
            X_aug[np.isnan(inds[:, k]) >= 0]
            + X[inds[np.isnan(inds[:, k]) >= 0, k]]
        )
        if Y is not None:
            Y_aug[np.isnan(inds[:, k]) >= 0] = (
                Y_aug[np.isnan(inds[:, k]) >= 0]
                + Y[inds[np.isnan(inds[:, k]) >= 0, k]]
            )

    if Y is not None:
        Y_aug = (Y_aug >= 1).astype(int)

    return X_aug, Y_aug


@dimensionalize
def random_cross_sum(X, Y=None, max_sum_series=5, random_seed=None):
    """Sum cross given time series randomly.

    Time series will be summed with others randomly. Time points at which at
    least one time series of summation is anomalous will be marked as
    anomalous.

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

    max_sum_series : int, optinonal
        Maximal number of time series to cross sum. Default: 5.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    N, n, c = X.shape
    rand = np.random.RandomState(random_seed)
    inds = rand.choice(range(-1, N), size=(N, max_sum_series))

    return cross_sum(X, Y, inds)


class CrossSum(_Augmentor):
    """Augmentor that sum cross given time series.

    Time series will be summed with others based on the given indices. Time
    points at which at least one time series of summation is anomalous will be
    marked as anomalous.

    Parameters
    ----------
    inds : numpy.array, optional
        Indices of time series to sum with. Matrix with shape (N, m), where N
        is the number of series, and m is the maximal number of series to sum
        with each series. The i-th output series is the sum of the i-th input
        series and the ind[i][j]-th time series for all j. Values of ind[i][j]
        can be NaN for series to be summed with less than m series.

    """

    def __init__(self, inds=None):
        super().__init__(augmentor_func=cross_sum, is_random=False, inds=inds)

    @property
    def inds(self):
        return self._params["inds"]

    @inds.setter
    def inds(self, inds):
        self._params["inds"] = inds


class RandomCrossSum(_Augmentor):
    """Augmentor that sums cross given time series randomly.

    Time series will be summed with others randomly. Time points at which at
    least one time series of summation is anomalous will be marked as
    anomalous.

    Parameters
    ----------
    max_sum_series : int, optinonal
        Maximal number of time series to cross sum. Default: 5.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    """

    def __init__(self, max_sum_series=5, random_seed=None):
        super().__init__(
            augmentor_func=random_cross_sum,
            is_random=True,
            max_sum_series=max_sum_series,
            random_seed=random_seed,
        )

    @property
    def max_sum_series(self):
        return self._params["max_sum_series"]

    @max_sum_series.setter
    def max_sum_series(self, max_sum_series):
        self._params["max_sum_series"] = max_sum_series

    @property
    def random_seed(self):
        return self._params["random_seed"]

    @random_seed.setter
    def random_seed(self, random_seed):
        self._params["random_seed"] = random_seed
