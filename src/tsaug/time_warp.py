"""
Time warping module
"""

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def random_time_warp(X, Y=None, n_speed_change=3, random_seed=None):
    """Warp time line of time series randomly.

    The speed at which the time line advances is a random smooth curve with a
    few speed changes.

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

    n_speed_change : int, optional
        Number of speed changes. Default: 3.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """
    N, n, c = X.shape
    if Y is not None:
        cl = Y.shape[2]

    rand = np.random.RandomState(random_seed)

    anchors = np.arange(0.5 / n_speed_change, 1, 1 / n_speed_change)
    R = (
        rand.uniform(
            low=-0.5 / n_speed_change,
            high=0.5 / n_speed_change,
            size=(N, n_speed_change),
        )
        * 0.5
    )
    anchor_values = R + np.arange(0.5 / n_speed_change, 1, 1 / n_speed_change)

    anchors = np.append([0], anchors)
    anchors = np.append(anchors, [1])
    anchors = anchors * (n - 1)

    anchor_values = np.concatenate([np.zeros((N, 1)), anchor_values], axis=1)
    anchor_values = np.concatenate([anchor_values, np.ones((N, 1))], axis=1)
    anchor_values = anchor_values * (n - 1)

    warp = PchipInterpolator(x=anchors, y=anchor_values, axis=1)(np.arange(n))

    X_aug = np.vstack(
        [
            interp1d(
                np.arange(n),
                Xi,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(warpi).reshape(1, n, c)
            for Xi, warpi in zip(X, warp)
        ]
    )

    if Y is None:
        Y_aug = None
    else:
        Y_aug = np.vstack(
            [
                interp1d(
                    np.arange(n),
                    Yi,
                    axis=0,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )(warpi).reshape(1, n, cl)
                for Yi, warpi in zip(Y, warp)
            ]
        )
        Y_aug = Y_aug.round().astype(int)

    return X_aug, Y_aug


class RandomTimeWarp(_Augmentor):
    """Augmentor that warps time line of time series randomly.

    The speed at which the time line advances is a random smooth curve with a
    few speed changes.

    Parameters
    ----------
    n_speed_change : int, optional
        Number of speed changes. Default: 3.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    """

    def __init__(self, n_speed_change=3, random_seed=None):
        super().__init__(
            augmentor_func=random_time_warp,
            is_random=True,
            n_speed_change=n_speed_change,
            random_seed=random_seed,
        )

    @property
    def n_speed_change(self):
        return self._params["n_speed_change"]

    @n_speed_change.setter
    def n_speed_change(self, n_speed_change):
        self._params["n_speed_change"] = n_speed_change

    @property
    def random_seed(self):
        return self._params["random_seed"]

    @random_seed.setter
    def random_seed(self, random_seed):
        self._params["random_seed"] = random_seed
