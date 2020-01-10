"""
Jitter module
"""
from typing import Tuple, Optional

import numpy as np
from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def random_jitter(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    dist: Optional[str] = "normal",
    strength: Optional[float] = 0.05,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Add random noise to time series.

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

    dist : str, optional
        Distribution the random noise follows. Either 'normal' or 'uniform'.
        Default: 'normal'.

    strength : float, optional
        Strength of noise. Default: 0.05.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    N: int
    n: int
    c: int
    N, n, c = X.shape
    rand = np.random.RandomState(random_seed)  # type: ignore # Not sure what type we need here
    scale: np.ndarray = strength * np.stack(
        [np.percentile(X, q=95, axis=1) - np.percentile(X, q=5, axis=1)] * n,
        axis=1,
    )

    if dist == "normal":
        R: np.ndarray = rand.normal(size=X.shape, scale=1 / 3)
    elif dist == "uniform":
        R = rand.uniform(size=X.shape, low=-1, high=1)
    else:
        raise ValueError("`dist` must be either 'normal' or 'uniform")

    X_aug: np.ndarray = X + scale * R

    if Y is None:
        Y_aug: Optional[np.ndarray] = None
    else:
        Y_aug = Y.copy()

    return X_aug, Y_aug


class RandomJitter(_Augmentor):
    """Augmentor that adds random noise to time series.

    Parameters
    ----------
    dist : str, optional
        Distribution the random noise follows. Either 'normal' or 'uniform'.
        Default: 'normal'.

    strength : float, optional
        Strength of noise. Default: 0.05.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    """

    def __init__(
        self,
        dist: Optional[str] = "normal",
        strength: Optional[float] = 0.05,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            augmentor_func=random_jitter,
            is_random=True,
            dist=dist,
            strength=strength,
            random_seed=random_seed,
        )

    @property
    def dist(self) -> Optional[str]:
        return self._params["dist"]

    @dist.setter
    def dist(self, dist: Optional[str]) -> None:
        self._params["dist"] = dist

    @property
    def strength(self) -> Optional[float]:
        return self._params["strength"]

    @strength.setter
    def strength(self, strength: Optional[float]) -> None:
        self._params["strength"] = strength

    @property
    def random_seed(self) -> Optional[int]:
        return self._params["random_seed"]

    @random_seed.setter
    def random_seed(self, random_seed: Optional[int]) -> None:
        self._params["random_seed"] = random_seed
