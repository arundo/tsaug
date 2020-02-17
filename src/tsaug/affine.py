"""
Affine transformation
"""

from typing import Tuple, Union, Optional

import numpy as np
from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def affine(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    a: Union[float, np.ndarray] = 1.0,
    b: Union[float, np.ndarray] = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Perform affine transformation to time series.

    A series x will be transformed to a*x+b, while binary label y will not be
    changed.

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

    a : float, or numpy.ndarray, optional
        Slope coefficients of the affine transformation. An array with
        shape (N,) or (N, c), where N is the number of series and c is the
        number of channels, or a scalar. Default: 1.0.

    b : float, or numpy.ndarray, optional
        Intercept coefficients of the affine transformation. An array with
        shape (N,) or (N, c), where N is the number of series and c is the
        number of channels, or a scalar. Default: 0.0.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """
    N = (
        0
    )  # type: int  # NOTE: this is a horrible hack to type hint for Python 3.5
    n = 0  # type: int
    c = 0  # type: int
    N, n, c = X.shape

    if isinstance(a, float) | isinstance(a, int):
        a = a * np.ones((N, c))
    if isinstance(b, float) | isinstance(b, int):
        b = b * np.ones((N, c))

    a = np.array(a)
    b = np.array(b)

    if a.ndim == 1:
        a = np.reshape(-1, 1)
    if b.ndim == 1:
        b = np.reshape(-1, 1)

    if a.shape != (N, c):
        raise ValueError("Wrong shape of a")
    if b.shape != (N, c):
        raise ValueError("Wrong shape of b")

    X_aug = X * a.reshape((N, 1, c)) + b.reshape((N, 1, c))  # type: np.ndarray

    if Y is None:
        Y_aug = None  # type: Optional[np.ndarray]
    else:
        Y_aug = Y.copy()

    return X_aug, Y_aug


@dimensionalize
def random_affine(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    max_a: float = 10.0,
    min_a: float = -10.0,
    max_b: float = 100.0,
    min_b: float = -100.0,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Perform affine transformation to time series with random coefficients.

    A series x will be transformed to a*x+b, while binary label y will not be
    changed. Coefficients `a` and `b` are randomly generated.

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

    max_a : float, optional
        Maximal value of slope cofficients of the affine transformation.
        Default: 10.0.

    min_a : float, optional
        Minimal value of slope cofficients of the affine transformation.
        Default: -10.0.

    max_b : float, optional
        Maximal value of intercept cofficients of the affine transformation.
        Default: 100.0.

    min_b : float, optional
        Minimal value of intercept cofficients of the affine transformation.
        Default: -100.0.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    N = (
        0
    )  # type: int  # NOTE: this is a horrible hack to type hint for Python 3.5
    n = 0  # type: int
    c = 0  # type: int
    N, n, c = X.shape
    rand = np.random.RandomState(random_seed)
    a = rand.uniform(low=min_a, high=max_a, size=(N, c))  # type: np.ndarray
    b = rand.uniform(low=min_b, high=max_b, size=(N, c))  # type: np.ndarray

    return affine(X, Y, a=a, b=b)


class Affine(_Augmentor):
    """Augmentor that performs affine transformation to time series.

    A series x will be transformed to a*x+b, while binary label y will not be
    changed.

    Parameters
    ----------
    a : float, or numpy.ndarray, optional
        Slope coefficients of the affine transformation. An array with
        shape (N,) or (N, c), where N is the number of series and c is the
        number of channels, or a scalar. Default: 1.0.

    b : float, or numpy.ndarray, optional
        Intercept coefficients of the affine transformation. An array with
        shape (N,) or (N, c), where N is the number of series and c is the
        number of channels, or a scalar. Default: 0.0.

    """

    def __init__(
        self,
        a: Union[float, np.ndarray] = 1.0,
        b: Union[float, np.ndarray] = 0.0,
    ) -> None:
        super().__init__(augmentor_func=affine, is_random=False, a=a, b=b)

    @property
    def a(self) -> Union[float, np.ndarray]:
        return self._params["a"]

    @a.setter
    def a(self, a: Union[float, np.ndarray]) -> None:
        self._params["a"] = a

    @property
    def b(self) -> Union[float, np.ndarray]:
        return self._params["b"]

    @b.setter
    def b(self, b: Union[float, np.ndarray]) -> None:
        self._params["b"] = b


class RandomAffine(_Augmentor):
    """
    Augmentor that performs affine transformation to time series with random
    coefficients.

    A series x will be transformed to a*x+b, while binary label y will not be
    changed. Coefficients `a` and `b` are randomly generated.

    Parameters
    ----------
    max_a : float, optional
        Maximal value of slope cofficients of the affine transformation.
        Default: 10.0.

    min_a : float, optional
        Minimal value of slope cofficients of the affine transformation.
        Default: -10.0.

    max_b : float, optional
        Maximal value of intercept cofficients of the affine transformation.
        Default: 100.0.

    min_b : float, optional
        Minimal value of intercept cofficients of the affine transformation.
        Default: -100.0.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    """

    def __init__(
        self,
        max_a: float = 10.0,
        min_a: float = -10.0,
        max_b: float = 100.0,
        min_b: float = -100.0,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            augmentor_func=random_affine,
            is_random=True,
            max_a=max_a,
            min_a=min_a,
            max_b=max_b,
            min_b=min_b,
            random_seed=random_seed,
        )

    @property
    def max_a(self) -> float:
        return self._params["max_a"]

    @max_a.setter
    def max_a(self, max_a: float) -> None:
        self._params["max_a"] = max_a

    @property
    def min_a(self) -> float:
        return self._params["min_a"]

    @min_a.setter
    def min_a(self, min_a: float) -> None:
        self._params["min_a"] = min_a

    @property
    def max_b(self) -> float:
        return self._params["max_b"]

    @max_b.setter
    def max_b(self, max_b: float) -> None:
        self._params["max_b"] = max_b

    @property
    def min_b(self) -> float:
        return self._params["min_b"]

    @min_b.setter
    def min_b(self, min_b: float) -> None:
        self._params["min_b"] = min_b

    @property
    def random_seed(self) -> int:
        return self._params["random_seed"]

    @random_seed.setter
    def random_seed(self, random_seed: int) -> None:
        self._params["random_seed"] = random_seed
