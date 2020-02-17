"""
Value warpping module
"""
from typing import Tuple, Optional

import numpy as np
from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def random_sidetrack(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    mode: Optional[str] = "multiplicative",
    initial_sidetrack: Optional[float] = None,
    max_sidetrack: Optional[float] = None,
    min_sidetrack: Optional[float] = None,
    step_mu: Optional[float] = 0.0,
    step_sigma: Optional[float] = 0.1,
    random_seed: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sidetrack time series from its original values.

    This augmentor has two modes: multiplicative mode sidetracks a time series
    by multiplying its values with random-walking multipliers, and additive
    mode sidetracks a time series by adding its values with random-walking
    additives.

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

    mode : str, optional
        Either "multiplicative" or "additive". Default: "multiplicative".

    initial_sidetrack : float, optional
        Sidetrack at the start of time series. If not given, 1.0 for
        multiplicative mode and 0.0 for additive mode. Default: None.

    max_sidetrack : float, optional
        Maximal sidetrack. The random walk will reflect downwards if it hits
        this limit. Default: None.

    min_sidetrack : float, optional
        Minimal sidetrack. The random walk will reflect upwards if it hits this
        limit. Default: None.

    step_mu : float, optional
        Mean of step size in random walk. Default: 0.0

    step_sigma : float, optional
        Standard deviation of step size in random walk. Default: 0.1.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    if mode not in ("multiplicative", "additive"):
        raise ValueError(
            "Argument `mode` must be either 'multiplicative' or 'additive'."
        )

    if initial_sidetrack is None:
        initial_sidetrack = 1.0 if (mode == "multiplicative") else 0.0

    if (min_sidetrack is not None) and (initial_sidetrack < min_sidetrack):
        raise ValueError(
            "Argument `initial_sidetrack` is less than `min_sidetrack`."
        )
    if (max_sidetrack is not None) and (initial_sidetrack > max_sidetrack):
        raise ValueError(
            "Argument `initial_sidetrack` is greater than `max_sidetrack`."
        )

    N = (
        0
    )  # type: int  # NOTE: this is a horrible hack to type hint for Python 3.5
    n = 0  # type: int
    c = 0  # type: int
    N, n, c = X.shape
    rand = np.random.RandomState(random_seed)
    M = (
        np.cumsum(
            rand.normal(size=X.shape, loc=step_mu, scale=step_sigma), axis=1
        )
        + initial_sidetrack
    )  # type: np.ndarray
    if (max_sidetrack is not None) and (min_sidetrack is not None):
        layer = (M - min_sidetrack) // (
            max_sidetrack - min_sidetrack
        )  # type: float
        M = M - layer * (max_sidetrack - min_sidetrack)
        M[layer % 2 == 1] = max_sidetrack + min_sidetrack - M[layer % 2 == 1]
    elif (max_sidetrack is not None) and (min_sidetrack is None):
        layer = (M >= max_sidetrack).astype(int)
        M[layer == 1] = 2 * max_sidetrack - M[layer == 1]
    elif (max_sidetrack is None) and (min_sidetrack is not None):
        layer = (M >= min_sidetrack).astype(int) - 1
        M[layer == -1] = 2 * min_sidetrack - M[layer == -1]
    else:
        pass

    if Y is None:
        if mode == "multiplicative":
            return X * M
        else:
            return X + M
    else:
        if mode == "multiplicative":
            return X * M, Y.copy()
        else:
            return X + M, Y.copy()


class RandomSidetrack(_Augmentor):
    """Augmentor that sidetracks time series from its original values.

    This augmentor has two modes: multiplicative mode sidetracks a time series
    by multiplying its values with random-walking multipliers, and additive
    mode sidetracks a time series by adding its values with random-walking
    additives.

    Parameters
    ----------
    mode : str, optional
        Either "multiplicative" or "additive". Default: "multiplicative".

    initial_sidetrack : float, optional
        Sidetrack at the start of time series. If not given, 1.0 for
        multiplicative mode and 0.0 for additive mode. Default: None.

    max_sidetrack : float, optional
        Maximal sidetrack. The random walk will reflect downwards if it hits
        this limit. Default: None.

    min_sidetrack : float, optional
        Minimal sidetrack. The random walk will reflect upwards if it hits this
        limit. Default: None.

    step_mu : float, optional
        Mean of step size in random walk. Default: 0.

    step_sigma : float, optional
        Standard deviation of step size in random walk. Default: 0.1.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.
    """

    def __init__(
        self,
        mode: Optional[str] = "multiplicative",
        initial_sidetrack: Optional[float] = None,
        max_sidetrack: Optional[float] = None,
        min_sidetrack: Optional[float] = None,
        step_mu: Optional[float] = 0,
        step_sigma: Optional[float] = 0.1,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            augmentor_func=random_sidetrack,
            is_random=True,
            mode=mode,
            initial_sidetrack=initial_sidetrack,
            max_sidetrack=max_sidetrack,
            min_sidetrack=min_sidetrack,
            step_mu=step_mu,
            step_sigma=step_sigma,
            random_seed=random_seed,
        )

    @property
    def mode(self) -> Optional[str]:
        return self._params["mode"]

    @mode.setter
    def mode(self, mode: Optional[str]) -> None:
        self._params["mode"] = mode

    @property
    def max_sidetrack(self) -> Optional[float]:
        return self._params["max_sidetrack"]

    @max_sidetrack.setter
    def max_sidetrack(self, max_sidetrack: Optional[float]) -> None:
        self._params["max_sidetrack"] = max_sidetrack

    @property
    def min_sidetrack(self) -> Optional[float]:
        return self._params["min_sidetrack"]

    @min_sidetrack.setter
    def min_sidetrack(self, min_sidetrack: Optional[float]) -> None:
        self._params["min_sidetrack"] = min_sidetrack

    @property
    def initial_sidetrack(self) -> Optional[float]:
        return self._params["initial_sidetrack"]

    @initial_sidetrack.setter
    def initial_sidetrack(self, initial_sidetrack: Optional[float]) -> None:
        self._params["initial_sidetrack"] = initial_sidetrack

    @property
    def step_mu(self) -> Optional[float]:
        return self._params["step_mu"]

    @step_mu.setter
    def step_mu(self, step_mu: Optional[float]) -> None:
        self._params["step_mu"] = step_mu

    @property
    def step_sigma(self) -> Optional[float]:
        return self._params["step_sigma"]

    @step_sigma.setter
    def step_sigma(self, step_sigma: Optional[float]) -> None:
        self._params["step_sigma"] = step_sigma

    @property
    def random_seed(self) -> Optional[int]:
        return self._params["random_seed"]

    @random_seed.setter
    def random_seed(self, random_seed: Optional[int]) -> None:
        self._params["random_seed"] = random_seed
