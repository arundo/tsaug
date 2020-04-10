from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline

from .base import _Augmenter, _default_seed


class Drift(_Augmenter):
    """
    Drift the value of time series.

    The augmenter drifts the value of time series from its original values
    randomly and smoothly. The extent of drifting is controlled by the maximal
    drift and the number of drift points.

    Parameters
    ----------
    max_drift : float or tuple, optional
        The maximal amount of drift added to a time series.

        - If float, all series (all channels if `per_channel` is True) are
          drifted with the same maximum.
        - If tuple, the maximal drift added to a time series (a channel if
          `per_channel` is True) is sampled from this interval randomly.

        Default: 0.5.

    n_drift_points : int or list, optional
        The number of time points a new drifting trend is defined in a series.

        - If int, all series (all channels if `per_channel` is True) have the
          same number of drift points.
        - If list, the number of drift points defined in a series (a channel if
          `per_channel` is True) is sampled from this list randomly.

    kind : str, optional
        How the noise is added to the original time series. It must be either
        'additive' or 'multiplicative'. Default: 'additive'.

    per_channel : bool, optional
        Whether to sample independent drifting trends for each channel in a time
        series or to use the same drifting trends for all channels in a time
        series. Default: True.

    normalize : bool, optional
        Whether the drifting trend is added to the normalized time series. If
        True, each channel of a time series is normalized to [0, 1] first.
        Default: True.

    repeats : int, optional
        The number of times a series is augmented. If greater than one, a series
        will be augmented so many times independently. This parameter can also
        be set by operator `*`. Default: 1.

    prob : float, optional
        The probability of a series is augmented. It must be in (0.0, 1.0]. This
        parameter can also be set by operator `@`. Default: 1.0.

    seed : int, optional
        The random seed. Default: None.

    """

    def __init__(
        self,
        max_drift: Union[float, Tuple[float, float]] = 0.5,
        n_drift_points: Union[int, List[int]] = 3,
        kind: str = "additive",
        per_channel: bool = True,
        normalize: bool = True,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = _default_seed,
    ):
        self.max_drift = max_drift
        self.n_drift_points = n_drift_points
        self.kind = kind
        self.per_channel = per_channel
        self.normalize = normalize
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return (
            "max_drift",
            "n_drift_points",
            "kind",
            "per_channel",
            "normalize",
        )

    @property
    def max_drift(self) -> Union[float, Tuple[float, float]]:
        return self._max_drift

    @max_drift.setter
    def max_drift(self, v: Union[float, Tuple[float, float]]) -> None:
        MAX_DRIFT_ERROR_MSG = (
            "Parameter `max_drift` must be a non-negative number "
            "or a 2-tuple of non-negative numbers representing an interval. "
        )
        if not isinstance(v, (float, int)):
            if isinstance(v, tuple):
                if len(v) != 2:
                    raise ValueError(MAX_DRIFT_ERROR_MSG)
                if (not isinstance(v[0], (float, int))) or (
                    not isinstance(v[1], (float, int))
                ):
                    raise TypeError(MAX_DRIFT_ERROR_MSG)
                if v[0] > v[1]:
                    raise ValueError(MAX_DRIFT_ERROR_MSG)
                if (v[0] < 0.0) or (v[1] < 0.0):
                    raise ValueError(MAX_DRIFT_ERROR_MSG)
            else:
                raise TypeError(MAX_DRIFT_ERROR_MSG)
        elif v < 0.0:
            raise ValueError(MAX_DRIFT_ERROR_MSG)
        self._max_drift = v

    @property
    def n_drift_points(self) -> Union[int, List[int]]:
        return self._n_drift_points

    @n_drift_points.setter
    def n_drift_points(self, n: Union[int, List[int]]) -> None:
        N_DRIFT_POINTS_ERROR_MSG = (
            "Parameter `n_drift_points` must be a positive integer "
            "or a list of positive integers."
        )
        if not isinstance(n, int):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(N_DRIFT_POINTS_ERROR_MSG)
                if not all([isinstance(nn, int) for nn in n]):
                    raise TypeError(N_DRIFT_POINTS_ERROR_MSG)
                if not all([nn > 0 for nn in n]):
                    raise ValueError(N_DRIFT_POINTS_ERROR_MSG)
            else:
                raise TypeError(N_DRIFT_POINTS_ERROR_MSG)
        elif n <= 0:
            raise ValueError(N_DRIFT_POINTS_ERROR_MSG)
        self._n_drift_points = n

    @property
    def per_channel(self) -> bool:
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p: bool) -> None:
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, p: bool) -> None:
        if not isinstance(p, bool):
            raise TypeError("Paremeter `normalize` must be boolean.")
        self._normalize = p

    @property
    def kind(self) -> str:
        return self._kind

    @kind.setter
    def kind(self, k: str) -> None:
        if not isinstance(k, str):
            raise TypeError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        if k not in ("additive", "multiplicative"):
            raise ValueError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        self._kind = k

    def _augment_core(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)

        if isinstance(self.n_drift_points, int):
            n_drift_points = set([self.n_drift_points])
        else:
            n_drift_points = set(self.n_drift_points)

        ind = rand.choice(
            len(n_drift_points), N * (C if self.per_channel else 1)
        )  # map series to n_drift_points

        drift = np.zeros((N * (C if self.per_channel else 1), T))
        for i, n in enumerate(n_drift_points):
            if not (ind == i).any():
                continue
            anchors = np.cumsum(
                rand.normal(size=((ind == i).sum(), n + 2)), axis=1
            )  # type: np.ndarray
            interpFuncs = CubicSpline(
                np.linspace(0, T, n + 2), anchors, axis=1
            )  # type: Callable
            drift[ind == i, :] = interpFuncs(np.arange(T))
        drift = drift.reshape((N, -1, T)).swapaxes(1, 2)
        drift = drift - drift[:, 0, :].reshape(N, 1, -1)
        drift = drift / abs(drift).max(axis=1, keepdims=True)
        if isinstance(self.max_drift, (float, int)):
            drift = drift * self.max_drift
        else:
            drift = drift * rand.uniform(
                low=self.max_drift[0],
                high=self.max_drift[1],
                size=(N, 1, C if self.per_channel else 1),
            )

        if self.kind == "additive":
            if self.normalize:
                X_aug = X + drift * (
                    X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)
                )
            else:
                X_aug = X + drift
        else:
            X_aug = X * (1 + drift)

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug
