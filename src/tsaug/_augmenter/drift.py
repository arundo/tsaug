import numpy as np
from scipy.interpolate import CubicSpline

from .base import _Augmentor


class Drift(_Augmentor):
    def __init__(
        self,
        max_drift=0.5,
        n_drift_points=3,
        kind="additive",
        per_channel=True,
        repeat=1,
        prob=1.0,
        seed=None,
    ):
        self.max_drift = max_drift
        self.n_drift_points = n_drift_points
        self.kind = kind
        self.per_channel = per_channel
        super().__init__(repeat=repeat, prob=prob, seed=seed)

    @staticmethod
    def _change_series_length():
        return False

    @property
    def max_drift(self):
        return self._max_drift

    @max_drift.setter
    def max_drift(self, v):
        if not isinstance(v, (float, int)):
            raise TypeError(
                "Parameter `max_drift` must be a non-negative number."
            )
        if v < 0:
            raise ValueError(
                "Parameter `max_drift` must be a non-negative number."
            )
        self._max_drift = v

    @property
    def n_drift_points(self):
        return self._n_drift_points

    @n_drift_points.setter
    def n_drift_points(self, n):
        if not isinstance(n, (float, int)):
            raise TypeError(
                "Parameter `n_drift_points` must be a positive integer."
            )
        if n < 0:
            raise ValueError(
                "Parameter `n_drift_points` must be a positive integer."
            )
        self._n_drift_points = n

    @property
    def per_channel(self):
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p):
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, k):
        if not isinstance(k, str):
            raise TypeError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        if k not in ("additive", "multiplicative"):
            raise ValueError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        self._kind = k

    def _augment_once(self, X, Y):
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)
        if self.per_channel:
            anchors = np.cumsum(
                rand.normal(size=(N, self.n_drift_points + 2, C)), axis=1
            )  # type: np.ndarray
        else:
            anchors = np.cumsum(
                rand.normal(size=(N, self.n_drift_points + 2, 1)), axis=1
            )

        interpFuncs = CubicSpline(
            np.linspace(0, T, self.n_drift_points + 2), anchors, axis=1
        )  # type: Callable

        drift = interpFuncs(np.arange(T))
        drift = drift - drift[:, 0, :].reshape(N, 1, -1)
        drift = drift / abs(drift).max(axis=1, keepdims=True) * self.max_drift

        if self.kind == "additive":
            X_aug = X + drift * abs(X).max(axis=1, keepdims=True)
        else:
            X_aug = X * (1 + drift)

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug
