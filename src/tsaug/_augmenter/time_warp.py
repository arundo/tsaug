import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from .base import _Augmenter


class TimeWarp(_Augmenter):
    def __init__(
        self,
        n_speed_change=3,
        maxmin_speed_ratio=3,
        repeats=1,
        prob=1.0,
        seed=None,
    ):
        self.n_speed_change = n_speed_change
        self.maxmin_speed_ratio = maxmin_speed_ratio
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls):
        return ("n_speed_change",)

    @property
    def n_speed_change(self):
        return self._n_speed_change

    @n_speed_change.setter
    def n_speed_change(self, n):
        if not isinstance(n, int):
            raise TypeError(
                "Parameter `n_speed_change` must be a positive integer."
            )
        if n <= 0:
            raise ValueError(
                "Parameter `n_speed_change` must be a positive integer."
            )
        self._n_speed_change = n

    @property
    def maxmin_speed_ratio(self):
        return self._maxmin_speed_ratio

    @maxmin_speed_ratio.setter
    def maxmin_speed_ratio(self, n):
        MAXMIN_SPEED_RATIO_ERROR_MSG = (
            "Parameter `maxmin_speed_ratio` must be a number greater or equal "
            " to 1.0, "
            "a 2-tuple of such numbers representing an interval, "
            "or a list of such numbers."
        )
        if not isinstance(n, (float, int)):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(MAXMIN_SPEED_RATIO_ERROR_MSG)
                if not all([isinstance(nn, (float, int)) for nn in n]):
                    raise TypeError(MAXMIN_SPEED_RATIO_ERROR_MSG)
                if not all([nn >= 0 for nn in n]):
                    raise ValueError(MAXMIN_SPEED_RATIO_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(MAXMIN_SPEED_RATIO_ERROR_MSG)
                if (not isinstance(n[0], (float, int))) or (
                    not isinstance(n[1], (float, int))
                ):
                    raise TypeError(MAXMIN_SPEED_RATIO_ERROR_MSG)
                if n[0] > n[1]:
                    raise ValueError(MAXMIN_SPEED_RATIO_ERROR_MSG)
                if (n[0] < 0) or (n[1] < 0):
                    raise ValueError(MAXMIN_SPEED_RATIO_ERROR_MSG)
            else:
                raise TypeError(MAXMIN_SPEED_RATIO_ERROR_MSG)
        elif n < 0:
            raise ValueError(MAXMIN_SPEED_RATIO_ERROR_MSG)
        self._maxmin_speed_ratio = n

    def _augment_core(self, X, Y):
        rand = np.random.RandomState(self.seed)
        N, T, C = X.shape
        if Y is not None:
            L = Y.shape[2]  # type: int

        anchors = np.arange(
            0,
            1 + 1 / (self.n_speed_change + 1) / 2,
            1 / (self.n_speed_change + 1),
        ) * (T - 1)

        if isinstance(self.maxmin_speed_ratio, (float, int)):
            maxmin_speed_ratio = np.ones(N) * self.maxmin_speed_ratio
        elif isinstance(self.maxmin_speed_ratio, tuple):
            maxmin_speed_ratio = rand.uniform(
                low=self.maxmin_speed_ratio[0],
                high=self.maxmin_speed_ratio[1],
                size=N,
            )
        else:
            maxmin_speed_ratio = rand.choice(self.maxmin_speed_ratio, size=N)
        anchor_values = rand.uniform(
            low=1 / maxmin_speed_ratio.reshape(N, 1),
            high=1.0,
            size=(N, self.n_speed_change + 1),
        )
        anchor_values = (
            anchor_values.cumsum(axis=1)
            / anchor_values.sum(axis=1, keepdims=True)
            * (T - 1)
        )
        anchor_values = np.concatenate(
            [np.zeros((N, 1)), anchor_values], axis=1
        )

        warp = PchipInterpolator(x=anchors, y=anchor_values, axis=1)(
            np.arange(T)
        )  # type: np.ndarray

        X_aug = np.vstack(
            [
                interp1d(
                    np.arange(T),
                    Xi,
                    axis=0,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )(warpi).reshape(1, T, C)
                for Xi, warpi in zip(X, warp)
            ]
        )  # type: np.ndarray

        if Y is None:
            Y_aug = None  # type: Optional[np.ndarray]
        else:
            Y_aug = np.vstack(
                [
                    interp1d(
                        np.arange(T),
                        Yi,
                        axis=0,
                        fill_value="extrapolate",
                        assume_sorted=True,
                    )(warpi).reshape(1, T, L)
                    for Yi, warpi in zip(Y, warp)
                ]
            )
            Y_aug = Y_aug.round().astype(int)

        return X_aug, Y_aug
