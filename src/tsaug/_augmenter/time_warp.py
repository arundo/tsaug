import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from .base import _Augmenter


class TimeWarp(_Augmenter):
    def __init__(self, n_speed_change=3, repeats=1, prob=1.0, seed=None):
        self.n_speed_change = n_speed_change
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

    def _augment_core(self, X, Y):
        rand = np.random.RandomState(self.seed)
        N, T, C = X.shape
        if Y is not None:
            L = Y.shape[2]  # type: int

        anchors = np.arange(
            0.5 / self.n_speed_change, 1, 1 / self.n_speed_change
        )  # type: np.ndarray
        R = (
            rand.uniform(
                low=-0.5 / self.n_speed_change,
                high=0.5 / self.n_speed_change,
                size=(N, self.n_speed_change),
            )
            * 0.5
        )  # type: np.ndarray
        anchor_values = R + np.arange(
            0.5 / self.n_speed_change, 1, 1 / self.n_speed_change
        )  # type: np.ndarray

        anchors = np.append([0], anchors)
        anchors = np.append(anchors, [1])
        anchors = anchors * (T - 1)

        anchor_values = np.concatenate(
            [np.zeros((N, 1)), anchor_values], axis=1
        )
        anchor_values = np.concatenate(
            [anchor_values, np.ones((N, 1))], axis=1
        )
        anchor_values = anchor_values * (T - 1)

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
