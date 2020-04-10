from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from .base import _Augmenter, _default_seed


class TimeWarp(_Augmenter):
    """
    Random time warping.

    The augmenter random changed the speed of timeline. The time warping is
    controlled by the number of speed changes and the maximal ratio of max/min
    speed.

    Parameters
    ----------
    n_speed_change : int, optional
        The number of speed changes in each series. Default: 3.

    max_speed_ratio : float, tuple, or list, optional
        The maximal ratio of max/min speed in the warpped time line. The time
        line of a series is more likely to be significantly warpeped if this
        value is greater.

        - If float, all series are warpped with the same ratio.
        - If list, each series is warpped with a ratio that is randomly sampled
          from the list.
        - If 2-tuple, each series is warpped with a ratio that is randomly
          sampled from the interval.

        Default: 3.0.

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
        n_speed_change: int = 3,
        max_speed_ratio: Union[float, Tuple[float, float], List[float]] = 3.0,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = _default_seed,
    ):
        self.n_speed_change = n_speed_change
        self.max_speed_ratio = max_speed_ratio
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("n_speed_change",)

    @property
    def n_speed_change(self) -> int:
        return self._n_speed_change

    @n_speed_change.setter
    def n_speed_change(self, n: int) -> None:
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
    def max_speed_ratio(
        self,
    ) -> Union[float, Tuple[float, float], List[float]]:
        return self._max_speed_ratio

    @max_speed_ratio.setter
    def max_speed_ratio(
        self, n: Union[float, Tuple[float, float], List[float]]
    ) -> None:
        MAX_SPEED_RATIO_ERROR_MSG = (
            "Parameter `max_speed_ratio` must be a number greater than 1.0, "
            "a 2-tuple of such numbers representing an interval, "
            "or a list of such numbers."
        )
        if not isinstance(n, (float, int)):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(MAX_SPEED_RATIO_ERROR_MSG)
                if not all([isinstance(nn, (float, int)) for nn in n]):
                    raise TypeError(MAX_SPEED_RATIO_ERROR_MSG)
                if not all([nn > 1.0 for nn in n]):
                    raise ValueError(MAX_SPEED_RATIO_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(MAX_SPEED_RATIO_ERROR_MSG)
                if (not isinstance(n[0], (float, int))) or (
                    not isinstance(n[1], (float, int))
                ):
                    raise TypeError(MAX_SPEED_RATIO_ERROR_MSG)
                if n[0] > n[1]:
                    raise ValueError(MAX_SPEED_RATIO_ERROR_MSG)
                if (n[0] <= 1.0) or (n[1] <= 1.0):
                    raise ValueError(MAX_SPEED_RATIO_ERROR_MSG)
            else:
                raise TypeError(MAX_SPEED_RATIO_ERROR_MSG)
        elif n <= 1.0:
            raise ValueError(MAX_SPEED_RATIO_ERROR_MSG)
        self._max_speed_ratio = n

    def _augment_core(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rand = np.random.RandomState(self.seed)
        N, T, C = X.shape
        if Y is not None:
            L = Y.shape[2]  # type: int

        anchors = np.arange(
            0,
            1 + 1 / (self.n_speed_change + 1) / 2,
            1 / (self.n_speed_change + 1),
        ) * (T - 1)

        if isinstance(self.max_speed_ratio, (float, int)):
            max_speed_ratio = np.ones(N) * self.max_speed_ratio
        elif isinstance(self.max_speed_ratio, tuple):
            max_speed_ratio = rand.uniform(
                low=self.max_speed_ratio[0],
                high=self.max_speed_ratio[1],
                size=N,
            )
        else:
            max_speed_ratio = rand.choice(self.max_speed_ratio, size=N)
        anchor_values = rand.uniform(
            low=0.0, high=1.0, size=(N, self.n_speed_change + 1)
        )
        anchor_values = anchor_values - (
            anchor_values.max(axis=1, keepdims=True)
            - max_speed_ratio.reshape(N, 1)
            * anchor_values.min(axis=1, keepdims=True)
        ) / (1 - max_speed_ratio.reshape(N, 1))
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
