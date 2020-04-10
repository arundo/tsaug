from typing import List, Optional, Tuple, Union

import numpy as np

from .base import _Augmenter, _default_seed


class Dropout(_Augmenter):
    """
    Dropout values of some random time points in time series.

    Single time points or sub-sequences could be dropped out.

    Parameters
    ----------
    p : float, tuple, or list, optional
        Probablity of the value of a time point to be dropped out.

        - If float, all series (all channels if `per_channel` is True) have the
          same probability.
        - If list, a series (a channel if `per_channel` is True) has a
          probability sampled from this list randomly.
        - If 2-tuple, a series (a channel if `per_channel` is True) has a
          probability sampled from this interval randomly.

        Default: 0.05.

    size : int, tuple, or list, optional
        Size of dropped out units.

        - If int, all dropped out units have the same size.
        - If list, a dropped out unit has size sampled from this list randomly.
        - If 2-tuple, a dropped out unit has size sampled from this interval
          randomly.

        Note that dropped out units could overlap which results in larger units
        effectively, though the probability is low if `p` is small.

        Default: 1.

    fill : str or float, optional
        How a dropped out value is filled.

        - If 'ffill', fill with the last previous value that is not dropped.
        - If 'bfill', fill with the first next value that is not dropped.
        - If 'mean', fill with the mean value of this channel in this series.
        - If float, fill with this value.

        Default: 'ffill'.

    per_channel : bool, optional
        Whether to sample dropout units independently for each channel in a time
        series or to use the same dropout units for all channels in a time
        series. Default: False.

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
        p: Union[float, Tuple[float, float], List[float]] = 0.05,
        size: Union[int, Tuple[int, int], List[int]] = 1,
        fill: Union[str, float] = "ffill",
        per_channel: bool = False,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = _default_seed,
    ):
        self.p = p
        self.size = size
        self.fill = fill
        self.per_channel = per_channel
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("p", "size", "fill", "per_channel")

    @property
    def p(self) -> Union[float, Tuple[float, float], List[float]]:
        return self._p

    @p.setter
    def p(self, n: Union[float, Tuple[float, float], List[float]]) -> None:
        P_ERROR_MSG = (
            "Parameter `p` must be a non-negative number, "
            "a 2-tuple of non-negative numbers representing an interval, "
            "or a list of non-negative numbers."
        )
        if not isinstance(n, (float, int)):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(P_ERROR_MSG)
                if not all([isinstance(nn, (float, int)) for nn in n]):
                    raise TypeError(P_ERROR_MSG)
                if not all([(nn >= 0.0) and (nn <= 1.0) for nn in n]):
                    raise ValueError(P_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(P_ERROR_MSG)
                if (not isinstance(n[0], (float, int))) or (
                    not isinstance(n[1], (float, int))
                ):
                    raise TypeError(P_ERROR_MSG)
                if n[0] > n[1]:
                    raise ValueError(P_ERROR_MSG)
                if (
                    (n[0] < 0.0)
                    or (n[0] > 1.0)
                    or (n[1] < 0.0)
                    or (n[1] > 1.0)
                ):
                    raise ValueError(P_ERROR_MSG)
            else:
                raise TypeError(P_ERROR_MSG)
        elif (n < 0.0) or (n > 1.0):
            raise ValueError(P_ERROR_MSG)
        self._p = n

    @property
    def size(self) -> Union[int, Tuple[int, int], List[int]]:
        return self._size

    @size.setter
    def size(self, n: Union[int, Tuple[int, int], List[int]]) -> None:
        SIZE_ERROR_MSG = (
            "Parameter `size` must be a positive integer, "
            "a 2-tuple of positive integers representing an interval, "
            "or a list of positive integers."
        )
        if not isinstance(n, int):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(SIZE_ERROR_MSG)
                if not all([isinstance(nn, int) for nn in n]):
                    raise TypeError(SIZE_ERROR_MSG)
                if not all([nn > 0 for nn in n]):
                    raise ValueError(SIZE_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(SIZE_ERROR_MSG)
                if (not isinstance(n[0], int)) or (not isinstance(n[1], int)):
                    raise TypeError(SIZE_ERROR_MSG)
                if n[0] >= n[1]:
                    raise ValueError(SIZE_ERROR_MSG)
                if (n[0] <= 0) or (n[1] <= 0):
                    raise ValueError(SIZE_ERROR_MSG)
            else:
                raise TypeError(SIZE_ERROR_MSG)
        elif n <= 0:
            raise ValueError(SIZE_ERROR_MSG)
        self._size = n

    @property
    def fill(self) -> Union[str, float]:
        return self._fill

    @fill.setter
    def fill(self, f: Union[str, float]) -> None:
        FILL_ERROR_MSG = (
            "Paramter `fill` must be a number or one of 'ffill', 'bfill', and "
            "'mean'."
        )
        if isinstance(f, str):
            if f not in ("ffill", "bfill", "mean"):
                raise ValueError(FILL_ERROR_MSG)
        elif not isinstance(f, (int, float)):
            raise TypeError(FILL_ERROR_MSG)

        self._fill = f

    @property
    def per_channel(self) -> bool:
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p: bool) -> None:
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    def _augment_core(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rand = np.random.RandomState(self.seed)
        N, T, C = X.shape

        if isinstance(self.size, int):
            size = [self.size]
        elif isinstance(self.size, tuple):
            size = list(range(self.size[0], self.size[1]))
        else:
            size = self.size

        if isinstance(self.p, (float, int)):
            p = np.ones(N * C if self.per_channel else N) * self.p
        elif isinstance(self.p, tuple):
            p = rand.uniform(
                low=self.p[0],
                high=self.p[1],
                size=(N * C if self.per_channel else N),
            )
        else:
            p = rand.choice(self.p, size=(N * C if self.per_channel else N))

        X_aug = X.copy()
        X_aug = X_aug.swapaxes(1, 2).reshape(N * C, T)
        if isinstance(self.fill, str) and (self.fill == "mean"):
            fill_value = X_aug.mean(axis=1)
        for s in size:
            # sample dropout blocks
            if self.per_channel:
                drop = (
                    rand.uniform(size=(N * C, T - s))
                    <= p.reshape(-1, 1) / len(size) / s
                )
            else:
                drop = (
                    rand.uniform(size=(N, T - s))
                    <= p.reshape(-1, 1) / len(size) / s
                )
                drop = np.repeat(drop, C, axis=0)
            ind = np.argwhere(drop)  # position of dropout blocks
            if ind.size > 0:
                if isinstance(self.fill, str) and (self.fill == "ffill"):
                    i = np.repeat(ind[:, 0], s)
                    j0 = np.repeat(ind[:, 1], s)
                    j1 = j0 + np.tile(np.arange(1, s + 1), len(ind))
                    X_aug[i, j1] = X_aug[i, j0]
                elif isinstance(self.fill, str) and (self.fill == "bfill"):
                    i = np.repeat(ind[:, 0], s)
                    j0 = np.repeat(ind[:, 1], s) + s
                    j1 = j0 - np.tile(np.arange(1, s + 1), len(ind))
                    X_aug[i, j1] = X_aug[i, j0]
                elif isinstance(self.fill, str) and (self.fill == "mean"):
                    i = np.repeat(ind[:, 0], s)
                    j = np.repeat(ind[:, 1], s) + np.tile(
                        np.arange(1, s + 1), len(ind)
                    )
                    X_aug[i, j] = fill_value[i]
                else:
                    i = np.repeat(ind[:, 0], s)
                    j = np.repeat(ind[:, 1], s) + np.tile(
                        np.arange(1, s + 1), len(ind)
                    )
                    X_aug[i, j] = self.fill

        X_aug = X_aug.reshape(N, C, T).swapaxes(1, 2)

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug
