import numpy as np

from .base import _Augmentor


class Dropout(_Augmentor):
    def __init__(
        self,
        p=0.05,
        size=1,
        fill="ffill",
        per_channel=False,
        repeats=1,
        prob=1.0,
        seed=None,
    ):
        self.p = p
        self.size = size
        self.fill = fill
        self.per_channel = per_channel
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, n):
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
    def size(self):
        return self._size

    @size.setter
    def size(self, n):
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
    def fill(self):
        return self._fill

    @fill.setter
    def fill(self, f):
        FILL_ERROR_MSG = (
            "Paramter `fill` must be a number or one of 'ffill', 'bfill', "
            "'mean', and 'median'."
        )
        if isinstance(f, str):
            if f not in ("ffill", "bfill", "mean", "median"):
                raise ValueError(FILL_ERROR_MSG)
        elif not isinstance(f, (int, float)):
            raise TypeError(FILL_ERROR_MSG)

        self._fill = f

    @property
    def per_channel(self):
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p):
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    def _augment_core(self, X, Y):
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
        if isinstance(self.fill, str) and (self.fill in ("mean", "median")):
            fill_value = (
                X_aug.mean(axis=1)
                if (self.fill == "mean")
                else X_aug.median(axis=1)
            )
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
                elif isinstance(self.fill, str) and (
                    self.fill in ("mean", "median")
                ):
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
