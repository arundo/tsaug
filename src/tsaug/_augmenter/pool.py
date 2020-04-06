import numpy as np

from .base import _Augmenter


class Pool(_Augmenter):
    def __init__(
        self,
        kind="ave",
        size=2,
        per_channel=False,
        repeats=1,
        prob=1.0,
        seed=None,
    ):
        self.kind = kind
        self.size = size
        self.per_channel = per_channel
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls):
        return ("kind", "size", "per_channel")

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, k):
        if not isinstance(k, str):
            raise TypeError(
                "Parameter `kind` must be one of 'max', 'min', and 'ave'."
            )
        if k not in ("max", "min", "ave"):
            raise ValueError(
                "Parameter `kind` must be one of 'max', 'min', and 'ave'."
            )
        self._kind = k

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

        if self.per_channel:
            kernel = rand.choice(size, size=N * C)
        else:
            kernel = rand.choice(size, size=N)
            kernel = np.repeat(kernel, C)

        X_aug = X.copy()
        X_aug = X_aug.swapaxes(1, 2).reshape(N * C, T)

        if self.kind == "max":
            pool_func = np.max
        elif self.kind == "min":
            pool_func = np.min
        else:
            pool_func = np.mean

        for s in np.unique(kernel):
            X_aug[kernel == s, : (s * int(T / s))] = np.repeat(
                pool_func(
                    X_aug[kernel == s, : (s * int(T / s))].reshape(
                        -1, int(T / s), s
                    ),
                    axis=2,
                    keepdims=True,
                ),
                s,
                axis=2,
            ).reshape(-1, s * int(T / s))
            if T % s:
                X_aug[kernel == s, (s * int(T / s)) :] = pool_func(
                    X_aug[kernel == s, (s * int(T / s)) :],
                    axis=1,
                    keepdims=True,
                )

        X_aug = X_aug.reshape(N, C, T).swapaxes(1, 2)

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug
