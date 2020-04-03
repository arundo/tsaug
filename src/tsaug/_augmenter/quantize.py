import numpy as np
from sklearn.cluster import KMeans

from .base import _Augmentor


class Quantize(_Augmentor):
    def __init__(
        self,
        n_levels=10,
        how="uniform",
        per_channel=False,
        repeats=1,
        prob=1.0,
        seed=None,
    ):
        self.n_levels = n_levels
        self.how = how
        self.per_channel = per_channel
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @staticmethod
    def _change_series_length():
        return False

    @property
    def n_levels(self):
        return self._n_levels

    @n_levels.setter
    def n_levels(self, n):
        N_LEVELS_ERROR_MSG = (
            "Parameter `n_levels` must be a positive integer, "
            "a 2-tuple of positive integers representing an interval, "
            "or a list of positive integers."
        )
        if not isinstance(n, int):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(N_LEVELS_ERROR_MSG)
                if not all([isinstance(nn, int) for nn in n]):
                    raise TypeError(N_LEVELS_ERROR_MSG)
                if not all([nn > 0 for nn in n]):
                    raise ValueError(N_LEVELS_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(N_LEVELS_ERROR_MSG)
                if (not isinstance(n[0], int)) or (not isinstance(n[1], int)):
                    raise TypeError(N_LEVELS_ERROR_MSG)
                if n[0] > n[1]:
                    raise ValueError(N_LEVELS_ERROR_MSG)
                if (n[0] <= 0) or (n[1] <= 0):
                    raise ValueError(N_LEVELS_ERROR_MSG)
            else:
                raise TypeError(N_LEVELS_ERROR_MSG)
        elif n <= 0:
            raise ValueError(N_LEVELS_ERROR_MSG)
        self._n_levels = n

    @property
    def how(self):
        return self._how

    @how.setter
    def how(self, h):
        HOW_ERROR_MSG = "Parameter `how` must be one of 'uniform', 'quantile', and 'kmeans'."
        if not isinstance(h, str):
            raise TypeError(HOW_ERROR_MSG)
        if h not in ["uniform", "quantile", "kmeans"]:
            raise ValueError(HOW_ERROR_MSG)
        self._how = h

    @property
    def per_channel(self):
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p):
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    def _augment_once(self, X, Y):
        rand = np.random.RandomState(self.seed)
        N, T, C = X.shape

        if isinstance(self.n_levels, int):
            n_levels = (np.ones((N, 1, C)) * self.n_levels).astype(int)
        elif isinstance(self.n_levels, list):
            if self.per_channel:
                n_levels = rand.choice(self.n_levels, size=(N, 1, C)).astype(
                    int
                )
            else:
                n_levels = rand.choice(self.n_levels, size=(N, 1, 1)).astype(
                    int
                )
                n_levels = np.repeat(n_levels, C, axis=2)
        else:
            if self.per_channel:
                n_levels = rand.choice(
                    range(self.n_levels[0], self.n_levels[1]), size=(N, 1, C)
                ).astype(int)
            else:
                n_levels = rand.choice(
                    range(self.n_levels[0], self.n_levels[1]), size=(N, 1, 1)
                ).astype(int)
                n_levels = np.repeat(n_levels, C, axis=2)

        if self.how == "uniform":
            series_min = X.min(axis=1, keepdims=True)
            series_max = X.max(axis=1, keepdims=True)
            series_range = series_max - series_min
            series_range[series_range == 0] = 1
            X_aug = (X - series_min) / series_range
            X_aug = X_aug * n_levels
            X_aug = X_aug.round()
            X_aug = X_aug.clip(0, n_levels - 1)
            X_aug = X_aug + 0.5
            X_aug = X_aug / n_levels
            X_aug = X_aug * series_range + series_min
        elif self.how == "quantile":
            n_levels = n_levels.flatten()
            X_aug = X.copy()
            X_aug = X_aug.swapaxes(1, 2).reshape((N * C, T))
            for i in range(len(X_aug)):
                bins = np.quantile(
                    X_aug[i, :], np.arange(n_levels[i] + 1) / n_levels[i]
                )
                bins_center = np.quantile(
                    X_aug[i, :], np.arange(0.5, n_levels[i]) / n_levels[i]
                )
                X_aug[i, :] = bins_center[
                    np.digitize(X_aug[i, :], bins).clip(0, n_levels[i] - 1),
                ]
            X_aug = X_aug.reshape(N, C, T).swapaxes(1, 2)
        else:
            n_levels = n_levels.flatten()
            X_aug = X.copy()
            X_aug = X.swapaxes(1, 2).reshape((N * C, T))
            model = KMeans(n_clusters=2, n_jobs=-1)
            for i in range(len(X_aug)):
                model.n_clusters = n_levels[i]
                ind = model.fit_predict(X_aug[i].reshape(-1, 1))
                X_aug[i, :] = model.cluster_centers_[ind, :].flatten()
            X_aug = X_aug.reshape(N, C, T).swapaxes(1, 2)

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug
