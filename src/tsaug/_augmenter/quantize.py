from typing import List, Optional, Tuple, Union

import numpy as np

from .base import _Augmenter, _default_seed


class Quantize(_Augmenter):
    """
    Quantize time series to a level set.

    Values in a time series are rounded to the nearest level in the level set.

    Parameters
    ----------
    n_levels : int, tuple, or list, optional
        The number levels in a level set.

        - If int, all series (all channels if `per_channel` is True) are
          quantized to a level set of this size.
        - If list, a series (a channel if `per_channel` is True) is quantized
          to a level set whose size is sampled from this list randomly.
        - If 2-tuple, a series (a channel if `per_channel` is True) is quantized
          to a level set whose size is sampled from this interval randomly.

        Default: 10.

    how : str, optional
        The method that a level set is defined.

        - If 'uniform', a level set is defined by uniformly discretizing the
          range of this channel in this series.
        - If 'quantile', a level set is defined by the quantiles of values in
          this channel in this series.
        - If 'kmeans', a level set is defined by k-means clustering of values
          in this channel in this series. Note that this method could be slow.

        Default: 'uniform'.

    per_channel : bool, optional
        Whether to sample a level set size for each channel in a time series or
        to use the same size for all channels in a time series. Only used if
        the level set size is not deterministic. Default: False.

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
        n_levels: Union[int, Tuple[int, int], List[int]] = 10,
        how: str = "uniform",
        per_channel: bool = False,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = _default_seed,
    ):
        self.n_levels = n_levels
        self.how = how
        self.per_channel = per_channel
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("n_levels", "how", "per_channel")

    @property
    def n_levels(self) -> Union[int, Tuple[int, int], List[int]]:
        return self._n_levels

    @n_levels.setter
    def n_levels(self, n: Union[int, Tuple[int, int], List[int]]) -> None:
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
                if n[0] >= n[1]:
                    raise ValueError(N_LEVELS_ERROR_MSG)
                if (n[0] <= 0) or (n[1] <= 0):
                    raise ValueError(N_LEVELS_ERROR_MSG)
            else:
                raise TypeError(N_LEVELS_ERROR_MSG)
        elif n <= 0:
            raise ValueError(N_LEVELS_ERROR_MSG)
        self._n_levels = n

    @property
    def how(self) -> str:
        return self._how

    @how.setter
    def how(self, h: str) -> None:
        HOW_ERROR_MSG = "Parameter `how` must be one of 'uniform', 'quantile', and 'kmeans'."
        if not isinstance(h, str):
            raise TypeError(HOW_ERROR_MSG)
        if h not in ["uniform", "quantile", "kmeans"]:
            raise ValueError(HOW_ERROR_MSG)
        self._how = h

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
                bins = np.percentile(
                    X_aug[i, :], np.arange(n_levels[i] + 1) / n_levels[i] / 100
                )
                bins_center = np.percentile(
                    X_aug[i, :],
                    np.arange(0.5, n_levels[i]) / n_levels[i] / 100,
                )
                X_aug[i, :] = bins_center[
                    np.digitize(X_aug[i, :], bins).clip(0, n_levels[i] - 1),
                ]
            X_aug = X_aug.reshape(N, C, T).swapaxes(1, 2)
        else:
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError(
                    "To use kmeans quantization, sklearn>=0.22 must be installed."
                )
            n_levels = n_levels.flatten()
            X_aug = X.copy()
            X_aug = X.swapaxes(1, 2).reshape((N * C, T))
            model = KMeans(n_clusters=2, n_jobs=-1, random_state=self.seed)
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
