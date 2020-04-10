from typing import Optional, Tuple

import numpy as np

from .base import _Augmenter, _default_seed


class Resize(_Augmenter):
    """
    Change the temporal resolution of time series.

    The resized time series is obtained by linear interpolation of the original
    time series.

    Parameters
    ----------
    size : int
        Length of the output series.

    repeats : int, optional
        The number of times a series is augmented. If greater than one, a series
        will be augmented so many times independently. This parameter can also
        be set by operator `*`. Default: 1.

    prob : float, optional
        The probability of a series is augmented. It must be in (0.0, 1.0]. If
        multiple output is expected, this value must be 1.0, so that all output
        have the same length. This parameter can also be set by operator `@`.
        Default: 1.0.

    seed : int, optional
        The random seed. Default: None.

    """

    def __init__(
        self,
        size: int,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = _default_seed,
    ):
        self.size = size
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("size",)

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, s: int) -> None:
        if not isinstance(s, int):
            raise TypeError("Parameter `size` must be a positive integer.")
        if s <= 0:
            raise ValueError("Parameter `size` must be a positive integer.")
        self._size = s

    def _augmented_series_length(self, T: int) -> int:
        return self.size

    def _augment(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overwrite the memory-expensive base method.
        """
        rand = np.random.RandomState(self.seed)
        if self.prob != 1.0:
            # it implies N == 1 and self.repeats == 1
            if rand.uniform() > self.prob:
                if Y is None:
                    return X.copy(), None
                else:
                    return X.copy(), Y.copy()

        T = X.shape[1]

        if self.size == T:
            X_aug = X.copy()
            if Y is None:
                Y_aug = None
            else:
                Y_aug = Y.copy()
            return X_aug, Y_aug

        if self.size == 1:
            X_aug = (X[:, :1, :] + X[:, -1:, :]) / 2
            if Y is None:
                Y_aug = None
            else:
                Y_aug = (Y[:, :1, :] + Y[:, -1:, :]) / 2
            return X_aug, Y_aug

        ind = np.arange(self.size - 1) / (self.size - 1) * (T - 1)
        ind_0 = ind.astype(int)
        ind_1 = ind_0 + 1
        weight_1 = ind - ind_0
        weight_0 = 1.0 - weight_1

        X_aug = X[:, ind_0, :] * weight_0.reshape(1, self.size - 1, 1) + X[
            :, ind_1, :
        ] * weight_1.reshape(1, self.size - 1, 1)
        X_aug = np.concatenate([X_aug, X[:, -1:, :]], axis=1)
        if self.repeats > 1:
            X_aug = np.repeat(X_aug, self.repeats, axis=0)

        if Y is None:
            Y_aug = None
        else:
            Y_aug = Y[:, ind_0, :] * weight_0.reshape(1, self.size - 1, 1) + Y[
                :, ind_1, :
            ] * weight_1.reshape(1, self.size - 1, 1)
            Y_aug = np.concatenate([Y_aug, Y[:, -1:, :]], axis=1)
            Y_aug = Y_aug.round().astype(int)
            if self.repeats > 1:
                Y_aug = np.repeat(Y_aug, self.repeats, axis=0)

        return X_aug, Y_aug

    def _augment_core(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        "Method _augment is overwritten, therefore this method is not needed."
        pass
