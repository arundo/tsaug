from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class _Augmentor(ABC):
    def __init__(
        self, repeats: int = 1, prob: float = 1.0, seed: Optional[int] = None
    ) -> None:
        self.repeats = repeats
        self.prob = prob
        self.seed = seed

    @property
    def repeats(self) -> int:
        return self._repeat

    @repeats.setter
    def repeats(self, M: int) -> None:
        if not isinstance(M, int):
            raise TypeError("Parameter `repeats` must be a positive integer.")
        if M <= 0:
            raise ValueError("Parameter `repeats` must be a positive integer.")
        self._repeat = M

    @property
    def prob(self) -> float:
        return self._prob

    @prob.setter
    def prob(self, p: float) -> None:
        if not isinstance(p, (int, float)):
            raise TypeError(
                "Parameter `prob` must be a positive number between 0 and 1."
            )
        if p > 1.0 or p <= 0.0:
            raise TypeError(
                "Parameter `prob` must be a positive number between 0 and 1."
            )
        self._prob = p

    @staticmethod
    @abstractmethod
    def _change_series_length() -> bool:
        return False

    def augment(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Augment time series

        Parameters
        ----------
        X : numpy array
            Time series to be augmented

        """

        # TODO: reshape X to (N, T, C)
        # TODO: reshape Y to (N, T, L)
        N, T, C = X.shape

        if Y is not None:
            Ny, Ty, L = Y.shape
            # check consistency between X and Y
            if N != Ny:
                raise ValueError(
                    "The numbers of series in X and Y are different."
                )
            if T != Ty:
                raise ValueError(
                    "The length of series in X and Y are different."
                )

        # if this augmenter changes series length AND multiple outputs are
        # expected, it must has prob equal to 1, otherwise the outputs may have
        # different length
        if (
            self._change_series_length()
            and ((self.repeats > 1) or (N > 1))
            and (self.prob != 1.0)
        ):
            raise RuntimeError(
                "This augmenter changes series length. "
                "When augmenting multiple series or multiple times, parameter "
                "`prob` must be 1.0, otherwise the output series may have "
                "different length."
            )

        # augment
        X_aug, Y_aug = self._augment(X, Y)

        # TODO: reshape X_aug, Y_aug

        if Y_aug is None:
            return X_aug
        else:
            return X_aug, Y_aug

    def _augment(self, X, Y):
        """
        The main part of augmentation, without pre- and post-processing.

        This method calls _augment_core which is the core algorithmic part. The
        process in this method handles `repeats` and `prob`.

        1. If `repeats` > 1, we first concatenate `repeats` copies of input
           into a 'super' input..
        2. Select series from the (super) input to be augmented.
        3. Apply _augment_core to the selected series.

        The problem of this strategy includes:

        1. The memory burden may be unnecessarily high for 'long-to-short'
           augmentation like cropping (say if cropping a 100-window from a
           100M-series, this strategy copies 100M-series `repeats` times).
        2. Some time-consuming computation may be duplicated, for example
           quantization with kmeans. Each series should only train a model once
           instead of `repeats` times.

        In those cases, the subclass of the augmentor should overwrite this
        method.

        """
        rand = np.random.RandomState(self.seed)
        N, T, C = X.shape
        ind = (
            rand.uniform(size=self.repeats * N) <= self.prob
        )  # indice of series to be augmented
        if Y is None:
            if self.repeats > 1:
                X_aug = (
                    np.stack([X.copy() for _ in range(self.repeats)], axis=0)
                    .swapaxes(0, 1)
                    .reshape((N * self.repeats, T, C))
                )
            else:
                X_aug = X.copy()
            Y_aug = None
            if ind.any():
                X_aug[ind, :], Y_aug = self._augment_core(X_aug[ind, :], None)
        else:
            L = Y.shape[2]
            if self.repeats > 1:
                X_aug = (
                    np.stack([X.copy() for _ in range(self.repeats)], axis=0)
                    .swapaxes(0, 1)
                    .reshape((N * self.repeats, T, C))
                )
                Y_aug = (
                    np.stack([Y.copy() for _ in range(self.repeats)], axis=0)
                    .swapaxes(0, 1)
                    .reshape((N * self.repeats, T, L))
                )
            else:
                X_aug = X.copy()
                Y_aug = Y.copy()
            if ind.any():
                X_aug[ind, :], Y_aug[ind, :] = self._augment_core(
                    X_aug[ind, :], Y_aug[ind, :]
                )
        return X_aug, Y_aug

    @abstractmethod
    def _augment_core(self, X, Y):
        """
        The core of augmentation.
        """
        pass
