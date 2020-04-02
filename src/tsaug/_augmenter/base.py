from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class _Augmentor(ABC):
    def __init__(
        self, repeat: int = 1, prob: float = 1.0, seed: Optional[int] = None
    ) -> None:
        self.repeat = repeat
        self.prob = prob
        self.seed = seed

    @property
    def repeat(self) -> int:
        return self._repeat

    @repeat.setter
    def repeat(self, M: int) -> None:
        if not isinstance(M, int):
            raise TypeError("Parameter `repeat` must be a positive integer.")
        if M <= 0:
            raise ValueError("Parameter `repeat` must be a positive integer.")
        self._repeat = M

    @property
    def prob(self) -> float:
        return self._prob

    @prob.setter
    def prob(self, p: float) -> None:
        if not isinstance(p, (int, float)):
            raise TypeError(
                "Parameter `prob` must be a float between 0 and 1."
            )
        if p > 1.0 or p < 0.0:
            raise TypeError(
                "Parameter `prob` must be a float between 0 and 1."
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
            and ((self.repeat > 1) or (N > 1))
            and (self.prob != 1.0)
        ):
            raise RuntimeError(
                "This augmenter changes series length. "
                "When augmenting multiple series or multiple times, parameter "
                "`prob` must be 1.0, otherwise the output series may have "
                "different length."
            )

        # augment
        if self.repeat == 1:
            X_aug, Y_aug = self._augment_once(X, Y)
        else:
            X_aug, Y_aug = self._augment_repeat(X, Y)

        # TODO: reshape X_aug, Y_aug

        if Y_aug is None:
            return X_aug
        else:
            return X_aug, Y_aug

    @abstractmethod
    def _augment_once(self, X, Y):
        pass

    def _augment_repeat(self, X, Y):
        """
        By default, if `repeat` > 1, we first concatenate `repeat` copies of
        input into a 'super' input, and then apply _augment_once to it. The
        problem of this strategy is that the memory burden may be unnecessarily
        high for 'long-to-short' augmentation like cropping. In that case, the
        augmentor should overwrite this method.
        """
        rand = np.random.RandomState(self.seed)
        N = len(X)
        ind = (
            rand.uniform(size=self.repeat * N) <= self.prob
        )  # indice of series to be augmented
        if Y is None:
            X_aug = np.vstack([X.copy()] * self.repeat)
            X_aug[ind, :], Y_aug = self._augment_once(X_aug[ind, :], None)
        else:
            X_aug = np.vstack([X.copy()] * self.repeat)
            Y_aug = np.vstack([Y.copy()] * self.repeat)
            X_aug[ind, :], Y_aug[ind, :] = self._augment_once(
                X_aug[ind, :], Y_aug[ind, :]
            )

        return X_aug, Y_aug
