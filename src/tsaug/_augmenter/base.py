from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np


class _Augmenter(ABC):
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

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, s):
        np.random.RandomState(s)  # try setting up seed
        self._seed = s

    def _augmented_series_length(self, T):
        """
        Return the length (2nd dimension) of augmented series.

        For most augmenters, the length of series will not be changed. If an
        augmenter does change the length of series, this method should be
        overwritten in the augmenter subclass.

        """
        return T

    def augment(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Augment time series.

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
            self._augmented_series_length(T) != T
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
        N = X.shape[0]
        ind = (
            rand.uniform(size=self.repeats * N) <= self.prob
        )  # indice of series to be augmented
        if Y is None:
            if self.repeats > 1:
                X_aug = np.repeat(X.copy(), self.repeats, axis=0)
            else:
                X_aug = X.copy()
            Y_aug = None
            if ind.any():
                X_aug[ind, :], Y_aug = self._augment_core(X_aug[ind, :], None)
        else:
            if self.repeats > 1:
                X_aug = np.repeat(X.copy(), self.repeats, axis=0)
                Y_aug = np.repeat(Y.copy(), self.repeats, axis=0)
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

    def copy(self):
        "Return a copy of this augmenter."
        return deepcopy(self)

    def __mul__(self, m: int):
        """
        Operator *
        """
        copy = self.copy()
        copy.repeats = copy.repeats * m
        return copy

    def __matmul__(self, p: float):
        """
        Operator @
        """
        copy = self.copy()
        copy.prob = copy.prob * p
        return copy

    def __add__(self, another_augmenter):
        """
        Operator +
        """
        if isinstance(another_augmenter, _Augmenter):
            return _AugmenterPipe([self.copy(), another_augmenter.copy()])
        elif isinstance(another_augmenter, _AugmenterPipe):
            return _AugmenterPipe(
                [self.copy()]
                + [augmenter.copy() for augmenter in another_augmenter.pipe]
            )
        else:
            raise TypeError(
                "An augmenter can only be connected by another augmenter or an "
                "augmentor pipeline."
            )

    def __len__(self) -> int:
        return 1


class _AugmenterPipe:
    def __init__(self, pipe):
        self._pipe = pipe

    @property
    def pipe(self):
        return self._pipe

    def augment(self, X, Y=None):
        """
        Augment time series.

        Parameters
        ----------
        X : numpy array
            Time series to be augmented

        """
        X_aug = X
        Y_aug = Y
        for augmenter in self._pipe:
            if Y_aug is None:
                X_aug = augmenter.augment(X_aug)
            else:
                X_aug, Y_aug = augmenter.augment(X_aug, Y_aug)
        if Y_aug is None:
            return X_aug
        else:
            return X_aug, Y_aug

    def __add__(self, another_augmenter):
        """
        Operator +
        """
        if isinstance(another_augmenter, _Augmenter):
            return _AugmenterPipe(
                [augmenter.copy() for augmenter in self.pipe]
                + [another_augmenter.copy()]
            )
        elif isinstance(another_augmenter, _AugmenterPipe):
            return _AugmenterPipe(
                [augmenter.copy() for augmenter in self.pipe]
                + [augmenter.copy() for augmenter in another_augmenter.pipe]
            )
        else:
            raise TypeError(
                "An augmenter can only be connected by another augmenter or an "
                "augmentor pipeline."
            )
