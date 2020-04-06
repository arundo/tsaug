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

    @classmethod
    @abstractmethod
    def _get_param_name(cls):
        return tuple()

    def _get_params(self):
        return {
            param_name: getattr(self, param_name)
            for param_name in self._get_param_name()
        }

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
            Time series to be augmented (T,), (N,T), or (N,T,C)

        """
        X_ERROR_MSG = (
            "Input X must be a numpy array with shape (T,), (N, T), or (N, T, "
            "C), where T is the length of a series, N is the number of series, "
            "and C is the number of a channels in a series."
        )

        Y_ERROR_MSG = (
            "Input Y must be a numpy array with shape (T,), (N, T), or (N, T, "
            "L), where T is the length of a series, N is the number of series, "
            "and L is the number of a segmentation classes."
        )

        if not isinstance(X, np.ndarray):
            raise TypeError(X_ERROR_MSG)
        ndim_x = X.ndim
        if ndim_x == 1:  # (T, )
            X = X.reshape(1, -1, 1)
        elif ndim_x == 2:  # (N, T)
            X = np.expand_dims(X, 2)
        elif ndim_x == 3:  # (N, T, C)
            pass
        else:
            raise ValueError(X_ERROR_MSG)

        if Y is not None:
            if not isinstance(Y, np.ndarray):
                raise TypeError(Y_ERROR_MSG)
            ndim_y = Y.ndim
            if ndim_y == 1:  # (T, )
                Y = Y.reshape(1, -1, 1)
            elif ndim_y == 2:  # (N, T)
                Y = np.expand_dims(Y, 2)
            elif ndim_y == 3:  # (N, T, L)
                pass
            else:
                raise ValueError(Y_ERROR_MSG)

        N, T, _ = X.shape

        if Y is not None:
            Ny, Ty, _ = Y.shape
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

        if ndim_x == 1:
            if self.repeats == 1:
                X_aug = X_aug.reshape(T)
            else:
                X_aug = X_aug.reshape(self.repeats, T)
        elif ndim_x == 2:
            X_aug = X_aug.reshape(N * self.repeats, T)

        if Y is not None:
            if ndim_y == 1:
                if self.repeats == 1:
                    Y_aug = Y_aug.reshape(T)
                else:
                    Y_aug = Y_aug.reshape(self.repeats, T)
            elif ndim_y == 2:
                Y_aug = Y_aug.reshape(N * self.repeats, T)

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

    def _copy(self):
        "Return a copy of this augmenter."
        return deepcopy(self)

    def __mul__(self, m: int):
        """
        Operator *
        """
        copy = self._copy()
        copy.repeats = copy.repeats * m
        return copy

    def __matmul__(self, p: float):
        """
        Operator @
        """
        copy = self._copy()
        copy.prob = copy.prob * p
        return copy

    def __add__(self, another_augmenter):
        """
        Operator +
        """
        if isinstance(another_augmenter, _Augmenter):
            return _AugmenterPipe([self._copy(), another_augmenter._copy()])
        elif isinstance(another_augmenter, _AugmenterPipe):
            return _AugmenterPipe(
                [self._copy()]
                + [augmenter._copy() for augmenter in another_augmenter]
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

    def __getitem__(self, ind):
        if isinstance(self._pipe.__getitem__(ind), _Augmenter):
            return self._pipe.__getitem__(ind)
        else:
            return _AugmenterPipe(self._pipe.__getitem__(ind))

    def __setitem__(self, ind, value):
        if isinstance(self._pipe.__getitem__(ind), _Augmenter) and isinstance(
            value, _Augmenter
        ):
            self._pipe.__setitem__(ind, value)
        else:
            raise NotImplementedError(
                "Setting multiple augmenters in an augmenter pipe is not "
                "supported yet."
            )

    def __iter__(self):
        return self._pipe.__iter__()

    def summary(self, show_params=False):
        """
        Print summary of this augmenter pipe.

        Parameters
        ----------
        show_params : bool, optional
            Whether show parameters of each augmenter in the summary table. If
            True, the table may be too wide to be readable. Default: False.

        """

        print(
            "{ind}\t{name}\t{repeats}\t{prob}\t{param}".format(
                ind="idx",
                name="augmenter",
                repeats="repeats".rjust(8),
                prob="prob".rjust(5),
                param="params" if show_params else "",
            )
        )
        print("=" * (120 if show_params else 45))
        for i, augmenter in enumerate(self):
            print(
                "{ind:3.0g}\t{name}\t{repeats:8.3g}\t{prob:5.3g}\t{param}".format(
                    ind=i,
                    name=augmenter.__class__.__name__.ljust(8),
                    repeats=augmenter.repeats,
                    prob=augmenter.prob,
                    param=augmenter._get_params() if show_params else "",
                )
            )

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
                [augmenter._copy() for augmenter in self]
                + [another_augmenter._copy()]
            )
        elif isinstance(another_augmenter, _AugmenterPipe):
            return _AugmenterPipe(
                [augmenter._copy() for augmenter in self]
                + [augmenter._copy() for augmenter in another_augmenter]
            )
        else:
            raise TypeError(
                "An augmenter can only be connected by another augmenter or an "
                "augmentor pipeline."
            )
