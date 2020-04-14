from typing import List, Optional, Tuple, Union

import numpy as np

from .base import _Augmenter, _default_seed
from .resize import Resize


class Crop(_Augmenter):
    """
    Crop random sub-sequences from time series.

    To guarantee all output series have the same length, if the crop size is not
    deterministic, all crops must be resize to a fixed length.

    Parameters
    ----------
    size : int, tuple, list
        The length of random crops.

        - If int, all crops have the same length.
        - If list, a crop from a series has a length sampled from this list
          randomly.
        - If 2-tuple, a crop from a series has a length sampled from this
          interval randomly.

    resize : int, optional
        The length that all crops are resized to. Only necessary if the crop
        size is not fixed.

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
        size: Union[int, Tuple[int, int], List[int]],
        resize: Optional[int] = None,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = _default_seed,
    ):
        self.size = size
        self.resize = resize
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("size", "resize")

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
    def resize(self) -> Optional[int]:
        return self._resize

    @resize.setter
    def resize(self, s: Optional[int]) -> None:
        if (s is not None) and (not isinstance(s, int)):
            raise TypeError("Parameter `resize` must be a positive integer.")
        if (s is not None) and (s <= 0):
            raise ValueError("Parameter `resize` must be a positive integer.")
        self._resize = s

    def _augmented_series_length(self, T: int) -> int:
        if isinstance(self.size, int):
            size = [self.size]
        elif isinstance(self.size, tuple):
            size = list(range(self.size[0], self.size[1]))
        else:
            size = self.size

        if self.resize is not None:
            resize = self.resize
        else:
            if len(size) > 1:
                raise ValueError(
                    "Parameter `resize` must be specified if parameter `size` "
                    "is not a single value."
                )
            else:
                resize = size[0]

        return resize

    def _augment(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overwrite the memory-expensive base method.
        """
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)

        if self.prob != 1.0:
            # it implies N == 1 and self.repeats == 1
            if rand.uniform() > self.prob:
                if Y is None:
                    return X.copy(), None
                else:
                    return X.copy(), Y.copy()

        if isinstance(self.size, int):
            size = [self.size]
        elif isinstance(self.size, tuple):
            size = list(range(self.size[0], self.size[1]))
        else:
            size = self.size

        if self.resize is not None:
            resize = self.resize
        else:
            if len(size) > 1:
                raise ValueError(
                    "Parameter `resize` must be specified if parameter `size` "
                    "is not a single value."
                )
            else:
                resize = size[0]

        X_aug = np.zeros((N * self.repeats, resize, C))
        if Y is None:
            Y_aug = None
        else:
            L = Y.shape[2]
            Y_aug = np.zeros((N * self.repeats, resize, L))

        crop_size = rand.choice(size, size=N * self.repeats)
        resizer = Resize(resize)
        for s in np.unique(crop_size):
            n = (crop_size == s).sum()
            crop_start = rand.choice(T - s + 1, size=n)
            X_aug[crop_size == s, :, :] = resizer.augment(
                X[
                    np.repeat(
                        np.repeat(np.arange(N), self.repeats)[crop_size == s],
                        s,
                    )
                    .reshape(n, s)
                    .astype(int),
                    (
                        crop_start.reshape(n, 1) + np.arange(s).reshape(1, s)
                    ).astype(int),
                    :,
                ].reshape((n, s, C))
            )
            if (Y is not None) and (Y_aug is not None):
                Y_aug[crop_size == s, :, :] = resizer.augment(
                    Y[
                        np.repeat(
                            np.repeat(np.arange(N), self.repeats)[
                                crop_size == s
                            ],
                            s,
                        )
                        .reshape(n, s)
                        .astype(int),
                        (
                            crop_start.reshape(n, 1)
                            + np.arange(s).reshape(1, s)
                        ).astype(int),
                        :,
                    ].reshape((n, s, L))
                )

        return X_aug, Y_aug

    def _augment_core(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        "Method _augment is overwritten, therefore this method is not needed."
        pass
