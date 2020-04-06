import numpy as np

from .base import _Augmenter
from .resize import Resize


class Crop(_Augmenter):
    def __init__(self, size, resize=None, repeats=1, prob=1.0, seed=None):
        self.size = size
        self.resize = resize
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls):
        return ("size", "resize")

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
    def resize(self):
        return self._resize

    @resize.setter
    def resize(self, s):
        if (s is not None) and (not isinstance(s, int)):
            raise TypeError("Parameter `resize` must be a positive integer.")
        if (s is not None) and (s <= 0):
            raise ValueError("Parameter `resize` must be a positive integer.")
        self._resize = s

    def _augmented_series_length(self, T):
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

    def _augment(self, X, Y):
        """
        Overwrite the memory-expensive base method.
        """
        # No need to handle prob, because it must be 1.0
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)

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
            if Y is not None:
                Y_aug[crop_size == s, :, :] = resizer.augment(
                    Y[
                        np.repeat(
                            np.repeat(np.arange(N), self.repeats)[
                                crop_size == s
                            ],
                            s,
                        ).astype(int),
                        (
                            crop_start.reshape(n, 1)
                            + np.arange(s).reshape(1, s)
                        ).astype(int),
                        :,
                    ].reshape((n, s, C))
                )

        return X_aug, Y_aug

    def _augment_core(self, X, Y):
        "Method _augment is overwritten, therefore this method is not needed."
        pass
