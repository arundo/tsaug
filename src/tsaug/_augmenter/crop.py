import numpy as np

from .base import _Augmentor


class Crop(_Augmentor):
    def __init__(self, size=None, repeat=1, prob=1.0, seed=None):
        self.size = size
        super().__init__(repeat=repeat, prob=prob, seed=seed)

    @staticmethod
    def _change_series_length():
        return True

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, s):
        if (s is not None) and not isinstance(s, int):
            raise TypeError("Parameter `size` must be a positive integer.")
        if (s is not None) and (s <= 0):
            raise ValueError("Parameter `size` must be a positive integer.")
        self._size = s

    def _augment_once(self, X, Y=None):
        N, T, _ = X.shape
        rand = np.random.RandomState(self.seed)
        size = max(
            1, min(T, self.size if (self.size is not None) else int(T / 2))
        )
        crop_start = rand.choice(T - size + 1, size=(N, self.repeat))
        X_aug = X[
            np.hstack([i * np.ones(self.repeat * size) for i in range(N)])
            .reshape(N, self.repeat, size)
            .astype(int),
            np.stack([crop_start + i for i in range(size)], axis=2).astype(
                int
            ),
            :,
        ].reshape((N * self.repeat, size, -1))

        if Y is None:
            Y_aug = None
        else:
            Y_aug = Y[
                np.hstack([i * np.ones(self.repeat * size) for i in range(N)])
                .reshape(N, self.repeat, size)
                .astype(int),
                np.stack([crop_start + i for i in range(size)], axis=2).astype(
                    int
                ),
                :,
            ].reshape((N * self.repeat, size, -1))

        return X_aug, Y_aug

    def _augment_repeat(self, X, Y=None):
        """
        Overwrite the memory-expensive base method.
        Repeat mode is handle by _augment_once
        """
        return self._augment_once(X, Y)
