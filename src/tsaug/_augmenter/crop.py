import numpy as np

from .base import _Augmentor


class Crop(_Augmentor):
    def __init__(self, size=None, repeats=1, prob=1.0, seed=None):
        self.size = size
        super().__init__(repeats=repeats, prob=prob, seed=seed)

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

    def _augment(self, X, Y):
        """
        Overwrite the memory-expensive base method.
        """
        # No need to handle prob, because it must be 1.0
        N, T, _ = X.shape
        rand = np.random.RandomState(self.seed)
        size = max(
            1, min(T, self.size if (self.size is not None) else int(T / 2))
        )
        crop_start = rand.choice(T - size + 1, size=(N, self.repeats))
        X_aug = X[
            np.hstack([i * np.ones(self.repeats * size) for i in range(N)])
            .reshape(N, self.repeats, size)
            .astype(int),
            np.stack([crop_start + i for i in range(size)], axis=2).astype(
                int
            ),
            :,
        ].reshape((N * self.repeats, size, -1))

        if Y is None:
            Y_aug = None
        else:
            Y_aug = Y[
                np.hstack([i * np.ones(self.repeats * size) for i in range(N)])
                .reshape(N, self.repeats, size)
                .astype(int),
                np.stack([crop_start + i for i in range(size)], axis=2).astype(
                    int
                ),
                :,
            ].reshape((N * self.repeats, size, -1))

        return X_aug, Y_aug

    def _augment_core(self, X, Y):
        "Method _augment is overwritten, therefore this method is not needed."
        pass
