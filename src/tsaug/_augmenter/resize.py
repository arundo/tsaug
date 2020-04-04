import numpy as np

from .base import _Augmentor


class Resize(_Augmentor):
    def __init__(self, size, repeats=1, prob=1.0):
        self.size = size
        super().__init__(repeats=repeats, prob=prob)

    @staticmethod
    def _change_series_length():
        return True

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, s):
        if not isinstance(s, int):
            raise TypeError("Parameter `size` must be a positive integer.")
        if s <= 0:
            raise ValueError("Parameter `size` must be a positive integer.")
        self._size = s

    def _augment(self, X, Y):
        """
        Overwrite the memory-expensive base method.
        """
        # No need to handle prob, because it must be 1.0
        N, T, C = X.shape
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
            X_aug = (
                np.stack([X_aug.copy() for _ in range(self.repeats)], axis=0)
                .swapaxes(0, 1)
                .reshape((N * self.repeats, T, C))
            )

        if Y is None:
            Y_aug = None
        else:
            L = Y.shape[2]
            Y_aug = Y[:, ind_0, :] * weight_0.reshape(1, self.size - 1, 1) + Y[
                :, ind_1, :
            ] * weight_1.reshape(1, self.size - 1, 1)
            Y_aug = np.concatenate([Y_aug, Y[:, -1:, :]], axis=1)
            Y_aug = Y_aug.round().astype(int)
            if self.repeats > 1:
                Y_aug = (
                    np.stack(
                        [Y_aug.copy() for _ in range(self.repeats)], axis=0
                    )
                    .swapaxes(0, 1)
                    .reshape((N * self.repeats, T, L))
                )

        return X_aug, Y_aug

    def _augment_core(self, X, Y):
        "Method _augment is overwritten, therefore this method is not needed."
        pass
