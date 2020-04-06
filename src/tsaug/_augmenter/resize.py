import numpy as np

from .base import _Augmenter


class Resize(_Augmenter):
    def __init__(self, size, repeats=1, prob=1.0):
        self.size = size
        super().__init__(repeats=repeats, prob=prob)

    @classmethod
    def _get_param_name(cls):
        return ("size",)

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

    def _augmented_series_length(self, T):
        return self.size

    def _augment(self, X, Y):
        """
        Overwrite the memory-expensive base method.
        """
        # No need to handle prob, because it must be 1.0
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

    def _augment_core(self, X, Y):
        "Method _augment is overwritten, therefore this method is not needed."
        pass
