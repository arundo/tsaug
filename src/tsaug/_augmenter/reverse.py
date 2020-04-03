from .base import _Augmentor


class Reverse(_Augmentor):
    def __init__(self, repeats=1, prob=1.0):
        super().__init__(repeats=repeats, prob=prob)

    @staticmethod
    def _change_series_length():
        return False

    def _augment_once(self, X, Y):
        X_aug = X[:, ::-1, :].copy()  # type: np.ndarray

        if Y is None:
            Y_aug = None  # type: Optional[np.ndarray]
        else:
            Y_aug = Y[:, ::-1, :].copy()

        return X_aug, Y_aug
