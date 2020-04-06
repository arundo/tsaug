from .base import _Augmenter


class Reverse(_Augmenter):
    def __init__(self, repeats=1, prob=1.0, seed=None):
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls):
        return tuple()

    def _augment_core(self, X, Y):
        X_aug = X[:, ::-1, :].copy()  # type: np.ndarray

        if Y is None:
            Y_aug = None  # type: Optional[np.ndarray]
        else:
            Y_aug = Y[:, ::-1, :].copy()

        return X_aug, Y_aug
