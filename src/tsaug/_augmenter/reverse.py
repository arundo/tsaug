from .base import _Augmenter, _default_seed


class Reverse(_Augmenter):
    """
    Reverse the time line of series.

    Parameters
    ----------
    repeats : int, optional
        The number of times a series is augmented. If greater than one, a series
        will be augmented so many times independently. This parameter can also
        be set by operator `*`. Default: 1.

    prob : float, optional
        The probability of a series is augmented. It must be in (0.0, 1.0]. This
        parameter can also be set by operator `@`. Default: 1.0.

    seed : int, optional
        The random seed. Default: None.

    """

    def __init__(self, repeats=1, prob=1.0, seed=_default_seed):
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
