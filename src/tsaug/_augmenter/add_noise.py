import numpy as np

from .base import _Augmentor


class AddNoise(_Augmentor):
    def __init__(
        self,
        loc=0.0,
        scale=0.1,
        distr="gaussian",
        kind="additive",
        per_channel=True,
        normalize=True,
        repeats=1,
        prob=1.0,
        seed=None,
    ):
        self.loc = loc
        self.scale = scale
        self.distr = distr
        self.kind = kind
        self.per_channel = per_channel
        self.normalize = normalize
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, n):
        LOC_ERROR_MSG = (
            "Parameter `loc` must be a number, "
            "a 2-tuple of numbers representing an interval, "
            "or a list of numbers."
        )
        if not isinstance(n, (float, int)):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(LOC_ERROR_MSG)
                if not all([isinstance(nn, int) for nn in n]):
                    raise TypeError(LOC_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(LOC_ERROR_MSG)
                if (not isinstance(n[0], (float, int))) or (
                    not isinstance(n[1], (float, int))
                ):
                    raise TypeError(LOC_ERROR_MSG)
                if n[0] > n[1]:
                    raise ValueError(LOC_ERROR_MSG)
            else:
                raise TypeError(LOC_ERROR_MSG)
        self._loc = n

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, n):
        SCALE_ERROR_MSG = (
            "Parameter `scale` must be a non-negative number, "
            "a 2-tuple of non-negative numbers representing an interval, "
            "or a list of non-negative numbers."
        )
        if not isinstance(n, (float, int)):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(SCALE_ERROR_MSG)
                if not all([isinstance(nn, (float, int)) for nn in n]):
                    raise TypeError(SCALE_ERROR_MSG)
                if not all([nn >= 0 for nn in n]):
                    raise ValueError(SCALE_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(SCALE_ERROR_MSG)
                if (not isinstance(n[0], (float, int))) or (
                    not isinstance(n[1], (float, int))
                ):
                    raise TypeError(SCALE_ERROR_MSG)
                if n[0] > n[1]:
                    raise ValueError(SCALE_ERROR_MSG)
                if (n[0] < 0) or (n[1] < 0):
                    raise ValueError(SCALE_ERROR_MSG)
            else:
                raise TypeError(SCALE_ERROR_MSG)
        elif n < 0:
            raise ValueError(SCALE_ERROR_MSG)
        self._scale = n

    @property
    def distr(self):
        return self._distr

    @distr.setter
    def distr(self, d):
        DISTR_ERROR_MSG = (
            "Parameter `distr` must be one of 'gaussian', 'laplace', and "
            "'uniform'."
        )
        if not isinstance(d, str):
            raise TypeError(DISTR_ERROR_MSG)
        if d not in ("gaussian", "laplace", "uniform"):
            raise ValueError(DISTR_ERROR_MSG)
        self._distr = d

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, k):
        if not isinstance(k, str):
            raise TypeError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        if k not in ("additive", "multiplicative"):
            raise ValueError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        self._kind = k

    @property
    def per_channel(self):
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p):
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, p):
        if not isinstance(p, bool):
            raise TypeError("Paremeter `normalize` must be boolean.")
        self._normalize = p

    def _augment_core(self, X, Y):
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)
        if self.distr == "gaussian":
            gen_noise = lambda size: rand.normal(0.0, 1.0, size=size)
        elif self.distr == "laplace":
            gen_noise = lambda size: rand.laplace(0.0, 1.0, size=size)
        else:
            gen_noise = lambda size: rand.uniform(
                low=-3 ** 0.5, high=3 ** 0.5, size=size
            )

        if isinstance(self.loc, (float, int)):
            loc = np.ones(N) * self.loc
        elif isinstance(self.loc, tuple):
            loc = rand.uniform(low=self.loc[0], high=self.loc[1], size=N)
        else:
            loc = rand.choice(self.loc, size=N)

        if isinstance(self.scale, (float, int)):
            scale = np.ones(N) * self.scale
        elif isinstance(self.scale, tuple):
            scale = rand.uniform(low=self.scale[0], high=self.scale[1], size=N)
        else:
            scale = rand.choice(self.scale, size=N)

        if self.per_channel:
            noise = gen_noise((N, T, C))
        else:
            noise = gen_noise((N, T, 1))
            noise = np.repeat(noise, C, axis=2)

        noise = noise * scale.reshape((N, 1, 1)) + loc.reshape((N, 1, 1))

        if self.kind == "additive":
            if self.normalize:
                X_aug = X + noise * (
                    X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)
                )
            else:
                X_aug = X + noise
        else:
            X_aug = X * noise

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug
