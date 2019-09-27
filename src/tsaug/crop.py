"""
Crop time series
"""

import numpy as np
from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def crop(X, Y=None, crop_start=0, crop_size=None):
    """Crop time series.

    Time series will be cropped based on given location and size of cropping
    window.

    Parameters
    ----------
    X : numpy.ndarray
        Time series to be augmented. Matrix with shape (n,), (N, n) or (N, n,
        c), where n is the length of each series, N is the number of series,
        and c is the number of channels.

    Y : numpy.ndarray, optional
        Binary labels of time series, where 0 represents a normal point and 1
        represents an anomalous points. Matrix with shape (n,), (N, n) or (N,
        n, cl), where n is the length of each series, N is the number of
        series, and cl is the number of classes (i.e. types of anomaly).
        Default: None.

    crop_start : int, or numpy.ndarray, optional
        Indices of the start of cropping windows at each time series. If an
        array of shape (N, m) where N is the number of series, m croppings will
        be performed at each series. Default: 0.

    crop_size : int, optional
        Size of cropping windows. If not given, n will be used. Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """
    N, n, c = X.shape
    if Y is not None:
        cl = Y.shape[2]

    if isinstance(crop_start, int):
        crop_start = (crop_start * np.ones(N)).astype(int)

    crop_start = np.array(crop_start).astype(int)

    if len(crop_start) != N:
        raise ValueError("Wrong shape of `crop_start`")

    crop_start = crop_start.reshape(N, -1)
    crops_per_series = crop_start.shape[1]

    if crop_size is None:
        crop_size = n
    crop_size = int(crop_size)
    if (crop_size > n) | (crop_size <= 0):
        raise ValueError("`crop_size` must between 1 and n")

    if (crop_start + crop_size > n).any():
        raise ValueError("Inconsistent value of `crop_start` and `crop_size`")

    X_aug = X[
        np.hstack(
            [i * np.ones(crops_per_series * crop_size) for i in range(N)]
        )
        .reshape(N, crops_per_series, crop_size)
        .astype(int),
        np.stack([crop_start + i for i in range(crop_size)], axis=2).astype(
            int
        ),
        :,
    ].reshape((N * crops_per_series, crop_size, c))

    if Y is None:
        Y_aug = None
    else:
        Y_aug = Y[
            np.hstack(
                [i * np.ones(crops_per_series * crop_size) for i in range(N)]
            )
            .reshape(N, crops_per_series, crop_size)
            .astype(int),
            np.stack(
                [crop_start + i for i in range(crop_size)], axis=2
            ).astype(int),
            :,
        ].reshape((N * crops_per_series, crop_size, cl))

    return X_aug, Y_aug


@dimensionalize
def random_crop(
    X, Y=None, crop_size=None, crops_per_series=1, random_seed=None
):
    """Crop time series randomly.

    Parameters
    ----------
    X : numpy.ndarray
        Time series to be augmented. Matrix with shape (n,), (N, n) or (N, n,
        c), where n is the length of each series, N is the number of series,
        and c is the number of channels.

    Y : numpy.ndarray, optional
        Binary labels of time series, where 0 represents a normal point and 1
        represents an anomalous points. Matrix with shape (n,), (N, n) or (N,
        n, cl), where n is the length of each series, N is the number of
        series, and cl is the number of classes (i.e. types of anomaly).
        Default: None.

    crop_size : int, optional
        Size of cropping windows. If not given, n will be used. Default: None.

    crops_per_series : int, optional
        Number of croppings to be performed at each series. Default: 1.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    N, n, c = X.shape
    rand = np.random.RandomState(random_seed)
    if crop_size is None:
        crop_size = n
    crop_size = int(crop_size)
    if (crop_size > n) | (crop_size <= 0):
        raise ValueError("`crop_size` must between 1 and n")

    crop_start = rand.choice(n - crop_size + 1, size=(N, crops_per_series))

    return crop(X, Y, crop_start=crop_start, crop_size=crop_size)


class Crop(_Augmentor):
    """Augmentor that crops time series.

    Time series will be cropped based on given location and size of cropping
    window.

    Parameters
    ----------
    crop_start : int, or numpy.ndarray, optional
        Indices of the start of cropping windows at each time series. If an
        array of shape (N, m) where N is the number of series, m croppings will
        be performed at each series. Default: 0.

    crop_size : int, optional
        Size of cropping windows. If not given, n will be used. Default: None.

    """

    def __init__(self, crop_start=0, crop_size=None):
        super().__init__(
            augmentor_func=crop,
            is_random=False,
            crop_start=crop_start,
            crop_size=crop_size,
        )

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, prob):
        if (prob != 1.0) & (prob != 0.0):
            raise ValueError(
                "Crop augmentor may change the length of time series, "
                + "therefore probability must be either 0 or 1"
            )
        self._prob = prob

    @property
    def crop_start(self):
        return self._params["crop_start"]

    @crop_start.setter
    def crop_start(self, crop_start):
        self._params["crop_start"] = crop_start

    @property
    def crop_size(self):
        return self._params["crop_size"]

    @crop_size.setter
    def crop_size(self, crop_size):
        self._params["crop_size"] = crop_size

    def _get_output_dim(
        self, input_N=(1, None), input_n=(1, None), input_c=(1, None)
    ):
        if isinstance(self.crop_start, np.ndarray) and (
            self.crop_start.ndim == 2
        ):
            crops_per_series = self.crop_start.shape[1]
        else:
            crops_per_series = 1
        output_N = (
            (input_N[0] * self.M * crops_per_series, None)
            if (input_N[0] is not None)
            else (None, input_N[1] * self.M * crops_per_series)
        )
        if self.crop_size is None:
            output_n = input_n
        else:
            output_n = (None, self.crop_size)
        return output_N, output_n, input_c


class RandomCrop(_Augmentor):
    """Augmentor that crops time series randomly.

    Parameters
    ----------
    crop_size : int, optional
        Size of cropping windows. If not given, n will be used. Default: None.

    crops_per_series : int, optional
        Number of croppings to be performed at each series. Default: 1.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    """

    def __init__(self, crop_size=None, crops_per_series=1, random_seed=None):
        super().__init__(
            augmentor_func=random_crop,
            is_random=True,
            crop_size=crop_size,
            crops_per_series=crops_per_series,
            random_seed=random_seed,
        )

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, prob):
        if (prob != 1.0) & (prob != 0.0):
            raise ValueError(
                "RandomCrop augmentor may change the length of time series, "
                + "therefore probability must be either 0 or 1"
            )
        self._prob = prob

    @property
    def crop_size(self):
        return self._params["crop_size"]

    @crop_size.setter
    def crop_size(self, crop_size):
        self._params["crop_size"] = crop_size

    @property
    def crops_per_series(self):
        return self._params["crops_per_series"]

    @crops_per_series.setter
    def crops_per_series(self, crops_per_series):
        self._params["crops_per_series"] = crops_per_series

    @property
    def random_seed(self):
        return self._params["random_seed"]

    @random_seed.setter
    def random_seed(self, random_seed):
        self._params["random_seed"] = random_seed

    def _get_output_dim(
        self, input_N=(1, None), input_n=(1, None), input_c=(1, None)
    ):
        output_N = (
            (input_N[0] * self.M * self.crops_per_series, None)
            if (input_N[0] is not None)
            else (None, input_N[1] * self.M * self.crops_per_series)
        )
        if self.crop_size is None:
            output_n = input_n
        else:
            output_n = (None, self.crop_size)
        return output_N, output_n, input_c
