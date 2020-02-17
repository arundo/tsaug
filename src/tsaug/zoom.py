"""
magnify
"""
from typing import Tuple, Optional, Union, Mapping

from collections import Counter
import numpy as np
from .dimensionalize import dimensionalize
from .crop import crop
from .resample import resample
from .augmentor import _Augmentor


@dimensionalize
def magnify(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    start: Union[int, np.ndarray] = 0,
    end: Optional[Union[int, np.ndarray]] = None,
    size: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Magnify time intervels of time series.

    This transformation does not change the number of time points in a series.

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

    start : int or numpy.ndarray, optional
        Indices of the starting positions of time windows to be magnified. If
        an integer, all series use the same value. Default: 0.

    end : int or numpy.ndarray, optional
        Indices of the ending positions of time windows to be magnified. If
        an integer, all series use the same value. Only used when argument
        `size` is not given. Default: n.

    size : int, optional
        Length of time windows to be magnified. If given, argument `end` will
        be ignored. Default: None

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    N = (
        0
    )  # type: int  # NOTE: this is a horrible hack to type hint for Python 3.5
    n = 0  # type: int
    c = 0  # type: int
    N, n, c = X.shape
    if Y is not None:
        cl = Y.shape[2]  # type: int

    if size is not None:
        if Y is None:
            X_crop = crop(
                X, crop_start=start, crop_size=size
            )  # type: np.ndarray
            return resample(X_crop, n_new=n)
        else:
            Y_crop = (
                []
            )  # type: np.ndarray  # See earlier variable attribution hack
            X_crop, Y_crop = crop(X, Y, crop_start=start, crop_size=size)
            return resample(X_crop, Y_crop, n_new=n)
    else:
        if end is None:
            end = n
        sizes = end - start  # type: Union[int, np.ndarray]
        if isinstance(sizes, int):
            return magnify(X, Y, start=start, size=end - start)
        else:
            counter = Counter(sizes)  # type: Mapping[int, int]
            X_zoom = np.zeros((N, n, c))  # type: np.ndarray
            if Y is None:
                Y_zoom = None  # type: Optional[np.ndarray]
            else:
                Y_zoom = np.zeros((N, n, cl))
            # if the windows to be magnified have different sizes across the input, we run over each size
            for size, count in counter.items():
                if (Y is None) or (Y_zoom is None):
                    X_zoom[sizes == size, :, :] = magnify(
                        X[sizes == size, :, :],
                        start=np.array([start] * count)
                        if isinstance(start, int)
                        else start[sizes == size],
                        size=size,
                    )
                else:
                    (
                        X_zoom[sizes == size, :, :],
                        Y_zoom[sizes == size, :, :],
                    ) = magnify(
                        X[sizes == size, :, :],
                        Y[sizes == size, :, :],
                        start=np.array([start] * count)
                        if isinstance(start, int)
                        else start[sizes == size],
                        size=size,
                    )
        return X_zoom, Y_zoom


@dimensionalize
def random_magnify(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    max_zoom: float = 2.0,
    min_zoom: float = 1.0,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Magnify random time intervels of time series.

    This transformation does not change the number of time points in a series.

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

    max_zoom : float, optional
        Maximal zooming factor. Default: 2.0.

    min_zoom : float, optional
        Minimal zooming factor. Default: 1.0.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    N = (
        0
    )  # type: int  # NOTE: this is a horrible hack to type hint for Python 3.5
    n = 0  # type: int
    c = 0  # type: int
    N, n, c = X.shape
    rand = np.random.RandomState(random_seed)

    if (max_zoom < 1.0) or (min_zoom < 1.0):
        raise ValueError("Zooming factor must be at least 1.0.")

    if max_zoom < min_zoom:
        raise ValueError("`max_zoom` must be greater or equal to `min_zoom`")

    max_size = max(int(round(n / min_zoom)), 1)  # type: int
    min_size = max(int(round(n / max_zoom)), 1)  # type: int

    if max_size == min_size:
        size = max_size  # type: int
        start = rand.choice(
            n - size + 1, size=N
        )  # type: Union[int, np.ndarray]
        return magnify(X, Y, start=start, size=size)
    else:
        sizes = rand.choice(
            range(min_size, max_size + 1), size=N
        )  # type: np.ndarray
        counter = Counter(sizes)  # type: Mapping[int, int]
        start = np.zeros(N)
        end = np.zeros(N)  # type: np.ndarray
        for size in range(min_size, max_size + 1):
            start[sizes == size] = rand.choice(
                n - size + 1, size=counter[size]
            )
            end[sizes == size] = start[sizes == size] + size
        return magnify(X, Y, start=start, end=end)


class Magnify(_Augmentor):
    """Augmentor that magnifies time intervels of time series.

    This transformation does not change the number of time points in a series.

    Parameters
    ----------
    start : int or numpy.ndarray, optional
        Indices of the starting positions of time windows to be magnified. If
        an integer, all series use the same value. Default: 0.

    end : int or numpy.ndarray, optional
        Indices of the ending positions of time windows to be magnified. If
        an integer, all series use the same value. Only used when argument
        `size` is not given. Default: n.

    size : int, optional
        Length of time windows to be magnified. If given, argument `end` will
        be ignored. Default: None

    """

    def __init__(
        self,
        start: Union[int, np.ndarray] = 0,
        end: Optional[Union[int, np.ndarray]] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(
            augmentor_func=magnify,
            is_random=False,
            start=start,
            end=end,
            size=size,
        )

    @property
    def start(self) -> Union[int, np.ndarray]:
        return self._params["start"]

    @start.setter
    def start(self, start: Union[int, np.ndarray]) -> None:
        self._params["start"] = start

    @property
    def end(self) -> Optional[Union[int, np.ndarray]]:
        return self._params["end"]

    @end.setter
    def end(self, end: Optional[Union[int, np.ndarray]]) -> None:
        self._params["end"] = end

    @property
    def size(self) -> Optional[int]:
        return self._params["size"]

    @size.setter
    def size(self, size: Optional[int]) -> None:
        self._params["size"] = size


class RandomMagnify(_Augmentor):
    """Augmentor that magnifies random time intervels of time series.

    This transformation does not change the number of time points in a series.

    Parameters
    ----------
    max_zoom : float, optional
        Maximal zooming factor. Default: 2.0.

    min_zoom : float, optional
        Minimal zooming factor. Default: 1.0.

    random_seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
        Default: None.

    """

    def __init__(
        self,
        max_zoom: float = 2.0,
        min_zoom: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            augmentor_func=random_magnify,
            is_random=True,
            max_zoom=max_zoom,
            min_zoom=min_zoom,
            random_seed=random_seed,
        )

    @property
    def max_zoom(self) -> float:
        return self._params["max_zoom"]

    @max_zoom.setter
    def max_zoom(self, max_zoom: float) -> None:
        self._params["max_zoom"] = max_zoom

    @property
    def min_zoom(self) -> float:
        return self._params["min_zoom"]

    @min_zoom.setter
    def min_zoom(self, min_zoom: float) -> None:
        self._params["min_zoom"] = min_zoom

    @property
    def random_seed(self) -> Optional[int]:
        return self._params["random_seed"]

    @random_seed.setter
    def random_seed(self, random_seed: Optional[int]) -> None:
        self._params["random_seed"] = random_seed
