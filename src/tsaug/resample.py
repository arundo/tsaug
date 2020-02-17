"""
Resample module
"""
from typing import Tuple, Optional, Union

import numpy as np
from .dimensionalize import dimensionalize
from .augmentor import _Augmentor


@dimensionalize
def resample(
    X: np.ndarray, Y: Optional[np.ndarray] = None, n_new: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resample from time series with new length.

    Time series will be transformed into new length. Values of time series is
    determined by linear interpolation.

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

    n_new : int, optional
        New length of time series. Default: n.

    Returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        Augmented time series and augmented labels (if argument `Y` exists).

    """

    n = X.shape[1]  # type: int

    if (n_new is None) or (n_new == n):
        if Y is None:
            return X.copy(), None
        else:
            return X.copy(), Y.copy()

    interp = (
        np.arange(1, n_new - 1) / (n_new - 1) * (n - 1)
    )  # type: np.ndarray
    interp_inds_0 = interp.astype(int)  # type: np.ndarray
    interp_inds_1 = interp_inds_0 + 1  # type: np.ndarray
    interp_weight_1 = interp - interp_inds_0  # type: np.ndarray
    interp_weight_0 = 1 - interp_weight_1  # type: np.ndarray

    X_aug = np.concatenate(
        [
            X[:, 0:1, :],
            X[:, interp_inds_0, :] * interp_weight_0.reshape(1, -1, 1)
            + X[:, interp_inds_1, :] * interp_weight_1.reshape(1, -1, 1),
            X[:, -1:, :],
        ],
        axis=1,
    )  # type: np.ndarray

    if Y is None:
        Y_aug = None  # type: Optional[np.ndarray]
    else:
        Y_aug = (
            np.concatenate(
                [
                    Y[:, 0:1, :],
                    Y[:, interp_inds_0, :] * interp_weight_0.reshape(1, -1, 1)
                    + Y[:, interp_inds_1, :]
                    * interp_weight_1.reshape(1, -1, 1),
                    Y[:, -1:, :],
                ],
                axis=1,
            )
            .round()
            .astype(int)
        )

    return X_aug, Y_aug


class Resample(_Augmentor):
    """Augmentor that resamples from time series with new length.

    Time series will be transformed into new length. Values of time series is
    determined by linear interpolation.

    Parameters
    ----------
    n_new : int, optional
        New length of time series. Default: n.

    """

    def __init__(self, n_new: Optional[int] = None) -> None:
        super().__init__(augmentor_func=resample, is_random=False, n_new=n_new)

    @property
    def prob(self) -> float:
        return self._prob

    @prob.setter
    def prob(self, prob: float) -> None:
        if (prob != 1.0) & (prob != 0.0):
            raise ValueError(
                "Resample augmentor may change the length of time series, "
                + "therefore probability must be either 0 or 1"
            )
        self._prob = prob

    @property
    def n_new(self) -> Optional[int]:
        return self._params["n_new"]

    @n_new.setter
    def n_new(self, n_new: Optional[int]) -> None:
        self._params["n_new"] = n_new

    def _get_output_dim(
        self,
        input_N: Union[
            Tuple[int, Optional[int]], Tuple[Optional[int], int]
        ] = (1, None),
        input_n: Union[
            Tuple[int, Optional[int]], Tuple[Optional[int], int]
        ] = (1, None),
        input_c: Union[
            Tuple[int, Optional[int]], Tuple[Optional[int], int]
        ] = (1, None),
    ) -> Tuple[
        Union[Tuple[int, Optional[int]], Tuple[Optional[int], int]],
        Union[Tuple[int, Optional[int]], Tuple[Optional[int], int]],
        Union[Tuple[int, Optional[int]], Tuple[Optional[int], int]],
    ]:
        output_N = (
            (input_N[0] * self.M, None)
            if (input_N[0] is not None)
            else (None, input_N[1] * self.M)
        )  # type: Union[Tuple[int, Optional[int]], Tuple[Optional[int], int]]
        if self.n_new is None:
            output_n = (
                input_n
            )  # type: Union[Tuple[int, Optional[int]], Tuple[Optional[int], int]]
        else:
            output_n = (None, self.n_new)
        return output_N, output_n, input_c
