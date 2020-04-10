"""
This module includes functions to visualize time series and segmentation mask.
"""

from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "To use the visualization module, matplotlib>=3 must be installed."
    )


def plot(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot time series and segmentation mask.

    This function requires matplotlib>=3.0.

    Parameters
    ----------
    X : numpy array
        Time series to be augmented. It must be a numpy array with shape
        (T,), (N, T), or (N, T, C), where T is the length of a series, N is
        the number of series, and C is the number of a channels in a series.

    Y: numpy array, optional
        Segmentation mask of the original time series. It must be a binary
        numpy array with shape (T,), (N, T), or (N, T, L), where T is the
        length of a series, N is the number of series, and L is the number
        of a segmentation classes. Default: None.

    Returns
    -------
    tuple (matplotlib Figure, matplotlib Axes)
        Figure and axes object of the plot.


    """
    X_ERROR_MSG = (
        "Input X must be a numpy array with shape (T,), (N, T), or (N, T, "
        "C), where T is the length of a series, N is the number of series, "
        "and C is the number of a channels in a series."
    )

    Y_ERROR_MSG = (
        "Input Y must be a numpy array with shape (T,), (N, T), or (N, T, "
        "L), where T is the length of a series, N is the number of series, "
        "and L is the number of a segmentation classes."
    )

    if not isinstance(X, np.ndarray):
        raise TypeError(X_ERROR_MSG)
    ndim_x = X.ndim
    if ndim_x == 1:  # (T, )
        X = X.reshape(1, -1, 1)
    elif ndim_x == 2:  # (N, T)
        X = np.expand_dims(X, 2)
    elif ndim_x == 3:  # (N, T, C)
        pass
    else:
        raise ValueError(X_ERROR_MSG)

    if Y is not None:
        if not isinstance(Y, np.ndarray):
            raise TypeError(Y_ERROR_MSG)
        ndim_y = Y.ndim
        if ndim_y == 1:  # (T, )
            Y = Y.reshape(1, -1, 1)
        elif ndim_y == 2:  # (N, T)
            Y = np.expand_dims(Y, 2)
        elif ndim_y == 3:  # (N, T, L)
            pass
        else:
            raise ValueError(Y_ERROR_MSG)

    N, T, _ = X.shape

    if Y is not None:
        Ny, Ty, L = Y.shape
        # check consistency between X and Y
        if N != Ny:
            raise ValueError("The numbers of series in X and Y are different.")
        if T != Ty:
            raise ValueError("The length of series in X and Y are different.")

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if Y is not None:
        Y = np.round(np.clip(Y, 0, 1))

    f, axes = plt.subplots(nrows=N, sharex=True, figsize=(16, 2 * len(X)))
    if N == 1:
        axes = [axes]

    if Y is None:
        Y = [None for _ in range(N)]

    clcolors = [
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1f77b4",
    ]

    for i, (Xk, Yk) in enumerate(zip(X, Y)):
        axes[i].plot(Xk)
        if Yk is not None:
            for j in range(L):
                windows = _get_event_windows(
                    Yk[:, j].clip(0, 1).round().astype(int)
                )
                for window in windows:
                    axes[i].axvspan(
                        window[0], window[1], alpha=0.4, color=clcolors[j % 10]
                    )

    return f, axes if (len(axes) > 1) else axes[0]


def _get_event_windows(Yk: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find continuous segmentation labels and group them
    """
    Yk_diff = np.diff(np.concatenate([[0], Yk, [0]]))
    start = np.argwhere(Yk_diff == 1).flatten()
    end = np.argwhere(Yk_diff == -1).flatten()
    return [(ss, ee - 1) for ss, ee in zip(start, end)]
