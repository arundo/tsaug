"""
This module includes functions to visualize time series and segmentation mask.
"""
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "To use the visualization module, matplotlib>=3 must be installed."
    )

import numpy as np

__all__ = ["plot"]


def plot(X, Y=None, f=None, ax=None, global_ylim=False, saveas=None):
    """
    Plot time series

    Args:
        X (numpy array): Time series to plot, N*n or N*n*c matrix. Each row
            represents a series, each column represents a time point, and the
            third dimension (if exists) represents channels.
        Y (numpy array): Anomaly labels, N*n or N*n*L matrix. Each row
            represents a series, each column represents a time point, and the
            third dimension (if exists) represents types of anomalies. Normal
            time points are represented by 0, while anomalies are 1. If it is
            N*n matrix, different types of anomalies may also be represented
            by different positive integers.
        f (matplotlib figure object): Figure to plot at. If not given, a figure
            object will be created. Default: None
        ax (matplotlib axes object or list of axes objects): Axes to plot
            every series. The number of axes in the list must be N. If not
            given, axes will be created. Default: None
        global_ylim (bool, optional): Whether to use the same scale for all y-
            axes. Default: False.
        saveas (str): Path to save the figure. If not given, then the figure
            will not be saved. Default: None

    """

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if Y is not None:
        Y = np.round(np.clip(Y, 0, 1))

    classes = {0, 1}
    if Y is not None:
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        if Y.ndim == 2:
            classes = classes.union(set(list(np.unique(Y))))

    if len(classes) > 0:
        classes.remove(0)

    if Y is not None:
        if Y.ndim == 2:
            Y = np.stack([(Y == cl).astype(int) for cl in classes], axis=2)
        num_classes = Y.shape[2]

    if Y is None:
        Y = [None] * len(X)

    if ax is None:
        if f is None:
            f, ax = plt.subplots(
                nrows=len(X), sharex=True, figsize=(15, 2 * len(X))
            )
        else:
            ax = f.subplots(nrows=len(X), sharex=True)
    if (len(X) == 1) & (not isinstance(ax, list)):
        ax = [ax]

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

    if global_ylim:
        ylim = (np.nanmin(X), np.nanmax(X))
        ylim = (
            ylim[0] - (ylim[1] - ylim[0]) / 20,
            ylim[1] + (ylim[1] - ylim[0]) / 20,
        )
    counter = 0
    for Xk, Yk in zip(X, Y):
        ax[counter].plot(Xk)
        if global_ylim:
            ax[counter].set_ylim(ylim[0], ylim[1])
        if Yk is None:
            counter += 1
            continue
        for clcounter in range(num_classes):
            if Yk is not None:
                anomaly_windows = _get_anomaly_windows(
                    Yk[:, clcounter].astype(int)
                )
                for anomaly_window in anomaly_windows:
                    ax[counter].axvspan(
                        anomaly_window[0],
                        anomaly_window[1],
                        alpha=0.4,
                        color=(
                            "r"
                            if num_classes == 1
                            else clcolors[clcounter % 10]
                        ),
                    )
        counter += 1

    if saveas is not None:
        f.savefig(saveas)


def _get_anomaly_windows(Yk):
    """
    Find continuous anomalies labels and group them
    """
    Yk_diff = np.diff(np.concatenate([[0], Yk, [0]]))
    start = np.argwhere(Yk_diff == 1).flatten()
    end = np.argwhere(Yk_diff == -1).flatten()
    return [(ss, ee - 1) for ss, ee in zip(start, end)]
