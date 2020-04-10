from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage.filters import convolve1d
from scipy.signal import get_window

from .base import _Augmenter, _default_seed


class Convolve(_Augmenter):
    """
    Convolve time series with a kernel window.

    Parameters
    ----------
    window : str, tuple, or list, optional
        The type of kernal window used for the convolution.

        - If str or tuple, it is a window type that can be passed to
          `scipy.signal.get_window`. See
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
          for more details.
        - If list, it is a list of such object. The type of a kernel window
          convolved with a time series is randomly sampled from this list.

        Default: "hann".

    size : int, list, tuple, optional
        Length of kernel windows.

        - If int, all series are convolved with windows of the same length.
        - If list, each series is convolved with a window with a size sampled
          from the list randomly.
        - If 2-tuple, each series is convolved with a window with a size sampled
          from the interval randomly.

        Default: 7.

    per_channel : bool, optional
        Whether to sample a kernel window for each channel in a time series or
        to use the same window for all channels in a time series. Only used if
        the kernel window is not deterministic. Default: False.

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

    def __init__(
        self,
        window: Union[str, Tuple, List[Union[str, Tuple]]] = "hann",
        size: Union[int, Tuple[int, int], List[int]] = 7,
        per_channel: bool = False,
        repeats: int = 1,
        prob: float = 1.0,
        seed: Optional[int] = _default_seed,
    ):
        self.window = window
        self.size = size
        self.per_channel = per_channel
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("window", "size", "per_channel")

    @property
    def window(self) -> Union[str, Tuple, List[Union[str, Tuple]]]:
        return self._window

    @window.setter
    def window(self, w: Union[str, Tuple, List[Union[str, Tuple]]]) -> None:
        WINDOW_ERROR_MSG = (
            "Parameter `window` must be a str or a tuple that can pass to "
            "`scipy.signal.get_window`, or a list of such objects. See "
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html "
            "for more details."
        )
        if not isinstance(w, list):
            try:
                get_window(w, 7)
            except TypeError:
                raise TypeError(WINDOW_ERROR_MSG)
            except ValueError:
                raise ValueError(WINDOW_ERROR_MSG)
            except:
                raise RuntimeError(WINDOW_ERROR_MSG)
        else:
            for ww in w:
                try:
                    get_window(ww, 7)
                except TypeError:
                    raise TypeError(WINDOW_ERROR_MSG)
                except ValueError:
                    raise ValueError(WINDOW_ERROR_MSG)
                except:
                    raise RuntimeError(WINDOW_ERROR_MSG)
        self._window = w

    @property
    def size(self) -> Union[int, Tuple[int, int], List[int]]:
        return self._size

    @size.setter
    def size(self, n: Union[int, Tuple[int, int], List[int]]) -> None:
        SIZE_ERROR_MSG = (
            "Parameter `size` must be a positive integer, "
            "a 2-tuple of positive integers representing an interval, "
            "or a list of positive integers."
        )
        if not isinstance(n, int):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(SIZE_ERROR_MSG)
                if not all([isinstance(nn, int) for nn in n]):
                    raise TypeError(SIZE_ERROR_MSG)
                if not all([nn > 0 for nn in n]):
                    raise ValueError(SIZE_ERROR_MSG)
            elif isinstance(n, tuple):
                if len(n) != 2:
                    raise ValueError(SIZE_ERROR_MSG)
                if (not isinstance(n[0], int)) or (not isinstance(n[1], int)):
                    raise TypeError(SIZE_ERROR_MSG)
                if n[0] >= n[1]:
                    raise ValueError(SIZE_ERROR_MSG)
                if (n[0] <= 0) or (n[1] <= 0):
                    raise ValueError(SIZE_ERROR_MSG)
            else:
                raise TypeError(SIZE_ERROR_MSG)
        elif n <= 0:
            raise ValueError(SIZE_ERROR_MSG)
        self._size = n

    @property
    def per_channel(self) -> bool:
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p: bool) -> None:
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    def _augment_core(
        self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)

        if isinstance(self.window, (str, tuple)):
            window_type = [self.window for _ in range(N * C)]
        else:
            if self.per_channel:
                window_type = [
                    self.window[i]
                    for i in rand.choice(len(self.window), N * C)
                ]
            else:
                window_type = [
                    self.window[i]
                    for i in rand.choice(len(self.window), N)
                    for _ in range(C)
                ]

        if isinstance(self.size, int):
            window_size = np.array([self.size for _ in range(N * C)])
        elif isinstance(self.size, tuple):
            if self.per_channel:
                window_size = rand.choice(
                    range(self.size[0], self.size[1]), N * C
                )
            else:
                window_size = rand.choice(range(self.size[0], self.size[1]), N)
                window_size = np.repeat(window_size, C)
        else:
            if self.per_channel:
                window_size = rand.choice(self.size, N * C)
            else:
                window_size = rand.choice(self.size, N)
                window_size = np.repeat(window_size, C)
        window_size = window_size.astype(int)

        X_aug = X.copy()
        X_aug = X_aug.swapaxes(1, 2).reshape(N * C, T)
        for ws in np.unique(window_size):
            for wt in set(window_type):
                window = get_window(window=wt, Nx=ws, fftbins=False)
                X_aug[
                    (window_size == ws) & [w == wt for w in window_type], :
                ] = (
                    convolve1d(
                        X_aug[
                            (window_size == ws)
                            & [w == wt for w in window_type],
                            :,
                        ],
                        window,
                        axis=1,
                    )
                    / window.sum()
                )
        X_aug = X_aug.reshape(N, C, T).swapaxes(1, 2)

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug
