import numpy as np
from scipy.ndimage.filters import convolve1d
from scipy.signal import convolve, get_window

from .base import _Augmentor


class Convolve(_Augmentor):
    def __init__(
        self,
        window="hann",
        size=7,
        per_channel=False,
        repeats=1,
        prob=1.0,
        seed=None,
    ):
        self.window = window
        self.size = size
        self.per_channel = per_channel
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @staticmethod
    def _change_series_length():
        return False

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, w):
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
    def size(self):
        return self._size

    @size.setter
    def size(self, n):
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
                if n[0] > n[1]:
                    raise ValueError(SIZE_ERROR_MSG)
                if (n[0] <= 0) or (n[1] <= 0):
                    raise ValueError(SIZE_ERROR_MSG)
            else:
                raise TypeError(SIZE_ERROR_MSG)
        elif n <= 0:
            raise ValueError(SIZE_ERROR_MSG)
        self._size = n

    @property
    def per_channel(self):
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p):
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    def _augment_core(self, X, Y):
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)

        if isinstance(self.window, str):
            window_type = np.array([self.window for _ in range(N * C)])
        else:
            if self.per_channel:
                window_type = rand.choice(self.window, N * C)
            else:
                window_type = rand.choice(self.window, N)
                window_type = np.vstack(
                    [window_type for _ in range(C)]
                ).T.flatten()

        if isinstance(self.size, int):
            window_size = np.array([self.size for _ in range(N * C)])
        elif isinstance(self.size, tuple):
            if self.per_channel:
                window_size = rand.choice(
                    range(self.size[0], self.size[1]), N * C
                )
            else:
                window_size = rand.choice(range(self.size[0], self.size[1]), N)
                window_size = np.vstack(
                    [window_size for _ in range(C)]
                ).T.flatten()
        else:
            if self.per_channel:
                window_size = rand.choice(self.size, N * C)
            else:
                window_size = rand.choice(self.size, N)
                window_size = np.vstack(
                    [window_size for _ in range(C)]
                ).T.flatten()
        window_size = window_size.astype(int)

        X_aug = X.copy()
        X_aug = X_aug.swapaxes(1, 2).reshape(N * C, T)
        for ws in np.unique(window_size):
            for wt in np.unique(window_type):
                window = get_window(window=wt, Nx=ws, fftbins=False)
                X_aug[(window_size == ws) & (window_type == wt), :] = (
                    convolve1d(
                        X_aug[(window_size == ws) & (window_type == wt), :],
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
