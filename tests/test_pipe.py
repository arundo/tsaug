import numpy as np
import pytest

from tsaug import (
    AddNoise,
    Convolve,
    Crop,
    Drift,
    Dropout,
    Pool,
    Quantize,
    Resize,
    Reverse,
    TimeWarp,
)

rand = np.random.RandomState(123)

N = 10
T = 100
C = 3
L = 2


X1 = np.random.uniform(size=T)
X2 = np.random.uniform(size=(N, T))
X3 = np.random.uniform(size=(N, T, C))

Y1 = np.random.choice(2, size=T).astype(int)
Y2 = np.random.choice(2, size=(N, T)).astype(int)
Y3 = np.random.choice(2, size=(N, T, L)).astype(int)


def test_pipe():
    augmenter = (
        AddNoise() * 2 @ 0.5
        + (Crop(size=int(T / 2)) * 2 + Drift())
        + (Dropout() @ 0.5 + Pool())
        + Quantize() * 2
    )
    augmenter.augment(X1)
    augmenter.augment(X1, Y1)
    augmenter.augment(X2)
    augmenter.augment(X2, Y2)
    augmenter.augment(X2, Y3)
    augmenter.augment(X3)
    augmenter.augment(X3, Y2)
    augmenter.augment(X3, Y3)
    augmenter.summary()

    assert len(augmenter) == 6

    exchange = Resize(size=int(T / 2)) * 2 @ 0.5
    augmenter[3] = exchange
    assert augmenter[3] is exchange
    exchange.resize = int(T / 3)
    exchange.repeats = 3
    exchange.prob = 0.4
    assert isinstance(augmenter[3], Resize)
    assert augmenter[3].resize == int(T / 3)
    assert augmenter[3].repeats == 3
    assert augmenter[3].prob == 0.4
