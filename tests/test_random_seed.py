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

augmenters = [
    AddNoise(),
    Convolve(),
    Crop(size=100),
    Drift(),
    Dropout(),
    Pool(),
    Quantize(),
    Resize(size=100),
    Reverse(),
    TimeWarp(),
]

N = 10
T = 100
C = 3
L = 2
M = 4

X1 = np.random.uniform(size=T)
X2 = np.random.uniform(size=(N, T))
X3 = np.random.uniform(size=(N, T, C))

Y1 = np.random.choice(2, size=T).astype(int)
Y2 = np.random.choice(2, size=(N, T)).astype(int)
Y3 = np.random.choice(2, size=(N, T, L)).astype(int)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X1_Y0(augmenter):
    """
    1D X, no Y
    """
    augmenter.seed = 0
    X_aug_0 = augmenter.augment(X1)
    X_aug_1 = augmenter.augment(X1)
    augmenter.seed = None
    X_aug_2 = augmenter.augment(X1)

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X1_Y1(augmenter):
    """
    1D X, 1D Y
    """
    augmenter.seed = 0
    X_aug_0, Y_aug_0 = augmenter.augment(X1, Y1)
    X_aug_1, Y_aug_1 = augmenter.augment(X1, Y1)
    augmenter.seed = None
    X_aug_2, Y_aug_2 = augmenter.augment(X1, Y1)

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y0(augmenter):
    """
    2D X, no Y
    """
    X_aug_0 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(X2)
    X_aug_1 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(X2)
    X_aug_2 = (augmenter[0](random_seed=1, **augmenter[1]) * M).run(X2)

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y2(augmenter):
    """
    2D X, 2D Y
    """
    X_aug_0, Y_aug_0 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(
        X2, Y2
    )
    X_aug_1, Y_aug_1 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(
        X2, Y2
    )
    X_aug_2, Y_aug_2 = (augmenter[0](random_seed=1, **augmenter[1]) * M).run(
        X2, Y2
    )

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y0(augmenter):
    """
    3D X, no Y
    """
    X_aug_0 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(X3)
    X_aug_1 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(X3)
    X_aug_2 = (augmenter[0](random_seed=1, **augmenter[1]) * M).run(X3)

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y2(augmenter):
    """
    3D X, 2D Y
    """
    X_aug_0, Y_aug_0 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(
        X3, Y2
    )
    X_aug_1, Y_aug_1 = (augmenter[0](random_seed=0, **augmenter[1]) * M).run(
        X3, Y2
    )
    X_aug_2, Y_aug_2 = (augmenter[0](random_seed=1, **augmenter[1]) * M).run(
        X3, Y2
    )

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()
