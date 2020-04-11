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
    Convolve(size=(7, 10)) * 10,
    Crop(size=10),
    Drift(),
    Dropout(),
    Pool(size=[2, 4, 8]) * 10,
    Quantize(n_levels=[10, 20, 30]) * 10,
    Reverse() @ 0.5 * 10,
    TimeWarp(),
]

N = 10
T = 1000
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

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)


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

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y0(augmenter):
    """
    2D X, no Y
    """
    augmenter.seed = 0
    X_aug_0 = augmenter.augment(X2)
    X_aug_1 = augmenter.augment(X2)
    augmenter.seed = None
    X_aug_2 = augmenter.augment(X2)

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y2(augmenter):
    """
    2D X, 2D Y
    """
    augmenter.seed = 0
    X_aug_0, Y_aug_0 = augmenter.augment(X2, Y2)
    X_aug_1, Y_aug_1 = augmenter.augment(X2, Y2)
    augmenter.seed = None
    X_aug_2, Y_aug_2 = augmenter.augment(X2, Y2)

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y0(augmenter):
    """
    3D X, no Y
    """
    augmenter.seed = 0
    X_aug_0 = augmenter.augment(X3)
    X_aug_1 = augmenter.augment(X3)
    augmenter.seed = None
    X_aug_2 = augmenter.augment(X3)

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y2(augmenter):
    """
    3D X, 2D Y
    """
    augmenter.seed = 0
    X_aug_0, Y_aug_0 = augmenter.augment(X3, Y2)
    X_aug_1, Y_aug_1 = augmenter.augment(X3, Y2)
    augmenter.seed = None
    X_aug_2, Y_aug_2 = augmenter.augment(X3, Y2)

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y3(augmenter):
    """
    3D X, 3D Y
    """
    augmenter.seed = 0
    X_aug_0, Y_aug_0 = augmenter.augment(X3, Y3)
    X_aug_1, Y_aug_1 = augmenter.augment(X3, Y3)
    augmenter.seed = None
    X_aug_2, Y_aug_2 = augmenter.augment(X3, Y3)

    assert np.array_equal(X_aug_0, X_aug_1)
    assert not np.array_equal(X_aug_0, X_aug_2)


def test_resize_random():
    augmenter = Resize(size=50) @ 0.5
    augmenter.seed = None
    X_aug = []
    for _ in range(10):
        X_aug.append(augmenter.augment(X1))
    assert any([np.array_equal(X_aug[0], X_aug[i]) for i in range(1, 10)])

    augmenter.seed = 0
    X_aug = []
    for _ in range(10):
        X_aug.append(augmenter.augment(X1))
    assert all([np.array_equal(X_aug[0], X_aug[i]) for i in range(1, 10)])
