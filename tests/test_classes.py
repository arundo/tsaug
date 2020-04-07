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
    X_aug = augmenter.augment(X1)
    assert X_aug.shape == (T,)

    # check X_aug is not a view of X
    Xc = X1.copy()
    X_aug[0] = 12345
    assert (Xc == X1).all()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X1_Y1(augmenter):
    """
    1D X, 1D Y
    """
    X_aug, Y_aug = augmenter.augment(X1, Y1)
    assert X_aug.shape == (T,)
    assert Y_aug.shape == (T,)

    # check X_aug is not a view of X
    Xc = X1.copy()
    X_aug[0] = 12345
    assert (Xc == X1).all()

    # check Y_aug is not a view of Y
    Yc = Y1.copy()
    Y_aug[0] = 12345
    assert (Yc == Y1).all()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y0(augmenter):
    """
    2D X, no Y
    """
    X_aug = augmenter.augment(X2)
    assert X_aug.shape == (N, T)

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert (Xc == X2).all()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y2(augmenter):
    """
    2D X, 2D Y
    """
    X_aug, Y_aug = augmenter.augment(X2, Y2)
    assert X_aug.shape == (N, T)
    assert Y_aug.shape == (N, T)

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert (Xc == X2).all()

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert (Yc == Y2).all()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y0(augmenter):
    """
    3D X, no Y
    """
    X_aug = augmenter.augment(X3)
    assert X_aug.shape == (N, T, C)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert (Xc == X3).all()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y2(augmenter):
    """
    3D X, 2D Y
    """
    X_aug, Y_aug = augmenter.augment(X3, Y2)
    assert X_aug.shape == (N, T, C)
    assert Y_aug.shape == (N, T)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert (Xc == X3).all()

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert (Yc == Y2).all()


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y3(augmenter):
    """
    3D X, 3D Y
    """
    X_aug, Y_aug = augmenter.augment(X3, Y3)
    assert X_aug.shape == (N, T, C)
    assert Y_aug.shape == (N, T, L)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert (Xc == X3).all()

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert (Yc == Y2).all()
