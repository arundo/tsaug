import numpy as np
import pytest

from tsaug import (
    Affine,
    Crop,
    CrossSum,
    Magnify,
    RandomAffine,
    RandomCrop,
    RandomCrossSum,
    RandomJitter,
    RandomMagnify,
    RandomSidetrack,
    RandomTimeWarp,
    RandomTrend,
    Resample,
    Reverse,
    Trend,
)

rand = np.random.RandomState(123)

augmentors = [
    Affine,
    RandomAffine,
    Crop,
    RandomCrop,
    RandomJitter,
    Resample,
    RandomTimeWarp,
    RandomSidetrack,
    Magnify,
    RandomMagnify,
    Reverse,
    CrossSum,
    RandomCrossSum,
    Trend,
    RandomTrend,
]

N = 10
n = 1024
c = 3
M = 4

X1 = np.random.uniform(size=n)
X2 = np.random.uniform(size=(N, n))
X3 = np.random.uniform(size=(N, n, c))

Y1 = np.random.choice(2, size=n).astype(int)
Y2 = np.random.choice(2, size=(N, n)).astype(int)


@pytest.mark.parametrize("augmentor", augmentors)
def test_X1_Y0(augmentor):
    """
    1D X, no Y
    """
    X_aug = augmentor().run(X1)
    assert X_aug.shape == (n,)

    # check X_aug is not a view of X
    Xc = X1.copy()
    X_aug[0] = 12345
    assert (Xc == X1).all()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X1_Y1(augmentor):
    """
    1D X, 1D Y
    """
    X_aug, Y_aug = augmentor().run(X1, Y1)
    assert X_aug.shape == (n,)
    assert Y_aug.shape == (n,)

    # check X_aug is not a view of X
    Xc = X1.copy()
    X_aug[0] = 12345
    assert (Xc == X1).all()

    # check Y_aug is not a view of Y
    Yc = Y1.copy()
    Y_aug[0] = 12345
    assert (Yc == Y1).all()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X2_Y0(augmentor):
    """
    2D X, no Y
    """
    X_aug = augmentor().run(X2)
    assert X_aug.shape == (N, n)

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert (Xc == X2).all()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X2_Y2(augmentor):
    """
    2D X, 2D Y
    """
    X_aug, Y_aug = augmentor().run(X2, Y2)
    assert X_aug.shape == (N, n)
    assert Y_aug.shape == (N, n)

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert (Xc == X2).all()

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert (Yc == Y2).all()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X3_Y0(augmentor):
    """
    3D X, no Y
    """
    X_aug = augmentor().run(X3)
    assert X_aug.shape == (N, n, c)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert (Xc == X3).all()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X3_Y2(augmentor):
    """
    3D X, 2D Y
    """
    X_aug, Y_aug = augmentor().run(X3, Y2)
    assert X_aug.shape == (N, n, c)
    assert Y_aug.shape == (N, n)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert (Xc == X3).all()

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert (Yc == Y2).all()
