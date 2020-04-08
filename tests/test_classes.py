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

augmenters = [
    AddNoise(repeats=M, prob=0.5),
    AddNoise(loc=(-1.0, 1.0), scale=(0.1, 0.2)),
    AddNoise(loc=[-1.0, 1.0], scale=[0.1, 0.2]),
    AddNoise(distr="laplace"),
    AddNoise(distr="uniform"),
    AddNoise(kind="multiplicative"),
    AddNoise(per_channel=False, normalize=False),
    Convolve(repeats=M, prob=0.5),
    Convolve(window=["hann", "blackman"]),
    Convolve(window=("gaussian", 1)),
    Convolve(size=(7, 11)),
    Convolve(size=[7, 11]),
    Convolve(per_channel=True),
    Crop(size=100),
    Drift(),
    Dropout(),
    Pool(),
    Quantize(),
    Resize(size=100),
    Reverse(),
    TimeWarp(),
]


@pytest.mark.parametrize("augmenter", augmenters)
def test_X1_Y0(augmenter):
    """
    1D X, no Y
    """
    X_aug = augmenter.augment(X1)
    if augmenter.repeats == 1:
        assert X_aug.shape == (T,)
    else:
        assert X_aug.shape == (M, T)

    # check X_aug is not a view of X
    Xc = X1.copy()
    X_aug[0] = 12345
    assert np.array_equal(Xc, X1)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X1_Y1(augmenter):
    """
    1D X, 1D Y
    """
    X_aug, Y_aug = augmenter.augment(X1, Y1)
    if augmenter.repeats == 1:
        assert X_aug.shape == (T,)
        assert Y_aug.shape == (T,)
    else:
        assert X_aug.shape == (M, T)
        assert Y_aug.shape == (M, T)

    # check X_aug is not a view of X
    Xc = X1.copy()
    X_aug[0] = 12345
    assert np.array_equal(Xc, X1)

    # check Y_aug is not a view of Y
    Yc = Y1.copy()
    Y_aug[0] = 12345
    assert np.array_equal(Yc, Y1)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y0(augmenter):
    """
    2D X, no Y
    """
    X_aug = augmenter.augment(X2)
    assert X_aug.shape == (N * augmenter.repeats, T)

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert np.array_equal(Xc, X2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y2(augmenter):
    """
    2D X, 2D Y
    """
    X_aug, Y_aug = augmenter.augment(X2, Y2)
    assert X_aug.shape == (N * augmenter.repeats, T)
    assert Y_aug.shape == (N * augmenter.repeats, T)

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert np.array_equal(Xc, X2)

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert np.array_equal(Yc, Y2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y0(augmenter):
    """
    3D X, no Y
    """
    X_aug = augmenter.augment(X3)
    assert X_aug.shape == (N * augmenter.repeats, T, C)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert np.array_equal(Xc, X3)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y2(augmenter):
    """
    3D X, 2D Y
    """
    X_aug, Y_aug = augmenter.augment(X3, Y2)
    assert X_aug.shape == (N * augmenter.repeats, T, C)
    assert Y_aug.shape == (N * augmenter.repeats, T)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert np.array_equal(Xc, X3)

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert np.array_equal(Yc, Y2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y3(augmenter):
    """
    3D X, 3D Y
    """
    X_aug, Y_aug = augmenter.augment(X3, Y3)
    assert X_aug.shape == (N * augmenter.repeats, T, C)
    assert Y_aug.shape == (N * augmenter.repeats, T, L)

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert np.array_equal(Xc, X3)

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert np.array_equal(Yc, Y2)
