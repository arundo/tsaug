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
    AddNoise(repeats=M, prob=0.5, seed=0),
    AddNoise(loc=(-1.0, 1.0), scale=(0.1, 0.2), seed=0),
    AddNoise(loc=[-1.0, 1.0], scale=[0.1, 0.2], seed=0),
    AddNoise(distr="laplace", seed=0),
    AddNoise(distr="uniform", seed=0),
    AddNoise(kind="multiplicative", seed=0),
    AddNoise(per_channel=False, normalize=False, seed=0),
    Convolve(repeats=M, prob=0.5, seed=0),
    Convolve(window=["hann", "blackman", ("gaussian", 1)], seed=0),
    Convolve(
        window=["hann", "blackman", ("gaussian", 1)], per_channel=True, seed=0
    ),
    Convolve(window=("gaussian", 1), seed=0),
    Convolve(size=(7, 11), seed=0),
    Convolve(size=(7, 11), per_channel=True, seed=0),
    Convolve(size=[7, 11], seed=0),
    Convolve(size=[7, 11], per_channel=True, seed=0),
    Convolve(per_channel=True, seed=0),
    Crop(size=int(T / 2), repeats=M, seed=0),
    Crop(size=(int(T / 3), T), resize=int(T / 2), seed=0),
    Crop(size=[int(T / 3), T], resize=int(T / 2), seed=0),
    Drift(repeats=M, prob=0.5, seed=0),
    Drift(max_drift=(0.5, 1.0), seed=0),
    Drift(n_drift_points=[3, 8], seed=0),
    Drift(kind="multiplicative", seed=0),
    Drift(per_channel=False, normalize=False, seed=0),
    Dropout(repeats=M, prob=0.5, seed=0),
    Dropout(p=(0.01, 0.1), size=(1, 5), seed=0),
    Dropout(p=[0.01, 0.02, 0.03], size=[1, 2, 3], seed=0),
    Dropout(fill="bfill", seed=0),
    Dropout(fill="mean", seed=0),
    Dropout(fill=0, seed=0),
    Dropout(per_channel=True, seed=0),
    Pool(repeats=M, prob=0.5, seed=0),
    Pool(kind="max", seed=0),
    Pool(kind="min", seed=0),
    Pool(size=(2, 8), seed=0),
    Pool(size=[2, 4, 6], seed=0),
    Pool(per_channel=True, seed=0),
    Quantize(repeats=M, prob=0.5, seed=0),
    Quantize(n_levels=(5, 10), seed=0),
    Quantize(n_levels=(5, 10), per_channel=True, seed=0),
    Quantize(n_levels=[5, 6, 7], seed=0),
    Quantize(n_levels=[5, 6, 7], per_channel=True, seed=0),
    Quantize(how="quantile", seed=0),
    Quantize(how="kmeans", seed=0),
    Quantize(per_channel=True, seed=0),
    Resize(size=int(T / 2), repeats=M, prob=1.0, seed=0),
    Reverse(repeats=M, prob=0.5, seed=0),
    TimeWarp(repeats=M, prob=0.5, seed=0),
    TimeWarp(max_speed_ratio=[3, 4, 5], seed=0),
    TimeWarp(max_speed_ratio=(3, 5), seed=0),
]


@pytest.mark.parametrize("augmenter", augmenters)
def test_X1_Y0(augmenter):
    """
    1D X, no Y
    """
    X_aug = augmenter.augment(X1)
    if augmenter.repeats == 1:
        assert X_aug.shape == (
            T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        )
    else:
        assert X_aug.shape == (
            M,
            T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        )

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
        assert X_aug.shape == (
            T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        )
        assert Y_aug.shape == (
            T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        )
    else:
        assert X_aug.shape == (
            M,
            T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        )
        assert Y_aug.shape == (
            M,
            T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        )

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
    assert X_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
    )

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
    assert X_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
    )
    assert Y_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
    )

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert np.array_equal(Xc, X2)

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert np.array_equal(Yc, Y2)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X2_Y3(augmenter):
    """
    2D X, 3D Y
    """
    X_aug, Y_aug = augmenter.augment(X2, Y3)
    assert X_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
    )
    assert Y_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        L,
    )

    # check X_aug is not a view of X
    Xc = X2.copy()
    X_aug[0, 0] = 12345
    assert np.array_equal(Xc, X2)

    # check Y_aug is not a view of Y
    Yc = Y3.copy()
    Y_aug[0, 0] = 12345
    assert np.array_equal(Yc, Y3)


@pytest.mark.parametrize("augmenter", augmenters)
def test_X3_Y0(augmenter):
    """
    3D X, no Y
    """
    X_aug = augmenter.augment(X3)
    assert X_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        C,
    )

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
    assert X_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        C,
    )
    assert Y_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
    )

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
    assert X_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        C,
    )
    assert Y_aug.shape == (
        N * augmenter.repeats,
        T if not isinstance(augmenter, (Resize, Crop)) else int(T / 2),
        L,
    )

    # check X_aug is not a view of X
    Xc = X3.copy()
    X_aug[0, 0, 0] = 12345
    assert np.array_equal(Xc, X3)

    # check Y_aug is not a view of Y
    Yc = Y2.copy()
    Y_aug[0, 0] = 12345
    assert np.array_equal(Yc, Y2)
