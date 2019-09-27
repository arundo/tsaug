import pytest
import numpy as np
from tsaug import (
    RandomAffine,
    RandomCrop,
    RandomJitter,
    RandomTimeWarp,
    RandomSidetrack,
    RandomMagnify,
    RandomCrossSum,
    RandomTrend,
)

rand = np.random.RandomState(123)

augmentors = [
    (
        RandomAffine,
        {"max_a": 100, "min_a": -100, "max_b": 1000, "min_b": -1000},
    ),
    (RandomCrop, {"crop_size": 300}),
    (RandomJitter, {"dist": "uniform", "strength": 0.1}),
    (RandomTimeWarp, {"n_speed_change": 2}),
    (
        RandomSidetrack,
        {
            "mode": "additive",
            "max_sidetrack": 3,
            "min_sidetrack": 0.3,
            "initial_sidetrack": 2,
            "step_mu": 0.01,
            "step_sigma": 0.01,
        },
    ),
    (RandomTrend, {"num_anchors": 3, "min_anchor": -10, "max_anchor": 10}),
    (RandomMagnify, {"max_zoom": 1.9, "min_zoom": 1.1}),
    (RandomCrossSum, {"max_sum_series": 3}),
]

N = 10
n = 1024
c = 2
M = 3

X1 = np.random.uniform(size=n)
X2 = np.random.uniform(size=(N, n))
X3 = np.random.uniform(size=(N, n, c))

Y1 = np.random.choice(2, size=n).astype(int)
Y2 = np.random.choice(2, size=(N, n)).astype(int)


@pytest.mark.parametrize("augmentor", augmentors[:-1])
def test_X1_Y0(augmentor):
    """
    1D X, no Y
    """
    X_aug_0 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(X1)
    X_aug_1 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(X1)
    X_aug_2 = (augmentor[0](random_seed=1, **augmentor[1]) * M).run(X1)

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmentor", augmentors[:-1])
def test_X1_Y1(augmentor):
    """
    1D X, 1D Y
    """
    X_aug_0, Y_aug_0 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(
        X1, Y1
    )
    X_aug_1, Y_aug_1 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(
        X1, Y1
    )
    X_aug_2, Y_aug_2 = (augmentor[0](random_seed=1, **augmentor[1]) * M).run(
        X1, Y1
    )

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X2_Y0(augmentor):
    """
    2D X, no Y
    """
    X_aug_0 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(X2)
    X_aug_1 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(X2)
    X_aug_2 = (augmentor[0](random_seed=1, **augmentor[1]) * M).run(X2)

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X2_Y2(augmentor):
    """
    2D X, 2D Y
    """
    X_aug_0, Y_aug_0 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(
        X2, Y2
    )
    X_aug_1, Y_aug_1 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(
        X2, Y2
    )
    X_aug_2, Y_aug_2 = (augmentor[0](random_seed=1, **augmentor[1]) * M).run(
        X2, Y2
    )

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X3_Y0(augmentor):
    """
    3D X, no Y
    """
    X_aug_0 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(X3)
    X_aug_1 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(X3)
    X_aug_2 = (augmentor[0](random_seed=1, **augmentor[1]) * M).run(X3)

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()


@pytest.mark.parametrize("augmentor", augmentors)
def test_X3_Y2(augmentor):
    """
    3D X, 2D Y
    """
    X_aug_0, Y_aug_0 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(
        X3, Y2
    )
    X_aug_1, Y_aug_1 = (augmentor[0](random_seed=0, **augmentor[1]) * M).run(
        X3, Y2
    )
    X_aug_2, Y_aug_2 = (augmentor[0](random_seed=1, **augmentor[1]) * M).run(
        X3, Y2
    )

    assert (X_aug_0 == X_aug_1).all()
    assert (X_aug_0 != X_aug_2).any()
