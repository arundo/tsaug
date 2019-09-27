import pytest
import numpy as np
from tsaug import (
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
)

rand = np.random.RandomState(123)

N = 10
n = 1024
c = 3

augmentors = [
    (Affine, {"a": 2, "b": 30}),
    (
        RandomAffine,
        {"max_a": 100, "min_a": -100, "max_b": 1000, "min_b": -1000},
    ),
    (Crop, {"crop_start": 2, "crop_size": 200}),
    (RandomCrop, {"crop_size": 300}),
    (RandomJitter, {"dist": "uniform", "strength": 0.1}),
    (Resample, {"n_new": 600}),
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
    (Magnify, {"start": 10, "end": 900, "size": 800}),
    (RandomMagnify, {"max_zoom": 1.9, "min_zoom": 1.1}),
    (Reverse, {}),
    (CrossSum, {"inds": rand.choice(range(-1, N), size=(N, 3))}),
    (RandomCrossSum, {"max_sum_series": 3}),
    (Trend, {"anchors": rand.uniform(size=(N, 3, c))}),
    (RandomTrend, {"num_anchors": 3, "min_anchor": -10, "max_anchor": 10}),
]


@pytest.mark.parametrize("augmentor", augmentors)
def test_param_setting(augmentor):
    aug = augmentor[0]()
    for key, value in augmentor[1].items():
        setattr(aug, key, value)
        if isinstance(value, np.ndarray):
            assert (aug._params[key] == value).all()
        else:
            assert aug._params[key] == value


def test_pipeline_M_prob_setting():
    augs = [augmentor[0]() for augmentor in augmentors]
    pipe = None
    for counter, aug in enumerate(augs):
        if pipe is None:
            pipe = aug * 2 @ 0.5
        else:
            if (counter == 2) | (counter == 3) | (counter == 5):
                pipe = pipe + aug * 2 @ 0.0
            else:
                pipe = pipe + aug * 2 @ 0.5
    pipe = pipe * 3

    prob_true = [0.5] * len(augs)
    prob_true[2] = 0
    prob_true[3] = 0
    prob_true[5] = 0

    assert [aug.M for aug in pipe._augmentor_list] == [2, 2 * 3] + [2] * (
        len(augs) - 2
    )
    assert [aug.prob for aug in pipe._augmentor_list] == prob_true
    for aug in augs:
        assert aug.M == 1
        assert aug.prob == 1.0


def test_pipeline_param_setting():
    augs = [augmentor[0]() for augmentor in augmentors]
    pipe = None
    for counter, aug in enumerate(augs):
        if pipe is None:
            pipe = aug * 2 @ 0.5
        else:
            if (counter == 2) | (counter == 3) | (counter == 5):
                pipe = pipe + aug * 2 @ 0.0
            else:
                pipe = pipe + aug * 2 @ 0.5
    pipe = pipe * 3

    for counter, aug in enumerate(augmentors):
        for key, value in aug[1].items():
            setattr(pipe[counter], key, value)
        setattr(pipe[counter], "prob", 0.0)
        setattr(pipe[counter], "M", 2)

    for counter, aug in enumerate(augmentors):
        for key, value in aug[1].items():
            if isinstance(value, np.ndarray):
                assert (getattr(pipe[counter], key) == value).all()
            else:
                assert getattr(pipe[counter], key) == value
        assert getattr(pipe[counter], "prob") == 0.0
        assert getattr(pipe[counter], "M") == 2

    for aug, aug_orig in zip(augmentors, augs):
        for key, value in aug[1].items():
            if isinstance(value, np.ndarray):
                assert (aug_orig._params[key] != value).any()
            else:
                assert aug_orig._params[key] != value
        assert aug_orig.prob == 1.0
        assert aug_orig.M == 1
