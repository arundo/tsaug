"""
Unit tests of anomaly adder
"""

import numpy as np
import pytest

from tsaug import resample


def test_resample_1D():
    X = np.arange(101)
    X_aug = resample(X, n_new=51)
    X_aug = X_aug.round(decimals=12)
    assert (X_aug == np.arange(0, 101, 2)).all()
    X_aug = resample(X, n_new=201)
    X_aug = X_aug.round(decimals=12)
    assert (X_aug == np.arange(0, 100.5, 0.5)).all()


def test_resample_2D():
    X = np.arange(101)
    X = np.vstack([X, X + 1])
    X_aug = resample(X, n_new=51)
    X_aug_true = np.vstack([np.arange(0, 101, 2), np.arange(1, 102, 2)])
    X_aug = X_aug.round(decimals=12)
    assert (X_aug == X_aug_true).all()

    X_aug = resample(X, n_new=201)
    X_aug_true = np.vstack(
        [np.arange(0, 100.5, 0.5), np.arange(1, 101.5, 0.5)]
    )
    X_aug = X_aug.round(decimals=12)
    assert (X_aug == X_aug_true).all()


def test_resample_3D():
    X = np.arange(101)
    X = np.vstack([X, X + 1])
    X = np.stack([X, X + 10, X + 20], axis=2)
    X_aug = resample(X, n_new=51)
    X_aug_true = np.vstack([np.arange(0, 101, 2), np.arange(1, 102, 2)])
    X_aug_true = np.stack(
        [X_aug_true, X_aug_true + 10, X_aug_true + 20], axis=2
    )
    X_aug = X_aug.round(decimals=12)
    assert (X_aug == X_aug_true).all()

    X_aug = resample(X, n_new=201)
    X_aug_true = np.vstack(
        [np.arange(0, 100.5, 0.5), np.arange(1, 101.5, 0.5)]
    )
    X_aug_true = np.stack(
        [X_aug_true, X_aug_true + 10, X_aug_true + 20], axis=2
    )
    X_aug = X_aug.round(decimals=12)
    assert (X_aug == X_aug_true).all()
