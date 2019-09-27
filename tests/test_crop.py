import numpy as np
from tsaug import Crop, RandomCrop


def test_crop_X1():
    X = np.arange(100)
    X_crops = Crop(crop_start=[[0, 90, 20]], crop_size=5).run(X)
    assert X_crops.shape == (3, 5)
    assert (X_crops[0, :] == np.arange(5)).all()
    assert (X_crops[1, :] == np.arange(90, 95)).all()


def test_crop_X2():
    X = np.arange(200).reshape(2, -1)
    X_crops = Crop(crop_start=[[0, 90, 20], [10, 30, 15]], crop_size=5).run(X)
    assert X_crops.shape == (6, 5)
    assert (X_crops[3, :] == np.arange(110, 115)).all()
    assert (X_crops[1, :] == np.arange(90, 95)).all()


def test_crop_X3():
    X = np.concatenate(
        [
            np.arange(200).reshape(2, -1, 1),
            -1 * np.arange(200).reshape(2, -1, 1),
        ],
        axis=2,
    )
    X_crops = Crop(crop_start=[[0, 90, 20], [10, 30, 15]], crop_size=5).run(X)
    assert X_crops.shape == (6, 5, 2)
    assert (X_crops[3, :, 0] == np.arange(110, 115)).all()
    assert (X_crops[1, :, 1] == -np.arange(90, 95)).all()


def test_random_crop_X1():
    X = np.arange(100)
    X_crops = RandomCrop(crop_size=5, crops_per_series=3).run(X)
    assert X_crops.shape == (3, 5)
    assert (np.diff(X_crops, axis=1) == 1).all()


def test_random_crop_X2():
    X = np.arange(200).reshape(2, -1)
    X_crops = RandomCrop(crop_size=5, crops_per_series=3).run(X)
    assert X_crops.shape == (6, 5)
    assert (np.diff(X_crops, axis=1) == 1).all()
    assert (X_crops[:3] < 100).all()
    assert (X_crops[3:] >= 100).all()


def test_random_crop_X3():
    X = np.concatenate(
        [
            np.arange(200).reshape(2, -1, 1),
            -1 * np.arange(200).reshape(2, -1, 1),
        ],
        axis=2,
    )
    X_crops = RandomCrop(crop_size=5, crops_per_series=3).run(X)
    assert X_crops.shape == (6, 5, 2)
    assert (abs(X_crops[:3]) < 100).all()
    assert (abs(X_crops[3:]) >= 100).all()
    assert (np.diff(X_crops[:, :, 0], axis=1) == 1).all()
    assert (np.diff(X_crops[:, :, 1], axis=1) == -1).all()

