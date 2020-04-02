"""
tsaug
=====

`tsaug` is a Python package for time series augmentation. It offers a set of
augmentation methods for time series with unified APIs, as well as operators to
connect multiple augmentors into a pipeline.

See https://arundo-tsaug.readthedocs-hosted.com complete documentation.

"""

__version__ = "0.2"

from ._augmenter.crop import Crop
from ._augmenter.time_warp import TimeWarp
