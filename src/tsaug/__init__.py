"""
tsaug
=====

`tsaug` is a Python package for time series augmentation. It offers a set of
augmentation methods for time series with unified APIs, as well as operators to
connect multiple augmentors into a pipeline.

See https://arundo-tsaug.readthedocs-hosted.com complete documentation.

"""

__version__ = "0.2"

from ._augmenter.add_noise import AddNoise
from ._augmenter.convolve import Convolve
from ._augmenter.crop import Crop
from ._augmenter.drift import Drift
from ._augmenter.dropout import Dropout
from ._augmenter.pool import Pool
from ._augmenter.quantize import Quantize
from ._augmenter.resize import Resize
from ._augmenter.reverse import Reverse
from ._augmenter.time_warp import TimeWarp
