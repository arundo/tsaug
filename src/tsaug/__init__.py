"""
tsaug
=====

`tsaug` is a Python package for time series augmentation. It offers a set of
augmentation methods for time series, as well as a simple API to connect
multiple augmenters into a pipeline.

See https://tsaug.readthedocs.io for complete documentation.

"""

__version__ = "0.2.1"

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
