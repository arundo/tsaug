"""
tsaug
=====

`tsaug` is a Python package for time series augmentation. It offers a set of
augmentation methods for time series with unified APIs, as well as operators to
connect multiple augmentors into a pipeline.

See https://arundo-tsaug.readthedocs-hosted.com complete documentation.

"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.1.1"

from .affine import Affine, RandomAffine, affine, random_affine
from .crop import Crop, RandomCrop, crop, random_crop
from .cross_sum import CrossSum, RandomCrossSum, cross_sum, random_cross_sum
from .jitter import RandomJitter, random_jitter
from .resample import Resample, resample
from .reverse import Reverse, reverse
from .sidetrack import RandomSidetrack, random_sidetrack
from .time_warp import RandomTimeWarp, random_time_warp
from .trend import RandomTrend, Trend, random_trend, trend
from .zoom import Magnify, RandomMagnify, magnify, random_magnify
