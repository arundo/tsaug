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
__version__ = "0.1"

from .time_warp import random_time_warp, RandomTimeWarp
from .zoom import magnify, random_magnify, Magnify, RandomMagnify
from .sidetrack import random_sidetrack, RandomSidetrack
from .resample import resample, Resample
from .jitter import random_jitter, RandomJitter
from .crop import crop, random_crop, Crop, RandomCrop
from .affine import affine, random_affine, Affine, RandomAffine
from .reverse import reverse, Reverse
from .cross_sum import cross_sum, random_cross_sum, CrossSum, RandomCrossSum
from .trend import trend, random_trend, Trend, RandomTrend
