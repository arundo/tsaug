# tsaug

[![Build Status](https://travis-ci.com/arundo/tsaug.svg?branch=master)](https://travis-ci.com/arundo/tsaug)
[![Documentation Status](https://readthedocs.org/projects/tsaug/badge/?version=stable)](https://tsaug.readthedocs.io/en/stable/?badge=stable)
[![Coverage Status](https://coveralls.io/repos/github/arundo/tsaug/badge.svg?branch=master&service=github)](https://coveralls.io/github/arundo/tsaug?branch=master)
[![PyPI](https://img.shields.io/pypi/v/tsaug)](https://pypi.org/project/tsaug/)
[![Downloads](https://pepy.tech/badge/tsaug)](https://pepy.tech/project/tsaug)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`tsaug` is a Python package for time series augmentation. It offers a set of
augmentation methods for time series, as well as a simple API to connect
multiple augmenters into a pipeline.

See https://tsaug.readthedocs.io complete documentation.

## Installation

Prerequisites: Python 3.5 or later.

It is recommended to install the most recent **stable** release of tsaug from PyPI.

```shell
pip install tsaug
```

Alternatively, you could install from source code. This will give you the **latest**, but unstable, version of tsaug.

```shell
git clone https://github.com/arundo/tsaug.git
cd tsaug/
git checkout develop
pip install ./
```

## Examples
A first-time user may start with two examples:

- [Augment a batch of multivariate time series](https://tsaug.readthedocs.io/en/stable/quickstart.html#augment-a-batch-of-multivariate-time-series)
- [Augment a 2-channel audio sequence](https://tsaug.readthedocs.io/en/stable/quickstart.html#augment-a-2-channel-audio-sequence)

Examples of every individual augmenter can be found [here](https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html)

For full references of implemented augmentation methods, please refer to [References](https://tsaug.readthedocs.io/en/stable/references.html).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate.

Please see [Contributing](https://tsaug.readthedocs.io/en/stable/developer.html) for more details.


## License

`tsaug` is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.