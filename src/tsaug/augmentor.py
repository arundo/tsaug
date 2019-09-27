import numpy as np


class _Augmentor:
    def __init__(self, augmentor_func=None, is_random=None, **kwargs):
        self._augmentor_func = augmentor_func
        self._is_random = is_random
        self._M = 1
        self._prob = 1.0
        self._params = kwargs

    def run(self, X, Y=None):
        """Perform augmentation to time series.

        Parameters
        ----------
        X : numpy.ndarray
            Time series to be augmented. Matrix with shape (n,), (N, n) or (N,
            n, c), where n is the length of each series, N is the number of
            series, and c is the number of channels.

        Y : numpy.ndarray, optional
            Binary labels of time series, where 0 represents a normal point and
            1 represents an anomalous points. Matrix with shape (n,), (N, n) or
            (N, n, cl), where n is the length of each series, N is the number
            of series, and cl is the number of classes (i.e. types of anomaly).
            Default: None.

        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray)
            Augmented time series and augmented labels (if argument `Y`
            exists).

        """
        if self._prob < 1:
            if X.ndim == 1:
                r = np.random.uniform(size=self._M)
            else:
                r = np.random.uniform(size=X.shape[0] * self._M)
        if Y is None:
            if self._M == 1:
                X_aug = X.copy()
            else:
                X_aug = np.vstack([X.copy()] * self._M)
            if self._prob == 1:
                X_aug = self._augmentor_func(X_aug, **self._params)
            elif (r <= self._prob).any():
                X_aug[r <= self._prob, :] = self._augmentor_func(
                    X_aug[r <= self._prob, :], **self._params
                )
            # if (X.ndim == 1) and (self._M == 1):
            #     X_aug = X_aug.flatten()
            return X_aug
        else:
            if self._M == 1:
                X_aug = X.copy()
                Y_aug = Y.copy()
            else:
                X_aug = np.vstack([X.copy()] * self._M)
                Y_aug = np.vstack([Y.copy()] * self._M)
            if self._prob == 1:
                X_aug, Y_aug = self._augmentor_func(
                    X_aug, Y_aug, **self._params
                )
            elif (r <= self._prob).any():
                X_aug[r <= self._prob, :], Y_aug[
                    r <= self._prob, :
                ] = self._augmentor_func(
                    X_aug[r <= self._prob, :],
                    Y_aug[r <= self._prob, :],
                    **self._params
                )
            # if (X.ndim == 1) and (self._M == 1):
            #     X_aug = X_aug.flatten()
            #     Y_aug = Y_aug.flatten()
            return X_aug, Y_aug

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, prob):
        if (prob < 0) or (prob > 1):
            raise ValueError("Probability must be between 0 and 1")
        self._prob = prob

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, M):
        self._M = M

    def copy(self):
        """Create a copy of this augmentor."""
        my_copy = self.__class__(**self._params)
        my_copy.M = self.M
        my_copy.prob = self.prob
        return my_copy

    def __add__(self, another_augmentor):
        if isinstance(another_augmentor, _AugmentorPipeline):
            return _AugmentorPipeline(
                [self] + another_augmentor._augmentor_list
            )
        elif isinstance(another_augmentor, _Augmentor):
            return _AugmentorPipeline([self, another_augmentor])
        else:
            raise TypeError(
                "An augmentor can only be added by another augmentor or an "
                "augmentor pipeline."
            )

    def __mul__(self, M):
        augmentor_copy = self.copy()
        augmentor_copy.M = augmentor_copy.M * M
        return augmentor_copy

    def __matmul__(self, prob):
        augmentor_copy = self.copy()
        augmentor_copy.prob = augmentor_copy.prob * prob
        return augmentor_copy

    def __len__(self):
        return 1

    def summary(self):
        """Print summary of this augmentor."""
        self._summary()

    def _summary(
        self,
        header=True,
        input_N=(1, None),
        input_n=(1, None),
        input_c=(1, None),
    ):
        if header:
            print("Augmentor \t M \t Prob \t Output Size \t Params")
            print("=" * 100)
        print(
            "{} \t {} \t {} \t {} \t {}".format(
                self.__class__.__name__,
                self._M,
                round(self._prob, 4),
                "({0}, {1}, {2})".format(
                    *[
                        ("{}{}".format(mul, symbol) if mul != 1 else symbol)
                        if (mul is not None)
                        else str(ab)
                        for (mul, ab), symbol in zip(
                            self._get_output_dim(
                                input_N=input_N,
                                input_n=input_n,
                                input_c=input_c,
                            ),
                            ("N", "n", "c"),
                        )
                    ]
                ),
                self._params,
            )
        )
        return self._get_output_dim(
            input_N=input_N, input_n=input_n, input_c=input_c
        )

    def _get_output_dim(
        self, input_N=(1, None), input_n=(1, None), input_c=(1, None)
    ):
        """
        Get output dimensions from input dimensions

        Could be overridden in some augmentors.

        Returns
        -------
        3-tuple of 2-tuples
            Multipliers of N, n, c, and absolute of N, n, c. None when not
            applicable.

        """
        output_N = (
            (input_N[0] * self.M, None)
            if (input_N[0] is not None)
            else (None, input_N[1] * self.M)
        )
        return output_N, input_n, input_c


class _AugmentorPipeline:
    def __init__(self, augmentor_list):
        self._augmentor_list = [
            augmentor.copy() for augmentor in augmentor_list
        ]

    def run(self, X, Y=None):
        """Perform augmentation to time series.

        Args:
            X (numpy array): Time series to be augmented. N*n*c or N*n matrix,
                where N is the number of series, n is the length of each
                series, and c is optionally the number of channels if the time
                series is multivariate.
            Y (numpy array, optional): Labels of time series. N*n binary
                matrix, where 0 represents normal and 1 represents anomalous.

        Returns:
            2-tuple:
            - numpy array: Augmented time series.
            - numpy array: Augmented labels (only when input Y exists).

        """
        if Y is None:
            X_aug = X.copy()
            for augmentor in self._augmentor_list:
                X_aug = augmentor.run(X_aug)
            return X_aug
        else:
            X_aug = X.copy()
            Y_aug = Y.copy()
            for augmentor in self._augmentor_list:
                X_aug, Y_aug = augmentor.run(X_aug, Y_aug)
            return X_aug, Y_aug

    def copy(self):
        return _AugmentorPipeline(self._augmentor_list)

    def __add__(self, another_augmentor):
        if isinstance(another_augmentor, _AugmentorPipeline):
            return _AugmentorPipeline(
                self._augmentor_list + another_augmentor._augmentor_list
            )
        elif isinstance(another_augmentor, _Augmentor):
            return _AugmentorPipeline(
                self._augmentor_list + [another_augmentor]
            )
        else:
            raise TypeError(
                "An augmentor pipeline can only be added by another augmentor "
                "pipeline or an augmentor."
            )

    def __mul__(self, M):
        # when operator * is applied to a augmentor pipeline, change the M of
        # the first random augmentor. If no random augmentor exist in the
        # pipeline, then change the M of the last augmentor.
        try:
            ind_mul = [
                augmentor._is_random for augmentor in self._augmentor_list
            ].index(True)
        except ValueError:
            ind_mul = len(self) - 1

        return _AugmentorPipeline(
            [
                augmentor * M if counter == ind_mul else augmentor * 1
                for counter, augmentor in enumerate(self._augmentor_list)
            ]
        )

    # def __matmul__(self, prob):
    #     return _AugmentorPipeline(
    #         [augmentor @ prob for augmentor in self._augmentor_list]
    #     )

    def __len__(self):
        return len(self._augmentor_list)

    def summary(self):
        """Print summary of this augmentor pipeline."""
        input_N = (1, None)
        input_n = (1, None)
        input_c = (1, None)
        for counter, augmentor in enumerate(self._augmentor_list):
            input_N, input_n, input_c = augmentor._summary(
                header=(counter == 0),
                input_N=input_N,
                input_n=input_n,
                input_c=input_c,
            )

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._augmentor_list[index]
        elif isinstance(index, slice):
            return _AugmentorPipeline(self._augmentor_list[index])
        else:
            raise TypeError("Index must be an integer or a slice.")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if not isinstance(value, _Augmentor):
                raise TypeError(
                    "An element of an augmentor pipeline must be set by an "
                    "augmentor object."
                )
            self._augmentor_list[index] = value
        elif isinstance(index, slice):
            if not isinstance(value, _AugmentorPipeline):
                raise TypeError(
                    "A slice of an augmentor pipeline must be set by an "
                    "augmentor pipeline."
                )
            if len(value) != len(self[index]):
                raise ValueError(
                    "The length of index must be equal to the length of "
                    "augmentor pipeline to set."
                )
            self._augmentor_list[index] = value._augmentor_list
        else:
            raise TypeError("Index must be an integer or a slice.")
