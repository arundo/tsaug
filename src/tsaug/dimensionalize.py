"""
decorator of dimensionalize X and Y input
"""

from functools import wraps


def dimensionalize(f):
    """
    Decorator of augmentor that converts X into 3D matrix and Y into 2D matrix
    if it exists, and converts output matrices back to original shapes.
    """

    @wraps(f)
    def g(X, Y=None, *args, **kwargs):

        if X.ndim == 1:
            Xndim = 1
            n = len(X)
            X = X.reshape((1, n, 1))
        elif X.ndim == 2:
            Xndim = 2
            N, n = X.shape
            X = X.reshape((N, n, 1))
        elif X.ndim == 3:
            Xndim = 3
        else:
            raise ValueError("Wrong shape of X")

        N, n, c = X.shape

        if Y is None:
            pass
        else:
            if Y.ndim == 1:
                Yndim = 1
                n = len(Y)
                Y = Y.reshape((1, n, 1))
            elif Y.ndim == 2:
                Yndim = 2
                N, n = Y.shape
                Y = Y.reshape((N, n, 1))
            elif Y.ndim == 3:
                Yndim = 3
            else:
                raise ValueError("Wrong shape of Y")

            if Y.shape[:2] != (N, n):
                raise ValueError("Inconsistent shape between X and Y")

        returns = f(X, Y, *args, **kwargs)
        if isinstance(returns, tuple):
            X_aug = returns[0]
            Y_aug = returns[1]
        else:
            X_aug = returns

        if Xndim == 1:
            if X_aug.shape[0] == 1:
                X_aug = X_aug.flatten()
            else:
                X_aug = X_aug[:, :, 0]
        elif Xndim == 2:
            X_aug = X_aug[:, :, 0]

        if Y is None:
            return X_aug
        else:
            if Yndim == 1:
                if Y_aug.shape[0] == 1:
                    Y_aug = Y_aug.flatten()
                else:
                    Y_aug = Y_aug[:, :, 0]
            elif Yndim == 2:
                Y_aug = Y_aug[:, :, 0]
            return X_aug, Y_aug

    return g
