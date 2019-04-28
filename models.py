import numpy as np
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, normalize: bool = False):
        self.normalize = normalize
        self.coefs = None
        self.F_inv = None

    def fit(self, x: np.ndarray, y: np.ndarray):

        observation_matrix = np.hstack([np.ones_like(x[:, 0]).reshape(-1, 1), x])
        F = observation_matrix.T @ observation_matrix
        self.F_inv = np.linalg.inv(F)

        self.coefs = self.F_inv @ observation_matrix.T @ y

        self._summarize(x, y)

        return self.coefs

    def predict(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        assert self.coefs is not None, "You must fit your model first"

        observation_matrix = np.concatenate(([1, ], x))
        observation_matrix = observation_matrix.reshape(1, -1)
        y = observation_matrix @ self.coefs
        return y

    def _summarize(self, x: np.ndarray, y: np.ndarray):
        observation_matrix = np.hstack([np.ones_like(x[:, 0]).reshape(-1, 1), x])

        self.y_estimated = observation_matrix@self.coefs
        self.errors = y - self.y_estimated
        self.rss = (self.errors.T@self.errors)[0, 0]
        self.rss0 = np.sum((y-np.mean(y))**2)
        self.R2 = 1 - self.rss/self.rss0  # Determination coefficient
