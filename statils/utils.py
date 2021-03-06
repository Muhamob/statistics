import numpy as np
from scipy import stats
import math
from typing import Union
from matplotlib import axes
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut


def summary(ksi: np.ndarray,
            eta: np.ndarray,
            inplace: bool = True,
            *args,
            **kwargs) -> Union[axes.Axes, None]:
    r"""
    Shows the summary of dataset, i.e. number of features, shows plots of eta

    :param ksi: observed model factors
    :param eta: observed model output
    :param inplace: either return axes.Axes or None
    :param args: args
    :param kwargs: kwargs
    :return: Union[axes.Axes, None], axes.Axes is plot of observed model output
    """
    n_samples, n_features = ksi.shape
    print(f"Number of samples: {n_samples}")
    print(f"Number of features/dimensionality of observation: {n_features}")

    fig, ax = plt.subplots(nrows=1, figsize=(14, 5))
    ax.plot(eta, color='k')
    ax.set_xlabel("# of observation")
    ax.set_ylabel(r"$\eta$")
    ax.grid(alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if inplace:
        plt.plot()
        return None
    else:
        return ax


def get_correlation_matrix(observations: np.ndarray):
    r"""
    Construct correlation matrix where

    .. math::
        corr_{ij} = \sum_{k} \dfrac{(x_i^k - \bar{x_i})(x_j^k - \bar{x_j})}{\sigma_i \sigma_j}

    :param observations: observations with shape [n_samples, n_features]
    :return:
    """
    means = observations.mean(axis=0)  # output shape (n_features, )
    obs_centered = observations - means.reshape(1, -1)
    covariance = obs_centered.T @ obs_centered
    sigmas = np.diag(covariance) ** 0.5

    """
    sigma_i * sigma_j
    used some broadcasting tricks, have to be tested
    TODO: write tests on simple examples
    """
    denominator = sigmas.reshape(-1, 1) * sigmas.reshape(1, -1)
    return covariance / denominator


def pretty_print(x: np.ndarray, precision: int = 5, *prefix):
    arr_string = np.array2string(x, precision=precision)
    print(*prefix, arr_string)


def pairwise_corr_matrix_ci(corr_matrix: np.ndarray,
                            n_samples: int,
                            beta: float = 0.95,
                            print_: bool = True) -> np.ndarray:
    r"""
    Compute pairwise correlation matrix confidence intervals based on Fisher transform

    .. math::
        \tilde{z} = \dfrac{1}{2} ln(\dfrac{1+\tilde{r}_{ij}}{1-\tilde{r}_{ij}}) = arth \tilde{r}_{ij}

    :param corr_matrix: numpy array of shape [n_features, n_features]
    :param n_samples: number of samples
    :param beta: confidence interval quantile
    :param print_: either to print or not
    :return: confidence intervals matrix with shape [n_features, n_features, 2]
    """
    u_b = stats.norm.ppf(0.5 * (1 + beta))
    z1 = np.arctanh(corr_matrix) - 0.5 * corr_matrix / (n_samples - 1) - \
         u_b / math.sqrt(n_samples - 3)
    z2 = np.arctanh(corr_matrix) - 0.5 * corr_matrix / (n_samples - 1) + \
         u_b / math.sqrt(n_samples - 3)

    lower_bound = np.tanh(z1)
    upper_bound = np.tanh(z2)

    ci = np.dstack([lower_bound, upper_bound])

    if print_:
        for i, ci_1 in enumerate(ci):
            for j, ci_2 in enumerate(ci_1[i + 1:]):
                pretty_print(ci_2, 5,
                             f'r_{i+1}_{j + i + 2} =',
                             round(corr_matrix[i, j+i+1], 5),
                             'in')

    return ci


def partial_pairwise_corr_matrix_ci(partial_corr_matrix: np.ndarray,
                                    df: int,
                                    beta: float = 0.95,
                                    print_: bool = True) -> np.ndarray:
    r"""
    Compute partial pairwise correlation matrix confidence intervals
    based on Fisher transform

    I don't know how to name it in English, but in Russian it is "Частный коэффициент корреляции"

    .. math::
        \rho_{ij} = \dfrac{-Q_{ij}}{\sqrt{Q_{ii} Q_{jj}}}

    :math:`Q_{ij}` is a algebraic appendix of pairwise correlation matrix

    alias for pairwise_corr_matrix_ci. Just replace n_samples to n_samples-(n_features-2)

    :param corr_matrix: numpy array of shape [n_features, n_features]
    :param df: degrees of freedom = n_samples - n_features
    :param beta: confidence interval quantile
    :param print: either to print or not
    :return: confidence intervals matrix with shape [n_features, n_features, 2]
    """
    ci = pairwise_corr_matrix_ci(partial_corr_matrix, df+2, beta, print_)
    return ci


def minor(arr: np.ndarray, i: int, j: int) -> np.ndarray:
    # ith row, jth column removed
    return arr[np.array(list(range(i))+list(range(i+1,arr.shape[0])))[:,np.newaxis],
               np.array(list(range(j))+list(range(j+1,arr.shape[1])))]


def compute_partial_corr_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
    r"""
    .. math::
        \rho_{ij} = \dfrac{-Q_{ij}}{\sqrt{Q_{ii} Q_{jj}}},

    :math:`Q_{ij}` is a algebraic appendix of pairwise correlation matrix

    :param covariance_matrix: pairwise covariance matrix

    :return: partial covariance matrix
    """
    partial_covariance_matrix = np.zeros_like(covariance_matrix)
    for i, row in enumerate(covariance_matrix):
        for j, elem in enumerate(row):
            Q_ij = (-1) ** (i + j) * np.linalg.det(minor(covariance_matrix, i, j))
            Q_jj = (-1) ** (j + j) * np.linalg.det(minor(covariance_matrix, j, j))
            Q_ii = (-1) ** (i + i) * np.linalg.det(minor(covariance_matrix, i, i))
            partial_covariance_matrix[i, j] = (-1) * Q_ij / np.sqrt(Q_ii * Q_jj)
    return partial_covariance_matrix


def make_observation_matrix(x: np.ndarray, n_features: int) -> np.ndarray:

    if len(x.shape) == 2:
        if x.shape[1] == n_features-1:
            observation_matrix = np.hstack((np.ones([x.shape[0], 1]), x))
        elif x.shape[1] != n_features:
            print(x.shape, n_features)
            raise Exception("There must be some mistake")
        else:
            observation_matrix = x
    elif len(x.shape) == 1:
        observation_matrix = np.concatenate(([1, ], x)).reshape(1, -1)
    else:
        raise Exception(f"dims of points is {len(x.shape)}, but must be 1 or 2 - dim")

    return  observation_matrix


def linear_model_confidence_interval(x: np.ndarray,
                                     coefficients: np.ndarray,
                                     F_inv: np.ndarray,
                                     rss: float,
                                     n_samples: int,
                                     quantile: float = 0.95):
    r"""
    TODO: Check if it works properly with multiple points x

    :param x:
    :param coefficients:
    :param F_inv:
    :param rss:
    :param n_samples:
    :param quantile:
    :return:
    """
    alpha = (1 - quantile) / 2
    n_features = F_inv.shape[0]

    observation_matrix = make_observation_matrix(x, n_features)

    df = n_samples - n_features

    quantiles = stats.t.ppf((alpha, 1 - alpha), df)

    tmp = np.diag(observation_matrix @ (F_inv @ observation_matrix.T))
    multiplier = np.sqrt(rss * (1 + tmp)) / math.sqrt(df)

    return observation_matrix @ coefficients + quantiles.reshape(-1, 1) * multiplier


def count_inversions(arr: Union[np.ndarray, list, tuple]) -> int:
    inversions = 0
    for i, elem in enumerate(arr):
        inv = 0
        for j in arr[i:]:
            if j>elem: # or less, TODO: check it
                inv += 1
        inversions += inv

    return inversions


def get_Kolmogorov_delta(x: np.ndarray):
    n_samples = len(x) if isinstance(x, (list, tuple)) else x.shape[0]
    theta1_hat = np.mean(x)
    theta2_hat = np.sqrt((1 / (n_samples - 1)) * np.sum((x - theta1_hat) ** 2))

    delta_f = 1.0 / n_samples
    f2 = stats.norm.cdf(x, loc=theta1_hat, scale=theta2_hat)
    f1 = delta_f*np.linspace(1, n_samples, n_samples)

    # return math.sqrt(n_samples) * np.max(np.abs(f2 - f1))
    return np.max(np.abs(f2 - f1))


def cross_validation_loo(
        x: np.ndarray,
        y: np.ndarray):
    r"""
    TODO: cv must have model as a parameter, now it is only for linear regression

    :param x:

    :param y:
    :return:
    """
    loo = LeaveOneOut()
    residuals = []

    observation_matrix = make_observation_matrix(x, x.shape[1]+1)

    for train_index, test_index in loo.split(observation_matrix.T):
        X_train, X_test = observation_matrix[train_index], observation_matrix[test_index]
        y_train, y_test = y[train_index], y[test_index]
        beta = np.linalg.pinv(X_train) @ y_train
        rss = (y_test - X_test @ beta) ** 2
        residuals.append(rss)

    rss0 = np.sum((y - y.mean()) ** 2)
    rss_star = sum(residuals)
    r_start_sq = 1 - rss_star / rss0

    return r_start_sq[0, 0]


def get_two_regression_cmp_test_delta(
        rss0: float,
        rss1: float,
        n_samples: int,
        n_features0: int,
        n_features1: int):

    delta = (rss0 - rss1) / rss1 * (n_samples - n_features1) / (n_features1 - n_features0)
    return delta
