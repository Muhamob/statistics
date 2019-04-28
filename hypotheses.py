import numpy as np
from scipy import stats
import math
from sklearn.model_selection import LeaveOneOut
import tqdm
from typing import Union
import utils


def corr_coef_significance_test(correlation_matrix: np.ndarray,
                                df: int,
                                alpha: float = 0.1,
                                print_: bool = True) -> Union[np.ndarray, None]:
    """
    H_0 : r_ij = 0
    H_1 : not H_0
    criterion:
        |r_ij| >= th(\dfrac{u_{\alpha}}{\sqrt{df-3}})
    :param correlation_matrix:
    :param df: in case of simple(Pearson coef) df = n_samples,
    if partitial correlation coefficient => n_samples - n_features + 2
    :param alpha: significance value
    :param print_: either to print or not threshold and u_a
    :return:
    """
    u_a = stats.norm.ppf(1-0.5*alpha)
    threshold = math.tanh(u_a/math.sqrt(df-3))
    passed = np.abs(correlation_matrix) < threshold

    if print_:
        print('u_a =', u_a)
        print('threshold =', threshold, '\n')
    else:
        return passed


def pearson_test(observations: np.ndarray,
                 alpha: float = 0.05) -> tuple:
    """
    Standart Pearson test from scipy stats
    Reccomended usage if n_samples > 500

    if p-value more than alpha then we may consider x_i and x_j not correlated

    :param observations: shape = [n_samples, n_features
    :param alpha:
    :return:
    """
    n_features = observations.shape[1]
    passed = np.empty([n_features, n_features], dtype=np.bool)
    p = np.empty([n_features, n_features])
    for i in range(n_features):
        for j in range(n_features):
            x_i = observations[:, i]
            x_j = observations[:, j]
            p_value = stats.pearsonr(x_i, x_j)[1]
            p[i, j] = p_value
            passed_ = True if p_value>alpha else False
            passed[i, j] = passed_

    return p, passed


def linear_regression_significance_test(coefficients: np.ndarray,
                                        F_inv: np.ndarray,
                                        rss: float,
                                        n_samples: int,
                                        alpha: float = 0.05,
                                        print_: bool = True):
    """
    H_0 : \beta_i = 0
    H_1 : \beta_i \ne 0

    If p-value < alpha then we accept alternative hypothesys

    :param coefficients:
    :param F_inv:
    :param rss:
    :param n_samples:
    :param alpha:
    :param print_:
    :return:
    """
    n_features = F_inv.shape[0]
    diag = np.diag(F_inv).reshape(-1, 1)
    delta = coefficients * math.sqrt(n_samples - n_features) / np.sqrt(diag * rss)
    degrees_of_freedom = n_samples - n_features
    p_values = stats.t.sf(np.abs(delta), degrees_of_freedom)

    if print_:
        utils.pretty_print(np.squeeze(p_values), 4, "p-значения теста на значимость коэффициентов ")
        utils.pretty_print(np.squeeze(p_values)>alpha, None, "Прохождение теста на значимость коэффициентов")
    else:
        return p_values


def deteremination_coef_significance_test(n_samples: int,
                                          n_features: int,
                                          r_2: float,
                                          alpha: float = 0.05,
                                          print_: bool = True):
    """

    :param n_samples:
    :param n_features:
    :param r_2:
    :param alpha:
    :param print_:
    :return:
    """
    delta = ((n_samples-n_features)/(n_features-1)) * (r_2 / (1-r_2))
    p_value = stats.f.sf(delta, n_features-1, n_samples-n_features)

    if print_:
        print(f"p-значение для теста на значимость коэффициента детерминации = {p_value}")
        hyp = 'H_0 : R=0' if p_value > alpha else 'H_1 : R^2 != 0'
        print(f"можно считать, что гипотеза {hyp} верна")
    else:
        return p_value


def randomness_test(err: Union[np.ndarray, list, tuple],
                    alpha: float = 0.05,
                    print_: bool = True):
    """
    H_0: err had drawn from random distribution
    H_1: not H_0

    If p-value < alpha then we accept alternative hypothesys

    :param err:
    :param alpha:
    :param print_:
    :return:
    """
    inversions = utils.count_inversions(err)
    n_samples = err.shape[0]
    delta = (inversions-n_samples*(n_samples-1)/4) / np.sqrt((n_samples**3)/36)
    p_value = stats.norm.sf(delta)
    if print_:
        hyp = 'H_0 : наблюдения случайные' if p_value > alpha else 'H_1 : наблюдения не случайные'
        print(f"p-значение для теста на случайность выборки = {p_value}")
        print(f"можно считать, что гипотеза {hyp} верна")
    else:
        return p_value
