import numpy as np
from typing import Union


__all__ = ['DEFAULT_N_FEATURES', 'DEFAULT_N_SAMPLES', 'DEFAULT_SIGMA', 'DEFAUL_LINEAR_FORM',
           'dataset_from_normal']



DEFAULT_N_FEATURES = 6
DEFAULT_N_SAMPLES = 50
DEFAULT_SIGMA = 1.5
DEFAUL_LINEAR_FORM = (2., 3., -2., 1., 1., -1.)


def dataset_from_normal(n_samples: int = DEFAULT_N_SAMPLES,
                        n_features: int = DEFAULT_N_FEATURES,
                        sigma: float = DEFAULT_SIGMA,
                        linear_form: Union[list, tuple, np.ndarray] = DEFAUL_LINEAR_FORM,
                        random_seed: int = 42) -> dict:
    """
    Generate normal distributed variable with uniform distributed observations
    :param n_samples: number of observations
    :param n_features: dimensionality of observation
    :param sigma: standart deviation of generated variable
    :param linear_form: array of coefficients in linear form
    :param random_seed: random seed used to initialize in numpy random generator
    :return:
    """
    np.random.seed(random_seed)

    # ksi ~ U[-1, 1]
    ksi = 2 * np.random.rand(n_samples, n_features-1) - 1

    # eta ~ N(L[ksi], sigma), where L is linear_form
    linear_form_ = np.array(linear_form, dtype=np.float32).reshape(-1, 1)
    # Make observation matrix which is || 1 x_{i0} x_{i1} ... x_{i(n_features)} ||_{i=1}^{n_samples}
    observation_matrix = np.hstack([np.ones_like(ksi[:, 0]).reshape(-1, 1), ksi])
    y_true = observation_matrix@linear_form_
    y_noise = np.random.normal(loc=0, scale=sigma, size=n_samples).reshape(-1, 1)
    eta = y_true + y_noise

    return dict(
        ksi=ksi,
        observation_matrix=observation_matrix,
        eta=eta,
        linear_form=linear_form_,
        random_seed=random_seed
    )


def dataset_from_normal_given_points(
        ksi: np.ndarray,
        n_samples: int = DEFAULT_N_SAMPLES,
        n_features: int = DEFAULT_N_FEATURES,
        sigma: float = DEFAULT_SIGMA,
        linear_form: Union[list, tuple, np.ndarray] = DEFAUL_LINEAR_FORM,
        random_seed: int = 43) -> dict:
    """
    Generate normal distributed variable with uniform distributed observations
    :param ksi: points where to sample
    :param n_samples: number of observations
    :param n_features: dimensionality of observation
    :param sigma: standart deviation of generated variable
    :param linear_form: array of coefficients in linear form
    :return:
    """

    # eta ~ N(L[ksi], sigma), where L is linear_form
    linear_form_ = np.array(linear_form, dtype=np.float32).reshape(-1, 1)
    # Make observation matrix which is || 1 x_{i0} x_{i1} ... x_{i(n_features)} ||_{i=1}^{n_samples}
    observation_matrix = np.hstack([np.ones_like(ksi[:, 0]).reshape(-1, 1), ksi])
    y_true = observation_matrix@linear_form_
    y_noise = np.random.normal(loc=0, scale=sigma, size=n_samples).reshape(-1, 1)
    eta = y_true + y_noise

    return dict(
        ksi=ksi,
        observation_matrix=observation_matrix,
        eta=eta,
        linear_form=linear_form_,
        random_seed=random_seed
    )
