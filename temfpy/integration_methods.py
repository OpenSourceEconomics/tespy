"""Integration Methods.
We provide a variety of numerical integration methods.
"""
import numpy as np
from scipy.stats import norm


def mc_integration(u_prime, cov, y, n_draws=None):
    r"""Calculate probit choice probabilities with Monte-Carlo Integration.

    Parameters
    ----------
    u_prime : np.array
              2d array of shape  comprising the deterministic part of utilities

    cov : np.array
          2d array of shape

    y : np.array
        1d array of shape with the observed choices.

    n_draws : int
              Number of draws for Monte-Carlo integration. If :math:`None`
              is specified the value is set to the product of the number
              of choices and 2000.

    Returns:
    --------
    choice_prob_obs : np.array
                      1d array of shape  comprising the choice probabilities.
    """
    n_obs = np.shape(u_prime)[0]
    n_choices = np.shape(u_prime)[1]

    if n_draws is None:
        n_draws = n_choices * 2000

    base_error = np.random.normal(size=(n_obs * n_draws, (n_choices - 1)))
    chol = np.linalg.cholesky(cov)
    errors = chol.dot(base_error.T)
    errors = errors.T.reshape(n_obs, n_draws, (n_choices - 1))
    extra_column_errors = np.zeros((n_obs, n_draws, 1))
    errors = np.append(errors, extra_column_errors, axis=2)

    u = u_prime.reshape(n_obs, 1, n_choices) + errors

    index_choices = np.argmax(u, axis=2)

    choices = np.zeros((n_obs, n_draws, n_choices))
    for i in range(n_obs):
        for j in range(len(index_choices[1])):
            choices[i, j, int(index_choices[i, j])] = 1

    choice_probs = np.average(choices, axis=1)

    choice_prob_obs = choice_probs[range(len(y)), y]

    return choice_prob_obs


def smc_integration(u_prime, cov, y, tau=1, n_draws=None, max_bound=1e250):
    r"""Calculate probit choice probabilities with smooth Monte-Carlo Integration.

    Parameters
    ----------
    u_prime : np.array
              2d array of shape :math:'(n_obs, n_choices)' comprising
              the deterministic part of utilities

    cov : np.array
          2d array of shape :math:'(n_choices - 1, n_choices - 1)'.

    y : np.array
        1d array of shape :math:'(n_obs)' with the observed choices.

    tau : int
          corresponds to the smoothing factor. It should be a positive number.
          For values close to zero the estimated smooth choice  probabilities lie
          in a wider interval which becomes symmetrically smaller for larger values of tau.
          The default is 1.

    n_draws : int
              Number of draws for smooth Monte-Carlo integration.

    max_bound : float
                Positive number indicating the maximum number that is allowed
                during the computation of the choice probabilities.
                The default is :math:`1e250`.

    Returns:
    --------
    choice_prob_obs : np.array
                      1d array of shape :math:'(n_obs)' comprising the choice
                      probabilities for the chosen alternative for each individual.
    """
    n_obs = np.shape(u_prime)[0]
    n_choices = np.shape(u_prime)[1]

    if n_draws is None:
        n_draws = n_choices * 2000

    base_error = np.random.normal(size=(n_obs * n_draws, (n_choices - 1)))
    chol = np.linalg.cholesky(cov)
    errors = chol.dot(base_error.T)
    errors = errors.T.reshape(n_obs, n_draws, (n_choices - 1))
    extra_column_errors = np.zeros((n_obs, n_draws, 1))
    errors = np.append(errors, extra_column_errors, axis=2)

    u = u_prime.reshape(n_obs, 1, n_choices) + errors

    u_max = np.max(u, axis=2)
    val_exp = np.clip(
        np.exp((u - u_max.reshape(n_obs, n_draws, 1)) / tau), 0, max_bound
    )
    smooth_dummy = val_exp / val_exp.sum(axis=2).reshape(n_obs, n_draws, 1)
    choice_probs = np.average(smooth_dummy, axis=1)

    choice_prob_obs = choice_probs[range(len(y)), y]

    return choice_prob_obs


def gauss_integration(u_prime, cov, y, degrees=25):
    r"""Calculate probit choice probabilities with Gauss-Laguerre Integration.

    Parameters
    ----------
    u_prime : np.array
              2d array of shape :math:'(n_obs, n_choices)' comprising
              the deterministic part of utilities

    cov : np.array
          2d array of shape :math:'(n_choices - 1, n_choices - 1)'

    y : np.array
        1d array of shape :math'(n_obs)' with the observed choices.

    degrees : int
              Order of the polynomial used for the approximation.

    Returns:
    --------
    choice_prob_obs : np.array
                      1d array of shape (n_obs) comprising the choice
                      probabilities for the chosen alternative for each individual.
    """
    n_obs = np.shape(u_prime)[0]
    n_choices = np.shape(u_prime)[1]

    x_k, w_k = np.polynomial.laguerre.laggauss(int(degrees))
    fraction = np.divide(w_k, np.sqrt(x_k))
    sqrt_x_k = np.sqrt(2 * x_k)

    u_choice = np.choose(y, u_prime.T)
    dif_array = u_prime - u_choice.reshape(n_obs, 1)

    u_dif = np.zeros((n_obs, n_choices - 1))
    for counter, choice in enumerate(y):
        u_dif[counter] = np.delete(dif_array[counter, :], choice)

    phi_neg_choice = norm.cdf(
        -sqrt_x_k.reshape(1, degrees, 1) - u_dif.reshape(n_obs, 1, n_choices - 1)
    )
    phi_pos_choice = norm.cdf(
        sqrt_x_k.reshape(1, degrees, 1) - u_dif.reshape(n_obs, 1, n_choices - 1)
    )

    choice_probs_interm = phi_neg_choice.prod(axis=2) + phi_pos_choice.prod(axis=2)
    choice_prob_obs = (1 / (2 * np.sqrt(np.pi))) * np.dot(choice_probs_interm, fraction)

    return choice_prob_obs
