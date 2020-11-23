"""Econometric Methods.
We provide a variety of econometric methods used in data science.
"""
import sys

import numpy as np
import pandas as pd
import patsy
import temfpy.integration_methods
from estimagic.optimization.optimize import maximize


def _multinomial_processing(formula, data, cov_structure):
    r"""Construct the inputs for the multinomial probit function.

    Parameters
    ----------
    formula : str
              A patsy formula comprising the independent variable and the dependent variables.

    data : pd.DataFrame
           A pandas data frame with shape :math:`n_obs \times n_var + 1`.

    cov_structure : str
                    Available options are 'iid' or 'free'.

    Returns:
    --------
    y : np.array
        1d numpy array of shape n_obs with the observed choices.

    x : np.array
        2d numpy array of shape :math:'(n_obs, n_var)' including the independent variables.

    params_df : pd.Series
                Random starting values for the parameters.
    """

    if (cov_structure != "iid") and (cov_structure != "free"):
        sys.exit("cov_structure must either be iid or free")

    y, x = patsy.dmatrices(formula, data, return_type="dataframe")
    data = pd.concat([y, x], axis=1).dropna()
    y, x = patsy.dmatrices(formula, data, return_type="dataframe")

    n_var = len(x.columns)
    n_choices = len(np.unique(y.to_numpy()))

    bethas = np.random.rand(n_var * (n_choices - 1)) * 0.1

    if cov_structure == "iid":

        index_tuples = []
        var_names = list(x.columns)
        for choice in range(n_choices - 1):
            index_tuples += [
                ("choice_{}".format(choice), "betha_{}".format(name))
                for name in var_names
            ]

        start_params = bethas

    else:
        covariance = np.eye(n_choices - 1)
        cov = []
        for i in range(n_choices - 1):
            for j in range(n_choices - 1):
                if j <= i:
                    cov.append(covariance[i, j])

        cov = np.asarray(cov)

        index_tuples = []
        var_names = list(x.columns)
        for choice in range(n_choices - 1):
            index_tuples += [
                ("choice_{}".format(choice), "betha_{}".format(name))
                for name in var_names
            ]

        j = (n_choices) * (n_choices - 1) / 2
        index_tuples += [("covariance", i) for i in range(int(j))]

        start_params = np.concatenate((bethas, cov))

    params_sr = pd.Series(
        data=start_params, index=pd.MultiIndex.from_tuples(index_tuples), name="value"
    )

    y = y - y.min()

    return (
        y.to_numpy(dtype=np.int64).reshape(len(y)),
        x.to_numpy(dtype=np.float64),
        params_sr,
    )


def _multinomial_probit_loglikeobs(params, y, x, cov_structure, integration_method):
    r"""Individual log-likelihood of the multinomial probit model.

    .. math::

    Parameters
    ----------
    formula : str
              A patsy formula comprising the dependent variable and the independent variables.

    y : np.array
        1d numpy array of shape :math:'n_obs' with the observed choices

    x : np.array
        2d numpy array of shape :math:'(n_obs, n_var)' including the independent variables.

    cov_structure : str
                    Available options are 'iid' or 'free'.

    integration_method : str
                         Either 'mc_integration', 'smooth_mc_integration'
                         or 'gauss_integration'


    Returns:
    --------
        loglikeobs : np.array
                     1d numpy array of shape :math:'(n_obs)' with
                     the respective likelihood contribution.
    """

    if (cov_structure != "iid") and (cov_structure != "free"):
        sys.exit("cov_structure must either be iid or free")

    n_var = np.shape(x)[1]
    n_choices = len(np.unique(y))

    if cov_structure == "iid":
        cov = np.eye(n_choices - 1) + np.ones((n_choices - 1, n_choices - 1))

    else:
        covariance = params["value"]["covariance"].to_numpy()

        cov = np.zeros((n_choices - 1, n_choices - 1))

        a = 0
        for i in range(n_choices - 1):
            k = i + 1
            cov[i, : (i + 1)] = covariance[a : (a + k)]
            a += k

        for i in range(n_choices - 1):
            cov[i, (i + 1) :] = cov[(i + 1) :, i]

    bethas = np.zeros((n_var, n_choices))

    for i in range(n_choices - 1):
        bethas[:, i] = params["value"]["choice_{}".format(i)].to_numpy()

    u_prime = x.dot(bethas)

    if cov_structure == "gauss_integration":
        choice_prob_obs = temfpy.integration_methods.gauss_integration(u_prime, cov, y)
    else:
        choice_prob_obs = getattr(temfpy.integration_methods, integration_method)(
            u_prime, cov, y
        )

    choice_prob_obs.clip(min=1e-250)

    loglikeobs = np.log(choice_prob_obs)

    return loglikeobs


def _multinomial_probit_loglike(
    params, y, x, cov_structure, integration_method="gauss_integration"
):
    r"""log-likelihood of the multinomial probit model.

    Parameters
    ----------
    formula : str
              A patsy formula comprising the dependent variable and the independent variables.

    y : np.array
        1d numpy array of shape :math:`n_obs` with the observed choices

    x : np.array
        2d numpy array of shape :math:`(n_obs, nvar)` including the independent variables.

    cov_structure : str
                    Available options are 'iid' or 'free'.

    integration_method : str
                         Either 'mc_integration', 'smooth_mc_integration'
                         or 'gauss_integration'

    Returns:
    --------
        loglike : float
                  The value of the log-likelihood function evaluated at the given parameters.
    """

    return _multinomial_probit_loglikeobs(
        params, y, x, cov_structure, integration_method
    ).sum()


def multinomial_probit(formula, data, cov_structure, integration_method, algorithm):
    r"""Multinomial probit model.

    .. math::
        i &= 1, \dots, n \\
        j &= 1, \dots, m \\
        \beta_j, X_i &\in \mathbb{R}^{k} \\
        Y_i^{*j} &= X_i^T \beta_j + \varepsilon_j \\
        Y_i &= \underset{j}{\mathrm{argmax}} \{Y_i^{*j} \mid j = 1, \dots, m \}


    Parameters
    ----------
    formula : str
              A patsy formula comprising the dependent and the independent variables.

    data : pandas.DataFrame
           A pandas data frame with shape :math:`(n, k+1)`
           including the dependent variable and the independent variables.

    cov_structure : str
                    Available options are 'iid' or 'free'.

    integration_method : str
                         Either 'mc_integration', 'smc_integration'
                         or 'gauss_integration'

    algorithm : str
                Available options are 'scipy_lbfgsb' or 'scipy_slsqp'


    Returns
    -------
    result: dic
                 Information of the optimization.


    Notes
    -----
        The function fits a multinomial probit model to discrete choice
        data via maximum likelihood estimation.


    References
    ----------
    Train, Kenneth E. Discrete choice methods with simulation.
    Cambridge university press, 2009.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> import temfpy.econometrics as tpe
    >>>
    >>> data = sm.datasets.spector.load_pandas().data
    >>> f = 'GRADE ~ GPA + TUCE + PSI'
    >>> np.random.seed(123)
    >>> data['GRADE'] = np.random.randint(4, size=(32,))
    >>> cov = 'iid'
    >>> integr = 'gauss_integration'
    >>> algo = 'scipy_lbfgsb'
    >>> solution = tpe.multinomial_probit(f, data, cov, integr , algo)
    """

    if (cov_structure != "iid") and (cov_structure != "free"):
        sys.exit("cov_structure must either be iid or free")

    y, x, params = _multinomial_processing(formula, data, cov_structure)

    params_df = pd.DataFrame(params, columns=["value"])

    if cov_structure == "iid":
        constraints = []

    else:
        constraints = [
            {"loc": "covariance", "type": "covariance"},
            {"loc": ("covariance", 0), "type": "fixed", "value": 1.0},
        ]

    result = maximize(
        _multinomial_probit_loglike,
        params_df,
        algorithm,
        criterion_kwargs={
            "y": y,
            "x": x,
            "cov_structure": cov_structure,
            "integration_method": integration_method,
        },
        constraints=constraints,
    )

    return result
