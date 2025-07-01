import pymc as pm
import numpy as np
import pytensor.tensor as pt
import scipy

from pytensor.tensor import TensorVariable
from typing import Optional, Tuple
from pymc.distributions.dist_math import check_parameters

# Komponen 1: Fungsi log PDF
def logp(y: TensorVariable,
         mu: TensorVariable,
         sigma: TensorVariable,
         nu: TensorVariable,
         alpha: TensorVariable,
         **kwargs):
    """
    Compute the log of the probability density function of the FSST distribution

    Parameters
    ----------
    y       : Random variable where the log-PDF is evaluated
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)
    nu      : Tail heaviness parameter (must be positive)

    Returns
    -------
        The log of the probability density evaluated at y
    """
    
    loglik1a = pm.logp(pm.StudentT.dist(nu), (alpha * (y - mu) / sigma))
    loglik2a = pm.logp(pm.StudentT.dist(nu), ((y - mu) / (sigma * alpha)))
    loglika = pt.switch(y < mu, loglik1a, loglik2a)
    loglika += pt.log(2 * alpha / (1 + alpha**2)) - pt.log(sigma)

    loglik1b = pm.logp(pm.Normal.dist(0, 1), (alpha * (y - mu) / sigma))
    loglik2b = pm.logp(pm.Normal.dist(0, 1), ((y - mu) / (sigma * alpha)))
    loglikb = pt.switch(y < mu, loglik1b, loglik2b)
    loglikb += pt.log(2 * alpha / (1 + alpha**2)) - pt.log(sigma)

    threshold_nu = 1e6
    logp = pt.switch(
        pt.eq(y, -np.inf),
        -np.inf,
        pt.switch(nu < threshold_nu, loglika, loglikb)
    )
    return check_parameters(
        logp,
        sigma > 0,
        nu > 0,
        alpha > 0,
        msg=f"sigma, nu, alpha must be more than 0"
    )

# Komponen 2: Fungsi log CDF
def logcdf(y: TensorVariable,
           mu: TensorVariable,
           sigma: TensorVariable,
           nu: TensorVariable,
           alpha: TensorVariable,
           **kwargs):
    """
    Compute the log of the cumulative density function of the FSST distribution

    Parameters
    ----------
    y       : Random variable where the log-CDF is evaluated
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)
    nu      : Tail heaviness parameter (must be positive)

    Returns
    -------
        The log of the cumulative density evaluated at y
    """
    
    cdf1 = (2 / (1 + alpha**2)) * pt.exp(pm.logcdf(pm.StudentT.dist(nu), (alpha * (y - mu) / sigma)))
    cdf2 = (1 / (1 + alpha**2)) * (1 + 2 * alpha**2 *
                                  (pt.exp(pm.logcdf(pm.StudentT.dist(nu), (y - mu) / (sigma * alpha))) - 0.5))

    cdf = pt.switch(y < mu, cdf1, cdf2)
    logcdf = pt.log(cdf)
    logcdf = pt.switch(
        pt.eq(y, -np.inf), 
        -np.inf,
        pt.switch(pt.eq(y, np.inf), 0.0, logcdf)
    )
    return check_parameters(
        logcdf,
        sigma > 0,
        nu > 0,
        alpha > 0,
        msg=f"sigma, nu, alpha must be more than 0"
    )

# Komponen 3: Fungsi random
def random(
      mu: np.ndarray | float,
      sigma: np.ndarray | float,
      nu: np.ndarray | float,
      alpha: np.ndarray | float,
      rng = np.random.default_rng(),
      size: Optional[Tuple[int]]=None):
    """
    Generate random samples from the FSST distribution

    Parameters
    ----------
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)
    nu      : Tail heaviness parameter (must be positive)
    rng     : Random number generator, defaults to np.random.default_rng()
    size    : Desired shape of the random samples

    Returns
    -------
        Random samples from the FSST distribution
    """
    
    if np.isnan(mu).any() or np.isnan(sigma).any() or np.isnan(nu).any() or np.isnan(alpha).any():
        raise ValueError("NaN value detected in parameters")

    if not np.all(sigma > 0):
        raise ValueError("sigma must be positive")
    if not np.all(nu > 0):
        raise ValueError("nu must be positive")
    if not np.all(alpha > 0):
        raise ValueError("alpha must be positive")

    u = rng.uniform(low=0, high=1, size=size)

    q1 = mu + (sigma / alpha) * scipy.stats.t.ppf(u * (1 + alpha**2) / 2, df=nu)
    q2 = mu + (sigma * alpha) * scipy.stats.t.ppf(((u * (1 + alpha**2) - 1) / (2 * alpha**2)) + 0.5, df=nu)

    q = np.where(u < (1 / (1 + alpha**2)), q1, q2)

    return np.asarray(q)

# Komponen Tambahan: Fungsi quantile(inverse cdf)
def quantile(p: TensorVariable,
             mu: TensorVariable,
             sigma: TensorVariable,
             nu: TensorVariable,
             alpha: TensorVariable):
    """
    Compute the quantile (inverse cumulative distribution function) of the MSNBurr distribution

    Parameters
    ----------
    p       : Probability value(s) between 0 and 1
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)
    nu      : Tail heaviness parameter (must be positive)

    Returns
    -------
        The quantile corresponding to the given probability.
    """
    
    q1 = mu + (sigma / alpha) * pm.icdf(pm.StudentT.dist(nu), p * (1 + alpha**2) / 2)
    q2 = mu + (sigma * alpha) * pm.icdf(pm.StudentT.dist(nu), ((p * (1 + alpha**2) - 1) / (2 * alpha**2)) + 0.5)

    q = pt.switch(p < (1 / (1 + alpha**2)), q1, q2)

    q = pt.switch(pt.eq(p, 0), -np.inf, q)
    q = pt.switch(pt.eq(p, 1), np.inf, q)
    q = pt.switch(p < 0, np.nan, q)
    q = pt.switch(p > 1, np.nan, q)

    return check_parameters(
        q,
        sigma > 0,
        nu > 0,
        alpha > 0,
        msg=f"sigma, nu, alpha must be more than 0"
    )

class fsst:
    """
    FSST Custom Distribution for PyMC
    
    This class implements a PyMC-compatible version of the FSST distribution using `CustomDist`. 

    Methods
    -------
    __new__(name, mu, sigma, alpha, observed=None, **kwargs)
        Creates a new instance of the MSNBurr distribution in a PyMC model.

    dist(mu, sigma, alpha, **kwargs)
        Distribution constructor for symbolic usage inside PyMC.

    icdf(p, mu, sigma, alpha)
        Static method to compute the quantile (inverse CDF) of the FSST distribution.

    random(mu, sigma, alpha, size=None)
        Static method to generate random samples from the FSST distribution.

    Examples
    --------
    >>> with pm.Model():
    ...     mu = pm.Normal('mu', 0, 10)
    ...     sigma = pm.HalfCauchy('sigma', 10)
    ...     alpha = pm.Gamma('alpha', 2, 0.1)
    ...
    ...     y = fsst(
    ...         'FSST',
    ...         mu=mu,
    ...         sigma=sigma,
    ...         alpha=alpha,
    ...         observed=[30, 35, 73]
    ...     )
    
    """
    def __new__(self, name:str, mu, sigma, nu, alpha, observed=None, **kwargs):
        return pm.CustomDist(
            name,
            mu, sigma, nu, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
            observed=observed,
            **kwargs
        )

    @classmethod
    def dist(cls, mu, sigma, nu, alpha, **kwargs):
        return pm.CustomDist.dist(
            mu, sigma, nu, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
            **kwargs
        )
        

    @staticmethod
    def icdf(p, mu, sigma, nu, alpha):
        return quantile(p, mu, sigma, nu, alpha)
    
    @staticmethod
    def random(mu, sigma, nu, alpha, size=None):
        return random(mu, sigma, nu, alpha, size=size)
