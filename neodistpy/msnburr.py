import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable
from typing import Optional, Tuple
from pymc.distributions.dist_math import check_parameters

def logp(y: TensorVariable, 
         mu: TensorVariable, 
         sigma: TensorVariable, 
         alpha: TensorVariable,
         **kwargs):
    """
    Compute the log of the probability density function of the MSNBurr distribution

    Parameters
    ----------
    y       : Random variable where the log-PDF is evaluated
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)

    Returns
    -------
        The log of the probability density evaluated at y
    """
    
    omega = (1 + (1 / alpha))**(alpha + 1) / np.sqrt(2 * np.pi)
    epart = -pt.log(alpha) - (omega / sigma * (y - mu))
    logpdf = pt.log(omega) - pt.log(sigma) - (omega / sigma * (y - mu)) - ((alpha + 1) * pt.log1pexp(epart))
    logpdf = pt.switch(
        pt.eq(y, -np.inf),
        -np.inf,
        logpdf
    )
    return check_parameters(
        logpdf,
        alpha > 0,
        sigma > 0,
        msg=f"alpha and sigma must be more than 0"
    )

def logcdf(y: TensorVariable, 
           mu: TensorVariable, 
           sigma: TensorVariable, 
           alpha: TensorVariable, 
           **kwargs):
    """
    Compute the log of the cumulative density function of the MSNBurr distribution

    Parameters
    ----------
    y       : Random variable where the log-CDF is evaluated
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)

    Returns
    -------
        The log of the cumulative density evaluated at y
    """
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    epart = -pt.log(alpha) - (omega/sigma*(y-mu))
    logcdf= -alpha*pt.log1pexp(epart)
    logcdf = pt.switch(
        pt.eq(y, -np.inf), 
        -np.inf,
        pt.switch(pt.eq(y, np.inf), 0.0, logcdf)
    )
    return check_parameters(
        logcdf,
        alpha > 0,
        sigma > 0,
        msg=f"alpha and sigma must more than 0",
    )

def random(
      mu: np.ndarray | float,
      sigma: np.ndarray | float,
      alpha: np.ndarray | float,
      rng = np.random.default_rng(),
      size: Optional[Tuple[int]]=None):
    """
    Generate random samples from the MSNBurr distribution

    Parameters
    ----------
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)
    rng     : Random number generator, defaults to np.random.default_rng()
    size    : Desired shape of the random samples

    Returns
    -------
        Random samples from the MSNBurr distribution
    """
    
    if np.isnan(mu).any() or np.isnan(sigma).any() or np.isnan(alpha).any():
        raise ValueError("NaN value detected in parameters")
    
    # Validasi parameter
    if not np.all(sigma > 0):
        raise ValueError("sigma must be positive")
    if not np.all(alpha > 0):
        raise ValueError("alpha must be positive")
    
    u = rng.uniform(low=0, high=1, size=size)
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    random = mu - sigma/omega*(np.log(alpha)+np.log((u**(-1/alpha))-1))
    return np.asarray(random)

def quantile(p, mu, sigma, alpha):
    """
    Compute the quantile (inverse cumulative distribution function) of the MSNBurr distribution

    Parameters
    ----------
    p       : Probability value(s) between 0 and 1
    mu      : Location parameter
    sigma   : Scale parameter (must be positive)
    alpha   : Shape parameter (must be positive)

    Returns
    -------
        The quantile corresponding to the given probability.
    """
    
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    q = mu - sigma/omega*(np.log(alpha)+np.log((p**(-1/alpha))-1))
    q = pt.switch(pt.eq(p, 0), -np.inf, q)
    q = pt.switch(pt.eq(p, 1), np.inf, q)
    q = pt.switch(p < 0, np.nan, q)
    q = pt.switch(p > 1, np.nan, q)
    return check_parameters(
        q,
        sigma > 0,
        alpha > 0,
        msg=f"sigma and alpha must be more than 0"
    )

class msnburr:
    """
    MSNBurr Custom Distribution for PyMC
    
    This class implements a PyMC-compatible version of the MSNBurr distribution using `CustomDist`. 

    Methods
    -------
    __new__(name, mu, sigma, alpha, observed=None, **kwargs)
        Creates a new instance of the MSNBurr distribution in a PyMC model.

    dist(mu, sigma, alpha, **kwargs)
        Distribution constructor for symbolic usage inside PyMC.

    icdf(p, mu, sigma, alpha)
        Static method to compute the quantile (inverse CDF) of the MSNBurr distribution.

    random(mu, sigma, alpha, size=None)
        Static method to generate random samples from the MSNBurr distribution.

    Examples
    --------
    >>> with pm.Model():
    ...     mu = pm.Normal('mu', 0, 10)
    ...     sigma = pm.HalfCauchy('sigma', 10)
    ...     alpha = pm.Gamma('alpha', 2, 0.1)
    ...
    ...     y = msnburr(
    ...         'MSNBurr',
    ...         mu=mu,
    ...         sigma=sigma,
    ...         alpha=alpha,
    ...         observed=[30, 35, 73]
    ...     )
    
    """
    def __new__(self, name:str, mu, sigma, alpha, observed=None, **kwargs):
        return pm.CustomDist(
            name,
            mu, sigma, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
            observed=observed,
            **kwargs
        )

    @classmethod
    def dist(cls, mu, sigma, alpha, **kwargs):
        return pm.CustomDist.dist(
            mu, sigma, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
        )
        
    @staticmethod
    def icdf(p, mu, sigma, alpha):
        return quantile(p, mu, sigma, alpha)
    
    @staticmethod
    def random(mu, sigma, alpha, size=None):
        return random(mu, sigma, alpha, size=size)
