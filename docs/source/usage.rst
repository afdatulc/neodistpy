Usage
=====

This section provides guidance on how to utilize the neodistpy package with 
PyMC and Bambi. Additionally, it demonstrates how to manually compute the PDF, 
CDF, and quantile using symbolic expressions from PyTensor.

Manual Evaluation of PDF, CDF, Quantile
---------------------------------------

The following example illustrates how to manually evaluate the log PDF, CDF, 
and quantile for the `fsst` distribution using PyTensor:

.. code-block:: python

    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    from neonormal import fsst

    mu = pt.scalar('mu')
    sigma = pt.scalar('sigma')
    nu = pt.scalar('nu')
    alpha = pt.scalar('alpha')
    x = pt.scalar('x')
    p = pt.scalar('p')

    rv = fsst.dist(mu=mu, sigma=sigma, nu=nu, alpha=alpha)

    logp_expr = pm.logp(rv, x)
    pdf = np.exp(logp_expr.eval({mu: 0.0, sigma: 1.0, nu: 5, alpha: 2, x: 0.5}))

    logcdf_expr = pm.logcdf(rv, x)
    cdf = np.exp(logcdf_expr.eval({mu: 0.0, sigma: 1.0, nu: 5, alpha: 2, x: 0.5}))

    quantile_expr = fsst.icdf(p, mu, sigma, nu, alpha)
    quantile = quantile_expr.eval({p: 0.9, mu: 0.0, sigma: 1.0, nu: 5, alpha: 2})

    print("PDF        :", pdf)
    print("CDF        :", cdf)
    print("Quantile   :", quantile)

Compiled Quantile Function
--------------------------

To direct evaluation, distribution functions can be compiled using `pm.compile` for 
improved efficiency, especially within iterative workflows:

.. code-block:: python

    rv_logp = pm.logp(rv, x)
    rv_logp_fn = pm.compile([x, mu, sigma, nu, alpha], rv_logp)
    logpdf = rv_logp_fn(x=0.5, mu=0, sigma=1, nu=5, alpha=2)
    print("PDF :", logpdf)

    rv_quantile = fsst.icdf(p, mu, sigma, nu, alpha)
    rv_quantile_fn = pm.compile([p, mu, sigma, nu, alpha], rv_quantile)
    q = rv_quantile_fn(p=0.9, mu=0, sigma=1, nu=1, alpha=2)
    print("Quantile:", q)


Using with PyMC
---------------

The following example demonstrates how to implement the `fsst` distribution 
in a PyMC model:

.. code-block:: python

    import pymc as pm
    from neonormal import fsst

    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfCauchy("sigma", 1)
        alpha = pm.Gamma("alpha", 2, 1)
        nu = pm.Gamma("nu", 2, 1)

        y = fsst("y", mu, sigma, nu, alpha, observed=np.random.randn(100))

Full Sampling Example
---------------------

The following example illustrates a complete PyMC model using the fsst distribution and performing posterior sampling:

.. code-block:: python

    with pm.Model():
        mu = pm.Normal('mu', 0, 5)
        sigma = pm.HalfCauchy('sigma', 5)
        alpha = pm.Gamma('alpha', 2, 1.5)
        nu = pm.Gamma('nu', 2, 15)

        fsst('FSST', mu, sigma, nu, alpha, observed=np.random.randn(100))
        trace= pm.sample()

    import arviz as az
    az.summary(trace, round_to=3, hdi_prob=0.95)

Using with Bambi
----------------

The `fsst` distribution is integrated as a custom family within this package, allowing direct use in Bambi models:

.. code-block:: python

    import bambi as bmb
    import pymc as pm
    import numpy as np
    from neonormal import fsst_family

    # Example dataset
    np.random.seed(123)
    df = pd.DataFrame({
        "x": np.random.normal(size=100),
        "y": 0.5 + 2 * np.random.normal(size=100)
    })

    model = bmb.Model("y ~ x", data=df, family=fsst_family)
    idata = model.fit()

    import arviz as az
    az.summary(idata)
