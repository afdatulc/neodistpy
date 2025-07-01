import numpy as np
import pytest
import pymc as pm
import pytensor.tensor as pt
from neodistpy import fsst

# Variabel simbolik
value = pt.scalar('value')
mu = pt.scalar('mu')
sigma = pt.scalar('sigma')
nu = pt.scalar('nu')
alpha = pt.scalar('alpha')
p = pt.scalar('p')

rv = fsst.dist(mu=mu, sigma=sigma, nu=nu, alpha=alpha)
logp_fn = pm.compile([value, mu, sigma, nu, alpha], pm.logp(rv, value))
logcdf_fn = pm.compile([value, mu, sigma, nu, alpha], pm.logcdf(rv, value))
quantile_fn = pm.compile([p, mu, sigma, nu, alpha], fsst.icdf(p, mu, sigma, nu, alpha))

# ---------------------------- Test 1: Inappropriate Parameters -----------------------------------

def test_invalid_pdf_parameter():
    assert np.isinf(logp_fn(0, 0, -1, 2, 2)) or np.isnan(logp_fn(0, 0, -1, 2, 2))
    assert np.isinf(logp_fn(0, 0, 1, -1, 2)) or np.isnan(logp_fn(0, 0, 1, -1, 2))
    assert np.isinf(logp_fn(0, 0, 1, 2, -1)) or np.isnan(logp_fn(0, 0, 1, 2, -1))

def test_invalid_cdf_parameter():
    assert np.isinf(logcdf_fn(0, 0, -1, 2, 2)) or np.isnan(logcdf_fn(0, 0, -1, 2, 2))
    assert np.isinf(logcdf_fn(0, 0, 1, -1, 2)) or np.isnan(logcdf_fn(0, 0, 1, -1, 2))
    assert np.isinf(logcdf_fn(0, 0, 1, 2, -1)) or np.isnan(logcdf_fn(0, 0, 1, 2, -1))

def test_invalid_quantile_parameter():
    with pytest.raises(Exception):
        quantile_fn(0.9, 0, -1, 1, 1).eval() 
    with pytest.raises(Exception):
        quantile_fn(0.9, 0, 1, -1, 1).eval()  
    with pytest.raises(Exception):
        quantile_fn(0.9, 0, 1, 1, -1).eval()  

def test_invalid_random_parameter():
    with pytest.raises(ValueError):
        fsst.random(0, -1, 2, 2)  
    with pytest.raises(ValueError):
        fsst.random(0, 1, -1, 2)  
    with pytest.raises(ValueError):
        fsst.random(0, 1, 2, -1)  

# ---------------------------- Test 2: Infinity Inputs --------------------------------------------

def test_infinity_input():
    assert not np.isnan(logp_fn(np.inf, 0, 1, 2, 2))
    assert not np.isnan(logp_fn(-np.inf, 0, 1, 2, 2))
    assert not np.isnan(logcdf_fn(np.inf, 0, 1, 2, 2)) 
    assert not np.isnan(logcdf_fn(-np.inf, 0, 1, 2, 2)) 
    
# ---------------------------- Test 3: Log Probability --------------------------------------------

def test_valid_range_pdf_cdf():
    y_vals = [-100, -10, -5, -1, -0.5, 0, 0.5, 1, 5, 10, 100]
    for y in y_vals:
        pdf = np.exp(logp_fn(y, 0, 1, 2, 2))
        cdf = np.exp(logcdf_fn(y, 0, 1, 2, 2))
        assert pdf >= 0, f"PDF at {y} is negative"
        assert 0 <= cdf <= 1, f"CDF at {y} out of range"

# ---------------------------- Test 4: Missing Values (NaN) ----------------------------------------

def test_nan_in_pdf():
    assert np.isnan(logp_fn(np.nan, 0, 1, 1, 1)) or np.isinf(logp_fn(0, 0, 1, np.nan, 1))
    assert np.isnan(logp_fn(0, np.nan, 1, 1, 1)) or np.isinf(logp_fn(0, 0, 1, np.nan, 1))
    assert np.isnan(logp_fn(0, 0, np.nan, 1, 1)) or np.isinf(logp_fn(0, 0, 1, np.nan, 1))
    assert np.isnan(logp_fn(0, 0, 1, np.nan, 1)) or np.isinf(logp_fn(0, 0, 1, np.nan, 1))
    assert np.isnan(logp_fn(0, 0, 1, 1, np.nan)) or np.isinf(logp_fn(0, 0, 1, 1, np.nan))

def test_nan_in_cdf():
    assert np.isnan(logcdf_fn(np.nan, 0, 1, 1, 1)) or np.isinf(logcdf_fn(np.nan, 0, 1, 1, 1))
    assert np.isnan(logcdf_fn(0, np.nan, 1, 1, 1)) or np.isinf(logcdf_fn(0, np.nan, 1, 1, 1))
    assert np.isnan(logcdf_fn(0, 0, np.nan, 1, 1)) or np.isinf(logcdf_fn(0, 0, np.nan, 1, 1))
    assert np.isnan(logcdf_fn(0, 0, 1, np.nan, 1)) or np.isinf(logcdf_fn(0, 0, 1, np.nan, 1))
    assert np.isnan(logcdf_fn(0, 0, 1, 1, np.nan)) or np.isinf(logcdf_fn(0, 0, 1, 1, np.nan))

def test_nan_in_quantile():
    assert np.isnan(quantile_fn(np.nan, 0, 1, 1, 1)) or np.isinf(quantile_fn(np.nan, 0, 1, 1, 1))
    assert np.isnan(quantile_fn(0.5, np.nan, 1, 1, 1)) or np.isinf(quantile_fn(0.5, np.nan, 1, 1, 1))
    assert np.isnan(quantile_fn(0.5, 0, np.nan, 1, 1)) or np.isinf(quantile_fn(0.5, 0, np.nan, 1, 1))
    with pytest.raises(ValueError):
        quantile_fn(0.5, 0, 1, np.nan, 1).eval()
    assert np.isnan(quantile_fn(0.5, 0, 1, 1, np.nan)) or np.isinf(quantile_fn(0.5, 0, 1, 1, np.nan))

def test_nan_in_random():
    with pytest.raises(ValueError):
        fsst.random(np.nan, 1, 1, 1)

    with pytest.raises(ValueError):
        fsst.random(0, np.nan, 1, 1)

    with pytest.raises(ValueError):
        fsst.random(0, 1, np.nan, 1)

    with pytest.raises(ValueError):
        fsst.random(0, 1, 1, np.nan)


# ---------------------------- Test 5: Quantile Function Behavior ---------------------------------
def test_quantile_0_1_behavior():
    assert np.isnan(quantile_fn(-0.1, 0, 1, 2, 2))  # p < 0
    assert np.isnan(quantile_fn(1.1, 0, 1, 2, 2))   # p > 1
    assert quantile_fn(0, 0, 1, 2, 2) == -np.inf    # p = 0
    assert quantile_fn(1, 0, 1, 2, 2) == np.inf     # p = 1

def test_quantile_inverse():
    p_val = 0.9
    q = quantile_fn(p_val, 0, 1, 2, 2)
    log_cdf = logcdf_fn(q, 0, 1, 2, 2)
    cdf = np.exp(log_cdf)
    assert np.isclose(cdf, p_val, atol=1e-2)

# ---------------------------- Test 6: RNG Convergence --------------------------------------------
def test_random_coverage():
    samples = fsst.random(mu=0, sigma=1, nu=2, alpha=2, size=(5000,))
    assert len(samples) == 5000
    min_val, max_val = samples.min(), samples.max()
    cdf_range = np.exp(logcdf_fn(max_val, 0, 1, 2, 2)) - np.exp(logcdf_fn(min_val, 0, 1, 2, 2))
    assert cdf_range >= 0.99
