import bambi as bmb
import pymc as pm
from .fsst import fsst
from .msnburr import msnburr
from .msnburr_iia import msnburr_iia


setattr(pm, "fsst", fsst)
setattr(pm, "msnburr", msnburr)
setattr(pm, "msnburr_iia", msnburr_iia)


fsst_likelihood = bmb.Likelihood(
    "fsst",
    parent="mu",
    params=["mu", "sigma", "nu", "alpha"]
)
fsst_links = {"mu": "identity", "sigma": "log", "nu": "log", "alpha": "log"}
fsst_family = bmb.Family("fsst",fsst_likelihood,fsst_links)
fsst_family.set_default_priors({
    "sigma": bmb.Prior("HalfCauchy", beta=5),
    "nu": bmb.Prior("Gamma", alpha=2, beta=0.05),
    "alpha": bmb.Prior("Gamma", alpha=2, beta=15)
})

msnburr_likelihood = bmb.Likelihood(
    "msnburr",
    parent="mu",
    params=["mu", "sigma", "alpha"]
)
msnburr_links = {"mu": "identity", "sigma": "log", "alpha": "log"}
msnburr_family = bmb.Family("msnburr",msnburr_likelihood,link=msnburr_links)
msnburr_family.set_default_priors({
    "sigma": bmb.Prior("HalfCauchy", beta=5),
    "alpha": bmb.Prior("Gamma", alpha=2, beta=15)
})

msnburr_iia_likelihood = bmb.Likelihood(
    "msnburr_iia",
    parent="mu",
    params=["mu", "sigma", "alpha"]
)
msnburr_iia_links = {"mu": "identity", "sigma": "log", "alpha": "log"}
msnburr_iia_family = bmb.Family("msnburr_iia", msnburr_iia_likelihood, msnburr_iia_links)
msnburr_iia_family.set_default_priors({
    "sigma": bmb.Prior("HalfCauchy", beta=5),
    "alpha": bmb.Prior("Gamma", alpha=2, beta=15)
})