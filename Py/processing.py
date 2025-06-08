from scipy.stats import beta
import numpy as np


# full chatgpt bis jetzt

def compute_ntr_posterior_quantile(data, quantile: float, name: str):
    """
    Compute a posterior quantile of the NTR beta distribution and store as a new slot.

    Parameters:
    - data: GrandPy object
    - quantile: float in [0,1], quantile to compute
    - name: str, name of the new slot
    """
    alpha = data.get_table(mode_slots="alpha", name_genes_by="Gene")
    beta_ = data.get_table(mode_slots="beta", name_genes_by="Gene")

    alpha = alpha.values
    beta_ = beta_.values

    q = beta.ppf(quantile, alpha, beta_)
    return data.with_slot(name, q)

def compute_ntr_posterior_lower(data, CI_size: float = 0.95, name: str = "lower"):
    """
    Compute lower bound of the NTR credible interval.
    """
    quantile = (1 - CI_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def compute_ntr_posterior_upper(data, CI_size: float = 0.95, name: str = "upper"):
    """
    Compute upper bound of the NTR credible interval.
    """
    quantile = 1 - (1 - CI_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def _compute_ntr_ci(data, CI_size: float = 0.95, name_lower: str = "lower", name_upper: str = "upper"):
    """
    Compute both lower and upper bounds of the credible interval.
    """
    data = compute_ntr_posterior_lower(data, CI_size, name_lower)
    data = compute_ntr_posterior_upper(data, CI_size, name_upper)
    return data
