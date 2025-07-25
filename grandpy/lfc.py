import warnings
from typing import Union, Callable

import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.special import digamma, polygamma
from scipy.stats import norm, beta


def empirical_bayes_prior(A: np.ndarray, B: np.ndarray, min_sd: float = 0.0) -> tuple[float, float]:
    """
    Estimates Empirical Bayes prior parameters (a, b) for LFC shrinkage.

    Parameters
    ----------
    A : np.ndarray
        Vector A of counts from condition A.

    B : np.ndarray
        Vector B of counts from condition B.

    min_sd : float
        Minimal standard deviation of the prior (see also Psi_LFC-function below).

    Returns
    -------
    a, b : float
        Estimated prior pseudocounts for A and B, respectively.
    """

    mask = (A > 0) | (B > 0)
    A0, B0 = A[mask], B[mask]

    diff = np.log(A0) - np.log(B0)
    x = np.median(diff)
    q_up = np.quantile(diff, norm.cdf(1))
    q_low = np.quantile(diff, norm.cdf(-1))
    y = max((q_up - x)**2, (-q_low + x)**2)

    if np.isinf(x) or np.isinf(y):
        A1, B1 = A + 1, B + 1
        diff1 = np.log(A1) - np.log(B1)
        x = np.mean(diff1)
        y = np.var(diff1)

    def obj(v):
        return (digamma(v[0]) - digamma(v[1]) - x)**2 + (polygamma(1, v[0]) + polygamma(1, v[1]) - y)**2

    result = minimize(obj, x0=[1.0, 1.0], method="Nelder-Mead")
    a, b = result.x

    sd = np.sqrt((polygamma(1, a) + polygamma(1, b)) / (np.log(2)**2))
    if sd < min_sd:

        def f_fun(f):
            return np.sqrt((polygamma(1, f*a) + polygamma(1, f*b)) / (np.log(2)**2)) - min_sd

        interval = 1.0 / (a + b)
        root = root_scalar(f_fun, bracket=[interval, 1.0])

        if not root.converged:
            raise RuntimeError("Could not inflate prior SD to min_sd.")

        else:
            f = root.root
            warnings.warn(f"Inflated prior by a factor of {1 / f:.2f}", RuntimeWarning)

        a *= f
        b *= f

    return a, b


def center_median(l: np.ndarray) -> np.ndarray:
    """
    Subtracts the median of the given vector (for normalizing log2 fold changes).

    Parameters
    ----------
    l : np.ndarray
        Vector of effect sizes (see also Psi_LFC-function below).

    Returns
    -------
    np.ndarray
        Normalized vector of the same shape as the input, where each element is shifted
        by substracting the median of l.
    """

    return l - np.nanmedian(l)


def norm_lfc(A: np.ndarray,
            B: np.ndarray,
            pseudo: tuple[float, float] = (1.0, 1.0),
            normalize_fun: Callable[[np.ndarray], np.ndarray] = center_median
             ) -> np.ndarray:
    """
    Computes the standard, normalized log2 fold change with given pseudocounts.

    Parameters
    ----------
    A : np.ndarray
        Vector A of counts from condition A.
    B : np.ndarray
        Vector B of counts from condition B.
    pseudo : tuple[float, float]
        Vector of length 2 of the pseudo counts.
    normalize_fun : Callable[[np.ndarray], np.ndarray]
        Function to normalize the obtained effect sizes.

    Returns
    -------
        Normalized LFCs.

    """

    lfc = np.log2(A + pseudo[0]) - np.log2(B + pseudo[1])

    return normalize_fun(lfc)


def psi_lfc(A: np.ndarray,
            B: np.ndarray,
            prior: tuple[float, float] = None,
            normalize_fun: Callable[[np.ndarray], np.ndarray] = center_median,
            cre: Union[bool, list[float]] = False,
            verbose: bool = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Computes the optimal effect size estimate and credible intervals if needed.

    Parameters
    ----------
    A : np.ndarray
        Vector A of counts from condition A.

    B : np.ndarray
        Vector B of counts from condition B.

    prior : tuple[float, float], optional
        Prior pseudocounts (a, b). If None, estimated via empirical_bayes_prior(A, B).

    normalize_fun : callable, optional
        Function to normalize raw LFCs (default: median centering).

    cre : bool, default False
        Compute credible intervals.

    verbose : bool, default False
        If True status updates are provided.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

        - lfc_centered: Shrunk, median‑centered log2 fold‑changes (length = #genes).
        - qlfc: Credible interval matrix (genes x len(cre)); only if cre is not False.
    """
    if prior is None:
        a, b = empirical_bayes_prior(A, B)
    else:
        a, b = prior

    if verbose:
        print(f"Using prior pseudocounts: a={a:.2f}, b={b:.2f}")

    lfc = (digamma(A + a) - digamma(B + b)) / np.log(2)

    lfc_centered = normalize_fun(lfc)

    if verbose:
        med_before = np.nanmedian(lfc)
        med_after = np.nanmedian(lfc_centered)
        print(f"Median before/after normalizing: {med_before:.4f} -> {med_after:.4f}")


    if cre is True:
        cre = [0.05, 0.95]

    if cre is not False:
        qs = cre if isinstance(cre, list) else [cre]
        a_posterior = A + a
        b_posterior = B + b

        proportion_q = np.vstack([beta.ppf(q, a_posterior, b_posterior) for q in qs]).T

        raw_qlfc = np.log2(proportion_q / (1.0 - proportion_q))

        ci_centered = normalize_fun(raw_qlfc)

        return lfc_centered, ci_centered

    return lfc_centered