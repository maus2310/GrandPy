from typing import Any, Union, Sequence, Literal, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.stats import t
from functools import cached_property
from dataclasses import dataclass
from scipy.optimize import least_squares, OptimizeResult

@np.vectorize
def f_old_equi(t, s, d):
    return s / d * np.exp(-d * t)

@np.vectorize
def f_new(t, s, d):
    return s / d * (1 - np.exp(-d * t))

def make_res_jac(t_old, v_old, t_new, v_new, chase: bool):
    def res_fun(par):
        s, d = par
        ro = np.zeros_like(v_old) if chase else v_old - f_old_equi(t_old, s, d)
        rn = v_new - (f_old_equi(t_new, s, d) if chase else f_new(t_new, s, d))
        return np.concatenate([ro, rn])
    def jac(par):
        s, d = par
        if chase:
            j_old_s = np.zeros_like(t_old)
            j_old_d = np.zeros_like(t_old)
        else:
            exp_o = np.exp(-d * t_old)
            j_old_s = -exp_o / d
            j_old_d = -(-s / d**2 * exp_o + s / d * (-t_old * exp_o))
        exp_n = np.exp(-d * t_new)
        if chase:
            j_new_s = -exp_n / d
            j_new_d = s / d ** 2 * exp_n - s / d * (t_new * exp_n)
        else:
            one_me = 1 - exp_n
            j_new_s = -one_me / d
            j_new_d = -(-s / d**2 * one_me + s / d * (t_new * exp_n))
        return np.vstack([np.concatenate([j_old_s, j_new_s]),
                          np.concatenate([j_old_d, j_new_d])]).T
    return res_fun, jac


@dataclass
class FitResult:
    """
    Container for the result of a kinetic parameter fit.

    This object stores fitted synthesis and degradation rates, predicted values, residuals,
    diagnostics like RMSE or log-likelihood, and confidence intervals.

    Attributes
    ----------
    t_old, v_old : np.ndarray
        Time points and observed values for old RNA.

    t_new, v_new : np.ndarray
        Time points and observed values for new RNA.

    chase : bool
        Whether the fit was done in chase mode (no v_old data).

    slot_total : float or None
        Total expression used in chase mode for back-calculation.

    opt_result : OptimizeResult
        The result object returned by `scipy.optimize.least_squares`.

    ci_size : float
        Size of the confidence interval for parameter estimation.

    Notes
    -----
    All derived quantities like predictions, residuals, or confidence intervals are computed as
    cached properties and lazily evaluated.

    See Also
    --------
    fit_kinetics_gene_least_squares : Function to generate this object.
    """
    t_old: np.ndarray = np.nan
    v_old: np.ndarray = np.nan
    t_new: np.ndarray = np.nan
    v_new: np.ndarray = np.nan
    chase: bool = False
    slot_total: float = None
    opt_result: OptimizeResult = None
    ci_size: float = 0.95

    # --- Core Parameters ---
    @cached_property
    def synthesis(self) -> float:
        return self.opt_result.x[0] if self.opt_result is not None else np.nan

    @cached_property
    def degradation(self) -> float:
        return self.opt_result.x[1] if self.opt_result is not None else np.nan

    @cached_property
    def inv_deg(self) -> float:
        return 1.0 / self.degradation if self.degradation > 0 else np.nan

    # --- Cached exponentials ---
    @cached_property
    def exp_old(self) -> np.ndarray:
        return np.exp(-self.degradation * self.t_old)

    @cached_property
    def exp_new(self) -> np.ndarray:
        return np.exp(-self.degradation * self.t_new)

    # --- Predictions ---
    @cached_property
    def pred_old(self) -> np.ndarray:
        if self.chase:
            return np.zeros_like(self.t_old)
        return self.synthesis * self.inv_deg * self.exp_old

    @cached_property
    def pred_new(self) -> np.ndarray:
        if self.chase:
            return self.synthesis * self.inv_deg * self.exp_new
        return self.synthesis * self.inv_deg * (1 - self.exp_new)

    # --- Residuals ---
    @cached_property
    def residuals_raw(self) -> np.ndarray:
        if self.chase:
            return self.v_new - self.pred_new
        return np.concatenate([
            self.v_old - self.pred_old,
            self.v_new - self.pred_new
        ])

    @cached_property
    def residuals(self) -> dict[str, list[float]]:
        if self.chase:
            expected = self.pred_new
            observed = self.v_new
        else:
            expected = np.concatenate([self.pred_old, self.pred_new])
            observed = np.concatenate([self.v_old, self.v_new])

        rel = expected / (observed + 1e-8)
        return {
            "Absolute": expected.tolist(),
            "Relative": rel.tolist()
        }

    # --- Metrics ---
    @cached_property
    def rmse(self) -> float:
        return np.sqrt(np.mean(self.residuals_raw ** 2)) if self.residuals_raw.size > 0 else np.nan

    @cached_property
    def rmse_old(self) -> float:
        if self.chase or self.v_old.size == 0:
            return np.nan
        return np.sqrt(np.mean((self.v_old - self.pred_old) ** 2))

    @cached_property
    def rmse_new(self) -> float:
        if self.v_new.size == 0:
            return np.nan
        return np.sqrt(np.mean((self.v_new - self.pred_new) ** 2))

    @cached_property
    def half_life(self) -> float:
        return np.log(2) / self.degradation if self.degradation > 0 else np.nan

    @cached_property
    def log_likelihood(self) -> float:
        return -0.5 * np.sum(self.residuals_raw ** 2)

    @cached_property
    def total_expr(self) -> float:
        if self.chase and self.slot_total is not None:
            return self.slot_total
        return np.sum(self.v_old) + np.sum(self.v_new)

    @cached_property
    def f0(self) -> float:
        return self.total_expr if self.degradation > 0 else np.nan

    @cached_property
    def ci_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.opt_result is None or self.opt_result.jac is None:
            return np.full(2, np.nan), np.full(2, np.nan)
        try:
            J = self.opt_result.jac
            dof = len(self.v_old) + len(self.v_new) - len(self.opt_result.x)
            cov = np.linalg.pinv(J.T @ J)
            se = np.sqrt(np.diag(cov))
            alpha = 1 - self.ci_size
            tval = t.ppf(1 - alpha / 2, df=dof)
            lower = self.opt_result.x - tval * se
            upper = self.opt_result.x + tval * se
            return lower, upper
        except Exception:
            return np.full(2, np.nan), np.full(2, np.nan)

    @cached_property
    def ci_lower(self) -> dict:
        low, _ = self.ci_bounds
        d = low[1]
        return {
            "Synthesis": max(0, low[0]),
            "Degradation": max(0, d),
            "Half-life": np.log(2) / d if d > 0 else np.nan
        }

    @cached_property
    def ci_upper(self) -> dict:
        _, up = self.ci_bounds
        d = up[1]
        return {
            "Synthesis": up[0],
            "Degradation": d,
            "Half-life": np.log(2) / max(0.01, d) if d > 0 else np.nan
        }

    # --- Serialization ---
    def to_dict(self, fields: list[str]) -> dict[str, object]:
        field_funcs = {
            "Synthesis": lambda: self.synthesis,
            "Degradation": lambda: self.degradation,
            "Half-life": lambda: self.half_life,
            "rmse": lambda: self.rmse,
            "rmse_old": lambda: self.rmse_old,
            "rmse_new": lambda: self.rmse_new,
            "log_likelihood": lambda: self.log_likelihood,
            "f0": lambda: self.f0,
            "total": lambda: self.total_expr,
            "residuals": lambda: self.residuals,
            "conf_lower": lambda: self.ci_lower,
            "conf_upper": lambda: self.ci_upper,
        }
        return {
            f: field_funcs[f]() if f in field_funcs else np.nan
            for f in fields
        }

    def to_series(self, condition: str = None, prefix: str = "", fields: list[str] = None) -> pd.Series:
        data = self.to_dict(fields)
        p = f"{prefix}_" if prefix else ""
        c = f"{condition}_" if condition else ""

        flat = {}
        for key, val in data.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    flat[f"{p}{c}{key}_{subkey}"] = subval
            else:
                flat[f"{p}{c}{key}"] = val

        return pd.Series(flat)



def fit_kinetics_gene_least_squares(
    t_new: np.ndarray,
    v_new: np.ndarray,
    t_old: np.ndarray,
    v_old: np.ndarray,
    *,
    ci_size: float = 0.95,
    maxiter: int = 250,
    chase: bool = False,
    slot_data: np.ndarray = None,
    slot_names: np.ndarray = None,
    gene: Union[str, int, Sequence[Union[str, int, bool]]] = None
) -> FitResult:
    """
   Fit synthesis and degradation rates to old/new RNA data for a single gene using non-linear least squares.

   This function implements a least-squares optimization to infer kinetic parameters from time-resolved data.
   Optionally supports "chase" mode for experiments where labeled RNA is decaying.

   Parameters
   ----------
   t_new : np.ndarray
       Time points corresponding to the new (nascent) RNA values.

   v_new : np.ndarray
       Observed values of new RNA.

   t_old : np.ndarray
       Time points corresponding to the old RNA values.

   v_old : np.ndarray
       Observed values of old RNA.

   ci_size : float, optional
       Confidence interval level (between 0 and 1), by default 0.95.

   maxiter : int, optional
       Maximum number of optimization iterations, by default 250.

   chase : bool, optional
       Whether to perform the fit in "chase" mode (only v_new is used), by default False.

   slot_data : np.ndarray, optional
       Optional total expression values used to estimate initial concentration in chase mode.

   slot_names : np.ndarray, optional
       Gene names corresponding to `slot_data`.

   gene : str or int or Sequence[str or int or bool], optional
       The gene symbol currently being fit. Either by their index, their symbol, their ensamble id, or a boolean mask.

   Returns
   -------
   FitResult
       A container object holding fitted parameters, residuals, confidence intervals, and diagnostics.

   See Also
   --------
   FitResult
       Object encapsulating result and post-fit statistics.
   """

    if t_new.size + t_old.size == 0:
        return FitResult(ci_size=ci_size)

    res_fun, jac = make_res_jac(t_old, v_old, t_new, v_new, chase)

    # Initial guess
    mean_new = np.mean(v_new)
    mean_old = np.mean(v_old)
    s0 = (mean_new + mean_old) * 0.5
    d0 = 0.1 if mean_new > 0 else 0.01
    x0 = [s0, d0]

    result = least_squares(
        res_fun,
        x0=x0,
        jac=jac,
        bounds=([0, 1e-4], [np.inf, np.inf]),
        max_nfev=maxiter
    )
    if not result.success or result.nfev >= maxiter:
        return FitResult(ci_size=ci_size)

    # Optional slot total
    lvl_val = None
    if chase and slot_data is not None and slot_names is not None:
        idx = slot_names == gene
        lvl_val = np.median(slot_data[idx]) if np.any(idx) else np.nan

    return FitResult(
        t_old=t_old, v_old=v_old,
        t_new=t_new, v_new=v_new,
        chase=chase, slot_total=lvl_val,
        opt_result=result, ci_size=ci_size
    )

def fit_kinetics_gene_least_squares_chase(
        t_new: np.ndarray,
        v_new: np.ndarray,
        t_old: np.ndarray,
        v_old: np.ndarray,
        *,
        ci_size: float = 0.95,
        maxiter: int = 250,
        slot_data: np.ndarray = None,
        slot_names: np.ndarray = None,
        gene: Union[str, int, Sequence[Union[str, int, bool]]] = None
) -> FitResult:
    """
    This function only exists due to parallelization issues.
    """
    return fit_kinetics_gene_least_squares(t_new=t_new, v_new=v_new, t_old=t_old, v_old=v_old, ci_size=ci_size,
                                           maxiter=maxiter, chase=True, slot_data=slot_data, slot_names=slot_names, gene=gene)


def fit_kinetics(
    fit_type: Literal["nlls", "ntr", "chase"] = "nlls",
    *,
    cond_vec: np.ndarray = None,
    new_mat: np.ndarray = None,
    old_mat: np.ndarray = None,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name_prefix: Union[str, None] = None,
    time: Union[np.ndarray, pd.Series, list] = None,
    slot_data: np.ndarray = None,
    slot_names: np.ndarray = None,
    ci_size: float = 0.95,
    return_fields: Sequence[str] = None,
    show_progress: bool = True,
    **kwargs
) -> dict[str, pd.DataFrame]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if show_progress:
        from tqdm import tqdm

    # choose fit function
    fit_func = {
        "nlls": fit_kinetics_gene_least_squares,
        "chase": fit_kinetics_gene_least_squares_chase
    }.get(fit_type)
    if fit_func is None:
        raise ValueError(f"Unknown fit type: {fit_type}")

    unique_conditions = np.unique(cond_vec)

    result = {}

    for cond in unique_conditions:
        idx = np.where(cond_vec == cond)[0]
        t_cond = time[idx]
        new_cond = new_mat[:, idx]
        old_cond = old_mat[:, idx]

        jobs = []
        with ProcessPoolExecutor() as executor:
            for gene_index, gene in enumerate(genes):
                new_vals = new_cond[gene_index, :]
                old_vals = old_cond[gene_index, :]

                job = executor.submit(
                    fit_func,
                    t_new=t_cond,
                    v_new=new_vals,
                    t_old=t_cond,
                    v_old=old_vals,
                    ci_size=ci_size,
                    gene=gene,
                    slot_data=slot_data,
                    slot_names=slot_names,
                    **kwargs
                )
                jobs.append((gene, job))

            rows = []
            symbols = []

            if show_progress:
                for gene, future in tqdm(jobs, desc=f"Fitting {cond}", total=len(jobs)):
                    res = future.result()
                    series = res.to_series(condition=cond, prefix=name_prefix, fields=return_fields)
                    rows.append(series.values)
                    symbols.append(gene)
            else:
                for gene, future in jobs:
                    res = future.result()
                    series = res.to_series(condition=cond, prefix=name_prefix, fields=return_fields)
                    rows.append(series.values)
                    symbols.append(gene)

        df = pd.DataFrame(np.vstack(rows), index=symbols, columns=series.index)
        df.index.name = "Symbol"
        result[f"{name_prefix}kinetics_{cond}"] = df

    return result



# Non parallel version of the for loop. Faster for less genes (ca. <1500), but a lot slower for a large amount of genes.

# rows = []
# symbols = []
# for gene_index, gene in enumerate(genes_to_fit):
#     new_vals = new_cond[gene_index, :]
#     old_vals = old_cond[gene_index, :]
#
#     res = fit_func(
#         t_new=t_cond,
#         v_new=new_vals,
#         t_old=t_cond,
#         v_old=old_vals,
#         ci_size=ci_size,
#         gene=gene,
#         slot_data=slot_data,
#         slot_names=slot_names,
#         **kwargs
#     )
#     series = res.to_series(condition=cond, prefix=name_prefix, fields=return_fields)
#     rows.append(series.values)
#     symbols.append(gene)