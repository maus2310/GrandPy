from typing import Any, Union, Sequence, Literal, Mapping, Callable, TYPE_CHECKING
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import t
from dataclasses import dataclass
from scipy.optimize import least_squares, OptimizeResult

from Py.slot_tool import ModeSlot

import time as time_lib

if TYPE_CHECKING:
    from Py.grandPy import GrandPy

def f_old_equi(t, s, d):
    return s / d * np.exp(-d * t)

def f_new(t, s, d):
    return s / d * (1 - np.exp(-d * t))

def res_fun_equi(par, t_old, v_old, t_new, v_new):
    s, d = par
    old_pred = f_old_equi(t_old, s, d) if len(t_old) > 0 else np.array([])
    new_pred = f_new(t_new, s, d) if len(t_new) > 0 else np.array([])
    return np.concatenate([
        v_old - old_pred,
        v_new - new_pred
    ])

def res_fun_chase(par, t_old, v_old, t_new, v_new):
    """
    Residual function for chase fitting:
    - old values are ignored (residuals = 0)
    - new values are compared against f_old_equi (decay model)

    Parameters
    ----------
    par : array-like
        [synthesis, degradation]
    t_old, v_old : np.ndarray
        Old RNA values (ignored, but size matters)
    t_new, v_new : np.ndarray
        New RNA values to fit
    """
    s, d = par
    # dummy zeros for old values (ignored)
    res_old = np.zeros_like(v_old)

    # residuals for new values compared to expected decay
    pred_new = f_old_equi(t_new, s, d)
    res_new = v_new - pred_new

    return np.concatenate([res_old, res_new])



# Spielereien mit einer Klasse und lazy evaluation

@dataclass
class FitResult:
    t_old: np.ndarray = np.nan
    v_old: np.ndarray = np.nan
    t_new: np.ndarray = np.nan
    v_new: np.ndarray = np.nan
    chase: bool = False
    slot_total: float = None
    opt_result: OptimizeResult = None

    @property
    def synthesis(self) -> float:
        return self.opt_result.x[0] if self.opt_result is not None else np.nan

    @property
    def degradation(self) -> float:
        return self.opt_result.x[1] if self.opt_result is not None else np.nan

    def _predicted(self, kind: str) -> np.ndarray:
        s, d = self.synthesis, self.degradation
        if kind == "old":
            return np.zeros_like(self.t_old) if self.chase else s / d * np.exp(-d * self.t_old)
        elif kind == "new":
            return s / d * np.exp(-d * self.t_new) if self.chase else s / d * (1 - np.exp(-d * self.t_new))
        raise ValueError("kind must be 'old' or 'new'")

    @property
    def residuals_raw(self) -> np.ndarray:
        if self.chase:
            return self.v_new - self._predicted("new")
        return np.concatenate([
            self.v_old - self._predicted("old"),
            self.v_new - self._predicted("new")
        ])

    def residuals(self) -> dict[str, list[float]]:
        if self.chase:
            expected = self._predicted("new")
            observed = self.v_new
        else:
            expected = np.concatenate([
                self._predicted("old"),
                self._predicted("new")
            ])
            observed = np.concatenate([self.v_old, self.v_new])

        rel = expected / (observed + 1e-8)
        return {
            "Absolute": expected.tolist(),
            "Relative": rel.tolist()
        }

    @property
    def rmse(self) -> float:
        res = self.residuals_raw
        return np.sqrt(np.mean(res ** 2)) if len(res) > 0 else np.nan

    @property
    def rmse_old(self) -> float:
        if self.chase or len(self.v_old) == 0:
            return np.nan
        res = self.v_old - self._predicted("old")
        return np.sqrt(np.mean(res ** 2))

    @property
    def rmse_new(self) -> float:
        if len(self.v_new) == 0:
            return np.nan
        res = self.v_new - self._predicted("new")
        return np.sqrt(np.mean(res ** 2))

    @property
    def half_life(self) -> float:
        d = self.degradation
        return np.log(2) / d if d > 0 else np.nan

    @property
    def log_likelihood(self) -> float:
        res = self.residuals_raw
        return -0.5 * np.sum(res ** 2)

    @property
    def total_expr(self) -> float:
        if self.chase and self.slot_total is not None:
            return self.slot_total
        return np.sum(self.v_old) + np.sum(self.v_new)

    @property
    def f0(self) -> float:
        return self.total_expr if self.degradation > 0 else np.nan

    @property
    def ci_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.opt_result is None or self.opt_result.jac is None:
            return np.full(2, np.nan), np.full(2, np.nan)

        try:
            J = self.opt_result.jac
            dof = len(self.v_old) + len(self.v_new) - len(self.opt_result.x)
            cov = np.linalg.inv(J.T @ J)
            se = np.sqrt(np.diag(cov))
            tval = t.ppf(0.975, df=dof)
            lower = self.opt_result.x - tval * se
            upper = self.opt_result.x + tval * se
            return lower, upper
        except Exception:
            return np.full(2, np.nan), np.full(2, np.nan)

    @property
    def ci_lower(self) -> dict:
        low, _ = self.ci_bounds
        d = low[1]
        return {
            "Synthesis": max(0, low[0]),
            "Degradation": max(0, d),
            "Half-life": np.log(2) / d if d > 0 else np.nan
        }

    @property
    def ci_upper(self) -> dict:
        _, up = self.ci_bounds
        d = up[1]
        return {
            "Synthesis": up[0],
            "Degradation": d,
            "Half-life": np.log(2) / max(0.01, d) if d > 0 else np.nan
        }

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
            "residuals": self.residuals,
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
        time: np.ndarray,
        new_values: np.ndarray,
        old_values: np.ndarray,
        *,
        use_new: np.ndarray = None,
        use_old: np.ndarray = None,
        ci_size: float = 0.95,
        maxiter: int = 250,
        compute_residuals: bool = True,
        chase: bool = False,
        slot_data: np.ndarray = None,
        slot_names: np.ndarray = None,
        gene: str = None,
) -> FitResult:

    if use_new is None:
        use_new = np.ones_like(time, dtype=bool)
    if use_old is None:
        use_old = np.ones_like(time, dtype=bool)

    t_new = time[use_new]
    v_new = new_values[use_new]
    t_old = time[use_old]
    v_old = old_values[use_old]

    if chase:
        res_fun = lambda p: res_fun_chase(p, t_old, v_old, t_new, v_new)
    else:
        res_fun = lambda p: res_fun_equi(p, t_old, v_old, t_new, v_new)

    if len(t_new) == 0 and len(t_old) == 0:
        return FitResult()

    x0 = np.array([1.0, 0.1])
    result = least_squares(
        res_fun,
        x0=x0,
        bounds=([0, 0.01], [np.inf, np.inf]),
        max_nfev=maxiter
    )

    if not result.success or result.nfev >= maxiter:
        return FitResult()


    # optional total expression (für chase)
    if chase and slot_data is not None and slot_names is not None:
        idx = (slot_names == gene)
        lvl_val = np.median(slot_data[idx]) if np.any(idx) else np.nan
    else:
        lvl_val = None

    return FitResult(
        t_old=t_old,
        v_old=v_old,
        t_new=t_new,
        v_new=v_new,
        chase=chase,
        slot_total=lvl_val,
        opt_result=result
    )



def fit_kinetics(
        data: "GrandPy",
        fit_type: Literal["nlls", "ntr", "lm", "chase"] = "nlls",
        *,
        slot: str = None,
        genes: Union[str, Sequence[str]] = None,
        name_prefix: Union[str, None] = None,
        time: Union[np.ndarray, pd.Series, list] = None,
        ci_size: float = 0.95,
        return_fields: Union[str, Sequence[str]] = None,
        **kwargs
) -> "GrandPy":
    """
    Fit kinetic models to all genes using a specified fitting method.
    """

    name_prefix = f"{name_prefix}_" if name_prefix else ""

    genes_to_fit = data.get_genes(genes)

    fit_func = {
        "nlls": fit_kinetics_gene_least_squares,
        "chase": lambda **kw: fit_kinetics_gene_least_squares(**kw, chase=True)
    }.get(fit_type.lower())
    if fit_func is None:
        raise ValueError(f"Unknown fit type: {fit_type}")

    full_time = np.array(time) if time is not None else data.coldata["Time"].values
    condition_vector = data.coldata["Condition"].values

    new_mode_slot = "ntr" if fit_type == "chase" else ModeSlot("new", slot)
    chase_slot_data = data.get_matrix(slot)

    results = {}

    for gene in genes_to_fit:
        new_expr = data.get_matrix(mode_slot=new_mode_slot, genes=gene)
        old_expr = data.get_matrix(mode_slot=ModeSlot("old", slot), genes=gene)

        gene_results = {}
        for cond in np.unique(condition_vector):
            cond_mask = condition_vector == cond

            res = fit_func(
                time=full_time[cond_mask],
                new_values=new_expr[cond_mask],
                old_values=old_expr[cond_mask],
                ci_size=ci_size,
                gene=gene,
                slot_data=chase_slot_data if fit_type == "chase" else None,
                slot_names=np.array(data._adata.var_names) if fit_type == "chase" else None,
                **kwargs
            )

            gene_results[cond] = res

        results[gene] = gene_results

    # --- Formatieren und speichern ---
    for cond in np.unique(condition_vector):
        per_gene_series = {
            gene: res[cond].to_series(
                condition=cond,
                prefix=name_prefix,
                fields=return_fields
            )
            for gene, res in results.items()
        }
        df = pd.DataFrame.from_dict(per_gene_series, orient="index")
        df.index.name = "Symbol"
        data = data.with_analysis(name=f"{name_prefix}kinetics_{cond}", table=df)

    return data



# Ursprüngliche FitKinetics funktion

# def fit_kinetics(
#         data: "GrandPy",
#         fit_type: Literal["nlls", "ntr", "lm", "chase"] = "nlls",
#         *,
#         slot: str = None,
#         genes: Union[str, Sequence[str]] = None,
#         name_prefix: Union[str, None] = None,
#         time: Union[np.ndarray, pd.Series, list] = None,
#         ci_size: float = 0.95,
#         return_fields: list[str] = None,
#         **kwargs
# ) -> "GrandPy":
#     """
#     Fit kinetic models to all genes using a specified fitting method.
#
#     Parameters
#     ----------
#     data: GrandPy
#         The GrandPy dataset.
#     fit_type: str, default "nlls"
#         The fitting method: one of "nlls", "ntr", "lm", "chase".
#     slot: str, optional
#         The data slot to use. Defaults to data.default_slot.
#     genes: str or list of str, optional
#         The genes to fit. Defaults to all genes. Either by their index, their symbol, their ensamble id, or a boolean mask.
#     name_prefix: str or None, optional
#         A prefix for the analysis name.
#     time: array-like, optional
#         Time points (same length as number of samples).
#     ci_size: float, default 0.95
#         Confidence interval size.
#     return_fields: list[str], default ["Synthesis", "Half-life"]
#         Fields to include in the output.
#     **kwargs:
#         Additional arguments passed to the fitting functions.
#
#     Returns
#     -------
#     GrandPy
#         A new GrandPy object with the fitted analysis added.
#     """
#     if slot is None:
#         slot = data.default_slot
#     if return_fields is None:
#         return_fields = ["Synthesis", "Half-life"]
#     name_prefix = f"{name_prefix}_" if name_prefix else ""
#
#     genes_to_fit = data.get_genes(genes)
#
#     fit_func = {
#         "nlls": fit_kinetics_gene_least_squares,
#         "chase": lambda **kw: fit_kinetics_gene_least_squares(**kw, chase=True)
#     }.get(fit_type.lower())
#     if fit_func is None:
#         raise ValueError(f"Unknown fit type: {fit_type}")
#
#     full_time = np.array(time) if time is not None else data.coldata["Time"].values
#     condition_vector = data.coldata["Condition"].values
#
#     new_mode_slot = "ntr" if fit_type == "chase" else ModeSlot("new", slot)
#
#     results = {}
#     for gene in genes_to_fit:
#         new_expr = data.get_matrix(mode_slot=new_mode_slot, genes=gene)
#         old_expr = data.get_matrix(mode_slot=ModeSlot("old", slot), genes=gene)
#
#         gene_results = {}
#
#         for cond in np.unique(condition_vector):
#             cond_mask = condition_vector == cond
#
#             res = fit_func(
#                 time=full_time[cond_mask],
#                 new_values=new_expr[cond_mask],
#                 old_values=old_expr[cond_mask],
#                 ci_size=ci_size,
#                 gene=gene,
#                 **kwargs
#             )
#
#             gene_results[cond] = res
#
#         results[gene] = gene_results
#
#     # --- Formatieren und speichern ---
#     for cond in np.unique(condition_vector):
#         per_gene_series = {
#             gene: _extract_fit_series(
#                 fit_result=fit_res,
#                 condition=cond,
#                 name_prefix=name_prefix,
#                 return_fields=return_fields,
#             )
#             for gene, fit_res in results.items()
#         }
#         df = pd.DataFrame.from_dict(per_gene_series, orient="index")
#         df.index.name = "Symbol"
#         data = data.with_analysis(name=f"{name_prefix}kinetics_{cond}", table=df)
#
#     return data
#
#
#
# def fit_kinetics_gene_least_squares(
#         time: np.ndarray,
#         new_values: np.ndarray,
#         old_values: np.ndarray,
#         *,
#         use_new: np.ndarray = None,
#         use_old: np.ndarray = None,
#         ci_size: float = 0.95,
#         maxiter: int = 250,
#         compute_residuals: bool = True,
#         chase: bool = False,
#         slot_data: np.ndarray = None,
#         slot_names: np.ndarray = None,
#         gene: str = None,
# ) -> dict:
#     """
#
#     Parameters
#     ----------
#     time
#     new_values
#     old_values
#     use_new
#     use_old
#     ci_size
#     maxiter
#     compute_residuals
#     chase
#     slot_data
#     slot_names
#     gene
#
#     Returns
#     -------
#
#     """
#
#     if use_new is None:
#         use_new = np.ones_like(time, dtype=bool)
#     if use_old is None:
#         use_old = np.ones_like(time, dtype=bool)
#
#     if chase:
#         res_fun = lambda p: res_fun_chase(p, t_old, v_old, t_new, v_new)
#     else:
#         res_fun = lambda p: res_fun_equi(p, t_old, v_old, t_new, v_new)
#
#     t_new = time[use_new]
#     v_new = new_values[use_new]
#     t_old = time[use_old]
#     v_old = old_values[use_old]
#
#     # Abbruch bei leeren Daten
#     if len(t_new) == 0 and len(t_old) == 0:
#         return {
#             "residuals": None,
#             "Synthesis": np.nan,
#             "Degradation": np.nan,
#             "Half-life": np.nan,
#             "conf.lower": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
#             "conf.upper": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
#             "f0": np.nan,
#             "logLik": np.nan,
#             "rmse": np.nan,
#             "rmse.new": np.nan,
#             "rmse.old": np.nan,
#             "total": np.nan,
#             "type": "equi"
#         }
#
#     # Trying to estimate init_d and init_s like this ended up just slowing the code down.
#     # try:
#     #     t0 = t_new[t_new > 0].min() if len(t_new[t_new > 0]) > 0 else t_new.min()
#     #     n_val = v_new[t_new == t0]
#     #     o_val = v_old[t_old == t0]
#     #     init_d = np.mean(-np.log(1 - (0.1 + n_val) / (0.2 + n_val + o_val)))
#     #     init_s = init_d * np.mean(o_val)
#     # except:
#     init_d = 0.1
#     init_s = 1.0
#
#     x0 = np.array([init_s, max(0.01, init_d)])
#
#
#     # --- Fit ---
#     result = least_squares(
#         res_fun,
#         x0=x0,
#         bounds=([0, 0.01], [np.inf, np.inf]),
#         max_nfev=maxiter
#     )
#
#     if not result.success or result.nfev >= maxiter:
#         return {
#             "residuals": None,
#             "Synthesis": np.nan,
#             "Degradation": np.nan,
#             "Half-life": np.nan,
#             "conf.lower": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
#             "conf.upper": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
#             "f0": np.nan,
#             "logLik": np.nan,
#             "rmse": np.nan,
#             "rmse.new": np.nan,
#             "rmse.old": np.nan,
#             "total": np.nan,
#             "type": "equi"
#         }
#
#     s, d = result.x
#
#     # --- Vorhersage und RMSE ---
#     pred = res_fun(result.x)
#     n = len(t_old)
#     rmse = np.sqrt(np.sum(pred**2) / (n + len(t_new)))
#     rmse_old = np.sqrt(np.sum(pred[:n] ** 2) / n) if n > 0 else np.nan
#     rmse_new = np.sqrt(np.sum(pred[n:] ** 2) / (len(pred) - n)) if len(pred) > n else np.nan
#
#     # --- Konfidenzintervalle ---
#     try:
#         J = result.jac
#         cov = np.linalg.inv(J.T @ J)
#         se = np.sqrt(np.diag(cov))
#         tval = t.ppf(1 - (1 - ci_size) / 2, df=len(pred) - len(result.x))
#         ci = np.vstack([result.x - tval * se, result.x + tval * se]).T
#     except Exception:
#         ci = np.full((2, 2), np.nan)
#
#     # --- Total Expression (optional aus Slot-Daten) ---
#     if chase and slot_data is not None and gene is not None and slot_names is not None:
#         idx = (slot_names == gene)
#         lvl_val = np.median(slot_data[idx]) if np.any(idx) else np.nan
#     else:
#         lvl_val = s / d
#
#     synthesis = lvl_val * d if chase else s
#
#     # --- Residuen ---
#     residuals_df = None
#     if compute_residuals:
#         expected = np.concatenate([
#             f_old_equi(t_old, s, d),
#             f_new(t_new, s, d)
#         ])
#         observed = np.concatenate([v_old, v_new])
#         rel_res = expected / (observed + 1e-8)
#         residuals_df = {
#             "Absolute": expected.tolist(),
#             "Relative": rel_res.tolist()
#         }
#
#     return {
#         "residuals": residuals_df,
#         "Synthesis": synthesis,
#         "Degradation": d,
#         "Half-life": np.log(2) / d if d > 0 else np.nan,
#         "conf.lower": {
#             "Synthesis": max(0, ci[0, 0]),
#             "Degradation": max(0, ci[1, 0]),
#             "Half-life": np.log(2) / ci[1, 1] if ci[1, 1] > 0 else np.nan
#         },
#         "conf.upper": {
#             "Synthesis": ci[0, 1],
#             "Degradation": ci[1, 1],
#             "Half-life": np.log(2) / max(0.01, ci[1, 0])
#         },
#         "f0": synthesis / d if d > 0 else np.nan,
#         "logLik": -0.5 * np.sum(pred ** 2),
#         "rmse": rmse,
#         "rmse.new": rmse_new,
#         "rmse.old": rmse_old,
#         "total": np.sum(v_old) + np.sum(v_new),
#         "type": "equi"
#     }