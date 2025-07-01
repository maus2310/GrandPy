from typing import Any, Union, Sequence, Literal, Mapping, Callable, TYPE_CHECKING
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.stats import t

from Py.slot_tool import ModeSlot
from Py.utils import _extract_fit_series

if TYPE_CHECKING:
    from Py.grandPy import GrandPy

def fit_kinetics(
        data: "GrandPy",
        fit_type: Literal["nlls", "ntr", "lm", "chase"] = "nlls",
        *,
        slot: str = None,
        genes: Union[str, Sequence[str]] = None,
        name_prefix: Union[str, None] = "kinetics",
        time: Union[np.ndarray, pd.Series, list] = None,
        ci_size: float = 0.95,
        return_fields: list[str] = None,
        return_extra: Callable[[dict], Mapping] = None,
        **kwargs
) -> "GrandPy":
    """
    Fit kinetic models to all genes using a specified fitting method.

    Parameters
    ----------
    data: GrandPy
        The GrandPy dataset.
    fit_type: str, default "nlls"
        The fitting method: one of "nlls", "ntr", "lm", "chase".
    slot: str, optional
        The data slot to use. Defaults to data.default_slot.
    genes: str or list of str, optional
        The genes to fit. Defaults to all genes. Either by their index, their symbol, their ensamble id, or a boolean mask.
    name_prefix: str or None, default "kinetics"
        Prefix for the analysis name.
    time: array-like, optional
        Time points (same length as number of samples).
    ci_size: float, default 0.95
        Confidence interval size.
    return_fields: list[str], default ["Synthesis", "Half-life"]
        Fields to include in the output.
    return_extra: Callable[[dict], dict]
        A function that returns extra fields to include in the result as a dictionary.
    **kwargs:
        Additional arguments passed to the fitting functions.

    Returns
    -------
    GrandPy
        A new GrandPy object with the fitted analysis added.
    """
    from joblib import Parallel, delayed

    if slot is None:
        slot = data.default_slot
    if return_fields is None:
        return_fields = ["Synthesis", "Half-life"]

    genes_to_fit = data.get_genes(genes)

    fit_func = {
        "nlls": fit_kinetics_gene_least_squares,
        "chase": lambda **kw: fit_kinetics_gene_least_squares(**kw, chase=True)
    }.get(fit_type.lower())

    if fit_func is None:
        raise ValueError(f"Unknown fit type: {fit_type}")

    # Hole globale Informationen
    full_time = np.array(time) if time is not None else data.coldata["Time"].values
    condition_vector = data.coldata["Condition"].values if "Condition" in data.coldata.columns else np.full(
        len(full_time), "Data")

    new_mode_slot = "ntr" if fit_type == "chase" else ModeSlot("new", slot)

    results = {}
    for gene in genes_to_fit:
        new_expr = data.get_matrix(mode_slot=new_mode_slot, genes=gene)
        old_expr = data.get_matrix(mode_slot=ModeSlot("old", slot), genes=gene).flatten()

        gene_results = {}

        for cond in np.unique(condition_vector):
            cond_mask = condition_vector == cond

            res = fit_func(
                time=full_time[cond_mask],
                new_values=new_expr[cond_mask],
                old_values=old_expr[cond_mask],
                ci_size=ci_size,
                gene=gene,
                **kwargs
            )

            gene_results[cond] = res

        results[gene] = gene_results

    # --- Formatieren und speichern ---
    for cond in np.unique(condition_vector):
        per_gene_series = {
            gene: _extract_fit_series(
                fit_result=fit_res,
                condition=cond,
                name_prefix=name_prefix,
                return_fields=return_fields,
                return_extra=return_extra
            )
            for gene, fit_res in results.items()
        }
        df = pd.DataFrame.from_dict(per_gene_series, orient="index")
        df.index.name = "Symbol"
        data = data.with_analysis(name=f"{name_prefix}_{cond}", table=df)

    return data


#

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
) -> dict:
    """
    NumPy-only version of kinetic model fitting.

    Parameters
    ----------
    (wie vorher)

    Returns
    -------
    dict
        Structured result as in original implementation.
    """

    if use_new is None:
        use_new = np.ones_like(time, dtype=bool)
    if use_old is None:
        use_old = np.ones_like(time, dtype=bool)

    if chase:
        res_fun = lambda p: res_fun_chase(p, t_old, v_old, t_new, v_new)
    else:
        res_fun = lambda p: res_fun_equi(p, t_old, v_old, t_new, v_new)

    t_new = time[use_new]
    v_new = new_values[use_new]
    t_old = time[use_old]
    v_old = old_values[use_old]

    # Abbruch bei leeren Daten
    if len(t_new) == 0 and len(t_old) == 0:
        return {
            "data": None,
            "residuals": None,
            "Synthesis": np.nan,
            "Degradation": np.nan,
            "Half-life": np.nan,
            "conf.lower": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
            "conf.upper": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
            "f0": np.nan,
            "logLik": np.nan,
            "rmse": np.nan,
            "rmse.new": np.nan,
            "rmse.old": np.nan,
            "total": np.nan,
            "type": "equi"
        }

    # --- Initialparameter ---
    try:
        t0 = t_new[t_new > 0].min() if len(t_new[t_new > 0]) > 0 else t_new.min()
        n_val = v_new[t_new == t0]
        o_val = v_old[t_old == t0]
        init_d = np.mean(-np.log(1 - (0.1 + n_val) / (0.2 + n_val + o_val)))
        init_s = init_d * np.mean(o_val)
    except Exception:
        init_d = 0.1
        init_s = 1.0

    x0 = np.array([init_s, max(0.01, init_d)])


    # --- Fit ---
    result = least_squares(
        res_fun,
        x0=x0,
        bounds=([0, 0.01], [np.inf, np.inf]),
        max_nfev=maxiter
    )

    if not result.success or result.nfev >= maxiter:
        return {
            "data": None,
            "residuals": None,
            "Synthesis": np.nan,
            "Degradation": np.nan,
            "Half-life": np.nan,
            "conf.lower": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
            "conf.upper": {"Synthesis": np.nan, "Degradation": np.nan, "Half-life": np.nan},
            "f0": np.nan,
            "logLik": np.nan,
            "rmse": np.nan,
            "rmse.new": np.nan,
            "rmse.old": np.nan,
            "total": np.nan,
            "type": "equi"
        }

    s, d = result.x

    # --- Vorhersage und RMSE ---
    pred = res_fun(result.x) #TODO checken ob hier auch res_fun für chase = True
    n = len(t_old)
    rmse = np.sqrt(np.sum(pred**2) / (n + len(t_new)))
    rmse_old = np.sqrt(np.sum(pred[:n] ** 2) / n) if n > 0 else np.nan
    rmse_new = np.sqrt(np.sum(pred[n:] ** 2) / (len(pred) - n)) if len(pred) > n else np.nan

    # --- Konfidenzintervalle ---
    try:
        J = result.jac
        cov = np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
        tval = t.ppf(1 - (1 - ci_size) / 2, df=len(pred) - len(result.x))
        ci = np.vstack([result.x - tval * se, result.x + tval * se]).T
    except Exception:
        ci = np.full((2, 2), np.nan)

    # --- Total Expression (optional aus Slot-Daten) ---
    if chase and slot_data is not None and gene is not None and slot_names is not None:
        idx = (slot_names == gene)
        lvl_val = np.median(slot_data[idx]) if np.any(idx) else np.nan
    else:
        lvl_val = s / d

    synthesis = lvl_val * d if chase else s

    # --- Residuen ---
    residuals_df = None
    if compute_residuals:
        expected = np.concatenate([
            f_old_equi(t_old, s, d),
            f_new(t_new, s, d)
        ])
        observed = np.concatenate([v_old, v_new])
        rel_res = expected / (observed + 1e-8)
        residuals_df = {
            "Absolute": expected.tolist(),
            "Relative": rel_res.tolist()
        }

    return {
        "data": {
            "Time": np.concatenate([t_old, t_new]).tolist(),
            "Value": np.concatenate([v_old, v_new]).tolist(),
            "Type": ["old"] * len(t_old) + ["new"] * len(t_new)
        },
        "residuals": residuals_df,
        "Synthesis": synthesis,
        "Degradation": d,
        "Half-life": np.log(2) / d if d > 0 else np.nan,
        "conf.lower": {
            "Synthesis": max(0, ci[0, 0]),
            "Degradation": max(0, ci[1, 0]),
            "Half-life": np.log(2) / ci[1, 1] if ci[1, 1] > 0 else np.nan
        },
        "conf.upper": {
            "Synthesis": ci[0, 1],
            "Degradation": ci[1, 1],
            "Half-life": np.log(2) / max(0.01, ci[1, 0])
        },
        "f0": synthesis / d if d > 0 else np.nan,
        "logLik": -0.5 * np.sum(pred ** 2),
        "rmse": rmse,
        "rmse.new": rmse_new,
        "rmse.old": rmse_old,
        "total": np.sum(v_old) + np.sum(v_new),
        "type": "equi"
    }


# def fit_kinetics_gene_least_squares(
#         data,
#         gene,
#         slot,
#         time,
#         ci_size,
#         chase = False,
#         steady_state = None,
#         use_old = True,
#         use_new = True,
#         maxiter = 250,
#         compute_residuals = True
# ) -> dict:
#     """
#
#     Parameters
#     ----------
#     data
#     gene
#     slot
#     time
#     chase
#     ci_size
#     steady_state
#     use_old
#     use_new
#     maxiter
#     compute_residuals
#
#     Returns
#     -------
#
#     """
#     if np.isscalar(use_new):
#         use_new = [use_new] * len(data.columns)
#     if np.isscalar(use_old):
#         use_old = [use_old] * len(data.columns)
#
#     def correct(df, mode):
#         if df["Value"].max() == 0:
#             if mode == "new":
#                 df.loc[df["Time"] == 1, "Value"] = 0.01
#             else:
#                 df["Value"] = 0.01
#         df["Type"] = "New" if mode == "new" else "Old"
#         return df
#
#     # --- NEW Data ---
#     mode_slot_new = "ntr" if chase else f"new_{slot}"
#     new_df = data.get_data(mode_slots=mode_slot_new, genes=gene, with_coldata=True, ntr_nan=False)
#     new_df = new_df.rename(columns={gene: "Value"})
#     new_df["use"] = use_new[:len(new_df)]
#
#     if "Condition" not in new_df.columns or new_df["Condition"].isnull().all():
#         new_df["Condition"] = "Data"
#         if isinstance(steady_state, bool):
#             steady_state = {"Data": steady_state}
#
#     if chase:
#         new_df = new_df[~new_df["no4sU"]]
#
#     new_by_cond = {k: correct(v, "new") for k, v in new_df.groupby("Condition")}
#
#     # --- OLD Data ---
#     old_df = data.get_data(mode_slots=f"old_{slot}", genes=gene, with_coldata=True, ntr_nan=False)
#     old_df = old_df.rename(columns={gene: "Value"})
#     old_df["use"] = use_old[:len(old_df)]
#
#     if "Condition" not in old_df.columns:
#         old_df["Condition"] = "Data"
#
#     if chase:
#         old_df = old_df[~old_df["no4sU"]]
#         old_df["use"] = False
#
#     old_by_cond = {k: correct(v, "old") for k, v in old_df.groupby("Condition")}
#
#     results = {}
#
#     for cond in new_by_cond:
#         ndf = new_by_cond[cond]
#         odf = old_by_cond.get(cond, pd.DataFrame())
#
#         # steady state prüfen
#         if steady_state is None:
#             equi = True
#         elif isinstance(steady_state, dict):
#             equi = bool(steady_state.get(cond, True))
#         else:
#             equi = bool(steady_state)
#
#         if equi:
#             result = fit_equi(
#                 new=ndf,
#                 old=odf,
#                 ci_size=ci_size,
#                 maxiter=maxiter,
#                 compute_residuals=compute_residuals,
#                 chase=chase,
#                 slot_data=None,
#                 gene=gene
#             )
#         else:
#             raise NotImplementedError("Non-equilibrium model not implemented.")
#
#         results[cond] = result
#
#     return list(results.values())[0] if "Condition" not in data.coldata.columns else results
#
#
# def fit_equi(
#     new: pd.DataFrame,
#     old: pd.DataFrame,
#     ci_size: float = 0.95,
#     maxiter: int = 100,
#     compute_residuals: bool = True,
#     chase: bool = False,
#     slot_data: pd.DataFrame = None,
#     gene: str = None
# ) -> dict:
#     """
#     Fit equilibrium kinetic model using least squares.
#
#     Parameters
#     ----------
#     new : pd.DataFrame
#         DataFrame with new RNA data.
#     old : pd.DataFrame
#         DataFrame with old RNA data.
#     ci_size : float
#         Confidence interval level.
#     maxiter : int
#         Maximum iterations for optimizer.
#     compute_residuals : bool
#         Whether to compute residual values.
#     chase : bool
#         Use chase residuals.
#     slot_data : pd.DataFrame
#         Optional full data slot for total expression estimation.
#     gene : str
#         Gene name for extraction from slot_data.
#
#     Returns
#     -------
#     dict
#         Fitting result.
#     """
#     from scipy.optimize import least_squares
#
#     # 1. Residuenfunktion wählen
#     res_fun = res_fun_chase if chase else res_fun_equi
#
#     # 2. Initialparameter schätzen
#     t_use = new.loc[new["use"], "Time"]
#     v_use = new.loc[new["use"], "Value"]
#
#     if (v_use > 0).sum() == 0:
#         t_init = t_use[t_use > 0].min()
#     else:
#         t_init = t_use[v_use > 0].min()
#
#     n_val = new.loc[new["Time"] == t_init, "Value"].values
#     o_val = old.loc[old["Time"] == t_init, "Value"].values
#     init_d = np.mean(-np.log(1 - (0.1 + n_val) / (0.2 + n_val + o_val)))
#     init_s = init_d * np.mean(o_val)
#
#     init_params = np.array([init_s, max(0.01, init_d)])
#
#     # 3. Least Squares Fit
#     result = least_squares(
#         lambda p: res_fun(p, old.loc[old["use"]], new.loc[new["use"]]),
#         x0=init_params,
#         bounds=([0, 0.01], [np.inf, np.inf]),
#         max_nfev=maxiter
#     )
#
#     if not result.success or result.nfev >= maxiter:
#         return {
#             "data": None,
#             "residuals": None if not compute_residuals else pd.DataFrame(),
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
#     # 4. Parameter, RMSEs
#     s, d = result.x
#     pred = res_fun(result.x, old.loc[old["use"]], new.loc[new["use"]])
#     rmse = np.sqrt(np.sum(pred**2) / (len(old.loc[old["use"]]) + len(new.loc[new["use"]])))
#
#     n = len(old.loc[old["use"]])
#     rmse_old = np.sqrt(np.sum(pred[:n] ** 2) / n) if n > 0 else np.nan
#     rmse_new = np.sqrt(np.sum(pred[n:] ** 2) / (len(pred) - n)) if len(pred) > n else np.nan
#
#     # 5. Konfidenzintervalle (Schätzung)
#     try:
#         from scipy.stats import t
#         J = result.jac
#         cov = np.linalg.inv(J.T @ J)
#         se = np.sqrt(np.diag(cov))
#         tval = t.ppf(1 - (1 - ci_size) / 2, df=len(pred) - len(init_params))
#         ci = np.vstack([result.x - tval * se, result.x + tval * se]).T
#     except Exception:
#         ci = np.full((2, 2), np.nan)
#
#     # 6. Residuen berechnen
#     residuals_df = None
#     if compute_residuals:
#         expected = res_fun(result.x, old, new)
#         observed = np.concatenate([old["Value"], new["Value"]])
#         rel_res = expected / (observed + 1e-8)  # avoid div by 0
#
#         residuals_df = pd.DataFrame({
#             "Name": list(old["Name"]) + list(new["Name"]),
#             "Type": ["old"] * len(old) + ["new"] * len(new),
#             "Absolute": expected,
#             "Relative": rel_res
#         })
#
#     # 7. Total & Synthese
#     total_expr = np.sum(new["Value"]) + np.sum(old["Value"])
#     if chase and slot_data is not None and gene is not None:
#         lvl = slot_data[(slot_data["Gene"] == gene) & (slot_data["Name"].isin(new.loc[new["use"], "Name"]))]
#         lvl_val = lvl["Value"].median() if not lvl.empty else np.nan
#         total_expr = lvl_val
#     else:
#         lvl_val = s / d
#
#     synthesis = lvl_val * d if chase else s
#
#     return {
#         "data": pd.concat([old.loc[old["use"]], new.loc[new["use"]]], ignore_index=True),
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
#         "total": total_expr,
#         "type": "equi"
#     }

# def res_fun_equi(par, old: pd.DataFrame, new: pd.DataFrame) -> np.ndarray:
#     s, d = par
#     old_pred = f_old_equi(old["Time"].values, s, d)
#     new_pred = f_new(new["Time"].values, s, d)
#     return np.concatenate([
#         old["Value"].values - old_pred,
#         new["Value"].values - new_pred
#     ])
