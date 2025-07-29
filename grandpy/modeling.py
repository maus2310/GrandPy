import os
import warnings
from collections.abc import Sequence, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Union, Literal, TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, least_squares, OptimizeResult, brentq
from scipy.stats import norm, t
from tqdm import tqdm

from .slot_tool import ModeSlot
from .utils import _ensure_list, _get_kinetics_data

if TYPE_CHECKING:
    from .grandPy import GrandPy


def _fit_kinetics(
    data: "GrandPy",
    fit_type: Literal["nlls", "ntr", "chase"] = "nlls",
    *,
    slot: str = None,
    name_prefix: Union[str, None] = None,
    return_fields: Union[str, Sequence[str]] = None,
    time: Union[str, np.ndarray, pd.Series, list] = "Time",
    ci_size: float = 0.95,
    genes: Union[str, Sequence[str]] = None,
    show_progress: bool = True,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    Wrapper for fit_kinetics_nlls, fit_kinetics_chase, and fit_kinetics_ntr.

    For detailed documentation, see GrandPy.fit_kinetics.
    """
    # Preprocess the parameters
    if slot is None:
        slot = data.default_slot
    if return_fields is None:
        return_fields = ["Synthesis", "Half-life"]
    return_fields = _ensure_list(return_fields)

    name_prefix = f"{name_prefix}_" if name_prefix else ""

    if isinstance(time, str):
        time = data.coldata[time]

    time = np.array(time).squeeze()

    fit_function = {
        "nlls": fit_kinetics_nlls,
        "ntr": fit_kinetics_ntr,
        "chase": fit_kinetics_chase,
    }.get(fit_type, None)

    if fit_function is None:
        raise ValueError(f"Unknown fit type: {fit_type}. Available functions are: `nlls`, `ntr`, `chase`.")

    kinetics = fit_function(data=data, slot=slot, name_prefix=name_prefix, return_fields=return_fields, time=time,
                            ci_size=ci_size, genes=genes, show_progress=show_progress, **kwargs)

    return kinetics


def get_dynamic_process_count(data_size: int, max_processes: int = None, exact_processes: bool = False) -> int:
    """
    Dynamically determine the number of processes based on data size and available CPUs.

    Parameters
    ----------
    data_size: int
        Size of the data.

    max_processes: int, optional
        Maximum limit for amount of processes.

    Returns
    -------
    int
        Recommended number of processes
    """
    if exact_processes:
        if max_processes is None:
            max_processes = 1
        return max(max_processes, 1)

    available_cores = max(os.cpu_count()-1, 1)

    if max_processes is None:
        # Arbitrary threshold for amount of processes approximated by testing. (Probably different for other systems)
        num_processes = min(available_cores, data_size // 1200)
    else:
        num_processes = min(max_processes, available_cores)

    num_processes = max(1, num_processes)

    return num_processes


# TODO: Verhalten von correct überprüfen
def correct_new(new_expressions: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Applies a correction to every row of `all_expressions`, where time = 1.
    """
    zero_rows = np.max(new_expressions, axis=1) == 0

    time_1 = time == 1

    if np.any(zero_rows) and np.any(time_1):
        new_expressions[zero_rows, time_1] = 0.01

    return new_expressions

def correct_old(old_expressions: np.ndarray) -> np.ndarray:
    """
    Applies a correction to every row of `all_expressions`.
    """
    zero_rows = np.max(old_expressions, axis=1) == 0

    if np.any(zero_rows):
        old_expressions[zero_rows, :] = 0.01

    return old_expressions



# ----- nlls and chase kinetic modeling -----
def fit_kinetics_nlls(
    data: "GrandPy",
    slot: str,
    *,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name_prefix: Union[str, None] = None,
    time: Union[np.ndarray, pd.Series, list] = None,
    ci_size: float = 0.95,
    return_fields: Sequence[str] = None,
    steady_state: Union[bool, Mapping[str, bool]] = True,
    max_processes: int = None,
    exact_processes: bool = False,
    show_progress: bool = True,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    For a detailed documentation, see GrandPy.fit_kinetics.
    """
    genes_to_fit = data.get_genes(genes)

    coldata = data.coldata

    condition_vector = coldata["Condition"].values
    unique_conditions = np.unique(condition_vector)

    sample_names = data.coldata["Name"].values

    # --- Decide on parallelisation ---
    datasize = len(genes_to_fit)

    max_workers = get_dynamic_process_count(datasize, max_processes, exact_processes)

    if max_workers == 1:
        parallel = False
    else:
        from concurrent.futures import ProcessPoolExecutor

        parallel = True

    # --- Map steady_state to each condition ---
    if isinstance(steady_state, bool):
        steady_state = {cond: steady_state for cond in unique_conditions}

    # --- Retrieve expression matrices ---
    new_expression = correct_new(data.get_matrix(mode_slot=ModeSlot("new", slot), genes=genes_to_fit), time=time)
    old_expression = correct_old(data.get_matrix(mode_slot=ModeSlot("old", slot), genes=genes_to_fit))

    result = {}

    # --- Call the fitting function for each gene for each condition. ---
    for condition in unique_conditions:
        idx = np.where(condition_vector == condition)[0]
        time_cond = time[idx]
        new_cond = new_expression[:, idx]
        old_cond = old_expression[:, idx]
        sample_names_cond = sample_names[idx]
        condition_steady_state = steady_state[condition]

        if parallel:
            jobs = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for gene_index, gene in enumerate(genes_to_fit):
                    new_values = new_cond[gene_index, :]
                    old_values = old_cond[gene_index, :]

                    job = executor.submit(
                        fit_kinetics_gene_least_squares,
                        new_values=new_values,
                        old_values=old_values,
                        time=time_cond,
                        ci_size=ci_size,
                        chase=False,
                        total_value=None,
                        steady_state=condition_steady_state,
                        sample_names=sample_names_cond,
                        **kwargs
                    )
                    jobs.append((gene, job))

                rows = []
                symbols = []

                for gene, future in tqdm(jobs, desc=f"Fitting {condition}", total=len(jobs), disable=not show_progress):
                    res = future.result()
                    series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                    rows.append(series.values)
                    symbols.append(gene)

        else:
            rows = []
            symbols = []

            gene_index_iterator = enumerate(genes_to_fit)
            gene_index_iterator = tqdm(gene_index_iterator, total=len(genes_to_fit), desc=f"Fitting {condition}", disable=not show_progress)

            for gene_index, gene in gene_index_iterator:
                new_values = new_cond[gene_index, :]
                old_values = old_cond[gene_index, :]

                res = fit_kinetics_gene_least_squares(
                    new_values=new_values,
                    old_values=old_values,
                    time=time_cond,
                    ci_size=ci_size,
                    chase=False,
                    total_value=None,
                    steady_state=condition_steady_state,
                    sample_names=sample_names_cond,
                    **kwargs
                )
                series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                rows.append(series.values)
                symbols.append(gene)

        df = pd.DataFrame(np.vstack(rows), index=symbols, columns=series.index)
        df.index.name = "Symbol"
        result[f"{name_prefix}kinetics_{condition}"] = df

    return result

def fit_kinetics_chase(
    data: "GrandPy",
    slot: str,
    *,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name_prefix: Union[str, None] = None,
    time: Union[np.ndarray, pd.Series, list] = None,
    ci_size: float = 0.95,
    return_fields: Sequence[str] = None,
    max_processes: int = None,
    exact_processes: bool = False,
    show_progress: bool = True,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    For a detailed documentation, see GrandPy.fit_kinetics.
    """
    genes_to_fit = data.get_genes(genes)

    coldata = data.coldata

    condition_vector = coldata["Condition"].values
    unique_conditions = np.unique(condition_vector)

    sample_names = data.coldata["Name"].values

    # --- Decide on parallelisation ---
    datasize = len(genes_to_fit)

    max_workers = get_dynamic_process_count(datasize, max_processes, exact_processes)

    if max_workers == 1:
        parallel = False
    else:
        from concurrent.futures import ProcessPoolExecutor

        parallel = True

    # --- Retrieve expression matrix ---
    new_expression = correct_new(data.get_matrix(mode_slot="ntr", genes=genes_to_fit), time=time)

    # --- chase-specific preprocessing ---
    no4sU_mask = data.coldata["no4sU"].values
    new_expression = new_expression[:, ~no4sU_mask]

    condition_vector = condition_vector[~no4sU_mask]
    time = time[~no4sU_mask]

    sample_names = sample_names[~no4sU_mask]

    slot_matrix = data.get_matrix(slot, genes=genes_to_fit)
    slot_matrix = slot_matrix[:, ~no4sU_mask]

    slot_values_per_gene = {
        gene: np.median(slot_matrix[i, :])
        for i, gene in enumerate(genes_to_fit)
    }

    result = {}

    # --- Call the fitting function for each gene for each condition. ---
    for condition in unique_conditions:
        idx = np.where(condition_vector == condition)[0]
        time_cond = time[idx]
        new_cond = new_expression[:, idx]
        sample_names_cond = sample_names[idx]

        if parallel:
            jobs = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for gene_index, gene in enumerate(genes_to_fit):
                    new_values = new_cond[gene_index, :]
                    total_value = slot_values_per_gene.get(gene, None)

                    job = executor.submit(
                        fit_kinetics_gene_least_squares,
                        new_values=new_values,
                        old_values=np.zeros_like(new_values),
                        time=time_cond,
                        ci_size=ci_size,
                        chase=True,
                        total_value=total_value,
                        steady_state=True,
                        sample_names=sample_names_cond,
                        **kwargs
                    )
                    jobs.append((gene, job))

                rows = []
                symbols = []

                for gene, future in tqdm(jobs, desc=f"Fitting {condition}", total=len(jobs), disable= not show_progress):
                    res = future.result()
                    series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                    rows.append(series.values)
                    symbols.append(gene)

        else:
            rows = []
            symbols = []

            gene_index_iterator = enumerate(genes_to_fit)
            gene_index_iterator = tqdm(gene_index_iterator, total=len(genes_to_fit),desc=f"Fitting {condition}", disable=not show_progress)

            for gene_index, gene in gene_index_iterator:
                new_values = new_cond[gene_index, :]

                res = fit_kinetics_gene_least_squares(
                    new_values=new_values,
                    old_values=np.zeros_like(new_values),
                    time=time_cond,
                    ci_size=ci_size,
                    chase=True,
                    total_value=slot_values_per_gene.get(gene, None),
                    steady_state=True,
                    sample_names=sample_names_cond,
                    **kwargs
                )
                series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                rows.append(series.values)
                symbols.append(gene)

        df = pd.DataFrame(np.vstack(rows), index=symbols, columns=series.index)
        df.index.name = "Symbol"
        result[f"{name_prefix}kinetics_{condition}"] = df

    return result


def fit_kinetics_gene_least_squares(
    new_values: np.ndarray,
    time: np.ndarray,
    old_values: np.ndarray = None,
    *,
    ci_size: float = 0.95,
    max_iter: int = 250,
    chase: bool = False,
    steady_state: bool = True,
    total_value: float = None,
    sample_names: list[str] = None,
) -> "FitResult":
    """
    Fits synthesis and degradation rates for a single gene and condition using non-linear least squares.

    This method implements a least-squares optimization to infer kinetic parameters from time-resolved data.
    Optionally supports "chase" mode for experiments where labeled RNA is decaying.

    Parameters
    ----------
    new_values : np.ndarray
        Observed values of new RNA.

    time : np.ndarray
        Time points

    old_values : np.ndarray, optional
        Observed values of old RNA. Irrelevant for chase.

    ci_size : float, default 0.95
        Confidence interval level.

    max_iter : int, default 250
        Maximum number of optimization iterations.

    chase : bool, default False
        Whether to perform the fit in "chase" mode (only v_new is used).

    steady_state : bool, default True
        Whether to use steady for a condition.

    total_value : float, optional
        Optional total expression values used to estimate initial concentration in chase mode.

    sample_names: list[str], optional
        Sample names for naming residuals.

    Returns
    -------
    FitResult
        A container object holding fitted parameters, residuals, confidence intervals, and diagnostics.

    See Also
    --------
    FitResult
        Object encapsulating result and post-fit statistics.
    """

    if time.size + time.size == 0:
        return FitResult()

    # --- Define residual and Jacobian functions ---
    if steady_state:
        res_fun, jac = get_residuals_and_jacobian_equi(time, old_values, new_values, chase)
        if chase:
            x0 = guess_chase_start(new_values, time)
            bounds = ([1e-8, 1e-3], [np.inf, 2.0])
        else:
            x0 = [np.mean(old_values), 0.1]
            bounds = ([0, 1e-4], [np.inf, np.inf])
    else:
        res_fun, jac = get_residuals_and_jacobian_nonequi(time, old_values, new_values)

        s0 = np.maximum(np.max(new_values[time > 0] / time[time > 0]), 1e-3)
        d0 = guess_d0_from_old(old_values, time)
        f00 = np.mean(old_values[time == 0]) if np.any(time == 0) else np.mean(old_values)

        x0 = [s0, d0, f00]
        bounds = ([0, 1e-4, 0], [np.inf, np.inf, np.inf])

    # --- Run least-squares optimization ---
    result = least_squares(
        res_fun,
        x0=x0,
        jac=jac,
        bounds=bounds,
        max_nfev=max_iter
    )
    if not result.success or result.nfev >= max_iter:
        warnings.warn(f"The least squares optimization failed with the following message: {result.message}")
        return FitResult()

    # --- Construct FitResult object ---
    return FitResult(
        time=time,
        new_values=new_values,
        old_values=old_values,
        chase=chase,
        slot_total=total_value,
        opt_result=result,
        ci_size=ci_size,
        steady_state=steady_state,
        sample_names=sample_names
    )


@dataclass
class FitResult:
    """
    Stores the result of kinetic parameter fitting for RNA expression data.

    Attributes
    ----------
    time : np.ndarray
        Time points.

    new_values : np.ndarray
        Observed values of new RNA.

    old_values : np.ndarray
        Observed values of old RNA.

    chase : bool
        Indicates if the experiment is in "chase" mode (only v_new is used).

    slot_total : float
        Optional total RNA expression value (used in chase mode).

    opt_result : OptimizeResult
        Optimization result from `scipy.optimize.least_squares`.

    ci_size : float
        Confidence level for confidence intervals (e.g., 0.95 for 95%).

    steady_state : bool
        Whether steady-state assumption was used during fitting.

    sample_names : list
        A List of sample names for naming residuals.
    """
    time: np.ndarray = np.nan
    new_values: np.ndarray = np.nan
    old_values: np.ndarray = np.nan
    chase: bool = False
    slot_total: float = None
    opt_result: OptimizeResult = None
    ci_size: float = 0.95
    steady_state: bool = True
    sample_names: list[str] = None

    # --- Core Parameters ---
    @property
    def synthesis(self) -> float:
        return self.opt_result.x[0] if self.opt_result is not None else np.nan

    @property
    def return_synthesis(self):
        if self.chase:
            return self.slot_total * self.degradation
        else:
            return self.synthesis

    @property
    def degradation(self) -> float:
        return self.opt_result.x[1] if self.opt_result is not None else np.nan

    @property
    def inv_deg(self) -> float:
        return 1.0 / self.degradation if self.degradation > 0 else np.nan

    # --- Cached exponentials ---
    @property
    def exp_old(self) -> np.ndarray:
        return np.exp(-self.degradation * self.time)

    @property
    def exp_new(self) -> np.ndarray:
        return np.exp(-self.degradation * self.time)

    # --- Predictions ---
    @property
    def pred_old(self) -> np.ndarray:
        if self.chase:
            return np.zeros_like(self.time)
        if self.steady_state:
            return self.synthesis * self.inv_deg * self.exp_old
        else:
            return self.f0 * np.exp(-self.degradation * self.time)

    @property
    def pred_new(self) -> np.ndarray:
        if self.chase:
            return self.synthesis * self.inv_deg * self.exp_new
        return self.synthesis * self.inv_deg * (1 - self.exp_new)

    # --- Residuals ---
    @property
    def residuals_raw(self) -> np.ndarray:
        return np.concatenate([
            self.old_values - self.pred_old,
            self.new_values - self.pred_new
        ])

    @property
    def residuals(self) -> dict[str, np.ndarray]:
        if self.chase:
            expected = self.pred_new
            observed = self.new_values
        else:
            expected = np.concatenate([self.pred_old, self.pred_new])
            observed = np.concatenate([self.old_values, self.new_values])

        absolute = observed - expected

        modval = np.concatenate([self.old_values, self.new_values,]) - absolute
        relative = absolute / (modval + 1e-10)

        return {
            "absolute": np.abs(absolute),
            "relative": relative
        }

    # --- Metrics ---
    @property
    def rmse(self) -> float:
        return np.sqrt(np.sum(self.residuals_raw ** 2)/self.residuals_raw.size)

    @property
    def rmse_old(self) -> float:
        if self.chase or self.old_values.size == 0:
            return np.nan
        return np.sqrt(np.mean((self.old_values - self.pred_old) ** 2))

    @property
    def rmse_new(self) -> float:
        if self.new_values.size == 0:
            return np.nan
        return np.sqrt(np.mean((self.new_values - self.pred_new) ** 2))

    @property
    def half_life(self) -> float:
        return np.log(2) / self.degradation if self.degradation > 0 else np.nan

    @property
    def log_likelihood(self) -> float:
        if self.chase:
            N = self.residuals_raw.size/2
        else:
            N = self.residuals_raw.size
        return -N * (np.log(2 * np.pi) + 1 - np.log(N) + np.log(sum(self.residuals_raw ** 2)))/2

    @property
    def total_expr(self) -> float:
        if self.chase and self.slot_total is not None:
            return self.slot_total
        return np.sum(self.old_values) + np.sum(self.new_values)

    @property
    def f0(self) -> float:
        if self.opt_result is None:
            return np.nan
        if self.steady_state:
            return self.return_synthesis / self.degradation if self.degradation > 0 else np.nan
        if len(self.opt_result.x) >= 3:
            return self.opt_result.x[2] # steady_state = False
        return np.nan

    @property
    def ci_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.opt_result is None or self.opt_result.jac is None:
            return (np.full_like(self.opt_result.x, np.nan),) * 2

        try:
            J = self.opt_result.jac
            x = self.opt_result.x

            if J.ndim != 2 or J.shape[1] != x.size:
                return (np.full_like(x, np.nan),) * 2

            dof = len(self.residuals_raw) - x.size
            if dof <= 0:
                return (np.full_like(x, np.nan),) * 2

            sigma2 = np.sum(self.residuals_raw ** 2) / dof

            JTJ = J.T @ J
            if np.linalg.cond(JTJ) > 1e12:
                return (np.full_like(x, np.nan),) * 2

            cov = sigma2 * np.linalg.inv(JTJ)
            se = np.sqrt(np.diag(cov))
            tval = t.ppf(1 - (1 - self.ci_size) / 2, df=dof)

            lower = x - tval * se
            upper = x + tval * se

            return lower, upper

        except (np.linalg.LinAlgError, ValueError):
            return (np.full_like(self.opt_result.x, np.nan),) * 2

    @property
    def ci_lower(self) -> dict:
        low, up = self.ci_bounds
        result = {}
        if len(low) >= 1:
            result["Synthesis"] = max(0, low[0])
        if len(low) >= 2:
            result["Degradation"] = max(0, low[1])
            d_up = up[1] if len(up) > 1 else np.nan
            result["Half-life"] = np.log(2) / d_up if d_up > 0 else np.nan
        return result

    @property
    def ci_upper(self) -> dict:
        low, up = self.ci_bounds
        result = {}
        if len(up) >= 1:
            result["Synthesis"] = up[0]
        if len(up) >= 2:
            result["Degradation"] = up[1]
            d_low = low[1] if len(low) > 1 else np.nan
            result["Half-life"] = np.log(2) / max(0, d_low) if d_low > 0 else np.nan
        return result


    def to_dict(self, fields: list[str]) -> dict[str, object]:
        field_funcs = {
            "Synthesis": lambda: self.return_synthesis,
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

        flat = {}

        for key, val in data.items():
            base_key = f"{prefix}{condition}_{key}"

            if key == "residuals":
                # isolated handling for residuals
                sample_names = np.concatenate([self.sample_names, self.sample_names])

                for res_type, values in val.items():
                    n_old = len(self.time)

                    for i, v in enumerate(values):
                        if self.chase:
                            kind = "new"
                            sample_name = sample_names[i].replace(".", "_")
                        else:
                            if i < n_old:
                                kind = "old"
                                sample_name = sample_names[i].replace(".", "_")
                            else:
                                kind = "new"
                                sample_name = sample_names[i].replace(".", "_")

                        full_key = f"{prefix}{sample_name}_{key}_{res_type}_{kind}"
                        flat[full_key] = v

            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    flat[f"{base_key}_{subkey}"] = subval

            elif isinstance(val, (list, np.ndarray)):
                for i, v in enumerate(val):
                    flat[f"{base_key}_{i}"] = v
            else:
                flat[base_key] = val

        return pd.Series(flat)


def get_residuals_and_jacobian_equi(time: np.ndarray, v_old: np.ndarray, v_new: np.ndarray, chase: bool):
    """
    Constructs residual and Jacobian functions assuming steady-state kinetics.

    Parameters
    ----------
    time : np.ndarray
        Time points for RNA.
    v_old : np.ndarray
        Observed values for old RNA.
    v_new : np.ndarray
        Observed values for new RNA.
    chase : bool
        If True, assumes a chase experiment (no old RNA used).

    Returns
    -------
    tuple
        A tuple (residual_function, jacobian_function) for least-squares optimization.
    """
    if chase:
        def residual_function(par):
            s, d = par
            pred = s / d * np.exp(-d * time)
            return v_new - pred

        def jacobian_function(par):
            s, d = par
            exp_n = np.exp(-d * time)

            j_new_s = -exp_n / d
            j_new_d = s / d ** 2 * exp_n + s / d * (time * exp_n)

            return np.vstack([j_new_s, j_new_d]).T

    else:
        def residual_function(par):
            s, d = par
            ro = v_old - f_old_equi(time, s, d)
            rn = v_new - f_new(time, s, d)
            return np.concatenate([ro, rn])

        def jacobian_function(par):
            s, d = par

            exp_o = np.exp(-d * time)
            j_old_s = -exp_o / d
            j_old_d = -(-s / d ** 2 * exp_o + s / d * (-time * exp_o))

            exp_n = np.exp(-d * time)
            one_me = 1 - exp_n

            j_new_s = -one_me / d
            j_new_d = -(-s / d ** 2 * one_me + s / d * (time * exp_n))

            return np.vstack([np.concatenate([j_old_s, j_new_s]),
                              np.concatenate([j_old_d, j_new_d])]).T

    return residual_function, jacobian_function

def get_residuals_and_jacobian_nonequi(time: np.ndarray, v_old: np.ndarray, v_new: np.ndarray):
    """
    Constructs residual and Jacobian functions assuming non steady-state kinetics.

    Parameters
    ----------
    time : np.ndarray
        Time points for RNA.
    v_old : np.ndarray
        Observed values for old RNA.
    v_new : np.ndarray
        Observed values for new RNA.

    Returns
    -------
    tuple
        A tuple (residual_function, jacobian_function) for least-squares optimization.
    """
    def residual_function(par):
        s, d, f0 = par
        ro = v_old - f_old_nonequi(time, f0, s, d)
        rn = v_new - f_new(time, s, d)
        return np.concatenate([ro, rn])

    def jacobian_function(par):
        s, d, f0 = par
        exp = np.exp(-d * time)

        # r_old derivatives
        j_old_s = np.zeros_like(time)
        j_old_d = f0 * time * exp
        j_old_f0 = -exp

        # r_new derivatives
        j_new_s = -(1 - exp) / d
        j_new_d = s / d ** 2 * (1 - exp) - s / d * time * exp
        j_new_f0 = np.zeros_like(time)

        J = np.vstack([
            np.concatenate([j_old_s, j_new_s]),
            np.concatenate([j_old_d, j_new_d]),
            np.concatenate([j_old_f0, j_new_f0])
        ]).T
        return J

    return residual_function, jacobian_function


def guess_chase_start(values_new: np.ndarray, time: np.ndarray):
    """
    Approximates the start values x0 for least_squares in a chase experiment using linear regression.
    """
    mask = (values_new > 0) & (time > 0)
    if np.count_nonzero(mask) < 2:
        return 1.0, 0.5

    y = np.log(np.maximum(values_new[mask], 1e-3))
    t = time[mask]

    try:
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(t, y, deg=1)
        slope = p.convert().coef[1]
        d0 = -slope
    except Exception:
        d0 = 0.1

    d0 = np.clip(d0, 1e-3, 2.0)
    s0 = np.clip(values_new[0] * d0, 1e-8, np.inf)

    return s0, d0

def guess_d0_from_old(values_old: np.ndarray, time: np.ndarray):
    """
    Approximates the degradation(d0) for least_squares in a non steady state using linear regression.
    """
    from numpy.polynomial import Polynomial

    mask = (values_old > 0) & (time > 0)
    if np.count_nonzero(mask) < 2:
        return 0.1
    y = np.log(np.maximum(values_old[mask], 1e-3))
    t = time[mask]
    p = Polynomial.fit(t, y, deg=1)
    slope = p.convert().coef[1]
    return np.clip(-slope, 1e-3, 2.0)


@np.vectorize
def f_old_equi(t: float, s: float, d: float) -> float:
    """
    Computes the expected amount of old RNA under steady-state assumptions.
    """
    return s / d * np.exp(-t * d)

@np.vectorize
def f_old_nonequi(t: float, f0: float, s: float, d: float):
    """
    Computes the expected amount of old RNA under non-steady-state.
    """
    return f0 * np.exp(-t * d)

@np.vectorize
def f_new(t: float, s: float, d: float) -> float:
    """
    Computes the expected amount of newly synthesized RNA at a given time.
    """
    return s / d * (1 - np.exp(-t * d))



# ----- ntr kinetic modeling -----
def fit_kinetics_ntr(
        data: "GrandPy",
        slot: str,
        *,
        genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
        name_prefix: Union[str, None] = None,
        time: Union[np.ndarray, pd.Series, list] = None,
        ci_size: float = 0.95,
        return_fields: Sequence[str] = None,
        exact_ci: bool = False,
        transformed_ntr_map: bool = True,
        show_progress: bool = True
) -> dict[str, pd.DataFrame]:
    """
    For detailed documentation, see grandpy.GrandPy.fit_kinetics.
    """
    if not ("alpha" in data.slots and "beta" in data.slots):
        raise ValueError("NTR-basierte Anpassung erfordert alpha-, beta-Slots.")

    genes_to_fit = data.get_genes(genes)

    condition_vector = data.coldata["Condition"].values
    unique_conditions = np.unique(condition_vector)

    # Paralellisation proved superfluous, as even for large datasets (>20000 genes) it showed no improvements in runtime over serialisation.
    # --- Decide on parallelisation ---
    # datasize = (len(genes_to_fit) * len(unique_conditions)) // 5
    #
    # max_workers = get_dynamic_process_count(datasize, max_processes)
    #
    # if max_workers == 1:
    #     parallel = False
    # else:
    #     from concurrent.futures import ProcessPoolExecutor
    #
    #     parallel = True

    # --- Retrieve matrices ---
    alpha = data.get_matrix(mode_slot="alpha", genes=genes_to_fit)
    beta = data.get_matrix(mode_slot="beta", genes=genes_to_fit)
    ntr = data.get_matrix(mode_slot="ntr", genes=genes_to_fit)
    total = data.get_matrix(mode_slot=slot, genes=genes_to_fit)

    result = {}

    for condition in unique_conditions:
        cond_mask = np.where(condition_vector == condition)[0]
        time_cond = time[cond_mask]

        alpha_cond = alpha[:, cond_mask]
        beta_cond = beta[:, cond_mask]
        ntr_cond = ntr[:, cond_mask]
        total_cond = total[:, cond_mask]

        rows = []
        symbols = []

        # if parallel:
        #     jobs = []
        #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
        #         for gene_index, gene in enumerate(genes_to_fit):
        #             alpha_values = alpha_cond[gene_index, :]
        #             beta_values = beta_cond[gene_index, :]
        #             ntr_values = ntr_cond[gene_index, :]
        #             total_values = total_cond[gene_index, :]
        #
        #             job = executor.submit(
        #                 fit_kinetics_gene_ntr,
        #                 alpha = alpha_values,
        #                 beta = beta_values,
        #                 time=time_cond,
        #                 ntr_values=ntr_values,
        #                 total_values=total_values,
        #                 ci_size=ci_size,
        #                 exact_ci=exact_ci,
        #                 transformed_ntr_map=transformed_ntr_map,
        #             )
        #             jobs.append((gene, job))
        #
        #         rows = []
        #         symbols = []
        #
        #         if show_progress:
        #             for gene, future in tqdm(jobs, desc=f"Fitting {condition}", total=len(jobs)):
        #                 res = future.result()
        #                 series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
        #                 rows.append(series.values)
        #                 symbols.append(gene)
        #         else:
        #             for gene, future in jobs:
        #                 res = future.result()
        #                 series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
        #                 rows.append(series.values)
        #                 symbols.append(gene)

        gene_iter = enumerate(genes_to_fit)
        gene_iter = tqdm(gene_iter, total=len(genes_to_fit), desc=f"Fitting {condition}", disable=not show_progress)

        for gene_index, gene_id in gene_iter:
            alpha_values = alpha_cond[gene_index, :]
            beta_values = beta_cond[gene_index, :]
            ntr_values = ntr_cond[gene_index, :]
            total_values = total_cond[gene_index, :]

            res = fit_kinetics_gene_ntr(
                alpha=alpha_values,
                beta=beta_values,
                time=time_cond,
                ntr_values=ntr_values,
                total_values=total_values,
                ci_size=ci_size,
                exact_ci=exact_ci,
                transformed_ntr_map=transformed_ntr_map,
            )

            series = res.to_series(
                condition=condition,
                prefix=name_prefix,
                fields=return_fields
            )
            rows.append(series.values)
            symbols.append(gene_id)

        df = pd.DataFrame(np.vstack(rows), index=symbols, columns=series.index)
        df.index.name = "Symbol"
        result[f"{name_prefix}kinetics_{condition}"] = df

    return result

def fit_kinetics_gene_ntr(
        alpha: np.ndarray,
        beta: np.ndarray,
        time: np.ndarray,
        ntr_values: np.ndarray,
        total_values: np.ndarray,
        ci_size: float = 0.95,
        transformed_ntr_map: bool = True,
        exact_ci: bool = False,
        total_function: Callable = np.median
) -> "NTRFitResult":
    """
    Fit degradation rate using the NTR model for a single gene and condition.

    Parameters
    ----------
    alpha : np.ndarray
        alpha values.

    beta : np.ndarray
        beta values.

    time : np.ndarray
        Time points.

    ntr_values : np.ndarray
        New-to-total ratios.

    total_values : np.ndarray
        Expression values used to compute synthesis rate.

    ci_size : float, default=0.95
        Confidence interval size.

    transformed_ntr_map : bool, default=True
        Whether the NTR has been transformed via MAP estimation.

    exact_ci : bool, default=False
        If True, compute exact confidence intervals via posterior integration.

    total_function : Callable, default=np.median
        Function to reduce total expression across time points (e.g., mean, median).

    Returns
    -------
    NTRFitResult
        Fitted model containing degradation/synthesis/half-life/etc.
    """
    from scipy.optimize import minimize_scalar

    def loglik(d):
        # time_mask = time > 0
        # alpha = alpha[time_mask]
        # beta = beta[time_mask]
        # time = time[time_mask]

        exp = np.exp(-time * d)
        safe_exp = np.clip(exp, 1e-10, 1 - 1e-10)
        log_term = np.log1p(-safe_exp)

        if transformed_ntr_map:
            return np.sum((alpha - 1) * log_term - (time * d) * (beta - 1))
        else:
            return np.sum((alpha - 1) * log_term - (time * d) * beta)

    bounds = (np.log(2) / 48, np.log(2) / 0.01)

    result = minimize_scalar(lambda d: -loglik(d), bounds=bounds, method="bounded")

    if not result.success:
        warnings.warn(f"The NTR optimization failed with the following message: {result.message}")

    f0 = total_function(total_values)

    time_mask = time > 0

    return NTRFitResult(
        result=result.x,
        time=time[time_mask],
        alpha=alpha[time_mask],
        beta=beta[time_mask],
        ntr=ntr_values[time_mask],
        exact_ci=exact_ci,
        ci_size=ci_size,
        f0 = f0,
        transformed_ntr_map=transformed_ntr_map,
    )

@dataclass
class NTRFitResult:
    """
    Stores the result of NTR-based RNA degradation fitting.

    Attributes
    ----------
    result : float
        Fitted degradation rate.

    time : np.ndarray
        Time points.

    alpha : np.ndarray
        Alpha values.

    beta : np.ndarray
        Beta values.

    ntr : np.ndarray
        New-total-ratio.

    f0 : float
        Estimated total RNA expression.

    exact_ci : bool
        Whether confidence intervals are computed via posterior integration.

    ci_size : float
        Confidence interval size.

    transformed_ntr_map : bool
        Whether transformed MAP estimates were used for NTR.
    """
    result: float
    time: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    ntr: np.ndarray
    f0: float
    exact_ci: bool = False
    ci_size: float = 0.95
    transformed_ntr_map: bool = True
    model_type: Literal["ntr"] = "ntr"

    def loglik(self, d):
        exp = np.exp(-self.time * d)
        safe_exp = np.clip(exp, 1e-10, 1 - 1e-10)
        log_term = np.log1p(-safe_exp)

        if self.transformed_ntr_map:
            return np.sum((self.alpha - 1) * log_term - (self.time * d) * (self.beta - 1))
        else:
            return np.sum((self.alpha - 1) * log_term - (self.time * d) * self.beta)

    @cached_property
    def degradation(self) -> float:
        return self.result

    @cached_property
    def synthesis(self) -> float:
        return self.f0 * self.degradation

    @cached_property
    def half_life(self) -> float:
        return np.log(2) / self.degradation if self.degradation > 0 else np.nan

    @cached_property
    def predicted_ntr(self) -> np.ndarray:
        return 1 - np.exp(-self.degradation * self.time)

    @cached_property
    def rmse(self) -> float:
        return sum(np.sqrt((1 - np.exp(-self.time * self.degradation) - self.ntr) ** 2)) / len(self.time)

    @cached_property
    def log_likelihood(self) -> float:
        return self.loglik(self.degradation)

    @cached_property
    def total_expr(self) -> float:
        return self.f0

    @cached_property
    def ci_lower(self) -> dict:
        lower_d, upper_d = self.confidence_intervals

        return {
            "Synthesis": self.f0 * lower_d,
            "Degradation": lower_d,
            "Half-life": np.log(2) / upper_d if upper_d > 0 else np.nan
        }

    @cached_property
    def ci_upper(self) -> dict:
        lower_d, upper_d = self.confidence_intervals

        return {
            "Synthesis": self.f0 * upper_d,
            "Degradation": upper_d,
            "Half-life": np.log(2) / lower_d if lower_d > 0 else np.nan
        }

    @cached_property
    def confidence_intervals(self) -> tuple[float, float]:
        """
        Approximate confidence interval for degradation using fast posterior CDF approximation.
        """
        if not self.exact_ci:
            from scipy.stats import chi2

            crit = chi2.ppf(self.ci_size, df=2) / 2
            cutoff = self.log_likelihood - crit

            lower = uniroot_safe(
                lambda d: self.loglik(d) - cutoff,
                self._ci_bounds[0],
                self.degradation
            )

            upper = uniroot_safe(
                lambda d: self.loglik(d) - cutoff,
                self.degradation,
                self._ci_bounds[1]
            )

            return lower, upper

        from scipy.interpolate import interp1d

        # --- FAST posterior CDF via precomputed grid ---
        n_grid = 150
        center = self.degradation
        span = center * 2
        grid = np.linspace(center - span / 2, center + span / 2, n_grid)
        logpost = np.array([self.loglik(d) for d in grid])
        post = np.exp(logpost - np.max(logpost))
        area = np.trapezoid(post, grid)
        cdf_vals = np.cumsum(post) * (grid[1] - grid[0]) / area

        cdf_interp = interp1d(grid, cdf_vals, bounds_error=False, fill_value=(0.0, 1.0))

        # Find lower/upper bounds via interpolation
        lower = uniroot_safe(lambda d: cdf_interp(d) - (1 - self.ci_size) / 2, *self._ci_bounds)
        upper = uniroot_safe(lambda d: cdf_interp(d) - (1 + self.ci_size) / 2, *self._ci_bounds)

        return lower, upper

    @cached_property
    def _ci_bounds(self):
        span = self.degradation * 3
        return max(1e-6, self.degradation - span), self.degradation + span

    def to_dict(self, fields: list[str]) -> dict[str, object]:
        field_funcs = {
            "Synthesis": lambda: self.synthesis,
            "Degradation": lambda: self.degradation,
            "Half-life": lambda: self.half_life,
            "conf_lower": lambda: self.ci_lower,
            "conf_upper": lambda: self.ci_upper,
            "f0": lambda: self.f0,
            "log_likelihood": lambda: self.log_likelihood,
            "rmse": lambda: self.rmse,
            "total": lambda: self.total_expr,
        }
        return {
            f: field_funcs[f]() if f in field_funcs else np.nan
            for f in fields
        }

    def to_series(self, condition: str = None, prefix: str = "", fields: list[str] = None) -> pd.Series:
        data = self.to_dict(fields)

        flat = {}

        for key, val in data.items():
            base_key = f"{prefix}{condition}_{key}"

            if isinstance(val, dict):
                for subkey, subval in val.items():
                    flat[f"{base_key}_{subkey}"] = subval
            elif isinstance(val, (list, np.ndarray)):
                for i, v in enumerate(val):
                    flat[f"{base_key}_{i}"] = v
            else:
                flat[base_key] = val

        return pd.Series(flat)


def uniroot_safe(fun, lower, upper):
    try:
        if fun(lower) * fun(upper) >= 0:
            return (lower + upper) / 2
        return brentq(fun, lower, upper)
    except Exception:
        return (lower + upper) / 2




# ----- time calibration functions -----
def compute_use_mask(HLs, totals, time, n_estimate):
    bin_edges = np.concatenate([np.linspace(0, 2 * np.max(time), num=5), [np.inf]])
    HL_cat = np.digitize(HLs, bin_edges, right=False)

    use_mask = np.zeros_like(totals, dtype=bool)
    unique_cats = np.unique(HL_cat)
    n_groups = len(unique_cats)
    group_threshold = int(np.ceil(n_estimate / n_groups))

    for cat in unique_cats:
        group_idx = np.where(HL_cat == cat)[0]
        group_totals = totals[group_idx]

        if len(group_totals) == 0:
            continue

        sorted_totals = np.sort(group_totals)[::-1]
        threshold_index = min(len(sorted_totals), group_threshold) - 1
        threshold = sorted_totals[threshold_index]

        use_mask[group_idx] = group_totals >= threshold

    return use_mask

# def select_top_genes_by_hl_and_expression(half_lives, totals, time_vector, gene_names, n_top_genes):
#     """
#     Selects about n top genes by half-life and expression.
#     """
#     max_time = np.max(time_vector)
#     bins = np.linspace(0, 2 * max_time, 5)
#     bins = np.concatenate([bins, [np.inf]])
#
#     hl_cat_indices = np.digitize(half_lives, bins, right=False)  # 1-based bin indices
#     n_bins = len(bins)
#
#     use_mask = np.zeros_like(totals, dtype=bool)
#
#     for bin_idx in range(1, n_bins + 1):
#         in_bin = hl_cat_indices == bin_idx
#         n_in_bin = np.sum(in_bin)
#         if n_in_bin == 0:
#             continue
#         threshold_index = min(n_in_bin - 1, int(np.ceil(n_top_genes / n_bins)))
#         sorted_indices = np.argsort(-totals[in_bin])
#         keep_indices = np.where(in_bin)[0][sorted_indices[:threshold_index + 1]]
#         use_mask[keep_indices] = True
#
#     return np.array(gene_names)[use_mask].tolist()

def _calibrate_effective_labeling_time_kinetic_fit(
    data,
    slot: str = None,
    time: str = "duration.4sU",
    name: str = "calibrated_time",
    n_top_genes: int = 1000,
    max_iterations: int = 10000,
    compute_confidence: bool = False,
    ci_size: float = 0.95,
    show_progress: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    For a detailed description, see grandpy.GrandPy.calibrate_effective_labeling_time_kinetic_fit.
    """
    if slot is None:
        slot = data.default_slot

    conditions = data.condition
    sample_names = data.columns
    result = pd.DataFrame(np.nan, index=pd.Index(sample_names, name="Name"), columns=[name])

    for cond in np.unique(conditions):
        eval_bar = tqdm(desc=f"Optimizing {cond}", dynamic_ncols=True, unit=" Iterations", disable=not show_progress)

        mask = np.array(conditions) == cond
        sub = data[:, mask].with_dropped_analyses()

        expr = sub.get_matrix(slot)
        totals = expr.sum(axis=1)
        gene_names = np.array(sub.genes)

        kinetics = _get_kinetics_data(
            sub,
            fit_type="nlls",
            slot=slot,
            time=time,
            return_fields="Half-life",
            show_progress=False,
            **kwargs
        ).get(f"kinetics_{cond}")

        half_live = kinetics.values.squeeze()

        use_mask_genes = compute_use_mask(half_live, totals, data.coldata[time].values, n_top_genes)

        # use_genes = select_top_genes_by_hl_and_expression(
        #     half_lives=half_live,
        #     totals=totals,
        #     time_vector=data.coldata[time].values,
        #     gene_names=gene_names,
        #     n_top_genes=n_top_genes
        # )

        sub = sub[use_mask_genes].with_dropped_analyses()
        init = sub.coldata[time]
        init_array = init.values

        use_mask_columns = (init_array > 0) & (init_array < init_array.max())
        if not np.any(use_mask_columns):
            continue

        def opt_fun(times):
            tt = init_array.copy()
            tt[use_mask_columns] = times
            kin = _get_kinetics_data(
                sub,
                fit_type="nlls",
                slot=slot,
                time=tt,
                return_fields="log_likelihood",
                show_progress=False,
                **kwargs
            ).get(f"kinetics_{cond}")

            loglik_array = kin[f"{cond}_log_likelihood"].values
            loglik_sum = np.sum(loglik_array)

            eval_bar.update(1)
            return -loglik_sum

        def opt_fun_scalar(mini):
            shifted = init_array[use_mask_columns] - mini
            return opt_fun(shifted)

        res_scalar = minimize_scalar(
            opt_fun_scalar,
            bounds=(0, init_array[use_mask_columns].min()),
            method='bounded'
        )
        mini = res_scalar.x
        init_array[use_mask_columns] -= mini

        # This would be much faster, but CIs are overestimated by this approximation.
        # if compute_confidence:
        #     res = minimize(
        #         opt_fun,
        #         init_array[use_mask],
        #         method="L-BFGS-B",
        #         options={"maxiter": max_iterations, "gtol": 1e-6}
        #     )
        #
        #     inv_hess = res.hess_inv
        #
        #     try:
        #         inv_hess = inv_hess.todense()
        #     except:
        #         inv_hess = np.array(inv_hess)
        #
        #     std_err = np.sqrt(np.diag(inv_hess))
        #     z = norm.ppf(1 - (1 - ci_size) / 2)
        #
        #     ci_half = std_err * z
        #
        #     conf_tt = np.zeros_like(init_array)
        #     conf_tt[use_mask] = ci_half
        #
        #     result.loc[sub.coldata.index, name + "_conf"] = conf_tt

        res = minimize(
            opt_fun,
            init_array[use_mask_columns],
            method="Nelder-Mead",
            options={'maxiter': max_iterations}
        )

        eval_bar.close()

        if not res.success:
            warnings.warn(f"The Optimization failed for {cond} with the following message: {res.message}")

        final_tt = init_array.copy()
        final_tt[use_mask_columns] = res.x
        result.loc[sub.coldata.index, name] = final_tt

        if compute_confidence:
            try:
                import numdifftools as nd

                bar = tqdm(desc=f"Computing confidence {cond}", unit=" Evaluations", dynamic_ncols=True, disable=not show_progress)

                # wrapper function for progress bar
                def wrapped_fun(x):
                    bar.update(1)
                    return opt_fun(x)

                hess = nd.Hessian(wrapped_fun)(res.x)
                bar.close()

                cov = np.linalg.inv(hess)
                std_err = np.sqrt(np.diag(cov))
                z = norm.ppf(1 - (1 - ci_size) / 2)
                ci_half = std_err * z

                conf_tt = np.zeros_like(init_array)
                conf_tt[use_mask_columns] = ci_half

                result.loc[sub.coldata.index, name + "_conf"] = conf_tt

            except Exception as e:
                warnings.warn(f"Confidence interval computation failed for {cond}: {e}")

    return result

