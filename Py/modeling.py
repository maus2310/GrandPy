import os
import warnings
import numpy as np
import pandas as pd
from collections.abc import Sequence, Mapping
from typing import Union, Literal, TYPE_CHECKING, Callable
from functools import cached_property
from dataclasses import dataclass
from scipy.optimize import least_squares, OptimizeResult, brentq


from Py.slot_tool import ModeSlot
from Py.utils import _ensure_list

if TYPE_CHECKING:
    from Py.grandPy import GrandPy


def fit_kinetics(
    data: "GrandPy",
    fit_type: Literal["nlls", "ntr", "chase"] = "nlls",
    *,
    slot: str = None,
    name_prefix: Union[str, None] = None,
    return_fields: Union[str, Sequence[str]] = None,
    time: Union[str, np.ndarray, pd.Series, list] = "Time",
    ci_size: float = 0.95,
    genes: Union[str, Sequence[str]] = None,
    max_processes: int = None,
    show_progress: bool = True,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    Wrapper for fit_kinetics_nlls, fit_kinetics_chase, and fit_kinetics_ntr.

    For detailed documentation, see grandPy.GrandPy.fit_kinetics.
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
                            ci_size=ci_size, genes=genes, max_processes=max_processes, show_progress=show_progress, **kwargs)

    return kinetics


def get_dynamic_process_count(data_size: int, max_processes: int = None) -> int:
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
    available_cores = max(os.cpu_count()-1, 1)

    if max_processes is None:
        # Arbitrary threshold for amount of processes approximated by testing. (Probably different for other systems)
        num_processes = min(available_cores, data_size // 1000)
    else:
        num_processes = min(max_processes, available_cores)

    num_processes = max(1, num_processes)

    return num_processes


def correct(all_expressions: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Applies a correction to every row of `all_expressions`.
    """
    for i in range(all_expressions.shape[0]):
        if np.max(all_expressions[i, :]) == 0:
            all_expressions[i, time == 1] = 0.01
    return all_expressions



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
    show_progress: bool = True,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    For a detailed documentation, see grandpy.GrandPy.fit_kinetics.
    """
    if show_progress:
        from tqdm import tqdm

    genes_to_fit = data.get_genes(genes)

    condition_vector = data.coldata["Condition"].values
    unique_conditions = np.unique(condition_vector)

    # --- Decide on parallelisation ---
    datasize = len(genes_to_fit) * len(unique_conditions)

    max_workers = get_dynamic_process_count(datasize, max_processes)

    if max_workers == 1:
        parallel = False
    else:
        from concurrent.futures import ProcessPoolExecutor

        parallel = True

    # --- Map steady_state to each condition ---
    if isinstance(steady_state, bool):
        steady_state = {cond: steady_state for cond in unique_conditions}

    # --- Retrieve expression matrices ---
    new_expression = correct(data.get_matrix(mode_slot=ModeSlot("new", slot), genes=genes_to_fit), time=time)
    old_expression = correct(data.get_matrix(mode_slot=ModeSlot("old", slot), genes=genes_to_fit), time=time)

    result = {}

    # --- Call the fitting function for each gene for each condition. ---
    for condition in unique_conditions:
        idx = np.where(condition_vector == condition)[0]
        time_cond = time[idx]
        new_cond = new_expression[:, idx]
        old_cond = old_expression[:, idx]
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
                        **kwargs
                    )
                    jobs.append((gene, job))

                rows = []
                symbols = []

                if show_progress:
                    for gene, future in tqdm(jobs, desc=f"Fitting {condition}", total=len(jobs)):
                        res = future.result()
                        series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                        rows.append(series.values)
                        symbols.append(gene)
                else:
                    for gene, future in jobs:
                        res = future.result()
                        series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                        rows.append(series.values)
                        symbols.append(gene)

        else:
            rows = []
            symbols = []

            gene_index_iterator = enumerate(genes_to_fit)
            if show_progress:
                gene_index_iterator = tqdm(gene_index_iterator, total=len(genes_to_fit), desc=f"Fitting {condition}")

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
    steady_state: Union[bool, Mapping[str, bool]] = True,
    max_processes: int = None,
    show_progress: bool = True,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    For a detailed documentation, see grandpy.GrandPy.fit_kinetics.
    """
    if show_progress:
        from tqdm import tqdm

    genes_to_fit = data.get_genes(genes)

    condition_vector = data.coldata["Condition"].values
    unique_conditions = np.unique(condition_vector)

    # --- Decide on parallelisation ---
    datasize = len(genes_to_fit) * len(unique_conditions)

    max_workers = get_dynamic_process_count(datasize, max_processes)

    if max_workers == 1:
        parallel = False
    else:
        from concurrent.futures import ProcessPoolExecutor

        parallel = True

    # --- Map steady_state to each condition ---
    if isinstance(steady_state, bool):
        steady_state = {cond: steady_state for cond in unique_conditions}

    for cond, state in steady_state.items(): # Ensure steady-state for all conditions
        if state != True:
            warnings.warn(f"'steady_state' for condition {cond} is set to False. This is not supported for pulse chase designs. Continuing with steady-state assumption.")
            steady_state[cond] = True

    # --- Retrieve expression matrix ---
    new_expression = correct(data.get_matrix(mode_slot="ntr", genes=genes_to_fit), time=time)

    # --- chase-specific preprocessing ---
    no4sU_mask = data.coldata["no4sU"].values
    new_expression = new_expression[:, ~no4sU_mask]

    condition_vector = condition_vector[~no4sU_mask]
    time = time[~no4sU_mask]

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
        condition_steady_state = steady_state[condition]

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
                        steady_state=condition_steady_state,
                        **kwargs
                    )
                    jobs.append((gene, job))

                rows = []
                symbols = []

                if show_progress:
                    for gene, future in tqdm(jobs, desc=f"Fitting {condition}", total=len(jobs)):
                        res = future.result()
                        series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                        rows.append(series.values)
                        symbols.append(gene)
                else:
                    for gene, future in jobs:
                        res = future.result()
                        series = res.to_series(condition=condition, prefix=name_prefix, fields=return_fields)
                        rows.append(series.values)
                        symbols.append(gene)

        else:
            rows = []
            symbols = []

            gene_index_iterator = enumerate(genes_to_fit)
            if show_progress:
                gene_index_iterator = tqdm(gene_index_iterator, total=len(genes_to_fit),
                                           desc=f"Fitting {condition}")

            for gene_index, gene in gene_index_iterator:
                new_values = new_cond[gene_index, :]

                res = fit_kinetics_gene_least_squares(
                    new_values=new_values,
                    old_values=np.zeros_like(new_values),
                    time=time_cond,
                    ci_size=ci_size,
                    chase=True,
                    total_value=slot_values_per_gene.get(gene, None),
                    steady_state=condition_steady_state,
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
) -> "FitResult":
    """
    Fits synthesis and degradation rates for a single gene and condition using non-linear least squares.

    This function implements a least-squares optimization to infer kinetic parameters from time-resolved data.
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
        Wheter to use steady for a condition.

    total_value : float, optional
       Optional total expression values used to estimate initial concentration in chase mode.

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
        steady_state=steady_state
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
    """
    time: np.ndarray = np.nan
    new_values: np.ndarray = np.nan
    old_values: np.ndarray = np.nan
    chase: bool = False
    slot_total: float = None
    opt_result: OptimizeResult = None
    ci_size: float = 0.95
    steady_state: bool = True

    # --- Core Parameters ---
    @cached_property
    def synthesis(self) -> float:
        return self.opt_result.x[0] if self.opt_result is not None else np.nan

    @cached_property
    def return_synthesis(self):
        if self.chase:
            return self.slot_total * self.degradation
        else:
            return self.synthesis

    @cached_property
    def degradation(self) -> float:
        return self.opt_result.x[1] if self.opt_result is not None else np.nan

    @cached_property
    def inv_deg(self) -> float:
        return 1.0 / self.degradation if self.degradation > 0 else np.nan

    # --- Cached exponentials ---
    @cached_property
    def exp_old(self) -> np.ndarray:
        return np.exp(-self.degradation * self.time)

    @cached_property
    def exp_new(self) -> np.ndarray:
        return np.exp(-self.degradation * self.time)

    # --- Predictions ---
    @cached_property
    def pred_old(self) -> np.ndarray:
        if self.chase:
            return np.zeros_like(self.time)
        return self.synthesis * self.inv_deg * self.exp_old

    @cached_property
    def pred_new(self) -> np.ndarray:
        if self.chase:
            return self.synthesis * self.inv_deg * self.exp_new
        return self.synthesis * self.inv_deg * (1 - self.exp_new)

    # --- Residuals ---
    @cached_property
    def residuals_raw(self) -> np.ndarray:
        return np.concatenate([
            self.old_values - self.pred_old,
            self.new_values - self.pred_new
        ])

    @cached_property
    def residuals(self) -> dict[str, np.ndarray]:
        if self.chase:
            expected = self.pred_new
            observed = self.new_values
        else:
            expected = np.concatenate([self.pred_old, self.pred_new])
            observed = np.concatenate([self.old_values, self.new_values])

        rel = expected / (observed + 1e-8)

        return {
            "Absolute": expected,
            "Relative": rel
        }

    # --- Metrics ---
    @cached_property
    def rmse(self) -> float:
        if self.chase:
            return np.sqrt(np.sum(self.residuals_raw ** 2)/(self.residuals_raw.size))
        return np.sqrt(np.sum(self.residuals_raw ** 2)/self.residuals_raw.size)
        # return np.sqrt(np.mean(self.residuals_raw ** 2)) if self.residuals_raw.size > 0 else np.nan

    @cached_property
    def rmse_old(self) -> float:
        if self.chase or self.old_values.size == 0:
            return np.nan
        return np.sqrt(np.mean((self.old_values - self.pred_old) ** 2))

    @cached_property
    def rmse_new(self) -> float:
        if self.new_values.size == 0:
            return np.nan
        return np.sqrt(np.mean((self.new_values - self.pred_new) ** 2))

    @cached_property
    def half_life(self) -> float:
        return np.log(2) / self.degradation if self.degradation > 0 else np.nan

    @cached_property
    def log_likelihood(self) -> float:
        if self.chase:
            N = self.residuals_raw.size/2
        else:
            N = self.residuals_raw.size
        return -N * (np.log(2 * np.pi) + 1 - np.log(N) + np.log(sum(self.residuals_raw ** 2)))/2
        # return -0.5 * np.sum(self.residuals_raw ** 2)

    @cached_property
    def total_expr(self) -> float:
        if self.chase and self.slot_total is not None:
            return self.slot_total
        return np.sum(self.old_values) + np.sum(self.new_values)

    @cached_property
    def f0(self) -> float:
        if self.opt_result is None:
            return np.nan
        if self.steady_state:
            return self.synthesis / self.degradation if self.degradation > 0 else np.nan
        if len(self.opt_result.x) >= 3:
            return self.opt_result.x[2]
        return np.nan

    @cached_property
    def ci_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.opt_result is None or self.opt_result.jac is None:
            return np.full(2, np.nan), np.full(2, np.nan)

        try:
            from scipy.stats import t

            J = self.opt_result.jac
            if J.shape[1] != 2:
                return np.full(2, np.nan), np.full(2, np.nan)

            res = self.residuals_raw
            dof = len(res) - J.shape[1]
            if dof <= 0:
                return np.full(2, np.nan), np.full(2, np.nan)

            sigma2 = np.sum(res ** 2) / dof
            cov = sigma2 * np.linalg.pinv(J.T @ J)
            se = np.sqrt(np.diag(cov))
            tval = t.ppf(1 - (1 - self.ci_size) / 2, df=dof)

            lower = self.opt_result.x[:2] - tval * se
            upper = self.opt_result.x[:2] + tval * se
            return lower, upper

        except (np.linalg.LinAlgError, ValueError):
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
                from collections import defaultdict

                for res_type, values in val.items():
                    n_old = len(self.time)
                    rep_counter = defaultdict(int)

                    old_i = 0
                    new_i = 0

                    for i, v in enumerate(values):
                        if self.chase:
                            kind = "New"
                            time = self.time[i]
                            key = (kind, time)
                            r = rep_counter[key]
                            rep_counter[key] += 1
                        else:
                            if i < n_old:
                                kind = "Old"
                                time = self.time[old_i]
                                key = (kind, time)
                                r = rep_counter[key]
                                rep_counter[key] += 1
                                old_i += 1
                            else:
                                kind = "New"
                                time = self.time[new_i]
                                key = (kind, time)
                                r = rep_counter[key]
                                rep_counter[key] += 1
                                new_i += 1

                        full_key = f"{base_key}_{res_type}_{kind}_t{time:g}_r{r}"
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
            ro = np.zeros_like(v_old) if chase else v_old - f_old_equi(time, s, d)
            rn = v_new - (f_old_equi(time, s, d) if chase else f_new(time, s, d))
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
        pred_old = f_old_nonequi(time, f0, s, d)
        pred_new = s / d * (1 - np.exp(-d * time))
        return np.concatenate([v_old - pred_old, v_new - pred_new])

    def jocobian_function(par):
        s, d, f0 = par
        exp_o = np.exp(-d * time)
        exp_n = np.exp(-d * time)

        # r_old derivatives
        j_old_s = np.zeros_like(time)
        j_old_d = f0 * time * exp_o
        j_old_f0 = -exp_o

        # r_new derivatives
        j_new_s = -(1 - exp_n) / d
        j_new_d = s / d ** 2 * (1 - exp_n) - s / d * time * exp_n
        j_new_f0 = np.zeros_like(time)

        J = np.vstack([
            np.concatenate([j_old_s, j_new_s]),
            np.concatenate([j_old_d, j_new_d]),
            np.concatenate([j_old_f0, j_new_f0])
        ]).T
        return J

    return residual_function, jocobian_function


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
        max_processes: int = None,
        show_progress: bool = True
) -> dict[str, pd.DataFrame]:
    """
    For detailed documentation, see grandpy.GrandPy.fit_kinetics.

    This function switches between serialisation and parallelisation based on `processes`.
    """
    if not ("alpha" in data.slots and "beta" in data.slots):
        raise ValueError("NTR-basierte Anpassung erfordert alpha-, beta-Slots.")

    if show_progress:
        from tqdm import tqdm

    genes_to_fit = data.get_genes(genes)

    condition_vector = data.coldata["Condition"].values
    unique_conditions = np.unique(condition_vector)

    # --- Decide on parallelisation ---
    # datasize = (len(genes_to_fit) * len(unique_conditions)) // 5
    #
    # max_workers = get_dynamic_process_count(datasize, max_processes)
    #
    # print(max_workers)
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
        if show_progress:
            gene_iter = tqdm(gene_iter, total=len(genes_to_fit), desc=f"Fitting {condition}")

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

    return NTRFitResult(
        result=result.x,
        time=time,
        alpha=alpha,
        beta=beta,
        ntr=ntr_values,
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
        return np.sqrt(np.mean((self.predicted_ntr - self.ntr) ** 2))

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
