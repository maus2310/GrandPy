from collections.abc import Sequence
from typing import TYPE_CHECKING, Union, Optional, Callable

import numpy as np
import pandas as pd

from .lfc import psi_lfc

if TYPE_CHECKING:
    from .core_grandpy import GrandPy



def _comp_hl(
    p,
    time : int = 1
):
    """
    Computes half-life from NTR-value p and time t
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = np.log(2) / (-1.0 / time * np.log(1 - p))
        hl = np.where(np.isfinite(hl), hl, np.nan)
    return hl

def _comp_tpm(
    count_matrix: np.ndarray,
    lengths: np.ndarray,
    subset: list[int] = None
) -> np.ndarray:
    """
    Computes the TPM (transcripts per million) from the given slot and gene lengths.

    Parameters
    ----------
    count_matrix : np.ndarray
        the count-matrix.

    lengths : np.ndarray
        Lengths of the genes.

    subset : int, list[int], np.ndarray, optional
        index or indices of genes to subset.

    Returns
    -------
    np.ndarray
        TPM-normalisierte Matrix (shape: Gene × Samples)
    """
    lengths = lengths.copy()
    zero_len = lengths == 0
    lengths[zero_len] = 1

    reads_per_kilo = count_matrix / (np.atleast_2d(lengths / 1000).transpose())

    if subset is not None:
        subset = np.atleast_1d(subset)
        scale = np.nansum(reads_per_kilo[subset, :], axis=0) / 1e6
    else:
        scale = np.nansum(reads_per_kilo, axis=0) / 1e6

    tpm = reads_per_kilo / scale

    tpm[zero_len, :] = np.nan

    return tpm


def _comp_fpkm(
        count_matrix: np.ndarray,
        lengths: np.ndarray,
        subset: Union[Sequence[int], np.ndarray, None] = None
) -> np.ndarray:
    """
    Calculates FPKM from a count matrix and transcript lengths.

    Parameters
    ----------
    count_matrix : np.ndarray
        Count matrix (Genes × Samples)

    lengths : np.ndarray
        Lengths of the transcripts.

    subset : Optional[Sequence[int]]
        Which genes should be used for scaling (RPM).

    Returns
    -------
    np.ndarray
        FPKM matrix
    """
    if subset is not None:
        subset = np.atleast_1d(subset)
        scale = np.nansum(count_matrix[subset, :], axis=0) / 1e6
    else:
        scale = np.nansum(count_matrix, axis=0) / 1e6

    scale[scale == 0] = np.nan

    rpm = count_matrix / scale

    zerolen = lengths == 0
    lengths = lengths.copy()
    lengths[zerolen] = 1

    fpkm = rpm / (lengths[:, np.newaxis] / 1000)
    fpkm[zerolen, :] = np.nan

    return fpkm

def _comp_rpm(
    count_matrix: np.ndarray,
    subset: Union[Sequence[int], np.ndarray, None] = None,
    factor: float = 1e6
) -> np.ndarray:

    """
    Computes the RPM (Reads per Million) from a count-matrix.

    Parameters
    ----------
    count_matrix : np.ndarray
        The matrix to compute.

    subset : list of indices, optional
        Only the given genes are used for the computation.

    factor : float, default 1e6
        scaling factor.

    Returns
    -------
    np.ndarray
        RPM-normalized matrix
    """

    count_matrix = np.asarray(count_matrix)

    if subset is not None:
        subset = np.atleast_1d(subset)
        scale = np.nansum(count_matrix[subset, :], axis=0) / factor
    else:
        scale = np.nansum(count_matrix, axis=0) / factor

    scale[scale == 0] = np.nan

    rpm = count_matrix / scale

    return rpm

def compute_ntr_posterior_quantile(
    data: "GrandPy",
    quantile: float,
    name: str
) -> "GrandPy":

    """
    Compute a posterior quantile of the NTR beta distribution and store as a new slot.

    Parameters:

    quantile: float in [0,1]
        quantile to compute
    name: str
        name of the new slot
    """

    from scipy.stats import beta

    alpha = data.get_matrix(mode_slot="alpha")
    beta_ = data.get_matrix(mode_slot="beta")

    q = beta.ppf(quantile, alpha, beta_)
    return data.with_slot(name, q)

def compute_ntr_posterior_lower(
    data: "GrandPy",
    ci_size: float = 0.95,
    name: str = "lower"
) -> "GrandPy":

    """
    Compute lower bound of the NTR credible interval.
    """
    quantile = (1 - ci_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def compute_ntr_posterior_upper(
    data: "GrandPy",
    ci_size: float = 0.95,
    name: str = "upper"
) -> "GrandPy":

    """
    Compute upper bound of the NTR credible interval.
    """
    quantile = 1 - (1 - ci_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def _compute_ntr_ci(
    data: "GrandPy",
    ci_size: float = 0.95,
    name_lower: str = "lower",
    name_upper: str = "upper"
) -> "GrandPy":

    """
    Compute both lower and upper bounds of the credible interval.
    """
    data = compute_ntr_posterior_lower(data, ci_size, name_lower)
    data = compute_ntr_posterior_upper(data, ci_size, name_upper)
    return data


def _compute_steady_state_half_lives(
    data: "GrandPy",
    time : str = None,
    name : str = "hl",
    columns : list[str] = None,
    max_hl: float = 48.0,
    ci_size : float = 0.95,
    compute_ci : bool = False,
    as_analysis : bool = False
) -> "GrandPy":
    """
    For a detailed documentation see GrandPy.compute_steady_state_half_lives.
    """
    if time is None:
        time = data.coldata["duration.4sU"]

    if isinstance(time, str):
        time = data.coldata[time]

    ntrs = data.get_table(mode_slot="ntr", name_genes_by="Symbol")

    if np.isscalar(time):
        time = pd.Series([time] * ntrs.shape[1], index=ntrs.columns)
    else:
        time = pd.Series(time, index=ntrs.columns)

    if columns is None:
        selected_columns = list(ntrs.columns)
    else:
        selected_columns = data.get_columns(columns)
        as_analysis = True

    time = time[selected_columns]

    if compute_ci:
        as_analysis = True

        existing_slots = data.slots
        needs_lower = "lower" not in existing_slots
        needs_upper = "upper" not in existing_slots

        if needs_lower or needs_upper:
            data = data.compute_ntr_ci(ci_size=ci_size)

        lower = data.get_table(mode_slot="lower", name_genes_by="Symbol")
        upper = data.get_table(mode_slot="upper", name_genes_by="Symbol")

        frames = []
        for col in selected_columns:
            col_time = time[col]
            lower_hl = np.minimum(_comp_hl(upper[col].values, col_time), max_hl)
            map_hl   = np.minimum(_comp_hl(ntrs[col].values, col_time), max_hl)
            upper_hl = np.minimum(_comp_hl(lower[col].values, col_time), max_hl)

            df = pd.DataFrame({
                f"{name}.Half-life.lower.{col}": lower_hl,
                f"{name}.Half-life.MAP.{col}": map_hl,
                f"{name}.Half-life.upper.{col}": upper_hl
            }, index=ntrs.index)
            frames.append(df)

        hls = pd.concat(frames, axis=1)

    else:
        hls = pd.DataFrame({
            col: np.minimum(_comp_hl(ntrs[col].values, time[col]), max_hl)
            for col in selected_columns
        }, index=ntrs.index)

    if as_analysis:
        return data.with_analysis(name=name, table=hls)
    else:
        return data.with_slot(name=name, value=hls)


def _filter_genes(
    data: "GrandPy",
    mode_slot: Union[str, "ModeSlot"] = "count",
    min_expression: int = 100,
    min_columns: int = None,
    min_condition: int = None,
    keep: Union[str, int, Sequence[Union[int, str]]] = None,
    use: Union[str, int, Sequence[Union[int, str, bool]]] = None,
    return_genes: bool = False
) -> Union["GrandPy", list[str]]:
    """
    For a detailed documentation see GrandPy.filter_genes.
    """
    if use is not None and keep is not None:
        raise ValueError("use and keep cannot be used together. Use only one of them!")

    if use is None:
        aggregation_matrix = None
        if min_condition is not None:
            aggregation_matrix = data.get_summarize_matrix(no4su=True, average=False)
            min_columns = min_condition

        matrix = data.get_table(mode_slot=mode_slot, summarize=aggregation_matrix)

        if min_columns is None:
            min_columns = min_condition if min_condition is not None else matrix.shape[1] / 2

        use_mask = (matrix >= min_expression).sum(axis=1) >= min_columns

        if keep is not None:
            keep_idx = data.get_index(keep)
            use_mask |= np.isin(np.arange(matrix.shape[0]), keep_idx)

        use = use_mask

    gene_idx = data.get_index(use)

    if return_genes:
        return data.get_genes(gene_idx)

    # apply adjusts slots according to changes to gene_info
    return data._apply(
        function_gene_info=lambda t: t.iloc[gene_idx, :]
    )

def _normalize(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "norm",
    slot: str = "count",
    set_to_default: bool = True,
    size_factors: Union[np.ndarray, Sequence[float]] = None,
    return_size_factors: bool = False
) -> Union["GrandPy", np.ndarray]:
    """
    For detailed documentation see GrandPy.normalize.
    """
    matrix_for_size = data.get_matrix(slot, genes=genes)

    if size_factors is None:
        if matrix_for_size.ndim == 1:
            matrix_for_size = matrix_for_size[np.newaxis, :]

        jitter = 1e-8
        safe_matrix = np.where(matrix_for_size == 0, jitter, matrix_for_size)
        log_mat = np.log(safe_matrix)
        log_geomeans = np.mean(log_mat, axis=1)

        n_cols = safe_matrix.shape[1]
        size_factors = np.zeros(n_cols)

        for i in range(n_cols):
            counts = safe_matrix[:, i]
            valid = np.isfinite(log_geomeans) & (counts > 0)
            diffs = np.log(counts[valid]) - log_geomeans[valid]
            size_factors[i] = np.exp(np.median(diffs))

    size_factors = np.asarray(size_factors)

    if return_size_factors:
        return size_factors

    matrix_for_normalization = data.get_matrix(slot)

    normalized_matrix = matrix_for_normalization / size_factors

    return data.with_slot(name=name, value=normalized_matrix, set_to_default=set_to_default)

def _normalize_fpkm(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "fpkm",
    slot: str = "count",
    set_to_default: bool = True,
    total_len: Union[str, np.ndarray, Sequence[int]] = "Length"
) -> "GrandPy":
    """
    For a detailed documentation see GrandPy.normalize_fpkm.
    """
    if isinstance(total_len, str):
        total_len = data.gene_info[total_len]

    total_len = np.asarray(total_len)

    mat = data.get_matrix(slot, force_numpy=True)

    if genes is not None:
        subset_indices = data.get_index(genes=genes)
    else:
        subset_indices = None

    fpkm = _comp_fpkm(count_matrix=mat, lengths=total_len, subset=subset_indices)

    return data.with_slot(name=name, value=fpkm, set_to_default=set_to_default)

def _normalize_tpm(
        data: "GrandPy",
        genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
        name: str = "tpm",
        slot: str = "count",
        set_to_default: bool = True,
        total_len: np.ndarray = None
) -> "GrandPy":
    """
    For a detailed documentation see GrandPy.normalize_tpm.
    """
    if total_len is None:
        total_len = np.asarray(data.gene_info["Length"])

    count_matrix = data.get_matrix(slot)

    if genes is not None:
        subset_indices = data.get_index(genes=genes)
    else:
        subset_indices = None

    tpm_matrix = _comp_tpm(count_matrix, total_len, subset=subset_indices)

    return data.with_slot(name, tpm_matrix, set_to_default=set_to_default)

def _normalize_rpm(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "rpm",
    slot: str = "count",
    set_to_default: bool = True,
    factor: float = 1e6
) -> "GrandPy":
    """
    For a detailed documentation see GrandPy.normalize_rpm.
    """
    mat = data.get_matrix(slot)

    if genes is not None:
        subset_indices = data.get_index(genes=genes)
    else:
        subset_indices = None

    rpm = _comp_rpm(count_matrix=mat, subset=subset_indices, factor=factor)

    return data.with_slot(name=name, value=rpm, set_to_default=set_to_default)


def _normalize_baseline(
    data: "GrandPy",
    baseline: str = None,
    name: str = "baseline",
    slot: str = "count",
    set_to_default: bool = False,
    lfc_fun: Callable = psi_lfc,
    **kwargs
) -> "GrandPy":
    """
    For a detailed description see GrandPy.normalize_baseline.
    """

    # get unique conditions
    unique_conditions = pd.unique(data.condition)

    # no reference given: use first condition as reference
    if baseline is None:
        baseline = data.get_references(reference=f"Condition == '{unique_conditions[0]}'")

    data_matrix = data.get_matrix(mode_slot=slot)

    baseline_result = np.zeros_like(data_matrix)

    for i, col_name in enumerate(baseline.columns):

        ref_mask = baseline[col_name].values.astype(bool)

        if not np.any(ref_mask):
            raise ValueError(f"no reference found for '{col_name}'.")

        # compute lfc comparison between current- & reference-cell
        A = data_matrix[:, i:i + 1]  # (n_genes, 1)
        B = data_matrix[:, ref_mask].mean(axis=1, keepdims=True)  # reference cell

        result = lfc_fun(
            A=A,
            B=B,
            **kwargs
        )

        if isinstance(result, tuple):
            result = result[0]

        baseline_result[:, i] = result[:, 0]

    # add result as a new slot to the GrandPy-object
    return data.with_slot(name, baseline_result, set_to_default=set_to_default)

def _compute_total_expression(
    data: "GrandPy",
    name: str = "total_expression",
    genes: Union[Sequence[str], str] = None,
    mode_slot: str = None
) -> "GrandPy":
    """
    For a detailed documentation see GrandPy.compute_steady_state_half_lives.
    """
    matrix = data.get_matrix(mode_slot=mode_slot, genes=genes)
    total_expression = matrix.sum(axis=0)

    return data.with_coldata(name= name, value = total_expression)


def _compute_expression_percentage(
    data: "GrandPy",
    name: str,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    slot: str = None,
    genes_total: Union[str, Sequence[str]] = None,
    slot_total: str = None,
    percent_to_float: bool = True,
) -> "GrandPy":
    """
    For a detailed description see GrandPy.compute_expression_percentage.
    """
    numerator = data.get_matrix(mode_slot=slot, genes=genes).sum(axis=0)
    denominator = data.get_matrix(mode_slot=slot_total, genes=genes_total).sum(axis=0)

    percentage = numerator / denominator
    if percent_to_float:
        percentage *= 100

    data = data.with_coldata(name=name, value=percentage)
    return data
