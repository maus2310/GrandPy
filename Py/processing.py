from numbers import Number
from scipy.stats import beta
import numpy as np
from math import log
import pandas as pd
from typing import TYPE_CHECKING, Union, Sequence
import warnings

if TYPE_CHECKING:
    from Py.grandPy import GrandPy
    from Py.slot_tool import ModeSlot

def _comp_hl(p, time=1):
    """Computes half-life from NTR-value p and time t"""
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = log(2) / (-1.0 / time * np.log(1 - p))
        hl = np.where(np.isfinite(hl), hl, np.nan)
    return hl

def _comp_tpm(cmat: np.ndarray, lengths: np.ndarray, subset: np.ndarray = None) -> np.ndarray:
    """Computes TPM from count matrix and transcript lengths"""
    lengths = lengths.copy()
    zerolen = lengths == 0
    lengths[zerolen] = 1

    rpk = cmat / (lengths[:, np.newaxis] / 1000)
    rpk[zerolen, :] = np.nan

    if subset is not None:
        scale = np.nansum(rpk[subset, :], axis=0) / 1e6
    else:
        scale = np.nansum(rpk, axis=0) / 1e6

    tpm = rpk / scale
    return tpm

def _comp_fpkm(cmat: np.ndarray, lengths: np.ndarray, subset: Sequence[Union[Number, str]] = None) -> np.ndarray:
    """Computes FPKM from count matrix and transcript lengths"""
    if subset is not None:
        scale = np.nansum(cmat[subset, :], axis=0) / 1e6
    else:
        scale = np.nansum(cmat, axis=0) / 1e6

    rpm = cmat / scale

    lengths = np.asarray(lengths)  # ← fix für pandas Series
    zero_len = lengths == 0
    lengths[zero_len] = 1

    fpkm = rpm / (lengths[:, np.newaxis] / 1000)
    fpkm[zero_len, :] = np.nan

    return fpkm

def _comp_rpm(cmat: np.ndarray, subset: np.ndarray = None, factor: float = 1e6) -> np.ndarray:
    """Computes RPM (reads per million) from count matrix"""
    if subset is not None:
        scale = np.nansum(cmat[subset, :], axis=0) / factor
    else:
        scale = np.nansum(cmat, axis=0) / factor

    rpm = cmat / scale
    return rpm

def compute_ntr_posterior_quantile(data: "GrandPy", quantile: float, name: str) -> "GrandPy":
    """
    Compute a posterior quantile of the NTR beta distribution and store as a new slot.

    Parameters:
    - data: GrandPy object
    - quantile: float in [0,1], quantile to compute
    - name: str, name of the new slot
    """
    alpha = data.get_matrix(mode_slot="alpha")
    beta_ = data.get_matrix(mode_slot="beta")

    q = beta.ppf(quantile, alpha, beta_)
    return data.with_slot(name, q)

def compute_ntr_posterior_lower(data: "GrandPy", ci_size: float = 0.95, name: str = "lower") -> "GrandPy":
    """
    Compute lower bound of the NTR credible interval.
    """
    quantile = (1 - ci_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def compute_ntr_posterior_upper(data: "GrandPy", ci_size: float = 0.95, name: str = "upper") -> "GrandPy":
    """
    Compute upper bound of the NTR credible interval.
    """
    quantile = 1 - (1 - ci_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def _compute_ntr_ci(data: "GrandPy", ci_size: float = 0.95, name_lower: str = "lower", name_upper: str = "upper") -> "GrandPy":
    """
    Compute both lower and upper bounds of the credible interval.
    """
    data = compute_ntr_posterior_lower(data, ci_size, name_lower)
    data = compute_ntr_posterior_upper(data, ci_size, name_upper)
    return data

# Beispielaufruf: sars = sars.compute_steady_state_half_lives(compute_ci=True)
#                 print(sars.get_analysis_table())
def _compute_steady_state_half_lives(
    data: "GrandPy",
    time = None,
    name="HL",
    columns=None,
    max_hl=48.0,
    ci_size=0.95,
    compute_ci=False,
    as_analysis=False
) -> "GrandPy":
    if time is None:
        time = data.coldata["Time"]

    if isinstance(time, str):
        time = data.coldata[time]

    ntrs = data.get_table(mode_slots="ntr", name_genes_by="Symbol")

    if np.isscalar(time):
        time = pd.Series([time] * ntrs.shape[1], index=ntrs.columns)
    else:
        time = pd.Series(time, index=ntrs.columns)

    if columns is None:
        selected_columns = list(ntrs.columns)
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    elif isinstance(columns, list):
        selected_columns = columns
    else:
        raise ValueError("Unsupported column specification for `columns`.")

    time = time[selected_columns]
    if len(selected_columns) != ntrs.shape[1]:
        as_analysis = True

    if compute_ci:
        as_analysis = True

        existing_slots = data.slots
        needs_lower = "lower" not in existing_slots
        needs_upper = "upper" not in existing_slots

        if needs_lower or needs_upper:
            data = data.compute_ntr_ci(ci_size=ci_size)

        lower = data.get_table(mode_slots="lower", name_genes_by="Symbol")
        upper = data.get_table(mode_slots="upper", name_genes_by="Symbol")

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
        return data.with_analysis(name, hls)
    else:
        return data.with_slot(name, hls)


def filter_genes(
    data: "GrandPy",
    mode_slot: Union[str, "ModeSlot"]=None,
    min_expression=100,
    min_columns=None,
    min_condition=None,
    use=None,
    keep=None,
    return_genes=False
) -> Union["GrandPy", list[str]]:

    if use is not None and keep is not None:
        raise ValueError("Do not specify both use and keep!")

    if mode_slot is None:
        mode_slot = data.default_slot

    if not data._check_slot(mode_slot):
        raise ValueError(f"Slot '{mode_slot}' unknown!")

    if use is None:
        aggregation_matrix = None
        if min_condition is not None:
            aggregation_matrix = data.get_summary_matrix(no4sU=True, average=False)

        matrix = data.get_table(mode_slots=mode_slot, summarize=aggregation_matrix)

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

# TODO: normalize hat Probleme, wenn Nullwerte in den Daten sind(sars.tsv).
def _normalize(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "norm",
    slot: str = "count",
    set_to_default: bool = True,
    size_factors: np.ndarray = None,
    return_size_factors: bool = False
) -> Union["GrandPy", np.ndarray]:
    """
    DESeq2-ähnliche Normalisierung einer Slot-Matrix durch Size Factors.

    Parameters
    ----------
    data : GrandPy
        Das GrandPy-Objekt.
    genes : list[str] oder bool-Maske, optional
        Gene zur Berechnung der Size Factors. Default: alle.
    name : str
        Name des neuen Slots.
    slot : str
        Slot zur Normalisierung, z.B. "count".
    set_to_default : bool
        Ob der neue Slot als default gesetzt wird.
    size_factors : np.ndarray, optional
        Falls gegeben, verwende diese Size Factors direkt.
    return_size_factors : bool
        Wenn True, gib die Size Factors zurück.

    Returns
    -------
    GrandPy oder np.ndarray
        Normalisiertes GrandPy-Objekt oder Size Factors.
    """

    matrix_for_size = data.get_matrix(slot, genes=genes)

    if size_factors is None:
        if matrix_for_size.ndim == 1:
            matrix_for_size = matrix_for_size[np.newaxis, :]

        jitter = 1e-8
        safe_matrix = np.where(matrix_for_size == 0, jitter, matrix_for_size)
        log_mat = np.log(safe_matrix)
        log_geomeans = np.mean(log_mat, axis=1)

        # Größe des Arrays
        n_cols = safe_matrix.shape[1]
        size_factors = np.zeros(n_cols)

        for i in range(n_cols):
            counts = safe_matrix[:, i]
            valid = np.isfinite(log_geomeans) & (counts > 0)
            diffs = np.log(counts[valid]) - log_geomeans[valid]
            size_factors[i] = np.exp(np.median(diffs))

    if return_size_factors:
        return size_factors

    matrix_for_normalization = data.get_matrix(slot)

    normalized_matrix = matrix_for_normalization / size_factors

    return data.with_slot(name, normalized_matrix, set_to_default=set_to_default)

def _normalize_fpkm(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "fpkm",
    slot: str = "count",
    set_to_default: bool = True,
    total_len = None ):

    if total_len is None:
        total_len = data.gene_info["Length"]

    genes = data.get_index(genes)
    mat_for_fpkm = data.get_matrix(slot, genes=genes)
    mask = np.zeros(len(mat_for_fpkm), dtype=bool)
    mask[genes] = True
    final_mat = _comp_fpkm(cmat = mat_for_fpkm, lengths=total_len, subset = mask)

    return data.with_slot(name, final_mat, set_to_default=set_to_default)