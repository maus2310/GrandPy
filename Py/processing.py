from scipy.stats import beta
import numpy as np
from math import log
import pandas as pd
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from Py.grandPy import GrandPy

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

def _comp_fpkm(cmat: np.ndarray, lengths: np.ndarray, subset: np.ndarray = None) -> np.ndarray:
    """Computes FPKM from count matrix and transcript lengths"""
    if subset is not None:
        scale = np.nansum(cmat[subset, :], axis=0) / 1e6
    else:
        scale = np.nansum(cmat, axis=0) / 1e6

    rpm = cmat / scale

    lengths = lengths.copy()
    zerolen = lengths == 0
    lengths[zerolen] = 1

    fpkm = rpm / (lengths[:, np.newaxis] / 1000)
    fpkm[zerolen, :] = np.nan

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
    alpha = data.get_table(mode_slots="alpha", name_genes_by="Gene")
    beta_ = data.get_table(mode_slots="beta", name_genes_by="Gene")

    alpha = alpha.values
    beta_ = beta_.values

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
        time = data.coldata["Time_hr"]

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
                f"Half-life.lower.{col}": lower_hl,
                f"Half-life.MAP.{col}": map_hl,
                f"Half-life.upper.{col}": upper_hl
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
