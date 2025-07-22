import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from typing import Union, TYPE_CHECKING, Optional, Callable, Sequence, List

from pydeseq2.preprocessing import deseq2_norm

from grandpy.utils import _ensure_list
from grandpy.grandPy import GrandPy
from grandpy.slot_tool import ModeSlot

from scipy.special import digamma, polygamma
from scipy.stats import norm, beta, gmean
from scipy.optimize import minimize, root_scalar

try:
    from pydeseq2.ds import DeseqDataSet, DeseqStats
except ImportError:
    DeseqDataSet = DeseqStats = None

if TYPE_CHECKING:
    from grandpy.grandPy import GrandPy

def _get_summary_matrix(
        data: "GrandPy",
        no4sU: bool = False,
        columns: Union[None, str, list[str]] = None,
        average: bool = True) -> pd.DataFrame:
    coldata = data.coldata
    sample_names = coldata.index.tolist()

    if "Condition" not in coldata.columns:
        raise ValueError("Object does not have 'Condition' information!")

    if columns is None:
        columns = sample_names
    else:
        columns = _ensure_list(columns)
        columns = [c for c in columns if c in sample_names]

    # Exclude 4sU-marked samples if requested
    if not no4sU:
        no4su_samples = coldata.index[coldata["no4sU"]]
        columns = list(set(columns) - set(no4su_samples))

    # Mapping: sample_name → condition
    condition_series = coldata.loc[columns, "Condition"]

    # Create indicator matrix
    unique_conditions = condition_series.unique()
    matrix = pd.DataFrame(
        {
            cond: (condition_series == cond).astype(float)
            for cond in unique_conditions
        },
        index=condition_series.index
    )

    # Ensure all samples included, fill 0 if not matched
    all_samples = pd.Index(sample_names)
    matrix = matrix.reindex(index=all_samples).fillna(0)

    # Drop columns with all zeros
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]

    if average:
        matrix = matrix.div(matrix.sum(axis=0), axis=1).fillna(0)

    return matrix


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


def Psi_LFC(A: np.ndarray,
            B: np.ndarray,
            prior: Optional[tuple[float, float] | None] = None,
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

    prior : tuple[float, float] | None
        Prior pseudocounts (a, b). If None, estimated via empirical_bayes_prior(A, B).

    normalize_fun : callable, optional
        Function to normalize raw LFCs (default: median centering).

    cre : bool
        Compute credible intervals.

    verbose : bool
        If True status updates will be provided.

    Returns
    -------
    lfc_centered : np.ndarray
        Shrunk, median‑centered log2 fold‑changes (length = #genes).

    qlfc : np.ndarray, optional
        Credible interval matrix (genes x len(cre)); only if cre is not False.
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


def compute_lfc(data: GrandPy,
                name_prefix: Optional[str] = None,
                contrasts: Optional[pd.DataFrame] = None,
                slot: str = "count",
                LFC_fun: Callable = Psi_LFC,
                mode: str = "total",
                normalization: Optional[Union[str, Sequence[float]]] = None,
                compute_M: bool = True,
                genes: Optional[list[str]] = None,
                verbose: bool = False,
                **kwargs) -> GrandPy:
    """
    Estimate log2 fold changes and optional M values for each contrast and store analyses.

    Parameters
    ----------
    data : GrandPy
        The grandPy object. Must contain a 'Condition' column in its coldata.
    name_prefix : str
        The prefix for the new analysis name; e.g. 'total' or 'new'.
    contrasts : pd.DataFrame, optional
        Contrast matrix defining comparisons (samples x contrasts; values 1, -1).
    slot : str
        The slot of the grandPy object to take the data from (e.g. "count").
    LFC_fun : function
        Function to compute the log2 fold changes (default Psi_LFC).
    mode : str
        Computes LFCs for "total", "new", or "old" RNA.
    normalization : str or sequence, optional
        If str: name of normalization slot (e.g. "total");
        if sequence: size factors per sample.
    compute_M : bool, default True
        If True, include the "M" column (base mean) for each contrast.
    genes : list of str, optional
        Restrict analysis to these genes; None means all genes.
    verbose : bool
        If True, status updates will be printed.
    **kwargs
        Passed to LFC_fun.

    Returns
    -------
    GrandPy
        New GrandPy object with one analysis per contrast. Each analysis
        adds two columns named "{prefix}_{contrast}_LFC" and
        "{prefix}_{contrast}_M".

    """

    if name_prefix is None:
        name_prefix = mode

    # prepare contrasts
    if contrasts is None:
        contrasts = data.get_contrasts()
    if isinstance(contrasts, dict):
        contrasts = pd.DataFrame(contrasts)
    valid = [col for col in contrasts.columns
             if (1 in contrasts[col].values and -1 in contrasts[col].values)]
    contrasts = contrasts.loc[:, valid]
    if contrasts.shape[1] == 0:
        raise ValueError("Contrasts do not define any comparison!")

    # retrieve raw expression matrix
    mode_slot_obj = ModeSlot(mode, slot)
    try:
        raw_mat = data.get_matrix(mode_slot=str(mode_slot_obj))
    except Exception:
        raise ValueError(f"Invalid mode slot: '{mode_slot_obj}'")

    # optional: subset genes
    gene_list = data.genes if genes is None else genes

    new_data = data

    for contrast in contrasts.columns:
        if verbose:
            print(f"Computing LFC and M for contrast '{contrast}'...")
        c = contrasts[contrast]
        A = np.where(c == 1)[0]
        B = np.where(c == -1)[0]

        # sum counts per group
        mat = raw_mat
        A_counts = np.sum(mat[:, A], axis=1)
        B_counts = np.sum(mat[:, B], axis=1)

        # define normalization shift
        if isinstance(normalization, (list, np.ndarray, pd.Series)):
            sf = np.asarray(normalization)
            shift = np.log2(sf[A].sum() / sf[B].sum())
            normalize_fun = lambda x: x - shift
        elif isinstance(normalization, str):
            norm_mat = data.get_matrix(mode_slot=f"{normalization}_{slot}")
            A_norm = np.sum(norm_mat[:, A], axis=1)
            B_norm = np.sum(norm_mat[:, B], axis=1)
            res = LFC_fun(A_norm, B_norm, normalize_fun=lambda i: i, **kwargs)
            raw_norm = res[0] if isinstance(res, tuple) else res
            med = np.median(raw_norm)
            normalize_fun = lambda x: x - med
        else:
            normalize_fun = center_median

        # compute LFC (and optional M)
        result = LFC_fun(A_counts, B_counts, normalize_fun=normalize_fun, **kwargs)
        lfc_vals = result[0] if isinstance(result, tuple) else result

        # create named columns
        lfc_col = f"{name_prefix}_{contrast}_LFC"
        table = pd.DataFrame({lfc_col: lfc_vals}, index=gene_list)
        if compute_M:
            M_vals = 10 ** (0.5 * (np.log10(A_counts + 0.5) + np.log10(B_counts + 0.5)))
            m_col = f"{name_prefix}_{contrast}_M"
            table[m_col] = M_vals

            if verbose:
                print(f"M-value for '{contrast}': {M_vals.mean():.2f}")

        # append analysis to object
        name = f"{name_prefix}_{contrast}"
        new_data = new_data.with_analysis(name=name, table=table)

    return new_data


# TODO: fit_type ...

#  In R: -- note: fitType='parametric', but the dispersion trend was not well captured by the
#    function: y = a/x + b, and a local regression fit was automatically substituted.
#    specify fitType='local' or 'mean' to avoid this message next time.
# but pydeseq2 only has the fitTypes "parametric" and "mean"

def pairwise_DESeq2(
    data: GrandPy,
    contrasts: pd.DataFrame,
    name_prefix: Optional[str] = None,
    separate: bool = False,
    mode: str = "total",
    slot: str = "count",
    normalization: Optional[Union[str, Sequence[float]]] = None,
    genes: Optional[List[str]] = None,
    verbose: bool = False) -> GrandPy:

    """
    Run DESeq2 (via pydeseq2) for each contrast defined in the contrast matrix.

    Parameters
    ----------
    data : GrandPy
        The GrandPy object containing expression data and metadata.

    contrasts : pd.DataFrame
        Matrix defining pairwise comparisons (samples x contrasts; 1/-1 values).

    name_prefix : str, optional
        Prefix for naming the output columns.

    separate : bool
        If True, run DESeq2 separately for each contrast (two-group comparisons).

    mode : str
        Expression mode to use, e.g., "total" or "new".

    slot : str
        Name of the slot to use for count data.

    normalization : str or sequence, optional
        Either slot name or size factors for normalization.

    genes : list of str, optional
        Restrict to this subset of genes.

    verbose : bool
        Print progress information.

    Returns
    -------
    GrandPy
        GrandPy object with DESeq2 results added to analysis table.

    Notes
    -----
    Uses fit_type="mean" for compatibility with pydeseq2.
    pydeseq2 does not currently support fit_type="local".
    """

    try:
        import pydeseq2
    except ImportError:
        raise ImportError("pydeseq2 is required but not installed!")

    if not np.all(contrasts.apply(lambda v: set([-1, 1]).issubset(set(v)), axis=0)):
        raise ValueError("Contrasts do not define any comparison!")

    mode_slot = ModeSlot(mode, slot)
    normalization_slot = mode_slot.slot if normalization is None else (
        ModeSlot(normalization, slot).slot if isinstance(normalization, str) else normalization)

    if verbose:
        print(f"Running pairwise_DESeq2 with mode='{mode}', slot='{slot}', normalization='{normalization_slot}'")
        print("Available slots:", list(data.slots.keys()))

    def format_column_names(base: str, columns):
        return [f"{base}_{col}" for col in columns]

    if separate:
        for contrast_name in contrasts.columns:
            if verbose:
                print(f"Running DESeq2 for contrast '{contrast_name}' (separate=True).")

            A_mask = contrasts[contrast_name] == 1
            B_mask = contrasts[contrast_name] == -1

            counts_A = np.round(data.get_matrix(mode_slot, genes=genes)[:, A_mask].astype(int))
            counts_B = np.round(data.get_matrix(mode_slot, genes=genes)[:, B_mask].astype(int))
            counts = np.hstack((counts_A, counts_B))

            if isinstance(normalization_slot, str):
                norm_counts = data.get_matrix(normalization_slot, genes=genes)[:, np.logical_or(A_mask, B_mask)]
                _, size_factors = deseq2_norm(pd.DataFrame(norm_counts))
                size_factors = size_factors.squeeze()
            elif normalization is None:
                norm_counts = data.get_matrix(mode_slot.slot, genes=genes)[:, np.logical_or(A_mask, B_mask)]
                _, size_factors = deseq2_norm(pd.DataFrame(norm_counts))
                size_factors = size_factors.squeeze()
            else:
                normalization_array = np.asarray(normalization_slot)
                if normalization_array.ndim == 0:
                    raise ValueError("Normalization array is 0-dimensional.")
                size_factors = normalization_array[np.logical_or(A_mask, B_mask)]

            cond_labels = np.array(["A"] * counts_A.shape[1] + ["B"] * counts_B.shape[1])
            coldata = pd.DataFrame({"comparison": cond_labels})
            counts_df = pd.DataFrame(counts).T

            dds = DeseqDataSet(counts=counts_df, metadata=coldata, design_factors="comparison", ref_level=None)
            dds.size_factors = size_factors
            dds.deseq2(fit_type="parametric")
            stats = DeseqStats(dds)
            stats.summary()

            result = stats.results_df.set_index(data.gene_info.index)
            base_columns = ["M", "S", "P", "Q", "LFC"]
            result_df = pd.DataFrame({base_columns[0]: result["baseMean"],
                                      base_columns[1]: result["stat"],
                                      base_columns[2]: result["pvalue"],
                                      base_columns[3]: result["padj"],
                                      base_columns[4]: result["log2FoldChange"]})

            if "_vs_" in contrast_name:
                parts = contrast_name.split("_vs_")
                contrast_readable = f"{parts[0]} vs {parts[1]}"
            else:
                contrast_readable = contrast_name

            analysis_name = f"{mode}_{contrast_readable}" if name_prefix is None else f"{name_prefix}_{contrast_readable}"
            result_df.columns = format_column_names(analysis_name, result_df.columns)

            data = data.with_analysis(analysis_name, result_df)

        return data

    # combined estimation
    group_assignments = {}
    condition_vector = np.full(contrasts.shape[0], np.nan, dtype=object)
    groups = []

    def find_or_add_group(c):
        c_tuple = tuple(c)
        if c_tuple in group_assignments:
            return group_assignments[c_tuple]
        for existing in groups:
            if np.any(np.logical_and(c, existing)):
                raise ValueError("Illegal intersection of contrasts for joint estimation of variance!")
        group_id = str(len(groups) + 1)
        groups.append(c)
        group_assignments[c_tuple] = group_id
        return group_id

    dds_contrasts = []
    for contrast_name in contrasts.columns:
        A_group = find_or_add_group(contrasts[contrast_name] == 1)
        B_group = find_or_add_group(contrasts[contrast_name] == -1)
        dds_contrasts.append((contrast_name, A_group, B_group))
        condition_vector[contrasts[contrast_name] == 1] = A_group
        condition_vector[contrasts[contrast_name] == -1] = B_group

    valid_samples = ~pd.isna(condition_vector)
    counts = np.round(data.get_matrix(mode_slot, genes=genes)[:, valid_samples].astype(int))

    if isinstance(normalization_slot, str):
        norm_counts = data.get_matrix(normalization_slot, genes=genes)[:, valid_samples]
        _, size_factors = deseq2_norm(pd.DataFrame(norm_counts))
        size_factors = size_factors.squeeze()
    elif normalization is None:
        norm_counts = data.get_matrix(mode_slot.slot, genes=genes)[:, valid_samples]
        _, size_factors = deseq2_norm(pd.DataFrame(norm_counts))
        size_factors = size_factors.squeeze()
    else:
        normalization_array = np.asarray(normalization_slot)
        if normalization_array.ndim == 0:
            raise ValueError("Normalization array is 0-dimensional.")
        size_factors = normalization_array[valid_samples]

    coldata = pd.DataFrame({"comparison": pd.Categorical(condition_vector[valid_samples], categories=[str(i + 1) for i in range(len(groups))])})
    counts_df = pd.DataFrame(counts).T

    dds = DeseqDataSet(counts=counts_df, metadata=coldata, design_factors="comparison", ref_level=None)
    dds.size_factors = size_factors
    dds.deseq2(fit_type="parametric")

    for contrast_name, A_group, B_group in dds_contrasts:
        stats = DeseqStats(dds, contrast=("comparison", A_group, B_group))
        stats.summary()

        result = stats.results_df.set_index(data.gene_info.index)
        base_columns = ["M", "S", "P", "Q", "LFC"]
        result_df = pd.DataFrame({base_columns[0]: result["baseMean"],
                                  base_columns[1]: result["stat"],
                                  base_columns[2]: result["pvalue"],
                                  base_columns[3]: result["padj"],
                                  base_columns[4]: result["log2FoldChange"]})

        if "_vs_" in contrast_name:
            parts = contrast_name.split("_vs_")
            contrast_readable = f"{parts[0]} vs {parts[1]}"
        else:
            contrast_readable = contrast_name

        analysis_name = f"{mode}_{contrast_readable}" if name_prefix is None else f"{name_prefix}_{contrast_readable}"
        result_df.columns = format_column_names(analysis_name, result_df.columns)

        data = data.with_analysis(analysis_name, result_df)

    return data


def pairwise(data: GrandPy,
             contrasts: pd.DataFrame,
             name_prefix: Optional[str] = None,
             LFC_fun=Psi_LFC,
             slot: str = "count",
             mode: str = "total",
             normalization: Optional[Union[str, Sequence[float]]] = None,
             genes: Optional[List[str]] = None,
             verbose: bool = False
             ) -> GrandPy:
    """
    Combined log2 fold change and Wald test differential expression analysis.
    This function performs both the LFC computation (via compute_lfc) and DESeq2 testing
    (via pairwise_DESeq2). Only valid contrasts with both -1 and 1 are used.

    Parameters
    ----------
    data : GrandPy
        The grandPy object containing expression data and metadata.

    contrasts : pd.DataFrame
        Contrast matrix defining comparisons (samples x contrasts; values 1, -1).

    name_prefix : str, optional
        Prefix for naming the output analysis tables.

    LFC_fun : function, optional
        Function to compute the log2 fold changes (default Psi_LFC).

    slot : str
        Slot name to extract counts (default: "count").

    mode : str
        Expression mode to analyze (e.g. "total", "new", "old").

    normalization : str or sequence, optional
        Normalization strategy; name of slot or numeric vector.

    genes : list, optional
        Subset of genes to analyze.

    verbose : bool
        Whether to print progress messages.


    Returns
    -------
    GrandPy
        The updated GrandPy object with new analysis results.
    """

    valid_contrasts = contrasts.loc[:, contrasts.apply(lambda v: set([-1, 1]).issubset(set(v)), axis=0)]
    if valid_contrasts.shape[1] == 0:
        raise ValueError("Contrasts do not define any comparison!")

    if name_prefix is None:
        name_prefix = mode

    data = compute_lfc(data,
                       contrasts=valid_contrasts,
                       name_prefix=name_prefix,
                       LFC_fun=LFC_fun,
                       slot=slot,
                       mode=mode,
                       normalization=normalization,
                       compute_M=False,
                       genes=genes,
                       verbose=verbose)

    data = pairwise_DESeq2(data,
                           contrasts=valid_contrasts,
                           name_prefix=name_prefix,
                           slot=slot,
                           mode=mode,
                           normalization=normalization,
                           genes=genes,
                           verbose=verbose)

    return data


def _get_contrasts(
        data: "GrandPy",
        contrast: Union[Sequence[str], str] = "Condition",
        columns: Union[Sequence[bool], bool] = None,
        group: Union[Sequence[str], str] = None,
        name_format: str = None,
        no4sU: bool = False,
        ) -> pd.DataFrame:

    coldata = data.coldata

    contrast = _ensure_list(contrast)

    if len(contrast) not in [1, 2, 3]:
        raise ValueError("Contrast must be of length 1, 2, or 3.")

    if contrast[0] not in coldata.columns:
        raise ValueError(f"Column {contrast[0]} not found in coldata.")
    col = contrast[0]

    use_mask = np.ones(len(coldata), dtype=bool)
    if columns is not None:
        use_mask = coldata.index.isin(columns) if isinstance(columns, list) else columns

    if name_format is None:
        name_format = "$A vs $B" if group is None else "$A vs $B.$GRP"

    if not no4sU and "no4sU" in coldata.columns:
        use_mask &= ~coldata["no4sU"].fillna(False)

    def make_name(a, b, grp=""):
        return (name_format
                .replace("$A", str(a))
                .replace("$B", str(b))
                .replace("$COL", col)
                .replace("$GRP", grp))

    def make_vector(a, b, use):
        re = np.zeros(len(coldata))
        re[(coldata[col] == a) & use] = 1
        re[(coldata[col] == b) & use] = -1
        return re

    def contrast_df_for_level_pairs(level_pairs, group_name, use_mask_local):
        base_data_frame = {}
        for a, b in level_pairs:
            vec = make_vector(a, b, use_mask_local)
            name = make_name(a, b, group_name)
            base_data_frame[name] = vec
        return pd.DataFrame(base_data_frame, index=coldata.index)

    if group is not None:
        all_dfs = []
        for grp_val in coldata[group].dropna().unique():
            group_mask = (coldata[group] == grp_val)
            use_mask_group = use_mask & group_mask
            if len(coldata.loc[use_mask_group]) < 2:
                continue
            levels = coldata.loc[use_mask_group, col].dropna().unique().tolist()
            if len(levels) < 2:
                continue
            if len(contrast) == 1:
                level_pairs = list(combinations(levels, 2))
            elif len(contrast) == 2:
                level_pairs = [(l, contrast[1]) for l in levels if l != contrast[1]]
            else:
                level_pairs = [(contrast[1], contrast[2])]
            df = contrast_df_for_level_pairs(level_pairs, group_name=str(grp_val), use_mask_local=use_mask_group)
            all_dfs.append(df)
        if not all_dfs:
            raise ValueError("No valid contrasts could be generated for any group.")
        contrast_df = pd.concat(all_dfs, axis=1)
    else:
        levels = coldata.loc[use_mask, col].dropna().unique().tolist()
        if len(contrast) == 1:
            level_pairs = list(combinations(levels, 2))
        elif len(contrast) == 2:
            level_pairs = [(l, contrast[1]) for l in levels if l != contrast[1]]
        else:
            level_pairs = [(contrast[1], contrast[2])]
        contrast_df = contrast_df_for_level_pairs(level_pairs, group_name="", use_mask_local=use_mask)

    # Remove zero-columns (all zeros)
    contrast_df = contrast_df.loc[:, ~(contrast_df == 0).all(axis=0)]

    # Remove uninformative contrasts (e.g., all values ≥ 0 or all ≤ 0)
    remove_mask = ((contrast_df >= 0).all(axis=0)) | ((contrast_df <= 0).all(axis=0))
    if remove_mask.any():
        removed_cols = contrast_df.columns[remove_mask].tolist()
        print(f"Removed uninformative contrasts: {', '.join(removed_cols)}")
        contrast_df = contrast_df.loc[:, ~remove_mask]

    return contrast_df
