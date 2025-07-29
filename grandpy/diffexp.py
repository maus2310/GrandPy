from itertools import combinations
from typing import Union, TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd
from pydeseq2.preprocessing import deseq2_norm

from .lfc import psi_lfc, center_median
from .slot_tool import ModeSlot, _parse_as_mode_slot
from .utils import _ensure_list

try:
    from pydeseq2.ds import DeseqDataSet, DeseqStats
except ImportError:
    DeseqDataSet = DeseqStats = None

if TYPE_CHECKING:
    from .grandPy import GrandPy

def _get_summary_matrix(
        data: "GrandPy",
        no4su: bool = False,
        columns: Union[None, str, list[str]] = None,
        average: bool = True) -> pd.DataFrame:
    coldata = data.coldata
    sample_names = coldata.index.tolist()

    if "Condition" not in coldata.columns:
        raise ValueError("A GrandPy object must contain a 'Condition' column in its coldata for summarization")

    if columns is None:
        columns = sample_names
    else:
        columns = data.get_columns(columns)

    # Exclude 4sU-marked samples
    if not no4su:
        no4su_samples = coldata.index[coldata["no4sU"]]
        columns = [col for col in columns if col not in no4su_samples]

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



def _compute_lfc(
    data: "GrandPy",
    name_prefix: str = None,
    contrasts: pd.DataFrame = None,
    mode_slot: Union[str, ModeSlot] = "count",
    lfc_function: Callable = psi_lfc,
    normalization: Union[str, Sequence[float]] = None,
    compute_m: bool = True,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    verbose: bool = False,
    **kwargs) -> "GrandPy":
    """
    Estimate log2 fold changes and optional M values for each contrast and store analyses.

    Parameters
    ----------
    data : GrandPy
        The grandPy object. Must contain a 'Condition' column in its coldata.
    name_prefix : str, optional
        The prefix for the new analysis name; e.g. 'total' or 'new'.
    contrasts : pd.DataFrame, optional
        Contrast matrix defining comparisons (samples x contrasts; values 1, -1).
    mode_slot: str or ModeSlot, default "count"
        The name of the data slot to take data from. Usually 'count', optionally with a mode.
        A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'.
    lfc_function : Callable, default psi_lfc
        Function to compute the log2 fold changes.
    normalization : str or sequence, optional
        If str: name of normalization slot (e.g. "total");
        if sequence: size factors per sample.
    compute_m: bool, default True
        If True, include the "M" column (base mean) for each contrast.
    genes : str or int or Sequence[str or int or bool], optional
        Restrict computation to this subset of genes. Either by their index, their symbol, their ensemble ID, or a boolean mask.
    verbose : bool, default False
        If True, status updates are printed.
    **kwargs
        Passed to `lfc_function`.

    Returns
    -------
    GrandPy
        A GrandPy instance containing an analysis for each contrast. Each analysis
        has two columns named "{prefix}_{contrast}_LFC" and
        "{prefix}_{contrast}_M".
    """
    mode_slot_obj = _parse_as_mode_slot(mode_slot)

    if name_prefix is None:
        name_prefix = mode_slot_obj.mode

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
    raw_mat = data.get_matrix(mode_slot=mode_slot_obj)

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
            norm_mat = data.get_matrix(mode_slot=f"{normalization}_{mode_slot_obj.slot}")
            A_norm = np.sum(norm_mat[:, A], axis=1)
            B_norm = np.sum(norm_mat[:, B], axis=1)
            res = lfc_function(A_norm, B_norm, normalize_fun=lambda i: i, **kwargs)
            raw_norm = res[0] if isinstance(res, tuple) else res
            med = np.median(raw_norm)
            normalize_fun = lambda x: x - med
        else:
            normalize_fun = center_median

        # compute LFC (and optional M)
        result = lfc_function(A_counts, B_counts, normalize_fun=normalize_fun, **kwargs)
        lfc_vals = result[0] if isinstance(result, tuple) else result

        # create named columns
        lfc_col = f"{name_prefix}_{contrast}_LFC"
        table = pd.DataFrame({lfc_col: lfc_vals}, index=pd.Index(gene_list, name="Symbol"))
        if compute_m:
            M_vals = 10 ** (0.5 * (np.log10(A_counts + 0.5) + np.log10(B_counts + 0.5)))
            m_col = f"{name_prefix}_{contrast}_M"
            table[m_col] = M_vals

            if verbose:
                print(f"M-value for '{contrast}': {M_vals.mean():.2f}")

        # append analysis to object
        name = f"{name_prefix}_{contrast}"
        new_data = new_data.with_analysis(name, table)

    return new_data


def _pairwise_DESeq2(
    data: "GrandPy",
    contrasts: pd.DataFrame,
    name_prefix: str = None,
    separate: bool = False,
    mode_slot: Union[str, ModeSlot] = "count",
    normalization: Union[str, Sequence[float]] = None,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    verbose: bool = False) -> "GrandPy":
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

    separate : bool, default False
        If True, run DESeq2 separately for each contrast (two-group comparisons).

    mode_slot: str or ModeSlot, default "count"
        The name of the data slot to take data from. Usually 'count', optionally with a mode.
        A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'.

    normalization : str or sequence, optional
        Either slot name or size factors for normalization.

    genes : str or int or Sequence[str or int or bool], optional
        Restrict computation to this subset of genes. Either by their index, their symbol, their ensemble ID, or a boolean mask.

    verbose : bool, default False
        Print progress information.

    Returns
    -------
    GrandPy
        A GrandPy instance containing pydeseq2 analysis results.

    Notes
    -----
    Uses fit_type="parametric" for compatibility with pydeseq2.
    pydeseq2 does not currently support fit_type="local" (as in grandR).
    """
    try:
        import pydeseq2
    except ImportError:
        raise ImportError("pydeseq2 is required but not installed!")

    mode_slot = _parse_as_mode_slot(mode_slot)

    if not np.all(contrasts.apply(lambda v: set([-1, 1]).issubset(set(v)), axis=0)):
        raise ValueError("Contrasts do not define any comparison!")

    normalization_slot = mode_slot.slot if normalization is None else (
        ModeSlot(normalization, mode_slot.slot).slot if isinstance(normalization, str) else normalization)

    if verbose:
        print(f"Running pairwise_DESeq2 with '{mode_slot}', normalization='{normalization_slot}'")
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

            dds = DeseqDataSet(counts=counts_df, metadata=coldata, design_factors="comparison", ref_level=None, quiet=True)
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

            analysis_name = f"{mode_slot.mode}_{contrast_readable}" if name_prefix is None else f"{name_prefix}_{contrast_readable}"
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

    dds = DeseqDataSet(counts=counts_df, metadata=coldata, design_factors="comparison", ref_level=None, quiet=True)
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

        analysis_name = f"{mode_slot.mode}_{contrast_readable}" if name_prefix is None else f"{name_prefix}_{contrast_readable}"
        result_df.columns = format_column_names(analysis_name, result_df.columns)

        data = data.with_analysis(analysis_name, result_df)

    return data


def _pairwise(
    data: "GrandPy",
    contrasts: pd.DataFrame,
    name_prefix: str = None,
    lfc_function=psi_lfc,
    mode_slot: Union[str, ModeSlot] = "count",
    normalization: Union[str, Sequence[float]] = None,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    verbose: bool = False,
    **kwargs
    ) -> "GrandPy":
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

    lfc_function : Callable, optional
        Function to compute the log2 fold changes (default Psi_LFC).

    mode_slot: str or ModeSlot, default "count"
        The name of the data slot to take data from. Usually 'count', optionally with a mode.
        A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'.

    normalization : str or sequence, optional
        Normalization strategy; name of slot or numeric vector.

    genes : str or int or Sequence[str or int or bool], optional
        Restrict computation to this subset of genes. Either by their index, their symbol, their ensemble ID, or a boolean mask.

    verbose : bool, default False
        Whether to print progress messages.

    **kwargs
        Passed to `lfc_function`.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary with analysis results.
    """
    mode_slot = _parse_as_mode_slot(mode_slot)

    valid_contrasts = contrasts.loc[:, contrasts.apply(lambda v: set([-1, 1]).issubset(set(v)), axis=0)]
    if valid_contrasts.shape[1] == 0:
        raise ValueError("Contrasts do not define any comparison!")

    if name_prefix is None:
        name_prefix = mode_slot.mode

    new_gp = _compute_lfc(data,
                          contrasts=valid_contrasts,
                          name_prefix=name_prefix,
                          lfc_function=lfc_function,
                          mode_slot=mode_slot,
                          normalization=normalization,
                          compute_m=False,
                          genes=genes,
                          verbose=verbose,
                          **kwargs)

    new_gp = _pairwise_DESeq2(new_gp,
                           contrasts=valid_contrasts,
                           name_prefix=name_prefix,
                           mode_slot=mode_slot,
                           normalization=normalization,
                           genes=genes,
                           verbose=verbose)

    return new_gp


def _get_contrasts(
        data: "GrandPy",
        contrast: Union[Sequence[str], str] = "Condition",
        columns: Union[str, int, Sequence[Union[str, int, bool]]] = None,
        group: Union[Sequence[str], str] = None,
        name_format: str = None,
        no4su: bool = False,
        ) -> pd.DataFrame:

    coldata = data.coldata

    contrast = _ensure_list(contrast)

    if len(contrast) not in [1, 2, 3]:
        raise ValueError("Contrast must be of length 1, 2, or 3.")

    if contrast[0] not in coldata.columns:
        raise ValueError(f"Column {contrast[0]} not found in coldata.")
    col = contrast[0]

    columns = data.get_columns(columns)
    use_mask = [elem in columns for elem in data.columns]

    if name_format is None:
        name_format = "A vs B" if group is None else "A vs B.GRP"

    if not no4su and "no4sU" in coldata.columns:
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
