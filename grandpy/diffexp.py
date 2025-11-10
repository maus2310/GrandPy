import contextlib
import io
from itertools import combinations
from typing import Union, Callable, Sequence, Literal

import numpy as np
import pandas as pd

from .lfc import psi_lfc, center_median
from .slot_tool import ModeSlot, _parse_as_mode_slot
from .utils import _ensure_list



def _get_summarize_matrix(
        data: "GrandPy",
        no4su: bool = False,
        columns: Union[None, str, list[str]] = None,
        average: bool = True) -> pd.DataFrame:
    """
    For detailed documentation see GrandPy.get_summarize_matrix.
    """
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

    # Mapping: sample_name -> condition
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
    prefix: str = None,
    contrasts: pd.DataFrame = None,
    mode_slot: Union[str, ModeSlot] = "count",
    lfc_function: Callable = psi_lfc,
    normalization: Union[str, Sequence[float]] = None,
    compute_m: bool = True,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    verbose: bool = False,
    **kwargs) -> "GrandPy":
    """
    For a detailed documentation, see GrandPy.compute_lfc.
    """
    mode_slot_obj = _parse_as_mode_slot(mode_slot)

    if prefix is None:
        prefix = mode_slot_obj.mode

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
        a = np.where(c == 1)[0]
        b = np.where(c == -1)[0]

        # sum counts per group
        mat = raw_mat
        a_counts = np.sum(mat[:, a], axis=1)
        b_counts = np.sum(mat[:, b], axis=1)

        # define normalization shift
        if isinstance(normalization, (list, np.ndarray, pd.Series)):
            sf = np.asarray(normalization)
            shift = np.log2(sf[a].sum() / sf[b].sum())
            normalize_fun = lambda x: x - shift
        elif isinstance(normalization, str):
            norm_mat = data.get_matrix(mode_slot=f"{normalization}_{mode_slot_obj.slot}")
            a_norm = np.sum(norm_mat[:, a], axis=1)
            b_norm = np.sum(norm_mat[:, b], axis=1)
            res = lfc_function(a_norm, b_norm, normalize_fun=lambda i: i, **kwargs)
            raw_norm = res[0] if isinstance(res, tuple) else res
            med = np.median(raw_norm)
            normalize_fun = lambda x: x - med
        else:
            normalize_fun = center_median

        # compute LFC (and optional M)
        result = lfc_function(a_counts, b_counts, normalize_fun=normalize_fun, **kwargs)
        lfc_vals = result[0] if isinstance(result, tuple) else result

        # create named columns
        lfc_col = "LFC"
        table = pd.DataFrame({lfc_col: lfc_vals}, index=pd.Index(gene_list, name="Symbol"))
        if compute_m:
            m_vals = 10 ** (0.5 * (np.log10(a_counts + 0.5) + np.log10(b_counts + 0.5)))
            m_col = "M"
            table[m_col] = m_vals

            if verbose:
                print(f"M-value for '{contrast}': {m_vals.mean():.2f}")

        # append analysis to object
        name = f"{prefix}_{contrast}"
        new_data = new_data.with_analysis(name, table)

    return new_data


def _pairwise_deseq2(
    data: "GrandPy",
    contrasts: pd.DataFrame,
    prefix: str = None,
    separate: bool = False,
    mode_slot: Union[str, ModeSlot] = "count",
    normalization: Literal["new", "n", "old", "o", "total", "t"] = None,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    verbose: bool = False) -> "GrandPy":
    """
    For a detailed documentation, see GrandPy.pairwise_deseq2.
    """
    try:
        import pydeseq2
        from pydeseq2.preprocessing import deseq2_norm
        from pydeseq2.ds import DeseqDataSet, DeseqStats
    except ImportError:
        raise ImportError("pydeseq2 is required for pairwise_deseq2 but not installed!")

    mode_slot = _parse_as_mode_slot(mode_slot)

    if not np.all(contrasts.apply(lambda v: {-1, 1}.issubset(set(v)), axis=0)):
        raise ValueError("Contrasts do not define any comparison!")

    normalization_slot = mode_slot.slot if normalization is None else (
        ModeSlot(normalization, mode_slot.slot).slot if isinstance(normalization, str) else normalization)

    if verbose:
        print(f"Running pairwise_DESeq2 with '{mode_slot}', normalization='{normalization_slot}'")
        print("Available slots:", list(data.slots))

    # def format_column_names(base: str, columns):
    #     return [f"{base}_{col}" for col in columns]

    if separate:
        for contrast_name in contrasts.columns:
            if verbose:
                print(f"Running DESeq2 for contrast '{contrast_name}' (separate=True).")

            a_mask = contrasts[contrast_name] == 1
            b_mask = contrasts[contrast_name] == -1

            counts_a = np.round(data.get_matrix(mode_slot, genes=genes)[:, a_mask].astype(int))
            counts_b = np.round(data.get_matrix(mode_slot, genes=genes)[:, b_mask].astype(int))
            counts = np.hstack((counts_a, counts_b))

            if isinstance(normalization_slot, str):
                norm_counts = data.get_matrix(normalization_slot, genes=genes)[:, np.logical_or(a_mask, b_mask)]
                _, size_factors = deseq2_norm(pd.DataFrame(norm_counts))
                size_factors = size_factors.squeeze()
            elif normalization is None:
                norm_counts = data.get_matrix(mode_slot.slot, genes=genes)[:, np.logical_or(a_mask, b_mask)]
                _, size_factors = deseq2_norm(pd.DataFrame(norm_counts))
                size_factors = size_factors.squeeze()
            else:
                normalization_array = np.asarray(normalization_slot)
                if normalization_array.ndim == 0:
                    raise ValueError("Normalization array is 0-dimensional.")
                size_factors = normalization_array[np.logical_or(a_mask, b_mask)]

            cond_labels = np.array(["A"] * counts_a.shape[1] + ["B"] * counts_b.shape[1])
            coldata = pd.DataFrame({"comparison": cond_labels})
            counts_df = pd.DataFrame(counts).T

            dds = DeseqDataSet(counts=counts_df, metadata=coldata, design_factors="comparison", ref_level=None, quiet=True)
            dds.size_factors = size_factors
            dds.deseq2(fit_type="parametric")
            stats = DeseqStats(dds, contrast=["comparison", "A", "B"])
            with contextlib.redirect_stdout(io.StringIO()):
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

            analysis_name = f"{mode_slot.mode}_{contrast_readable}" if prefix is None else f"{prefix}_{contrast_readable}"
            result_df.columns = base_columns

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
        a_group = find_or_add_group(contrasts[contrast_name] == 1)
        b_group = find_or_add_group(contrasts[contrast_name] == -1)
        dds_contrasts.append((contrast_name, a_group, b_group))
        condition_vector[contrasts[contrast_name] == 1] = a_group
        condition_vector[contrasts[contrast_name] == -1] = b_group

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
        stats = DeseqStats(dds, contrast=["comparison", A_group, B_group])
        with contextlib.redirect_stdout(io.StringIO()):
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

        analysis_name = f"{mode_slot.mode}_{contrast_readable}" if prefix is None else f"{prefix}_{contrast_readable}"
        result_df.columns = base_columns

        data = data.with_analysis(analysis_name, result_df)

    return data


def _pairwise(
    data: "GrandPy",
    contrasts: pd.DataFrame,
    prefix: str = None,
    lfc_function=psi_lfc,
    mode_slot: Union[str, ModeSlot] = "count",
    normalization: Union[str, Sequence[float]] = None,
    separate: bool = False,
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    verbose: bool = False,
    **kwargs
    ) -> "GrandPy":
    """
    For detailed documentation, see GrandPy.pairwise.
    """
    mode_slot = _parse_as_mode_slot(mode_slot)

    valid_contrasts = contrasts.loc[:, contrasts.apply(lambda v: {-1, 1}.issubset(set(v)), axis=0)]
    if valid_contrasts.shape[1] == 0:
        raise ValueError("Contrasts do not define any comparison!")

    if prefix is None:
        prefix = mode_slot.mode

    new_gp = _pairwise_deseq2(data,
                              contrasts=valid_contrasts,
                              prefix=prefix,
                              mode_slot=mode_slot,
                              normalization=normalization,
                              separate=separate,
                              genes=genes,
                              verbose=verbose)

    new_gp = _compute_lfc(new_gp,
                          contrasts=valid_contrasts,
                          prefix=prefix,
                          lfc_function=lfc_function,
                          mode_slot=mode_slot,
                          normalization=normalization,
                          compute_m=False,
                          genes=genes,
                          verbose=verbose,
                          **kwargs)

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
    use_mask = pd.Series(pd.Index(data.columns).isin(columns), index=data.columns)

    if name_format is None:
        name_format = "$A vs $B" if group is None else "$A vs $B.$GRP"

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
