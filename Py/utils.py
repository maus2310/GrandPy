import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from collections.abc import Sequence
from typing import Union, Iterable, Callable, Literal, TYPE_CHECKING

from matplotlib.style.core import available

if TYPE_CHECKING:
    from grandPy import GrandPy

# Public utility functions
def concat(
        objects: Sequence["GrandPy"],
        *,
        axis: Literal["gene_info", 0, "coldata", 1] = 1,
        join: Literal["inner", "outer"] = "inner",
        merge: Union[Literal["same", "unique", "first", "only"], Callable] = "unique",
) -> "GrandPy":
    """
    Concatenates all given objects along a given axis. Uses `unique` for metadata and plots.

    Analyses will be concatenated if their names are identical. Otherwise, they are dropped.

    Parameters
    ----------
    objects : Sequence[GrandPy]
        The GrandPy objects to be concatenated.

    axis: {"gene_info" or 0 or "coldata" or 1}, default 1
        The axis along which to concatenate.

    join: {"inner" or "outer"}, default "inner"
        How to align values when concatenating. If "outer", the union of the other axis is taken. If "inner", the intersection.

    merge: {"same" or "unique" or "first" or "only"} or Callable, default "unique"
        How elements not aligned to the axis being concatenated along are selected.
        Currently implemented strategies include:

        * `None`: No elements are kept.
        * `"same"`: Elements that are the same in each of the objects.
        * `"unique"`: Elements for which there is only one possible value.
        * `"first"`: The first element seen at each from each position.
        * `"only"`: Elements that show up in only one of the objects.

    Returns
    -------
    GrandPy
        A new concatenated GrandPy object.
    """
    from collections import Counter

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="(Observation|Variable) names are not unique.*",
                                category=UserWarning, module="anndata")

        if axis == 0 or axis == "gene_info":
            axis = "var"
            analysis_axis = 1
        elif axis == 1 or axis == "coldata":
            axis = "obs"
            analysis_axis = 0
        else:
            raise ValueError(f"axis must be either 0, 'gene_info' or 1, 'coldata' not {axis}.")

        adatas = [obj._adata for obj in objects]
        new_adata = ad.concat(adatas, axis=axis, join=join, merge=merge, uns_merge="unique")

    merged_analyses = {}

    if all(obj.analyses is not None for obj in objects):
        analyses = [a for obj in objects for a in obj.analyses]
        duplikates = [item for item, count in Counter(analyses).items() if count > 1]

        for duplikate in duplikates:
            dfs = [obj.get_analysis_table(duplikate, with_gene_info=False) for obj in objects]
            merged_df = pd.concat(dfs, axis=analysis_axis, join=join)
            merged_analyses[duplikate] = merged_df

    new_adata.uns["analyses"] = merged_analyses

    return objects[0]._dev_replace(anndata=new_adata)


# Private utility functions
def _to_sparse(matrix: Union[pd.DataFrame, np.ndarray, sp.csr_matrix]) -> sp.csr_matrix:
    """
    Convert the given matrix to a csr_matrix.

    Parameters
    ----------
    matrix: Union[pd.DataFrame, np.ndarray, sp.csr_matrix]]
        The dense matrix to convert.

    Returns
    -------
    scipy.sparse.csr_matrix
        The sparse matrix in CSR format.
    """
    from scipy.sparse import csr_matrix

    if isinstance(matrix, sp.csr_matrix):
        return matrix
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    try:
        sparse_matrix = csr_matrix(matrix)
    except ValueError:
        raise ValueError(
            "Matrix could not be converted to a sparse matrix. Use numpy.ndarray or a pandas.DataFrame only containing numbers")

    return sparse_matrix

def _make_unique(series: pd.Series, warn = True) -> pd.Series:
    """
        Ensures all values in a Series are unique by appending suffixes to duplicates.

        Parameters
        ----------
        series : pd.Series
            Input Series containing potentially non-unique values (e.g., gene symbols).

        Returns
        -------
        pd.Series
            Series with unique values. Duplicates are renamed by appending '_1', '_2', etc.
        """
    counts = {}
    result = []

    if series.is_unique:
        return series

    else:
        if warn:
            duplicates_list = series[series.duplicated()].unique()
            warnings.warn(f"{len(duplicates_list)} Duplicate gene symbols found: {', '.join(duplicates_list[:5])} (first 5); they have been renamed to ensure uniqueness (e.g., MATR3 → MATR3_1).")

        for val in series:
            if val not in counts:
                counts[val] = 0
                result.append(val)
            else:
                counts[val] += 1
                result.append(f"{val}_{counts[val]}")
    return pd.Series(result, index=series.index)

def _reindex_by_index_name(df: pd.DataFrame, by: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the index of a DataFrame by a column from `by`, inferred from the index name of `df`.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to sort.

    by: pd.DataFrame
        The DataFrame to sort by, that contains a column with the same name as the index name of `df`.

    Returns
    -------
    pd.DataFrame
        The sorted DataFrame.
    """
    colname_to_sort_by = df.index.name

    if colname_to_sort_by is None:
        warnings.warn("Tried to reindex (see pandas) a DataFrame, by a column inferred from the index name, "
                      "but DataFrame did not have an index name. Now reindexing by a default.")
        sort_by = by.index

    else:
        sort_by = by[colname_to_sort_by]
        sort_by = pd.Index(sort_by)

    sorted_df = df.reindex(sort_by)

    return sorted_df

def _subset_dense_or_sparse(
    matrix: Union[np.ndarray, sp.csr_matrix],
    row_indices: list[int],
    column_indices: list[int],
    force_numpy: bool = True
) -> Union[np.ndarray, sp.csr_matrix]:
    """
    Subsets dense or sparse matrices by given row and column indices.

    Parameters
    ----------
    matrix: np.ndarray or sp.csr_matrix
        The matrix to subset.

    row_indices: list[int]
        The row indices to subset by.

    column_indices: list[int]
        The column indices to subset by.

    force_numpy: bool, default=True
        If True, the result is always returned as a np.ndarray.
        Otherwise, the result retains the type of the input (sparse or dense).

    Returns
    -------
    np.ndarray or sp.csr_matrix
        The subsetted matrix, in the desired format.
    """
    if sp.issparse(matrix):
        subset = matrix[np.ix_(row_indices, column_indices)]
        return subset.toarray() if force_numpy else subset
    else:
        subset = matrix[np.ix_(row_indices, column_indices)]
        return subset

def _ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str) or not isinstance(obj, Iterable):
        return [obj]
    return list(obj)


# alternative fit_kinetics function
def _get_kinetics_data(
    data,
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
    This function is almost the same as `GrandPy.fit_kinetics`.
    The only difference is that it returns the kinetics data instead of a GrandPy object.
    """
    from Py.modeling import fit_kinetics

    kinetics = fit_kinetics(data=data, fit_type=fit_type, slot=slot, genes=genes, name_prefix=name_prefix, time=time,
                            ci_size=ci_size, return_fields=return_fields, show_progress=show_progress, **kwargs)

    return kinetics
