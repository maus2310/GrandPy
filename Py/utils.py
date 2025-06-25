import warnings
from typing import Union, Iterable
import numpy as np
import pandas as pd
import scipy.sparse as sp


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


def _subset_dense_or_sparse(matrix: Union[np.ndarray, sp.csr_matrix], row_indices: list[int], column_indices: list[int]) -> np.ndarray:
    """
    Subsets dense or sparse matrices by given row and column indices.

    Not optimal for sparse matrices, as they are converted to np.ndarray.

    Parameters
    ----------
    matrix: np.ndarray or sp.csr_matrix
        The matrix to subset.

    row_indices: list[int]
        The row indices to subset by.

    column_indices: list[int]
        The column indices to subset by.

    Returns
    -------
    np.ndarray
        The subsetted matrix.
    """
    if sp.issparse(matrix):
        # matrix = matrix.tocsr()
        data_subset = matrix[np.ix_(row_indices, column_indices)].toarray()
    else:
        data_subset = matrix[np.ix_(row_indices, column_indices)]

    return data_subset.squeeze()



def _ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str) or not isinstance(obj, Iterable):
        return [obj]
    return list(obj)