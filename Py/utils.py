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
            duplicates_list = series[series.duplicated()].unique()[:10]
            warnings.warn(f"Duplicate gene symbols found: {', '.join(duplicates_list)}; they have been renamed to ensure uniqueness (e.g., MATR3 → MATR3_1).")

        for val in series:
            if val not in counts:
                counts[val] = 0
                result.append(val)
            else:
                counts[val] += 1
                result.append(f"{val}_{counts[val]}")
    return pd.Series(result, index=series.index)


def _ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str) or not isinstance(obj, Iterable):
        return [obj]
    return list(obj)