import warnings
from os import PathLike
from collections.abc import Sequence
from typing import Union, Iterable, Callable, Literal, TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:
    from grandpy import GrandPy

# --- Public utility functions ---
def concat(
        objects: Sequence["GrandPy"],
        *,
        axis: Literal["gene_info", 0, "coldata", 1] = 0,
        join: Literal["inner", "outer"] = "inner",
        merge: Union[Literal["same", "unique", "first", "only"], Callable] = "unique",
        analysis_prefixes: Sequence[str] = None
) -> "GrandPy":
    """
    Concatenates all given objects along a given axis. Uses 'first'` for metadata and plot functions.
    Analyses are all kept with an added prefix to avoid collisions.

    Parameters
    ----------
    objects : Sequence[GrandPy]
        The GrandPy objects to be concatenated.

    axis: {"gene_info" or 0 or "coldata" or 1}, default 0
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

    analysis_prefixes: Sequence[str], optional
        The prefixes added to the analyses of each instance.
        The list has to have the same length as the number of objects.
        By default 'dataset<n>', where n is the index of the object in `objects`.

        To disable this behavior, set to [""] * len(objects).
        Then analyses of the object coming first will be kept in case of a name collision.

    Returns
    -------
    GrandPy
        A new concatenated GrandPy object.
    """
    def add_prefix_to_analyses(grandpy_objects, prefixes):
        """
        Helper function, for adding prefixes to all analyses.
        """
        modified_objects = []

        if len(grandpy_objects) != len(prefixes):
            raise ValueError(f"The length of prefixes must match the number of objects to concatenate."
                             f"Length objects: {len(grandpy_objects)}, Length prefixes: {len(prefixes)}.")

        for obj, prefix in zip(reversed(grandpy_objects), reversed(prefixes)):
            if obj._anndata.uns.get("analyses", {}) == {}:
                modified_objects.append(obj)

            else:
                new_analyses = obj._anndata.uns['analyses'].copy()

                prefix = f"{prefix}_" if prefix != "" else ""

                new_analyses = {
                    f"{prefix}{key}": value for key, value in new_analyses.items()
                }

                new_obj = obj._dev_replace(analyses=new_analyses)
                modified_objects.append(new_obj)

        return modified_objects[::-1]

    if axis == 0 or axis == "gene_info":
        axis = "var"
    elif axis == 1 or axis == "coldata":
        axis = "obs"
    else:
        raise ValueError(f"Axis must be either 0, 'gene_info' or 1, 'coldata' not {axis}.")

    if analysis_prefixes is None:
        analysis_prefixes = [f"dataset{n}" for n in range(len(objects))]

    objects = add_prefix_to_analyses(objects, analysis_prefixes)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="(Observation|Variable) names are not unique.*",
                                category=UserWarning, module="anndata")
        adatas = [obj._anndata for obj in objects]
        new_adata = ad.concat(adatas, axis=axis, join=join, merge=merge, uns_merge="first")

    new_adata.obs.index = _make_unique(pd.Series(new_adata.obs.index))
    new_adata.var.index = _make_unique(pd.Series(new_adata.var.index))

    return objects[0]._dev_replace(anndata=new_adata)


def anndata_to_grandpy(anndata: ad.AnnData, transpose: bool = True) -> "GrandPy":
    """
    Create a GrandPy instance from an AnnData instance.

    Parameters
    ----------
    anndata: ad.AnnData
        The AnnData to convert.

    transpose: bool, default True
        If True, all Matrizes in the AnnData are transposed. (see Notes)
        Otherwise, they remain in their original form.

    Notes
    -----
    The internal AnnData has to be transposed, relative to what you would usually expect.
    Meaning obs has to contain the column metadata (coldata) and var the gene metadata (gene_info).

    See Also
    --------
    GrandPy.to_anndata
        Convert the GrandPy instance to AnnData.

    Returns
    -------
    GrandPy
        A GrandPy instance built from the AnnData.
    """
    from core_grandpy import GrandPy

    if transpose:
        adata = anndata.T

        if adata.uns.get("analyses", None) is not None:
            for name, analysis in adata.uns["analyses"].items():
                adata.uns["analyses"][name] = analysis.T
    else:
        adata = anndata

    return GrandPy(
        prefix=adata.uns.get("prefix", None),
        gene_info=adata.obs,
        coldata=adata.var,
        slots=adata.layers,
        metadata=adata.uns.get("metadata", None),
        analyses=adata.uns.get("analyses", None),
        plots=adata.uns.get("plots", None),
    )


def read_h5ad(path: Union[PathLike[str], str]) -> "GrandPy":
    """
    Construct a GrandPy instance from a file.

    Notes
    -----
    Stored plot function can currently not be saved to a file.

    See Also
    --------
    GrandPy.write_h5ad: Write a GrandPy instance to a file.

    Parameters
    ----------
    path: PathLike[str] or str
        The path to the file.

    Returns
    -------
    GrandPy
        A GrandPy instance loaded from the file.
    """
    anndata = ad.read_h5ad(path)

    return anndata_to_grandpy(anndata, transpose=False)



# --- Private utility functions ---
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
            warnings.warn(f"{len(duplicates_list)} Duplicates found: {', '.join(duplicates_list[:5])} (first 5); they have been renamed to ensure uniqueness (e.g., MATR3 → MATR3_1).")

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
    if df.index.name is None:
        df.index.name = by.index.name or "index"

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
    time: Union[str, np.ndarray, pd.Series, Sequence] = "duration.4sU",
    ci_size: float = 0.95,
    genes: Union[str, Sequence[str]] = None,
    show_progress: bool = False,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """
    This function is almost the same as `GrandPy.fit_kinetics`.
    The only difference is that it returns the kinetics data instead of a GrandPy object.
    """
    from .modeling import _fit_kinetics

    kinetics = _fit_kinetics(data=data, fit_type=fit_type, slot=slot, genes=genes, name_prefix=name_prefix, time=time,
                             ci_size=ci_size, return_fields=return_fields, show_progress=show_progress, **kwargs)

    return kinetics
