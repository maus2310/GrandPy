import copy
import re
import warnings
from collections.abc import Sequence, Mapping
from os import PathLike
from typing import Any, Union, Literal, Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from grandpy.analysis_tool import AnalysisTool
from grandpy.lfc import psi_lfc
from grandpy.plot_tool import PlotTool, Plot
from grandpy.slot_tool import SlotTool, ModeSlot
from grandpy.utils import _ensure_list, _make_unique, _reindex_by_index_name, _subset_dense_or_sparse


class GrandPy:
    """
    Create a GrandPy instance.

    Data is typically loaded using the `read_grand()` function, which parses preprocessed grandR-compatible
    data formats into a GrandPy object.

    Notes
    -----
    GrandPy objects are designed to be immutable. Most changes are made through `with_...` methods.
    Simple getters are implemented as properties, more complex ones through `get_...` methods.

    Examples
    --------
    Read a GrandPy object from a file.

    >>> import grandpy as gp
    >>> sars = gp.read_grand("./data/sars.tsv", design=("Condition", "Time", "Replicate"))
    >>> sars
    GrandPy:
    Read from ./data/sars.tsv
    1045 genes, 12 samples/cells
    Available data slots: count, ntr, alpha, beta
    Available analyses: None
    Available plots: None
    Default data slot: count

    See Also
    --------
    read_grand
        Create a GrandPy instance from a file.

    Parameters
    ----------
    prefix: str, optional
        Path to the data file.

    gene_info: pd.DataFrame, optional
        Genes and their metadata.

    coldata: pd.DataFrame, optional
        Samples or cells and their metadata.

    slots: dict[str, Union[np.ndarray, sp.csr_matrix]]], optional
        Name and the corresponding data matrix.

    metadata: dict[str, Any], optional
        Metadata about the data and file.

    analyses: dict[str, pd.DataFrame], optional
        Results from analyzing functions.

    plots: dict[str, dict[str, Plot]], optional
        Plot functions.
    """

    def __init__(
            self,
            prefix: str = None,
            gene_info: pd.DataFrame = None,
            coldata: pd.DataFrame = None,
            slots: dict[str, Union[np.ndarray, sp.csr_matrix]] = None,
            metadata: dict[str, Any] = None,
            analyses: dict[str, pd.DataFrame] = None,
            plots: dict[str, dict[str, Plot]] = None
    ):
        # Enforce that necessary things exist
        if gene_info is None:
            raise ValueError("GrandPy object must have gene_info.")
        if coldata is None:
            raise ValueError("GrandPy object must have coldata.")
        if slots is None:
            raise ValueError("GrandPy object must have slots (data).")
        if "count" not in slots:
            raise ValueError("GrandPy object must have a count slot.")

        # obs and var are swapped to allow the data to be gene x sample instead of sample x gene
        self._anndata = ad.AnnData(
            X = sp.csr_matrix(np.zeros(slots["count"].shape)),
            obs = gene_info,
            var = coldata,
        )
        self._is_sparse = True if sp.issparse(slots["count"]) else False

        self.__initialize_slots(slots)
        self.__initialize_uns_data(prefix, metadata, analyses, plots)
        self.__ensure_no4sU_column()
        self.__ensure_condition_column()

    def __initialize_slots(self, slots=None):
        for key, matrix in slots.items():
            self._anndata.layers[key] = matrix

    def __initialize_uns_data(self, prefix: str, metadata: dict[str, Any], analyses: dict[str, pd.DataFrame], plots: dict[str, dict[str, Plot]]):
        if metadata.get('default_slot') is None:
                metadata["default_slot"] = "count"

        self._anndata.uns['prefix'] = prefix if prefix is not None else "Unknown"
        self._anndata.uns['metadata'] = metadata
        self._anndata.uns['analyses'] = analyses if analyses is not None else {}
        self._anndata.uns['plots'] = plots if plots is not None else {}

    def __ensure_no4sU_column(self):
        if 'no4sU' not in self._anndata.var.columns:
            warnings.warn("No 'no4sU' entry in coldata, assuming all samples/cells as 4sU treated! "
                          "If the column is supposed to already exist, consider renaming it (see GrandPy.with_coldata())")
            self._anndata.var["no4sU"] = False

    def __ensure_condition_column(self):
        if 'Condition' not in self._anndata.var.columns:
            warnings.warn("No 'Condition' column in coldata, assuming all samples/cells as 'Control'! "
                          "Consider adding one. (see GrandPy.with_condition())")
            self._anndata.var["Condition"] = "Control"


    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self._anndata.uns['prefix']}\n"
            f"{self._anndata.n_obs} genes, {self._anndata.n_vars} samples/cells\n"
            f"Available data slots: {self.slots}\n"
            f"Available analyses: {self.analyses}\n"
            f"Available plots: {self.plots}\n"
            f"Default data slot: {self.default_slot}\n"
        )

    def __repr__(self):
        return (
            f"GrandPy object: {self._anndata.n_obs} genes, {self._anndata.n_vars} samples/cells\n"
            f"Available data slots: {self.slots}, default: {self.default_slot}\n"
            f"Available analyses: {self.analyses}\n"
            f"Available plots: {self.plots}\n"
        )

    def __len__(self):
        return self._anndata.n_obs

    def __contains__(self, key):
        return key in self._anndata.layers

    def __getitem__(self, items):
        new_adata = self._anndata.copy()
        new_adata = new_adata[items]

        # Reorders all existing analyses according to the new genes present in gene_info
        if new_adata.uns.get("analyses") is not None:
            for key in new_adata.uns["analyses"].keys():
                new_adata.uns["analyses"][key] = _reindex_by_index_name(new_adata.uns["analyses"][key], new_adata.obs)

        return self._dev_replace(anndata = new_adata)



    # ----- fundamental transformation methods -----
    def with_replaced_parameters(
            self,
            *,
            gene_info: pd.DataFrame = None,
            coldata: pd.DataFrame = None,
            slots: Mapping[str, Union[np.ndarray, sp.csr_matrix]] = None,
            metadata: Mapping[str, Any] = None,
            analyses: Mapping[str, Any] = None,
            plots: Mapping[str, Any] = None,
            anndata: ad.AnnData = None,
            **kwargs
    ) -> "GrandPy":
        """
        Replaces the specified parameters.

        This function is useful when you want to modify GrandPy objects on your own.

        Notes
        -----
        When trying to replace a slot (`slots` or `kwargs`), it has to be a numpy ndarray.

        Examples
        --------
        Replace the coldata of the GrandPy instance 'sars'.

        >>> sars = sars.with_replaced_parameters(coldata=new_coldata_dataframe)

        Replace the slot 'count'.

        >>> sars = sars.with_replaced_parameters(count=new_count_ndarray)

        See Also
        --------
        GrandPy.to_anndata
            Retrieves the internal anndata instance.

        Parameters
        ----------
        gene_info: pd.DataFrame, optional
            A new gene_info DataFrame.

        coldata: pd.DataFrame, optional
            A new coldata DataFrame.

        slots: dict[str, Union[np.ndarray, sp.csr_matrix]], optional
            A new dictionary of slots.

        metadata: dict[str, any], optional
            Replaces all metadata. If no `default_slot` is specified, the default slot will be set to 'count'.

        analyses: dict[str, Any], optional
            Replaces all analyses.

        plots: dict[str, any], optional
            Replaces all plots, gene and global.

        anndata: AnnData, optional
            Use with caution. GrandPy uses anndata internally. This will replace the whole anndata instance.
            Anndata can be retrieved from the instance via GrandPy.to_anndata().

            If both `anndata` and any of the other parameters are specified, `anndata` will be replaced first,
            followed by the rest, now in the new instance.

        **kwargs: np.ndarray or sp.csr_matrix
            Replaces a specific slot.

        Returns
        -------
        GrandPy
            A GrandPy instance with the given parameters replaced.
        """
        if anndata is None:
            anndata = self._anndata.copy()
        else:
            anndata = anndata.copy()

        new_gp = self.__class__(
            prefix = anndata.uns.get('prefix'),
            gene_info = gene_info.copy() if gene_info is not None else anndata.obs,
            coldata = coldata.copy() if coldata is not None else anndata.var,
            slots = copy.deepcopy(slots) if slots is not None else anndata.layers,
            metadata = copy.deepcopy(metadata) if metadata is not None else anndata.uns.get("metadata"),
            analyses = copy.deepcopy(analyses) if analyses is not None else anndata.uns.get("analyses"),
            plots = copy.deepcopy(plots) if plots is not None else anndata.uns.get("plots")
        )
        for param_name, param_value in kwargs.items():
            if param_name in new_gp.slots:
                new_gp._anndata.layers[param_name] = copy.deepcopy(param_value)
            else:
                warnings.warn(f"Tried to replace a slot with '{param_name}', but the slot was not found. Available slots: {new_gp.slots}")

        return new_gp

    def _dev_replace(
            self,
            *,
            prefix: str = None,
            gene_info: pd.DataFrame = None,
            coldata: pd.DataFrame = None,
            slots: Mapping[str, Union[np.ndarray, sp.csr_matrix]] = None,
            metadata: Mapping[str, Any] = None,
            analyses: Mapping[str, Any] = None,
            plots: Mapping[str, Any] = None,
            anndata: ad.AnnData = None
    ) -> "GrandPy":
        """
        This function is 'with_replaced_parameters' for internal use.

        This function does not copy provided parameters. Parts can become mutable if not handled correctly!

        Parameters
        ----------
        prefix: str, optional
            A new prefix.

        gene_info: pd.DataFrame, optional
            A new gene_info DataFrame.

        coldata: pd.DataFrame, optional
            A new coldata DataFrame.

        slots: dict[str, Union[np.ndarray, sp.csr_matrix]], optional
            A new dictionary of slots.

        metadata: dict[str, any], optional
            Replaces all metadata.

        analyses: dict[str, Any], optional
            Replaces all analyses.

        plots: dict[str, any], optional
            Replaces all plots, gene and global.

        anndata: AnnData, optional
            Particularly dangerous parameter. GrandPy uses anndata internally. This will replace the whole anndata instance.

        Returns
        -------
        GrandPy
            A GrandPy object with the given parameters replaced.
        """
        if anndata is None:
            anndata = self._anndata.copy()

        return self.__class__(
            prefix = prefix if prefix is not None else anndata.uns.get('prefix'),
            gene_info = gene_info if gene_info is not None else anndata.obs,
            coldata = coldata if coldata is not None else anndata.var,
            slots = slots if slots is not None else anndata.layers,
            metadata = metadata if metadata is not None else anndata.uns.get("metadata"),
            analyses = analyses if analyses is not None else anndata.uns.get("analyses"),
            plots = plots if plots is not None else anndata.uns.get("plots")
        )

    def copy(self) -> "GrandPy":
        """
        Copy the instance.

        Returns
        -------
        GrandPy
            A copy of the current instance.
        """
        return self.with_replaced_parameters()


    def write_h5ad(self, path: Union[PathLike[str], str], compression: Literal["gzip", "lzf"] = None) -> None:
        """
        Save the instance as a h5ad file.

        Notes
        -----
        Stored plot functions can currently not be saved to a file.

        See Also
        --------
        read_h5ad: Load a GrandPy instance from a h5ad file.

        Parameters
        ----------
        path: PathLike[str] or str
            The path where the file will be saved.

        compression: Literal['gzip' or 'lzf'], optional
            The compression to be used. If None, no compression is used. Generally 'gzip' compresses more, but 'lzf' is faster.
            Both of them are lossless. For more information and other compression methods, see h5py documentation.
            (https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data)
        """
        anndata = self._anndata.copy()

        anndata.uns["plots"] = {}

        anndata.write(path, convert_strings_to_categoricals=False, compression=compression)



    # ----- Basic properties and methods -----
    @property
    def title(self) -> str:
        """
        Get a title for the GrandPy object.
        The title is derived from the prefix.
        """
        prefix = self._anndata.uns.get('prefix')
        if prefix is None:
            raise KeyError("Title not available. Please specify a prefix when initializing the GrandPy object")
        return Path(prefix).name

    @property
    def shape(self) -> tuple[int]:
        """
        Get the dimension of the slots.
        """
        return self._anndata.X.shape

    @property
    def dim_names(self) -> tuple[list[str], list[str]]:
        """
        Get the column and row names of the slots.
        """
        row_names = self.gene_info.index.tolist()
        column_names = self.coldata.index.tolist()
        return row_names, column_names

    @property
    def default_slot(self) -> str:
        """
        Get the name of the default slot

        See Also
        --------
        GrandPy.with_default_slot
            Set a default slot.
        """
        return self.metadata.get('default_slot')

    def with_default_slot(self, name: str) -> "GrandPy":
        """
        Sets the default slot to `name`.

        Parameters
        ----------
        name: str
            Sets the default slot to this slot.

        Returns
        -------
        "GrandPy"
            Returns a GrandPy instance having the new default slot.
        """
        if name not in self.slots:
            raise ValueError(f"The name '{name}' is not a valid slot name. Please use one of the following names: {self.slots}")

        new_metadata = self.metadata
        new_metadata['default_slot'] = name

        return self._dev_replace(metadata=new_metadata)



    # ----- All slot methods ------
    @property
    def __slot_tool(self) -> SlotTool:
        return SlotTool(self._anndata, self._is_sparse)

    @property
    def slots(self) -> list[str]:
        """
        Get the names of all available slots.

        See Also
        --------
        GrandPy.get_table:
            Get the data from slots. (genes x samples)

        GrandPy.get_data:
            Get the data from slots. (samples x genes)
        """
        return self.__slot_tool.slots()

    def __check_slot(self, slot: Union[str, ModeSlot], *, allow_ntr: bool = True) -> bool:
        """
        Checks if a given slot exists in the data slots.

        Parameters
        ----------
        slot: str or ModeSlot
            The slot to be checked.

        allow_ntr: bool, default True
            If True, the slot "ntr" is allowed as input.

        Returns
        -------
        bool:
            True if the slot exists, False otherwise.
        """
        return self.__slot_tool._check_slot(slot, allow_ntr=allow_ntr)

    def __resolve_mode_slot(self, mode_slot: Union[str, ModeSlot], *, allow_ntr: bool = True, ntr_nan: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Checks whether the given slot is valid and computes the resulting mode slot if a mode was specified.

        Mode slots can be specified in the following formats: ModeSlot('<mode>', '<slot>'), '<mode>_<slot>', or '<slot>'.

        Parameters
        ----------
        mode_slot: str or ModeSlot
            A slot or a mode slot to be resolved.

        allow_ntr: bool, default True
            If True, the slot "ntr" is allowed as input.

        ntr_nan: bool, default False
            If True, the slot "ntr" is treated as NaN for no4sU samples/cells.

        Returns
        -------
        Union[np.ndarray, sp.csr_matrix]
            The resulting slot after the mode has been applied.
        """
        return self.__slot_tool._resolve_mode_slot(mode_slot, allow_ntr=allow_ntr, ntr_nan=ntr_nan)

    def with_dropped_slots(self, slots_to_remove: Union[str, Sequence[str]]) -> "GrandPy":
        """
        Returns a GrandPy instance with specified slot(s) removed.

        See Also
        --------
        GrandPy.with_slot:
            Add a new slot.

        Parameters
        ----------
        slots_to_remove: str or Sequence[str]
            One or more slots to remove from the data.

        Returns
        ----------
        GrandPy
            A GrandPy instance with specified slot(s) removed.
        """
        slots_to_remove = _ensure_list(slots_to_remove)

        new_slots, new_metadata = self.__slot_tool.with_dropped_slots(slots_to_remove)

        return self._dev_replace(slots=new_slots, metadata=new_metadata)

    def with_slot(self, name: str, value: Union[np.ndarray, pd.DataFrame, sp.csr_matrix], *, set_to_default = False) -> "GrandPy":
        """
        Returns a GrandPy instance with the new slot added. Will overwrite if the slot already exists and give a warning.

        Notes
        -----
        Recommended: use this function with DataFrames for security.

        It can only check the order of genes and samples/cells if the given matrix is a pandas DataFrame.
        Otherwise, the given matrix is expected to have rows and columns in the same order as existing slots.

        See Also
        --------
        GrandPy.get_table:
            Get the data from slots. (genes x samples)

        GrandPy.get_data:
            Get the data from slots. (samples x genes)

        Parameters
        ----------
        name: str
            Name of the new slot.

        value: np.ndarray or pd.DataFrame or sp.csr_matrix
            The data to be added as a new slot.

        set_to_default: bool, default False
            If True, sets the new slot as the default slot.

        Returns
        -------
        GrandPy
            A GrandPy instance with the new slot added.
        """
        new_slots, new_metadata = self.__slot_tool.with_slot(name, value, set_to_default=set_to_default)

        return self._dev_replace(slots=new_slots, metadata=new_metadata)

    def with_ntr_slot(self, as_ntr: str, save_ntr_as: str = None) -> "GrandPy":
        """
        Set a different slot as the new 'ntr' slot. If save_ntr_as is not set, the former ntr slot will be removed.
        The slot 'ntr' will be used for `mode_slots` and other functions relating to ntr.

        Examples
        --------
        Save the slot 'upper' (upper ntr) as the new 'ntr' slot for the GrandPy object 'sars'.

        >>> sars = sars.with_ntr_slot('upper')
        >>> print(sars.slots)
        ['count', 'ntr', 'alpha', 'beta', 'upper', 'lower']

        Notice the former 'ntr' was overwritten.
        It can be saved under a new name with the 'save_ntr_as' paramter.

        >>> sars = sars.with_ntr_slot('upper', save_ntr_as='normal_ntr')
        >>> print(sars.slots)
        ['count', 'ntr', 'alpha', 'beta', 'upper', 'lower', 'normal_ntr']

        Parameters
        ----------
        as_ntr: str
            The name of the slot to be set as the new 'ntr' slot.

        save_ntr_as: str, optional
            The name you want the old ntr slot to be saved as. If None, the old ntr slot will be deleted.

        Returns
        -------
        GrandPy
            A GrandPy instance with a new 'ntr' slot.

        Raises
        ------
        ValueError
            When the given name for ntr is not valid.
        """
        new_slots = self.__slot_tool.with_ntr_slot(as_ntr, save_ntr_as=save_ntr_as)

        return self._dev_replace(slots=new_slots)



    # ----- All analysis methods -----
    @property
    def __analysis_tool(self):
        return AnalysisTool(self._anndata)

    @property
    def analyses(self) -> list[str]:
        """
        Get the names of all stored analyses.

        See Also
        --------
        GrandPy.get_analysis_table:
            Retrieve analysis data.

        GrandPy.get_analysis:
            Get the names of analyses matching a pattern.

        GrandPy.with_dropped_analyses:
            Remove analyses with a regex pattern.

        GrandPy.with_analysis:
            Add an analysis to the instance. Usually not to be used directly.
        """
        return self.__analysis_tool.analyses()

    def get_analyses(self, pattern: Union[str, int, Sequence[Union[str, int, bool]]] = None, regex: bool = True, description: bool = False) -> list[str]:
        """
        Get the names of analyses. Either by regex, names, indices, or a boolean mask.

        See Also
        --------
        GrandPy.get_analysis_table:
            Retrieve analysis data.

        GrandPy.analyses:
            Get a list of all available analyses.

        GrandPy.with_dropped_analyses:
            Remove analyses with a regex pattern.

        GrandPy.with_analysis:
            Add an analysis to the instance. Usually not to be used directly.

        Parameters
        ----------
        pattern: str or int or Sequence[str or int or bool], optional
            Names of analyses to be retrieved. Can be a regex, names, indices, or a boolean mask.

        regex: bool, default True
            If True, `name` will be interpreted as a regular expression or a list of regular expressions.

        description: bool, default False
            If True, the names of selected analyses will be returned with their column names.

        Returns
        -------
        list[str]
            A list containing the names of all matching analyses.
        """
        return self.__analysis_tool.get_analyses(pattern, regex=regex, description=description)

    def with_analysis(self, name: str, table: pd.DataFrame) -> "GrandPy":
        """
        Returns a GrandPy instance with added analyses.

        Not to be used directly in most cases, instead it is called by analysis methods.

        Notes
        -----
        The index of `table` has to be named "Symbol", containing the gene symbols.

        See Also
        --------
        GrandPy.get_analysis_table:
            Retrieve analysis data.

        GrandPy.analyses:
            Get the names of all stored analyses.

        GrandPy.get_analysis:
            Get the names of analyses matching a pattern.

        GrandPy.with_dropped_analyses:
            Remove analyses with a regex pattern.

        Parameters
        ----------
        name: str
            The name of the analysis.

        table: pd.DataFrame
            A DataFrame containing the analysis data. Has to contain gene names or symbols.

        Returns
        -------
        A GrandPy instance with added analyses.
        """
        new_analyses = self.__analysis_tool.with_analysis(name, table)

        return self._dev_replace(analyses=new_analyses)

    def with_dropped_analyses(self, pattern: Union[str, Sequence[str]] = None) -> "GrandPy":
        """
        Returns a GrandPy instance with analyses matching the pattern removed.

        See Also
        --------
        GrandPy.get_analysis_table:
            Retrieve analysis data.

        GrandPy.analyses:
            Get the names of all stored analyses.

        GrandPy.get_analysis:
            Get the names of analyses matching a pattern.

        GrandPy.with_analysis:
            Add an analysis to the instance. Usually not to be used directly.

        Parameters
        ----------
        pattern: str or Sequence[str], optional
            One or multiple regex patterns to match analyses. If None, all analyses will be removed.

        Returns
        -------
            A GrandPy instance with removed analyses.
        """
        new_analyses = self.__analysis_tool.drop_analyses(pattern)

        return self._dev_replace(analyses=new_analyses)



    # ----- All plot methods -----
    @property
    def __plot_tool(self):
        return PlotTool(self._anndata)

    @property
    def plots(self) -> dict[str, list[str]]:
        """
        Get available plot names.

        See Also
        --------
        GrandPy.with_plot
            Add a plot function.

        GrandPy.with_dropped_plots
            Remove plots matching a regex pattern.

        Returns
        -------
        dict[str, list[str]]
            A dictionary mapping plot types('gene', 'global') to plot names.
        """
        return self.__plot_tool.plots()

    def with_plot(self, name: str, function: Union[Plot, Callable]) -> "GrandPy":
        """
        Returns a GrandPy instance with a plot added. Either a global or gene plot.
        Global plots only take a GrandPy object. Gene plots additionally require a gene.

        Examples
        --------
        Store a global plot function in the object:

        >>> sars = sars.with_plot(
        ...     "scatter",
        ...     lambda data: plot_scatter(data, x = "Mock.1h.A", y = "SARS.1h.A", mode_slot = "new_count")
        ...)
        >>> print(sars.plots)
        {'global': ['scatter']}

        Alternative version for adding plots using a Plot object:

        >>> sars = sars = sars.with_plot(
            ...     "scatter",
            ...     Plot(
            ...         function = plot_scatter,
            ...         parameters = {"x": "Mock.1h.A", "y": "SARS.1h.A", "mode_slot": "new_count"},
            ...         plot_type = "global"
            ...     )
            ... )
        >>> print(sars.plots)
        {'global': ['scatter']}

        Storing a gene plot function in the object:

        >>> sars = sars.with_plot("old_vs_new", lambda data, gene: plot_gene_old_vs_new(data, gene, slot = "count"))
        >>> print(sars.plots)
        {'gene': ['old_vs_new']}

        Executing stored plot functions:

        >>> sars.plot_global("scatter")
        >>> sars.plot_gene("old_vs_new", gene="Mock.1h.A")

        See Also
        --------
        Plot
            A class used to store a plot function.

        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_dropped_plots
            Remove plots matching a regex.

        GrandPy.plot_global
            Execute a stored global plot function.

        GrandPy.plot_gene
            Execute a stored gene plot function for a given gene.

        Parameters
        ----------
        name: str
            A name for the plot.

        function: Plot or Callable
            A Plot object, or a funktion, that takes a GrandPy object and optionally a gene as input and returns a plot.

        Returns
        -------
        GrandPy
            A new GrandPy object with a plot added.
        """
        new_plots = self.__plot_tool.with_plot(name, function)

        return self._dev_replace(plots=new_plots)

    def plot_gene(self, name: str, gene: str):
        """
        Executes a stored plot function for a given gene.

        See Also
        --------
        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_plot
            Add a plot function.

        GrandPy.with_dropped_plots
            Remove plots matching a regex.

        GrandPy.plot_global
            Executes a stored global plot function.

        Parameters
        ----------
        name: str
            The name of the stored plot.

        gene: str
            The name of a gene.
        """
        try:
            return self._anndata.uns["plots"]["gene"][name](self, gene)
        except KeyError:
            raise KeyError(f"No plot named, '{name}' was found. These are all available gene plots: {self.plots.get('gene', None)}")

    def plot_global(self, name: str):
        """
        Executes a stored global plot function.

        See Also
        --------
        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_plot
            Add a plot function.

        GrandPy.with_dropped_plots
            Remove plots matching a regex.

        GrandPy.plot_gene
            Executes a stored plot function for a given gene.

        Parameters
        ----------
        name: str
            The name of the stored plot.
        """
        try:
            return self._anndata.uns["plots"]["global"][name](self)
        except KeyError:
            raise KeyError(f"No plot named '{name}' was found. These are all available global plots: {self.plots.get('global', None)}")

    def with_dropped_plots(self, pattern: str = None) -> "GrandPy":
        """
        Returns a GrandPy instance with plot names matching the regex pattern removed.

        See Also
        --------
        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_plot
            Add a plot function.

        Parameters
        ----------
        pattern: str, optional
            A regular expression matching plot names to be dropped.

        Returns
        -------
        GrandPy
            A GrandPy instance with plot names matching the pattern removed.
        """
        new_plots = self.__plot_tool.with_dropped_plots(pattern)

        return self._dev_replace(plots=new_plots)



    # ----- Methods relating to coldata, gene_info or metadata -----
    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get the metadata about the GrandPy instance.
        """
        return self._anndata.uns.get('metadata').copy()


    @property
    def gene_info(self) -> pd.DataFrame:
        """
        Get the gene_info DataFrame.

        See Also
        --------
        GrandPy.with_gene_info
            Modify the gene_info.

        GrandPy.coldata
            Get the coldata DataFrame.
        """
        return self._anndata.obs.copy()

    def with_gene_info(self, value: Union[Mapping, pd.Series, pd.DataFrame, np.ndarray, Sequence], name:str = None) -> "GrandPy":
        """
        Returns a GrandPy instance with modified gene_info. If 'name' does not already exist as a column in `gene_info`, it will be added.

        Otherwise, the column 'name' will be replaced by the given value or updated if a dictionary was given.

        See Also
        --------
        GrandPy.gene_info
            Get the gene_info DataFrame.

        GrandPy.with_replaced_parameters
            Replace whole parts of the instance, such as gene_info.

        Parameters
        ----------
        value : Mapping or pd.Series or pd.DataFrame or np.ndarray or Sequence
            The values to assign can be any iterable.
            Can also be a dictionary when trying to update a column.

            If 'name' is None, 'value' is expected to be a pandas Series or DataFrame.

        name : str, optional
            The name of the column to be modified.

        Returns
        -------
        GrandPy
            A GrandPy instance with updated gene_info.
        """
        new_geneinfo = self.gene_info

        if name is None:
            if not isinstance(value, (pd.Series, pd.DataFrame)):
                raise ValueError("If column is None, value must be a pandas Series or DataFrame.")

            if isinstance(value, pd.Series):
                value = value.to_frame()

            value.index = _make_unique(value.index, warn=False)

            for name in value.columns:
                new_geneinfo[name] = value[name]

            return self._dev_replace(gene_info=new_geneinfo)


        if name in new_geneinfo.columns:
            if isinstance(value, Mapping):
                if all(v in new_geneinfo[name].values for v in value.keys()):
                    for match_value, new_val in value.items():
                        new_geneinfo.loc[new_geneinfo[name] == match_value, name] = new_val
                else:
                    for row_index, new_value in value.items():
                        if row_index in new_geneinfo.index:
                            new_geneinfo.at[row_index, name] = new_value
                return self._dev_replace(gene_info = new_geneinfo)

        new_geneinfo[name] = value

        return self._dev_replace(gene_info = new_geneinfo)

    @property
    def genes(self) -> list[str]:
        """
        Get the gene symbols.

        These names are used as the row names of the data slots and the row names of gene_info.

        See Also
        --------
        GrandPy.get_genes
            Retrieve specified symbols or ensemble IDs.

        GrandPy.with_updated_symbols
            Update gene symbols by deriving them from the ensemble IDs.
        """
        return self.gene_info["Symbol"].tolist()

    def get_genes(self, genes: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, get_gene_symbols: bool = True, regex: bool = False) -> list[str]:
        """
        Get symbols or ensemble IDs.

        If no genes are specified, all genes are returned.

        See Also
        --------
        GrandPy.get_columns
            Get the sample/cell names.

        GrandPy.gene_info
            Get the gene_info DataFrame.

        GrandPy.get_index
            Get the index of gene names/symbols.

        Parameters
        ----------
        genes: str or int or Sequence[str or int or bool], optional
            Genes to be retrieved. Either by their index, their symbol, their ensemble ID, a boolean mask, or a regex.

        get_gene_symbols: bool, default True
            If True, gene symbols will be returned.
            Otherwise, gene names (Ensemble IDs) will be returned.

        regex: bool, default False
            If True, `genes` will be interpreted as a regular expression or a list of regular expressions.

        Returns
        -------
        list[str]
            A list containing the specified genes.
        """
        if get_gene_symbols:
            if genes is None:
                return self.genes

            indices = self.get_index(genes, regex=regex)
            return self.gene_info.iloc[indices]["Symbol"].tolist()

        else:
            if genes is None:
                return self.gene_info["Gene"].tolist()

            indices = self.get_index(genes, regex=regex)
            return self.gene_info.iloc[indices]["Gene"].tolist()

    def get_index(self, genes: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, regex: bool = False) -> list[int]:
        """
        Get the index of symbols or ensemble IDs, chosen by higher number of matches.

        Either by gene name, symbol, index, or a boolean mask.

        Parameters
        ----------
        genes: str or int or Sequence[str or int or bool], optional
            Specifies which indices to return.

        regex: bool, default False
            If True, `gene` will be interpreted as a regular expression or a list of regular expressions.

        Returns
        -------
        list[int]
            A list containing the specified indices.
        """
        gene_info = self.gene_info
        n = len(gene_info)

        if genes is None:
            return list(range(n))

        gene_col = gene_info["Gene"].astype(str)
        symbol_col = gene_info["Symbol"].astype(str)

        genes = _ensure_list(genes)

        if not genes:
            return []

        if any(pd.isna(genes)):
            warnings.warn("NaN values removed from gene input.")
            genes = [g for g in genes if pd.notna(g)]

        if regex:
            gene_mask = np.array([False] * len(gene_col))
            symbol_mask = np.array([False] * len(symbol_col))

            for pattern in genes:
                gene_mask_current = gene_col.str.contains(pattern, regex=True, case=False, na=False).values
                symbol_mask_current = symbol_col.str.contains(pattern, regex=True, case=False, na=False).values

                gene_mask = gene_mask | gene_mask_current
                symbol_mask = symbol_mask | symbol_mask_current

            gene_matches = gene_mask.sum()
            symbol_matches = symbol_mask.sum()

            if gene_matches > symbol_matches:
                return np.flatnonzero(gene_mask).tolist()
            else:
                return np.flatnonzero(symbol_mask).tolist()

        # Boolean mask
        if all(isinstance(g, (bool, np.bool)) for g in genes):
            if len(genes) != n:
                raise ValueError(f"Boolean mask length({n}) does not match gene count({len(genes)}).")
            return list(np.flatnonzero(genes))

        # Integer index
        if all(isinstance(g, (int, np.integer)) for g in genes):
            for g in genes:
                if not 0 <= g < n:
                    raise IndexError(f"Gene index out of range. Must be between 0 and {n - 1}. Got {g}.")
            return genes

        # Matching by Gene/Symbol string
        genes_str = pd.Series(genes, dtype=str)

        matches_gene = genes_str[genes_str.isin(set(gene_col))]
        matches_symbol = genes_str[genes_str.isin(set(symbol_col))]

        use_col = gene_col if len(matches_gene) >= len(matches_symbol) else symbol_col
        ref_map = pd.Series(np.arange(n), index=use_col)

        found = genes_str[genes_str.isin(ref_map.index)]
        missing = genes_str[~genes_str.isin(ref_map.index)]

        if not missing.empty:
            preview = ", ".join(missing.head(5))
            more = " ..." if len(missing) > 5 else ""
            warnings.warn(f"Could not find {len(missing)} genes (e.g. {preview}{more})")

        return ref_map.loc[found].tolist()

    def with_updated_symbols(self, species: str = "human") -> "GrandPy":
        """
        Adds or updates(if it already exists) the column 'Symbols' in the gene_info.
        Symbols are derived from the column 'Gene', containing ensemble IDs.

        See Also
        --------
        GrandPy.genes
            Get the gene symbols.

        Parameters
        ----------
        species: str, default "human"
            The species the genes belong to. See mygene(https://pypi.org/project/mygene/) for supported species.
            The most common species include: 'human', 'mouse', 'rat', 'zebrafish', 'fruitfly', 'worm', 'yeast', 'chicken'.

        Returns
        -------
        GrandPy
            A GrandPy instance with the column 'Symbols' added to the gene_info.
        """
        import mygene
        import logging

        gene_info = self.gene_info
        genes = self.get_genes(get_gene_symbols=False)

        mg = mygene.MyGeneInfo()

        # This is done to suppress a specific warning mygene will output otherwise
        logging.getLogger('biothings.client').setLevel(logging.ERROR)

        # Query: get the symbols in batches (a single large request is not possible with mygene)
        try:
            result = mg.querymany(
                genes,
                scopes="ensembl.gene",
                fields="symbol",
                species=species,
                as_dataframe=True,
            )
        except Exception as e:
            raise RuntimeError(f"MyGeneInfo request failed: {e}")

        if "symbol" not in result:
            warnings.warn("No valid ensembl IDs were found. Check if the 'species' parameter is set correctly, "
                          "and that gene_info contains the column 'Gene' with valid Ensemble IDs.")
            return self

        result = _make_unique(result["symbol"].dropna())
        result.index.name = "Gene"
        result = _reindex_by_index_name(result, gene_info)

        if gene_info.get("Symbol", None) is not None:
            new_gene_info = gene_info.set_index("Gene", drop=False)

            result.name = "Symbol"

            new_gene_info.update(result)

            result = new_gene_info["Symbol"]

        return self.with_gene_info(name="Symbol", value=result.values)

    def get_classified_genes(self, classification_label: str) -> list:
        """
        Returns a list of gene names corresponding to the given classification label.

        Examples
        --------
        Retrieve all genes with the `Type` 'Unknown' from the GrandPy instance 'sars'.

        >>> self.get_classified_genes("Unknown")
        ['ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF10', 'ORF1ab', 'S']

        Parameters
        ----------
        classification_label : str
            The classification label to use.

        Returns
        -------
        list:
            A list of gene symbols corresponding to the given classification label.
        """
        return self.gene_info[self.gene_info["Type"] == classification_label].get("Symbol").tolist()

    def get_significant_genes(
            self,
            analysis = None,
            criteria: str = None,
            *,
            regex=True,
            use_symbols: bool = True,
            as_table: bool = False,
            with_gene_info: bool = True
    ) -> Union[list[str], pd.DataFrame]:
        """
        Return significant genes based on analysis results.

        Examples
        --------
        Perform an analysis on the GrandPy instance 'sars'.

        >>> sars = sars.pairwise(sars.get_contrasts())
        >>> sars.get_analyses(description=True)
        {'total_Mock vs SARS': ['M', 'S', 'P', 'Q', 'LFC']}

        Get the genes with Q < 0.05 and an LFC >= 1.

        >>> sig_genes = sars.get_significant_genes(criteria="Q < 0.05 & LFC >= 1")
        >>> print(len(sig_genes))
        45

        Parameters
        ----------
        analysis : str or list[str], optional
            Names of the analysis results to evaluate.

        criteria : str, optional
            String expression evaluated against `analysis`. By default, 'Q<0.05 & abs(LFC)>=1'.

        regex : bool, default True
            If True, `analysis` is evaluated as a regular expression.

        use_symbols : bool, default True
            Whether to use gene symbols or ensemble IDs in the result.

        as_table : bool, default False
            If True, return the genes as a DataFrame with their respective significance values.

        with_gene_info : bool, default True
            Whether to include gene info columns in output. Only relevant if `as_table` is True.

        Returns
        -------
        Union[list[str], pd.DataFrame]
            A list of significant gene symbols or a DataFrame containing the respective significance values.
        """
        analyses = self.get_analyses(analysis, regex=regex)
        result = self.gene_info

        if analysis not in analyses:
            raise ValueError(f"analysis '{analysis}' not found. Available analyses are {analyses}")

        for name in analyses:
            tab = self.get_analysis_table(
                analyses=name,
                regex=False,
                with_gene_info=False,
                prefix_by_analyses=False
            )

            if criteria is None:
                use = (tab["Q"] < 0.05) & (tab["LFC"].abs() >= 1)
            else:
                try:
                    use = tab.eval(criteria)
                except Exception as e:
                    raise ValueError(f"Could not evaluate criteria '{criteria}': {e}")

            use = use.fillna(False)
            result[name] = use

        if not use_symbols:
            result = result.set_index("Gene", drop=False)

        if not as_table:
            result = result.iloc[:, self.gene_info.shape[1]:]

            classes = set(result.dtypes)
            if len(classes) != 1:
                raise ValueError("The criteria evaluation returned mixed data types. Results should either be numeric or boolean.")

            dtype = list(classes)[0]
            if pd.api.types.is_bool_dtype(dtype):
                result = result.any(axis=1)
                return result[result].index.tolist()

            elif pd.api.types.is_numeric_dtype(dtype):
                if result.shape[1] > 1:
                    raise ValueError("Multiple numeric values present, can only return as a table.")
                return result.sort_values(by=result.columns[0], ascending=False).index.tolist()

        if not with_gene_info:
            result = result.iloc[:, self.gene_info.shape[1]:]

        result = result.sort_values(by=result.columns[-1], ascending=False)

        return result


    @property
    def coldata(self) -> pd.DataFrame:
        """
        Get the coldata DataFrame.

        See Also
        --------
        GrandPy.with_coldata
            Modify the coldata.

        GrandPy.gene_info
            Get the gene_info DataFrame.
        """
        return self._anndata.var.copy()

    def with_coldata(self, value: Union[Mapping, pd.Series, pd.DataFrame, np.ndarray, Sequence], name: str = None, ) -> "GrandPy":
        """
        Returns a GrandPy instance with modified coldata. If 'name' does not already exist as a column in `coldata`, it will be added.

        Otherwise, the column 'name' will be replaced by the given 'value' or updated if a dictionary was given.

        If 'name' is None or not given, 'value' is expected to be a pandas Series or DataFrame.

        See Also
        --------
        GrandPy.coldata
            Get the coldata DataFrame.

        GrandPy.with_replaced_parameters
            Replace whole parts of the instance, such as coldata.

        Parameters
        ----------
        value : Mapping or pd.Series or pd.DataFrame or np.ndarray or Sequence
            The values to assign can be any iterable or array-like.
            Can also be a dictionary when trying to update a column.

            If 'name' is None, 'value' is expected to be a pandas Series or DataFrame.

        name : str, optional
            The name of the column to be modified.

        Returns
        -------
        GrandPy
            A GrandPy instance with updated coldata.
        """
        new_coldata = self.coldata

        if name is None:
            if not isinstance(value, (pd.Series, pd.DataFrame)):
                raise ValueError("If column is None, value must be a pandas Series or DataFrame.")

            if isinstance(value, pd.Series):
                value = value.to_frame()

            for name in value.columns:
                new_coldata[name] = value[name]

            return self._dev_replace(coldata=new_coldata)

        if name in new_coldata.columns:
            if isinstance(value, Mapping):
                if all(v in new_coldata[name].values for v in value.keys()):
                    for match_value, new_val in value.items():
                        new_coldata.loc[new_coldata[name] == match_value, name] = new_val
                else:
                    for row_index, new_value in value.items():
                        if row_index in new_coldata.index:
                            new_coldata.at[row_index, name] = new_value
                    return self._dev_replace(coldata = new_coldata)

        new_coldata[name] = value

        return self._dev_replace(coldata = new_coldata)

    @property
    def condition(self) -> list[str]:
        """
        Get the condition of all samples/cells in the coldata.

        See Also
        --------
        GrandPy.with_condition
            Set the condition of all samples/cells.
        """
        return self.coldata['Condition'].tolist()

    def with_condition(self, value: Union[str, Sequence[str], pd.Series, Mapping]) -> "GrandPy":
        """
        Set new values for all samples/cells in the coldata.

        Examples
        --------
        Get the condition for all samples from the GrandPy instance 'sars'.

        >>> sars.condition
        ['Mock', 'Mock', 'Mock', 'Mock', 'Mock', 'Mock', 'SARS', 'SARS', 'SARS', 'SARS', 'SARS', 'SARS']

        Set the condition for all samples to 'Control'.

        >>> sars = sars.with_condition("Control")
        >>> sars.condition
        ['Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control']

        Construct the condition from the coldata columns 'Condition' and 'Replicate'.

        >>> sars = sars.with_condition(["Condition", "Replicate"])
        >>> sars.condition
        ['Control A', 'Control A', 'Control A', 'Control B', 'Control A', 'Control A', 'Control A', 'Control A', 'Control A', 'Control B', 'Control A', 'Control A']

        Parameters
        ----------
        value: str or Sequence[str] or pd.Series or Mapping
            The conditions to be set for the samples/cells.
            Can also construct the name from other columns in `coldata`, if their names are given.

        Returns
        -------
        GrandPy
            A GrandPy instance with the specified condition.
        """
        new_coldata = self.coldata

        if isinstance(value, Mapping):
            for k, v in value.items():
                new_coldata.loc[k, "Condition"] = v
            return self._dev_replace(coldata = new_coldata)

        value = _ensure_list(value)

        if all(v in new_coldata.columns for v in value):
            new_coldata['Condition'] = new_coldata[value].astype(str).agg(" ".join, axis=1)
        else:
            if len(value) == 1:
                value = value * len(new_coldata.index)
            elif len(value) != len(new_coldata.index):
                raise ValueError(
                    f"Number of values ({len(value)}) does not match number of samples/cells ({len(new_coldata.index)})")

            new_coldata['Condition'] = value

        return self._dev_replace(coldata = new_coldata)

    @property
    def columns(self) -> list[str]:
        """
        Get the sample/cell names.

        These names are used as the column names of the data slots and the row names of the coldata.

        See Also
        --------
        GrandPy.get_columns
            Retrieve specified columns

        GrandPy.with_renamed_columns
            Rename coldata columns according to a given mapping.

        GrandPy.with_swapped_columns
            Swap the values of two coldata columns. Do this if they were mislabeled
        """
        return self.coldata.index.tolist()

    def get_columns(self, columns: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, reorder: bool = False) -> list[str]:
        """
        Get sample/cell names.

        If no columns are specified, all sample/cell names are returned.

        Parameters
        ----------
        columns: str or int or Sequence[str or int or bool], optional
            Samples/cells to be retrieved. Either by their index, their name, or a boolean mask.

        reorder: bool, default False
            If True, the returned list will be in the same order as the original column data.

            Otherwise, the returned list will be in the same order as the input.

        Returns
        -------
        list[str]
            A list containing the specified samples/cells.

        See Also
        --------
        GrandPy.get_genes
            Get the gene symbols/names.

        GrandPy.coldata
            Get the entire coldata DataFrame.
        """
        coldata = self.coldata

        if columns is None:
            return self.columns

        columns = _ensure_list(columns)

        if columns == []:
            return []

        if all(isinstance(column, int) for column in columns):
            result = coldata.iloc[columns]

        else:
            try:
                result = coldata.loc[columns, :]
            except KeyError as e:
                raise e

        if reorder:
            result = result.reindex(coldata.index).dropna(how="all", axis=0)

        return list(result.index)

    def with_renamed_columns(self, mapping: Mapping) -> "GrandPy":
        """
        Returns a new GrandPy object with columns in slots renamed and corresponding rows in coldata.

        Parameters
        ----------
        mapping: Mapping
            Columns will be renamed according to this mapping.

        Returns
        -------
        GrandPy
            A new GrandPy object with renamed columns.

        See Also
        --------
        GrandPy.with_swapped_columns
            Swaps columns in coldata and the corresponding in all slots.

        GrandPy.get_columns
            Returns the names of columns, either by index, name, or a boolean mask.
        """
        coldata = self.coldata

        missing = [key for key in mapping if key not in coldata.columns]
        if missing:
            warnings.warn(f"Following rows cannot be renamed, as they do not exist in coldata: {missing}")

        new_coldata = coldata.rename(mapping)

        return self._dev_replace(coldata = new_coldata)

    def with_swapped_columns(self, column1: Union[str, int], column2: Union[str, int]) -> "GrandPy":
        """
        Returns a new GrandPy object with the values of two columns swapped in all slots.
        Swaps only the values, not the entire columns.

        This is what you call when samples/cells have been mislabeled.

        Parameters
        ----------
        column1: str or int
            sample/cell to swap with.

        column2: str or int
            sample/cell to swap with.

        Returns
        -------
        GrandPy
            A new GrandPy object with values for two columns swapped.

        See Also
        --------
        GrandPy.with_renamed_columns
            Rename the columns of all slots.
        """
        def swap(matrix, col1, col2):
            """
            Helper function that swaps columns for ndarray or csr_matrix and rows for a DataFrame.
            """
            if isinstance(matrix, np.ndarray):
                matrix = matrix.copy()
                matrix[:, [col1, col2]] = matrix[:, [col2, col1]]

            elif sp.issparse(matrix):
                # Convert to CSC for efficient column slicing
                csc = matrix.tocsc(copy=True)

                idx = [csc[:, i] for i in range(csc.shape[1])]
                idx[col1], idx[col2] = idx[col2], idx[col1]
                matrix = sp.hstack(idx).tocsr()

            else:
                raise TypeError("A Matrix in the GrandPy Object has an unexpected type. Only numpy ndarray and scipy sparse matrices are supported")

            return matrix

        if isinstance(column1, str):
            column1 = self.coldata.index.get_loc(column1)
        if isinstance(column2, str):
            column2 = self.coldata.index.get_loc(column2)

        return self._apply(swap, col1=column1, col2=column2)

    def get_references(
            self,
            reference: Union[str, Callable[[pd.Series], bool]] = None,
            reference_function: Callable[[pd.Series], Sequence[str]] = None,
            *,
            group: Union[str, Sequence[str]] = None,
            columns: Union[str, Sequence[str]] = None,
            as_dict: bool = False
    ) -> Union[pd.DataFrame, dict[str, Sequence[str]]]:
        """
        Find reference samples within groups using a condition or custom function.

        Examples
        --------
        Obtain the corresponding no4sU sample for each sample for the GrandPy instance 'sars'.

        >>> ref = sars.get_references(reference="no4sU")
        >>> ref.iloc[0:7, 0:7]
        Name          Mock.no4sU.A  Mock.1h.A  Mock.2h.A  Mock.2h.B  Mock.3h.A  Mock.4h.A  SARS.no4sU.A
        Name
        Mock.no4sU.A          True       True       True       True       True       True          True
        Mock.1h.A            False      False      False      False      False      False         False
        Mock.2h.A            False      False      False      False      False      False         False
        Mock.2h.B            False      False      False      False      False      False         False
        Mock.3h.A            False      False      False      False      False      False         False
        Mock.4h.A            False      False      False      False      False      False         False
        SARS.no4sU.A          True       True       True       True       True       True          True

        Obtain the corresponding sample in the 'Mock' condition for each sample.

        >>> ref == sars.get_references(reference="Condition == 'Mock'", group="duration.4sU.original"))
        >>> ref.iloc[0:7, 0:7]
        Name          Mock.no4sU.A  Mock.1h.A  Mock.2h.A  Mock.2h.B  Mock.3h.A  Mock.4h.A  SARS.no4sU.A
        Name
        Mock.no4sU.A          True      False      False      False      False      False          True
        Mock.1h.A            False       True      False      False      False      False         False
        Mock.2h.A            False      False       True       True      False      False         False
        Mock.2h.B            False      False       True       True      False      False         False
        Mock.3h.A            False      False      False      False       True      False         False
        Mock.4h.A            False      False      False      False      False       True         False
        SARS.no4sU.A         False      False      False      False      False      False         False

        Now we do the same thing again, but pay attention to replicates.

        >>> ref == sars.get_references(reference="Condition == 'Mock'",
        ...                            group=["duration.4sU.original", "Replicate"])
        >>> ref.iloc[0:7, 0:7]
        Name          Mock.no4sU.A  Mock.1h.A  Mock.2h.A  Mock.2h.B  Mock.3h.A  Mock.4h.A  SARS.no4sU.A
        Name
        Mock.no4sU.A          True      False      False      False      False      False          True
        Mock.1h.A            False       True      False      False      False      False         False
        Mock.2h.A            False      False       True      False      False      False         False
        Mock.2h.B            False      False      False       True      False      False         False
        Mock.3h.A            False      False      False      False       True      False         False
        Mock.4h.A            False      False      False      False      False       True         False
        SARS.no4sU.A         False      False      False      False      False      False         False

        Parameters
        ----------
        reference : str or Callable[[pd.Series], bool], optional
            A condition to define reference samples. Can be:
            - a string evaluated against the columns of coldata (via `pandas.query`)
            - a function taking a row and returning True for reference samples, False otherwise.

        reference_function : Callable[[pd.Series], Sequence[str]], optional
            A function that returns a list of references for each sample (row-wise).
            If specified, `reference` is ignored.

        group : str or Sequence[str], optional
            One or more columns in `coldata` used to group samples before searching for references.

        columns : str or Sequence[str], optional
            Limit the evaluation to specific columns from `coldata`.

        as_dict : bool, default False
            If True, return a dictionary mapping each sample to its references.
            If False, return a DataFrame indicating references. (samples x samples)

        Returns
        -------
        Union[pd.DataFrame, dict[str, Sequence[str]]]
            Either a reference matrix or a dictionary mapping samples to their references.

        Raises
        ------
        ValueError
            When neither `reference` nor `reference_function` is provided; When `reference` is a string and the query on the DataFrame fails.

        TypeError
            When `reference` is neither a string nor a callable.
        """

        def map_refs_by_group(df, selected_refs):
            """
            Helper to assign found references to their groups.
            """
            group_series = df["__group__"]
            group_to_refs = {}
            for grp in group_series.unique():
                refs = [r for r in selected_refs if group_series[r] == grp]
                group_to_refs[grp] = refs
            return group_to_refs

        def apply_reference_function(df_subset, group_series):
            """
            Applies a reference_function to each row within group.
            """
            ref_matrix = pd.DataFrame(False, index=df_subset.index, columns=df_subset.index)

            for grp, gdf in df_subset.groupby(group_series):
                for idx, row in gdf.iterrows():
                    try:
                        references = _ensure_list(reference_function(row))
                        ref_matrix.loc[references, idx] = True
                    except Exception as e:
                        raise ValueError(f"Error in reference_function for sample '{idx}': {e}")
            return ref_matrix

        def format_ref_output(group_series, group_to_refs, as_list):
            """
            Converts grouped references to matrix or dictionary.
            """
            samples = group_series.index
            if as_list:
                return {sample: group_to_refs.get(group_series[sample], []) for sample in samples}

            matrix = pd.DataFrame(False, index=samples, columns=samples)
            for sample in samples:
                refs = group_to_refs.get(group_series[sample], [])
                matrix.loc[refs, sample] = True
            return matrix

        df = self.coldata.copy()
        columns = _ensure_list(columns) if columns is not None else list(df.columns)
        group_cols = _ensure_list(group) if group is not None else []

        # Add group label column
        df["__group__"] = (
            df[group_cols].astype(str).agg("_".join, axis=1) if group_cols else "GROUP"
        )
        print(df)
        df_subset = df[columns].copy()

        if reference_function:
            return apply_reference_function(df_subset, df["__group__"])

        if reference is None:
            raise ValueError("Either `reference` or `reference_function` must be provided.")

        if isinstance(reference, str):
            try:
                mask_df = df_subset.eval(reference, engine="python")
                if isinstance(mask_df, pd.Series) and mask_df.dtype == bool:
                    selected_refs = df_subset.index[mask_df].tolist()
                else:
                    raise ValueError("Query did not return a boolean mask.")
            except Exception as e:
                raise ValueError(f"Invalid reference query string: {e}")
            group_to_refs = map_refs_by_group(df, selected_refs)
        elif callable(reference):
            group_to_refs = {
                grp: gdf.index[gdf.apply(reference, axis=1)].tolist()
                for grp, gdf in df.groupby("__group__")
            }
        else:
            raise TypeError("`reference` must be a string or a callable.")

        return format_ref_output(df["__group__"], group_to_refs, as_dict)


    def _apply(
            self,
            function: Callable = lambda x: x,
            *,
            function_gene_info: Callable = None,
            function_coldata: Callable = None,
            **kwargs
    ) -> "GrandPy":
        """
        Returns a new GrandPy object with the given function applied to each data slot.
        Can also apply a function to the gene_info and coldata DataFrames.

        When trying to change the order of genes or sample/cells, do so via `function_gene_info` or `function_coldata`.`

        Trying to do the same with `function` will swap the values but keep the order.

        Parameters
        ----------
        function: Callable, default lambda x: x
            Function to apply to each data slot.
        function_gene_info: Callable, default None
            Function to apply to the gene_info DataFrame.
        function_coldata: Callable, default None
            Function to apply to the coldata DataFrame.
        **kwargs:
            Additional keyword arguments to pass to the function.

        Returns
        -------
        GrandPy
            New GrandPy object with transformed data.

        """
        old_geneinfo_index = self.gene_info.index
        old_coldata_index = self.coldata.index

        new_adata = self._anndata.copy()

        # Apply function to gene_info
        if function_gene_info is not None:
            new_geneinfo = function_gene_info(new_adata.obs, **kwargs)
            new_geneinfo_index = new_geneinfo.index
        else:
            new_geneinfo = new_adata.obs
            new_geneinfo_index = old_geneinfo_index

        # Apply function to coldata
        if function_coldata is not None:
            new_coldata = function_coldata(new_adata.var, **kwargs)
            new_coldata_index = new_coldata.index
        else:
            new_coldata = new_adata.var
            new_coldata_index = old_coldata_index


        row_indices = old_geneinfo_index.get_indexer_for(new_geneinfo_index)
        column_indices = old_coldata_index.get_indexer_for(new_coldata_index)

        new_layers = {}

        for key in self._anndata.layers.keys():
            matrix = function(self._anndata.layers[key], **kwargs)

            matrix = matrix[np.ix_(row_indices, column_indices)]

            new_layers[key] = matrix

        new_analyses = {}
        # Also fix analysis reindexing if needed
        if new_adata.uns['analyses'] is not None:
            new_adata.uns['analyses'] = {
                key: _reindex_by_index_name(value, new_geneinfo)
                for key, value in new_adata.uns['analyses'].items()
            }

        return self._dev_replace(gene_info=new_geneinfo, coldata=new_coldata, slots=new_layers, analyses=new_analyses)


    def get_matrix(
            self,
            mode_slot: Union[str, ModeSlot] = None,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            columns: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            *,
            ntr_nan: bool = False,
            force_numpy: bool = True
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Get the data from a data slot as a numpy array, without row or column names.

        This function is mostly not needed, as get_table() or get_data() are usually better suited and more versatile.

        See Also
        --------
        GrandPy.get_table
            Get the data from slots, coldata can be concatenated. (genes x samples)

        GrandPy.get_data
            Get the data from slots, gene_info can be concatenated. (samples x genes)

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis results.

        Parameters
        ----------
        mode_slot: str or ModeSlot
            The name of the data slot. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int or bool]
            The genes to be retrieved. Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        columns: str or int or Sequence[str or int or bool]
            The samples/cells to be retrieved. Either by names, indices, or a boolean mask.

        ntr_nan: bool, default False
            If True, ntr values for no4sU will be set to NaN.
            Otherwise, they remain 0.

            This has an impact on all slots when they are in a different mode than 'total',
            as 'new' and 'old' are calculated by multiplying with ntr.

        force_numpy: bool, default True
            If True, return will always be a numpy ndarray, regardless of the type of the slots.
            Otherwise, return the data in their actual type.

        Returns
        -------
        Union[np.ndarray, sp.csr_matrix]
            A data matrix, without column or row names. (genes x samples)
        """
        if mode_slot is None:
            mode_slot = self.default_slot

        data = self.__resolve_mode_slot(mode_slot=mode_slot, ntr_nan=ntr_nan)

        row_indices = self.get_index(genes)
        column_indices = [self.coldata.index.get_loc(column) for column in self.get_columns(columns)]

        data_subset = _subset_dense_or_sparse(data, row_indices, column_indices, force_numpy=force_numpy)

        return data_subset

    def get_data(
            self,
            mode_slot: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            columns: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            *,
            with_coldata: bool = True,
            name_genes_by: str = "Symbol",
            by_rows: bool = False,
            ntr_nan: bool = False
    ) -> pd.DataFrame:
        """
        Get a DataFrame containing the data from data slots, optionally with the corresponding coldata.

        See Also
        --------
        GrandPy.get_table
            Get the data from slots, coldata can be concatenated. (genes x samples)

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis results.

        GrandPy.get_matrix
            Get the data from slots, but in their 'raw' form. (genes x samples)

        Parameters
        ----------
        mode_slot: str or ModeSlot or Sequence[str or ModeSlot], optional
            The name of the data slots to be retrieved. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int or bool]
            The genes to be retrieved. Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        columns: str or int or Sequence[str or int or bool], optional
            The samples/cells to be retrieved. Either by names, indices, or a boolean mask.

        with_coldata: bool, default True
            If True, the coldata DataFrame will be concatenated to the result.

        name_genes_by: str, default "Symbol"
            A column in the gene_info DataFrame to be used as the name of the genes.
            Usually either 'Symbol'(Symbols) or 'Gene'(Ensembl IDs).

        by_rows: bool, default False
            If True, add rows if there are multiple `genes` / `mode_slots`.
            Otherwise, add columns.

        ntr_nan: bool, default False
            If True, ntr values for no4sU will be set to NaN.
            Otherwise, they remain 0.

            This has an impact on all slots when they are in a different mode than 'total',
            as 'new' and 'old' are calculated by multiplying with ntr.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified data for the genes and columns. (samples x genes)
        """
        coldata = self.coldata
        gene_info = self.gene_info

        if mode_slot is None:
            mode_slot = self.default_slot

        mode_slot = _ensure_list(mode_slot)

        row_indices = [coldata.index.get_loc(column) for column in self.get_columns(columns)]
        column_indices = self.get_index(genes)

        row_names = coldata.iloc[row_indices].index.tolist()
        column_names = gene_info.iloc[column_indices][name_genes_by].tolist()

        result_df = pd.DataFrame()

        if not by_rows:
            for slot_name in mode_slot:
                all_data = self.__resolve_mode_slot(slot_name, ntr_nan=ntr_nan).T
                data_subset = _subset_dense_or_sparse(all_data, row_indices, column_indices)

                if len(mode_slot) > 1:
                    local_column_names = [f"{name}_{slot_name}" for name in column_names]
                else:
                    local_column_names = column_names

                processed_data = pd.DataFrame(data_subset, index=row_names, columns=local_column_names)
                result_df = pd.concat([result_df, processed_data], axis=1)

            if with_coldata:
                result_df = pd.concat([coldata.iloc[row_indices], result_df], axis=1)

            result_df.index.name = name_genes_by
            return result_df

        else:
            for slot_name in mode_slot:
                all_data = self.__resolve_mode_slot(slot_name, ntr_nan=ntr_nan).T
                data_subset = _subset_dense_or_sparse(all_data, row_indices, column_indices)

                df = pd.DataFrame(data_subset, index=row_names, columns=column_names)
                df.index.name = "Name"
                df_melted = df.reset_index().melt(id_vars='Name', var_name='Gene', value_name='Value')
                df_melted["Slot"] = slot_name

                result_df = pd.concat([result_df, df_melted], axis=0)

            if with_coldata:
                result_df = coldata.reset_index(drop=True).merge(
                    result_df,
                    on="Name",
                    how="right"
                )

            return result_df

    def get_table(
            self,
            mode_slot: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            columns: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            *,
            with_gene_info: bool = False,
            name_genes_by: str = "Symbol",
            summarize: pd.DataFrame = None,
            prefix: str = None,
            ntr_nan: bool = True,
            reorder_columns: bool = False
    ) -> pd.DataFrame:
        """
        Get a DataFrame containing the data from data slots, optionally with the corresponding gene_info.

        See Also
        --------
        GrandPy.get_data:
            Get the data from slots, coldata can be concatenated. (samples x genes)

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis results.

        GrandPy.get_summarize_matrix:
            Get a summarization matrix for averaging or aggregation. Can be provided to get_table via `summarize`.

        GrandPy.get_matrix:
            Get the data from slots, but in their 'raw' form. (genes x samples)

        Parameters
        ----------
        mode_slot: str or ModeSlot or Sequence[str or ModeSlot], optional
            The name of the data slots to be retrieved. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int or bool]
            The genes to be retrieved. Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        columns: str or int or Sequence[str or int or bool], optional
            The samples/cells to be retrieved. Either by names, indices, or a boolean mask.

        with_gene_info: bool, default False
            If True, the gene_info DataFrame will be concatenated to the result.

        name_genes_by: str, default "Symbol"
            A column in the gene_info DataFrame to be used as the name of the genes.
            Usually either 'Symbol'(Symbols) or 'Gene'(Ensembl IDs).

        summarize: pd.DataFrame, default None
            A summary DataFrame. This can be retrieved via GrandPy.get_summarize_matrix().
            `columns` will be ignored if provided.

        prefix: str, default None
            Will be prepended to all column names.

        ntr_nan: bool, default False
            If True, ntr values for no4sU will be set to NaN.
            Otherwise, they remain 0.

            This has an impact on all slots when they are in a different mode than 'total',
            as 'new' and 'old' are calculated by multiplying with ntr.

        reorder_columns: bool, default False
            If True, the columns in the result will be in the same order as in the object.
            Otherwise, they will be in the same order as the input.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified data for the genes and columns. (genes x samples)
        """
        gene_info = self.gene_info
        coldata = self.coldata

        if mode_slot is None:
            mode_slot = self.default_slot

        mode_slot = _ensure_list(mode_slot)

        gene_indices = self.get_index(genes)
        gene_names = gene_info.iloc[gene_indices][name_genes_by].tolist()

        if summarize is not None:
            summarize = summarize.loc[:, (summarize != 0).sum(axis=0) > 0]
            column_indices = list(range(summarize.shape[1]))
            column_names = summarize.columns.tolist() if summarize.columns.notnull().all() else [f"group_{i}" for i in column_indices]
        else:
            column_ids = self.get_columns(columns, reorder=reorder_columns)
            column_indices = [coldata.index.get_loc(c) for c in column_ids]
            column_names = coldata.loc[column_ids, :].index.tolist()

        result_df = pd.DataFrame()

        for slot_name in mode_slot:
            all_data = self.__resolve_mode_slot(slot_name, ntr_nan=ntr_nan)

            if summarize is not None:
                matrix = all_data @ summarize.values
            else:
                matrix = all_data

            data_subset = _subset_dense_or_sparse(matrix, row_indices=gene_indices, column_indices=column_indices)

            # Column names (add suffix if multiple slots)
            if len(mode_slot) > 1:
                local_colnames = [f"{col}_{slot_name}" for col in column_names]
            else:
                local_colnames = column_names

            slot_df = pd.DataFrame(data_subset, index=gene_names, columns=local_colnames)
            result_df = pd.concat([result_df, slot_df], axis=1)

        if with_gene_info:
            gene_info_block = gene_info.iloc[gene_indices].copy()
            gene_info_block.index = result_df.index
            result_df = pd.concat([gene_info_block, result_df], axis=1)

        if prefix is not None:
            result_df.columns = [f"{prefix}{col}" for col in result_df.columns]

        result_df.index = gene_names
        result_df.index.name = name_genes_by
        return result_df

    def get_analysis_table(
            self,
            analyses: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            genes: Union[str, int, Sequence[Union[str, int]]] = None,
            columns: Union[str, Sequence[str]] = None,
            *,
            regex: bool = True,
            with_gene_info: bool = True,
            name_genes_by: str = "Symbol",
            prefix_by_analyses: bool = True,
            by_rows: bool = False,
    ) -> pd.DataFrame:
        """
        Get a DataFrame containing analysis results, optionally with the corresponding gene_info.

        Examples
        --------
        Perform any analysis on the GrandPy instance 'sars'.

        >>> sars = sars.fit_kinetics(return_fields="Synthesis")
        >>> sars.analyses
        ['kinetics_Mock', 'kinetics_SARS']

        Retrieve all analyses.

        >>> sars.get_analysis_table(with_gene_info=False)
                        kinetics_Mock_Synthesis  kinetics_SARS_Synthesis
        Symbol
        UHMK1                175.303203             3.123868e+02
        ATF3                  34.018585             4.843992e+02
        ...                         ...                      ...
        ORF1ab               792.905313             1.546125e+06
        S                    522.247609             9.717529e+05

        Only retrieve specific results for the first three genes.

        >>> sars.get_analysis_table(analyses="kinetics_Mock", columns="Synthesis",
        ...                         genes=[0,1,2], prefix_by_analyses=False, with_gene_info=False)
                 Synthesis
        Symbol
        UHMK1   175.303203
        ATF3     34.018585
        PABPC4  213.387547

        See Also
        --------
        GrandPy.analyses
            Get the names of all stored analyses.

        GrandPy.with_dropped_analysis
            Drop analyses from an object with a regex.

        GrandPy.get_analyses
            Get the names of analyses. Either by a regex, names, indices, or a boolean mask.

        GrandPy.with_analysis
            Add a new analysis to the object. Usually not called directly.

        Parameters
        ----------
        analyses: str or int or Sequence[str or int or bool], optional
            The analyses to be retrieved. Either by name, index, or a boolean mask.

        genes: str or int or Sequence[str or int], optional
            The genes for which to retrieve the analysis tables. Either by gene symbols, names(Ensembl ids), or indices.

        columns: str or Sequence[str], optional
            A regular expression or a list of regexes to match the names of the columns in the analysis tables.

        regex: bool, default True
            If True, `analyses` will be interpreted as a regular expression.

        with_gene_info: bool, default True
            If True, the gene_info DataFrame will be concatenated to the result.

        name_genes_by: str, default "Symbol"
            The name of the column in the gene_info DataFrame to be used as the index.
            Usually either 'Symbol'(Symbols) or 'Gene'(Ensembl IDs).

            This will be ignored if `by_rows` is True.

        prefix_by_analyses: bool, default True
            If True, the columns in the result will be prefixed by the name of their analysis.

            This will be set to False if `by_rows` is True

        by_rows: bool, default False
            If True, add rows if there are analyses for multiple conditions.
            Otherwise, add columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified analyses for the genes and columns.
        """
        if by_rows:
            prefix_by_analyses = False

        analyses = self.get_analyses(analyses, regex=regex)

        row_indices = self.get_index(genes)
        gene_info = self.gene_info

        if genes is not None:
            gene_info = gene_info.iloc[row_indices]

        result_df = pd.DataFrame()
        result_rows = []

        for name in analyses:
            analysis_data = self._anndata.uns["analyses"][name].copy()

            if prefix_by_analyses:
                analysis_data.columns = [name + "_" + col for col in analysis_data.columns]

            if genes is not None:
                analysis_data = analysis_data.loc[gene_info.index]

            if columns is not None:
                columns = _ensure_list(columns)
                if regex:
                    matching_columns = [col for col in analysis_data.columns if any(re.search(pat, col) for pat in columns)]
                else:
                    matching_columns = [col for col in columns if col in analysis_data.columns]
            else:
                matching_columns = analysis_data.columns

            selected_data = analysis_data[matching_columns].copy()

            if by_rows:
                selected_data = selected_data.copy()
                selected_data.insert(0, "Analysis", name)

                if with_gene_info:
                    merged = pd.concat([gene_info.reset_index(drop=True), selected_data.reset_index(drop=True)], axis=1)
                else:
                    merged = selected_data.reset_index(drop=True)

                result_rows.append(merged)
            else:
                result_df = pd.concat([result_df, selected_data], axis=1)

        if by_rows:
            result_df = pd.concat(result_rows, axis=0).reset_index(drop=True)
        else:
            if with_gene_info:
                result_df = pd.concat([gene_info, result_df], axis=1)
            result_df.columns = _make_unique(pd.Series(result_df.columns), warn=False)
            result_df.index = pd.Index(gene_info[name_genes_by])

        return result_df



    # ----- Methods on the whole object -----
    def merge(
            self,
            other: "GrandPy",
            *,
            axis: Literal["gene_info", 0, "coldata", 1] = 0,
            join: Literal["inner", "outer"] = "inner",
            merge: Union[Literal["same", "unique", "first", "only"], Callable] = "unique",
            analysis_prefixes: Sequence[str] = None
    ) -> "GrandPy":
        """
        Merge the 'other' instance with the current instance along a given axis. Uses 'first' for metadata and plots.
        Analyses are all kept with an added prefix to avoid collisions.

        Parameters
        ----------
        other: GrandPy
            The object to merge with the current instance.

        axis: {"gene_info" or 0 or "coldata" or 1}, default 0
            The axis along which to merge.

        join: {"inner" or "outer"}, default "inner"
            How to align values when merging. If "outer", the union of the other axis is taken. If "inner", the intersection.

        merge: {"same" or "unique" or "first" or "only"} or Callable, default "unique"
            How elements not aligned to the axis being concatenated along are selected.
            Currently implemented strategies include:

            * `None`: No elements are kept.
            * `"same"`: Elements that are the same in each of the objects.
            * `"unique"`: Elements for which there is only one possible value.
            * `"first"`: The first element seen at each from each position.
            * `"only"`: Elements that show up in only one of the objects.

        analysis_prefixes: tuple(str, str), optional
            The prefixes added to the analyses of each instance. Has to have length of 2.
            By default: 'dataset0', 'dataset1'.

            To disable this behavior, set to ("", "").
            Then analyses of the object coming first will be kept in case of a name collision.

        Returns
        -------
        GrandPy
            A merged GrandPy object.
        """
        from .utils import concat

        objects = [self, other]

        return concat(objects, axis=axis, join=join, merge=merge, analysis_prefixes=analysis_prefixes)

    def split(self, by: str = "Condition") -> list:
        """
        Split the GrandPy object into a list of GrandPy objects based on a column in coldata.

        Parameters
        ----------
        by : str, default "Condition"
            Column in coldata to split by.

        Returns
        -------
        list of GrandPy
            One GrandPy object per unique value in the specified column.
        """
        if by not in self.coldata.columns:
            raise ValueError(f"Column '{by}' not found in coldata.")

        result = []
        for group in self.coldata[by].unique():
            mask = self.coldata[by] == group
            subset = self._anndata[:, mask]
            obj = self._dev_replace(anndata=subset)
            obj.group = group

            result.append(obj)

        return result

    def to_anndata(self, x: Union[str, ModeSlot] = None, transpose: bool = False) -> ad.AnnData:
        """
        Extracts an Anndata instance from GrandPy.

        In the unstructured data (uns), `analyses`, `metadata` and the `prefix` are stored.

        Notes
        -----
        When you want the AnnData instance to be scanpy compatible, use `transpose` = True.
        For this, the internal AnnData is transposed and plots are removed.

        See Also
        --------
        anndata_to_grandpy
            Returns a GrandPy instance from a given AnnData.

        Parameters
        ----------
        x: str or ModeSlot, optional
            The name of the slot to be set as the main data matrix X; by default `default_slot`.
            Can also be a `ModeSlot`.

        transpose: bool, default False
            If False, the anndata will be returned as stored internally.
            Otherwise, it will be transposed to make it scanpy compatible.

        Returns
        -------
        ad.AnnData
            An AnnData instance containing the data.
        """
        if x is None:
            x = self.default_slot

        adata = self._anndata.copy()
        adata.X = self.get_matrix(x, force_numpy=False)

        if not transpose:
            return adata

        if self.analyses is not None:
            for name, analysis in adata.uns["analyses"].items():
                adata.uns["analyses"][name] = analysis.copy().T

        adata.uns.pop("plots", None)

        return adata.T




    # ----- processing -----
    def normalize(
            self,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            name: str = "norm",
            slot: str = "count",
            *,
            set_to_default: bool = True,
            size_factors: Union[np.ndarray, Sequence[float]]=None,
            return_size_factors: bool = False
    ) -> Union["GrandPy",np.ndarray]:
        """
        Normalize gene expression values using size factors.

        This method computes expression values by normalizing raw counts using size factors,
        which account for differences in sequencing depth and gene composition across samples.
        It is suitable for comparing gene expression levels across samples with varying number of reads.

        See Also
        --------
        GrandPy.normalize_fpkm
            Normalize data using FPKM.

        GrandPy.normalize_tpm
            Normalize data using TPM.

        GrandPy.normalize_rpm
            Normalize data using RPM.

        Parameters
        ----------
        genes : str or int or Sequence[str or int or bool], optional
            A subsets of genes for calculation the size factor. All genes will be normalized regardless.
            Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        name : str, default "norm"
            Name of the slots where the normalized data will be stored.

        slot : str, default "count"
            The name of the slot to normalize.

        set_to_default : bool, default True
            If True, set the normalized slot as the default.

        size_factors : np.ndarray or Sequence[float], optional
            Precomputed size factors to use for normalization.
            If None, size factors are computed automatically.

        return_size_factors : bool, default False
            If True, return the size factors used for normalization.

        Returns
        -------
        Union[GrandPy, np.ndarray]
            The size factors used for normalization if `return_size_factors` is True.
            Otherwise, a GrandPy object with the normalized data added as a slot.
        """
        from .processing import _normalize

        return _normalize(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default,
                          size_factors=size_factors, return_size_factors=return_size_factors)

    def normalize_fpkm(
            self,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            name: str = "norm",
            slot: str = "count",
            *,
            set_to_default: bool = True,
            total_len: Union[str, np.ndarray[int]] = "Length"
    ) -> "GrandPy":
        """
        Normalize gene expression data using the FPKM method (Fragments Per Kilobase Million).

        This method computes FPKM values by normalizing raw counts for both gene length and total number of reads in a sample.
        It is suitable for comparing expression levels across genes within the same sample.

        See Also
        --------
        GrandPy.normalize
            Normalize data using size factors.

        GrandPy.normalize_tpm
            Normalize data using TPM.

        GrandPy.normalize_rpm
            Normalize data using RPM.

        Parameters
        ----------
        genes : str or int or Sequence[str or int or bool], optional
            A subsets of genes for calculation the scaling factor. All genes will be normalized regardless.
            Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        name : str, default "norm"
            The name of the data slot where the normalized values will be stored.

        slot : str, default "count"
            The name of the slot containing raw counts to normalize.

        set_to_default : bool, default True
            If True, sets the resulting normalized slot as the default slot.

        total_len : str or np.ndarray or Sequence[int], default "Length"
            Either the name of a column in the gene_info containing the lengths or the lengths themselves.

        Returns
        -------
        GrandPy
            A GrandPy object with the normalized data added as a slot.
        """
        from .processing import _normalize_fpkm

        return _normalize_fpkm(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default,
                               total_len=total_len)

    def normalize_tpm(
            self,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            name: str = "tpm",
            slot: str = "count",
            *,
            set_to_default: bool = True,
            total_len=None
    ) -> "GrandPy":
        """
        Normalize gene expression data using the TPM method (Transcripts Per Million).

        This method computes TPM values by normalizing raw counts for both gene length and total number of reads across all samples.
        It is suitable for comparing gene expression levels across different samples, accounting for varying sequencing depths.

        See Also
        --------
        GrandPy.normalize
            Normalize data using size factors.

        GrandPy.normalize_fpkm
            Normalize data using FPKM.

        GrandPy.normalize_rpm
            Normalize data using RPM.

        Parameters
        ----------
        genes : str or int or Sequence[str or int or bool], optional
            A subsets of genes for calculation the scaling factor. All genes will be normalized regardless.
            Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        name: str, default "tpm"
            The name of the slot that the tpm-normalization will create.

        slot: str, default "count"
            The name of the slot containing raw counts to normalize.

        set_to_default: bool, default True
            Whether to set the new slot as the default.

        total_len: np.ndarray, optional
            array with the transcript length of the genes

        Returns
        -------
        GrandPy
            A GrandPy object with the normalized data added as a slot.
        """
        from .processing import _normalize_tpm

        return _normalize_tpm(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default, total_len=total_len)

    def normalize_rpm(
            self,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            name: str = "rpm",
            slot: str = "count",
            *,
            set_to_default: bool = True,
            factor: int = 1e6
    ):
        """
        Normalize gene expression data using the RPM method (Reads Per Million).

        This method computes RPM values by normalizing raw counts by the total number of reads in a sample, scaled to millions.
        It is suitable for comparing gene expression levels across samples with varying sequencing depths.

        See Also
        --------
        GrandPy.normalize
            Normalize data using size factors.

        GrandPy.normalize_fpkm
            Normalize data using FPKM.

        GrandPy.normalize_tpm
            Normalize data using TPM.

        Parameters
        ----------
        genes : str or int or Sequence[str or int or bool], optional
            A subsets of genes for calculation the scaling factor. All genes will be normalized regardless.
            Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        name: str, default "rpm"
            The name of the slot that the tpm-normalization will create.

        slot: str, default "count"
            The name of the slot containing raw counts to normalize.

        set_to_default: bool, default True
            Whether to set the new slot as the default.

        factor : int, default 1e6
            The rpm scaling factor. A million by default.

        Returns
        -------
        GrandPy
            A GrandPy object with the normalized data added as a slot.
        """
        from .processing import _normalize_rpm

        return _normalize_rpm(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default, factor=factor)


    # TODO: DOC STRINGS für diese 5 methoden
    def compute_ntr_ci(self, ci_size: float = 0.95, name_lower: str = "lower", name_upper: str = "upper")-> "GrandPy":
        from .processing import _compute_ntr_ci

        return _compute_ntr_ci(self, ci_size, name_lower, name_upper)

    def compute_steady_state_half_lives(self, time=None, name="HL", columns=None, max_hl=48.0, ci_size=0.95, compute_ci=False, as_analysis=False) -> "GrandPy":
        from .processing import _compute_steady_state_half_lives

        return _compute_steady_state_half_lives(self, time, name=name ,columns=columns, max_hl=max_hl, ci_size=ci_size, compute_ci=compute_ci, as_analysis=as_analysis)

    def compute_absolute(self, dilution: float= 4e4, volume: float = 10.0, slot: str = "tpm", name: str = "absolute") -> "GrandPy":
        """
        Estimate absolute molecule counts from TPM data using ERCC spike-ins.

        This function approximates absolute transcript counts per cell by scaling TPM values
        based on the total ERCC signal. The method is inspired by `monocle::relative2abs`.

        Parameters
        ----------
        self : GrandPy
            A GrandPy data object containing gene expression matrices and gene annotations.

        dilution : float, default 4e4
             The dilution factor of the ERCC spike-in mix (molecules/µL).

        volume : float, default 10.0
             The volume (in µL) of ERCC spike-in solution added to each sample.

        slot : str, default "tpm"
             The name of the data slot containing TPM values to be converted to absolute counts.

        name : str, default "absolute"
             The name of the new slot in which to store the computed absolute expression matrix.

        Returns
        -------
        GrandPy
             A copy of the input `data` object with the new absolute expression matrix added as a slot.

        Raises
         ------
        ValueError
             If no ERCC genes are found in the dataset.
        """
        from .processing import _compute_absolute

        return _compute_absolute(self, dilution=dilution, volume=volume, slot=slot, name=name)

    def compute_total_expression(self, column: str = "total_expression", genes: Union[str, Sequence[str]] = None, mode_slot: str = None) -> "GrandPy":
        from .processing import _compute_total_expression

        return _compute_total_expression(self, column=column, genes=genes, mode_slot=mode_slot)

    def compute_expression_percentage(self,name: str, genes: Union[str, Sequence[str]] = None, slot: str = None, genes_total: Union[str, Sequence[str]] = None, slot_total: str = None, float_to_percent: bool = True):
        from .processing import _compute_expression_percentage

        return _compute_expression_percentage(self, name=name, genes=genes, slot=slot, genes_total=genes_total, slot_total=slot_total, float_to_percent=float_to_percent)


    def filter_genes(
        self,
        mode_slot: Union[str, ModeSlot] = "count",
        *,
        min_expression: int = 100,
        min_columns: int = None,
        min_condition: int = None,
        keep: Union[str, int, Sequence[Union[int, str]]] = None,
        use: Union[str, int, Sequence[Union[int, str, bool]]] = None,
        return_genes: bool = False
    ) -> Union["GrandPy", list[int]]:
        """
        Filter genes based on expression/value thresholds.

        Examples
        --------
        Get the number of genes in an unfiltered dataset.

        >>> len(sars.genes)
        19659

        Keep genes with at least 100 counts for at least half the columns(samples) in 'count'. This is the default behavior.

        >>> filtered = sars.filter_genes(min_expression = 100, min_columns=6, mode_slot = "count")
        >>> len(filtered.genes)
        9162

        Keep genes with >1000 counts in both conditions.

        >>> filtered = sars.filter_genes(min_expression=1000, min_condition=2)
        >>> len(filtered.genes)
        5611

        Apply the same filter, but force keeping a gene.

        >>> filtered = sars.filter_genes(min_expression=1000, min_condition=2, keep=["ATF3"])
        >>> len(filtered.genes)
        5612

        Retrieve the viral genes.

        >>> filtered = sars.filter_genes(use = sars.get_classified_genes("Unknown"))
        >>> len(filtered.genes)
        11

        Parameters
        ----------
        mode_slot: str or ModeSlot, default "count"
            The name of the data slot used for filtering.

            A mode('new'|'old'|'total') can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'.

        min_expression : Number, default 100
            Minimum value threshold to consider a gene expressed.

        min_columns : int, optional
            Minimum number of samples the gene must meet `min_expression` in.
            Defaults to half the number of columns in the matrix.
            Will be ignored if `min_condition` is provided

        min_condition : int, optional
            If set, values will not be compared by columns, but by conditions.
            Meaning all columns belonging to the same condition will be summed up before filtering.
            `min_columns` will be ignored.

        keep : str or int or Sequence[str or int], optional
            Genes to force-keep, regardless of threshold filtering.

        use : str or int or Sequence[bool or int or str], optional
            If provided, only genes specified in `use` will be kept. (Basically just subsetting)
            Filtering will be ignored completely.

        return_genes : bool, default False
            If True, return the list of selected gene indices instead of a filtered GrandPy object.

        Returns
        -------
        list[str] or GrandPy
            A filtered GrandPy object or a list of selected genes, depending on the `return_genes` parameter.

        Raises
        ------
        ValueError
            If both `keep` and `use` are specified.
        """
        from .processing import _filter_genes

        return _filter_genes(self, mode_slot, min_expression=min_expression, min_columns=min_columns,
                             min_condition=min_condition, use=use, keep=keep, return_genes=return_genes)



    # ----- modeling -----
    def fit_kinetics(
            self,
            fit_type: Literal["nlls", "ntr", "chase"] = "nlls",
            *,
            slot: str = None,
            time: Union[str, np.ndarray, pd.Series, Sequence] = "duration.4sU",
            prefix: Union[str, None] = "kinetics",
            return_fields: Union[str, Sequence[str]] = None,
            ci_size: float = 0.95,
            genes: Union[str, Sequence[str]] = None,
            show_progress: bool = True,
            **kwargs
    ) -> "GrandPy":
        """
        Fit kinetic models to gene expression data.

        This method fits mass-action kinetic models of RNA dynamics using one of several approaches.
        Fits are performed separately per condition.

        The `"nlls"` and `"chase"` methods require normalized input.
        The `"ntr"` method is independent of normalization but assumes steady-state kinetics.

        Notes
        -----
        This function decides dynamically how many processes to use for `nlls` and `chase`.
        By default, up to the number of CPU cores. (see the `max_processes` and `exact_processes` parameters.)

        See Also
        --------
        GrandPy.get_analysis_table
            Retrieves stored analyses.

        GrandPy.normalize
            Normalizes the expression data.

        plot_gene_progressive_timecourse
            Plots the kinetic model of a gene.

        GrandPy.calibrate_effective_labeling_time_kinetic_fit
            Calibrates the effective labeling time using kinetic models.

        GrandPy.calibrate_effective_labeling_time_match_halflives
            Calibrates the effective labeling time using known halflives from genes.

        Parameters
        ----------
        fit_type: {'nlls' or 'ntr' or 'chase'}, default 'nlls'
            The type of model to fit:

            - `"nlls"`: Fit synthesis and degradation rates using non-linear least squares.
            - `"ntr"`: Estimate degradation from new-to-total ratios, assuming steady-state.
            - `"chase"`: Fit degradation from decay of labeled RNA only.

        slot: str, optional
            Name of the data slot used to extract old/new RNA expression. Defaults to the default slot.

        time: str or np.ndarray or pd.Series or Sequence, default "duration.4sU"
            Either a column name in `coldata` or something array-like containing timepoints.

        prefix: str, default "kinetics"
            Prefix added to the name of the fit result.

        return_fields: str or Sequence[str], default ["Synthesis", "Half-life"]
            Names of result fields to extract from each fit. The following options are available:

            - `"Synthesis"`: Estimated synthesis rate.
            - `"Degradation"`: Estimated degradation rate.
            - `"Half-life"`: Calculated half-life.
            - `"log_likelihood"`: Log-likelihood of the fit.
            - `"f0"`: Initial transcript abundance.
            - `"total"`: Total RNA abundance.
            - `"conf_lower"`: Lower bounds of confidence intervals for s, d, and half-life.
            - `"conf_upper"`: Upper bounds of confidence intervals for s, d, and half-life.
            - `"rmse"`: Root mean square error over all timepoints.

            Additional fields for `"nlls"` and `"chase"`:

            - `"rmse_old"`: RMSE for old RNA timepoints.
            - `"rmse_new"`: RMSE for new RNA timepoints.
            - `"residuals"`: Dictionary of raw and relative residuals.

        ci_size: float, default 0.95
            The size of confidence intervals.

        genes: Union[str or int or Sequence[str or int or bool], optional
            Gene(s) to fit. Uses all by default. Specified either by their index, their symbol, their ensamble id, or a boolean mask.

        show_progress: bool, default True
            If True, a progress bar will be displayed.

        **kwargs
            Additional parameters passed to the model-specific fitting function.

            For `"nlls"`:
                - max_iter: Maximum number of optimization iterations, by default 250.
                - steady_state: Whether to use the steady-state model. It can be set for each condition individually by using a dict. By default, True
                - max_processes: The maximum number of processes this function will use. If None or not provided, it can start as many as cores are available.
                - exact_processes: If True, exactly `max_processes` will be used.

            For `"ntr"`:
                - transformed_ntr_map: If True, use the transformed NTR MAP estimator instead of the MAP of the transformed posterior; by default, True.
                - exact_ci: Whether to use exact confidence intervals; by default False.
                - total_function: Function to reduce total expression across time points (e.g., mean, median); by default `numpy.median`.

            For `"chase"`:
                - max_iter: Maximum number of optimization iterations, by default 250.
                - max_processes: The maximum number of processes this function will use. If None or not provided, it can start as many as cores are available.
                - exact_processes: If True, exactly `max_processes` will be used.

        Returns
        -------
        GrandPy
            A GrandPy instance with analysis results added per condition.
        """
        from .modeling import _fit_kinetics

        kinetics = _fit_kinetics(data=self, fit_type=fit_type, slot=slot, return_fields=return_fields,
                                 prefix=prefix, time=time, ci_size=ci_size, genes=genes,
                                 show_progress=show_progress, **kwargs)

        new_gp = self
        for name, analysis in kinetics.items():
            new_gp = new_gp.with_analysis(name, analysis)

        return new_gp


    def calibrate_effective_labeling_time_kinetic_fit(
            self,
            slot: str = None,
            time: str = "duration.4sU",
            name: str = "calibrated_time",
            n_top_genes: int = 1000,
            max_iterations: int = 10000,
            compute_confidence: bool = False,
            ci_size: float = 0.95,
            show_progress: bool = True,
            **kwargs
    ) -> "GrandPy":
        """
        Uses the non linear least squares kinetic model to calibrate the effective labeling time.

        The NTRs of each sample might be systematically too small or large. This function identifies such systematic
        deviations and computes labeling durations without systematic deviations.

        Can optionally compute confidence intervals for the estimation.

        Parameters
        ----------
        slot : str, optional
            The name of the slot used for calibration (Usually normalized counts). Defaults to the default slot.

        time: str, default "duration.4sU"
            A column name in `coldata` containing timepoints.

        name : str, default "calibrated_time"
            The name to assign to the calibrated time column in coldata.

        ci_size : float, default 0.95
            The confidence interval size for the calibration process.

        compute_confidence : bool, default False
            Whether to compute the confidence intervals for the labeling time.
            This will add a considerable amount of time to the computation.

        n_top_genes : int, default 1000
            Uses the n top genes for calibration. (Selected by expression and half-life)
            More genes make the calibration more accurate, but each iteration is slower.

        max_iterations : int, default 10000
            The number of maximum iterations for the calibration process.

        show_progress : bool, default True
            If True, the progress will be displayed as the number of iterations. (This doesn't necessarily match `max_iterations`)

        **kwargs : dict
            Additional keyword arguments to pass to fit_kinetics.

        Notes
        -----
        For large enough datasets, fit_kinetics will run in parallel. For control over this, see `kwargs` and GrandPy.fit_kinetics.

        See Also
        --------
        GrandPy.fit_kinetics: Fits kinetic models to gene expression data.

        Returns
        -------
        GrandPy
            A GrandPy instance containing the effective labeling time information in coldata.
        """
        from .modeling import _calibrate_effective_labeling_time_kinetic_fit

        new_columns = _calibrate_effective_labeling_time_kinetic_fit(data=self, time=time, name=name, slot=slot,
                                                                     n_top_genes=n_top_genes, max_iterations=max_iterations,
                                                                     compute_confidence=compute_confidence,
                                                                     ci_size=ci_size, show_progress=show_progress, **kwargs)

        return self.with_coldata(value = new_columns)



    # ----- diffexp -----
    def get_summarize_matrix(
            self,
            *,
            no4su: bool = False,
            columns: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            average: bool = True
    ) -> pd.DataFrame:
        """
        Get a summarization matrix for averaging or aggregation.

        If this matrix is multiplied with a count table,
        either the average (average=TRUE) or the sum (average=FALSE) of all columns
        belonging to the same Condition is computed.

        See Also
        --------
        GrandPy.get_table
            Retrieve the data for slots. get_summarize_matrix is mainly used as input for get_table.

        Parameters
        ----------
        no4su : bool, default False
            If True, no4sU columns will be included in the summary matrix.
            Otherwise, they will be ignored and returned as zeros.

        columns: str or int or Sequence[str or int or bool], optional
            Samples/cells to be included. Either by their index, their name, or a boolean mask.

        average : bool, default True
            If True, normalizes columns to sum to 1 (i.e., compute group-wise average).

        Returns
        -------
        pd.DataFrame
            A (samples × conditions) matrix indicating group membership (optionally normalized).
        """
        from .diffexp import _get_summarize_matrix

        return _get_summarize_matrix(data=self, no4su=no4su, columns=columns, average=average)


    def get_contrasts(
            self,
            contrast: Union[str, Sequence[str]] = "Condition",
            columns: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            group: Union[Sequence[str], str] = None,
            name_format: str = None,
            no4su: bool = True
    ) -> pd.DataFrame:
        """
        Generate contrast matrix for differential comparisons.

        Parameters
        ----------
        self : pd.DataFrame
            DataFrame containing sample metadata (e.g., conditions, groups).

        contrast : str or Sequence[str]
            Defines the contrast logic:
            - [condition_column] → all pairwise contrasts
            - [condition_column, reference_level] → all vs. reference
            - [condition_column, level_A, level_B] → specific comparison A vs B

        columns: str or int or Sequence[str or int or bool], optional
            Samples/cells considered in contrast computation. Either by their index, their name, or a boolean mask.

        group : str, optional
            Column name in `coldata` to compute contrasts within each group level separately.

        name_format : str, optional
            Format string for naming contrast columns.
            Supports placeholders: "$A", "$B", "$COL", "$GRP".
            Default: "$A vs $B" (or "$A vs $B.$GRP" if `group` is set)

        no4su : bool, default False
            If True, no4sU samples are included in contrast computation.

        Returns
        -------
        a GrandPy object
            A contrast matrix where rows are samples and columns represent contrasts.
            Values are -1, 0, or 1 depending on contrast design.
        """
        from .diffexp import _get_contrasts

        return _get_contrasts(self, contrast = contrast, columns = columns, group = group, name_format = name_format, no4su= no4su)


    def compute_lfc(
            self,
            contrasts: pd.DataFrame = None,
            mode_slot: Union[str, ModeSlot] = "count",
            prefix: str = None,
            lfc_function: Callable = psi_lfc,
            normalization: Union[str, Sequence[float]] = None,
            compute_m: bool = True,
            genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            verbose: bool = False,
            **kwargs
    ) -> "GrandPy":
        """
        Estimate log2 fold changes and optional M values for each contrast.

        See Also
        --------
        GrandPy.get_contrasts
            Get a contrast matrix.

        GrandPy.get_analysis_table
            Retrieves stored analyses.

        GrandPy.pairwise
            Combined log2 fold change and Wald test differential expression analysis.

        GrandPy.pairwise_deseq2
            Run PyDESeq2 for each contrast defined in the contrast matrix.

        psi_lfc
            Computes the optimal effect size estimate and credible intervals if needed.

        norm_lfc
            Computes the standard, normalized log2 fold change with given pseudocounts.

        Parameters
        ----------
        contrasts : pd.DataFrame, optional
            Contrast matrix defining comparisons (samples x contrasts; values 1, -1). Obtained via `get_contrasts`.

        mode_slot: str or ModeSlot, default "count"
            The name of the data slot to take data from. Usually 'count', optionally with a mode.
            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'.

        prefix : str, optional
            Prefix added to the name of the analyses.

        lfc_function : Callable, default psi_lfc
            Function to compute the log2 fold changes. Implemented: psi_lfc, norm_lfc.

        normalization : str or Sequence[str], optional
            - If str: name of normalization slot (e.g. "total")
            - If sequence: size factors per sample.

        compute_m : bool, default True
            If True, include the "M" column (base mean) for each contrast.

        genes : str or int or Sequence[str or int or bool], optional
            Restrict computation to this subset of genes. Either by their index, their symbol, their ensemble ID, or a boolean mask.

        verbose : bool, default False
            If True, status updates will be printed.

        **kwargs
            Passed to `lfc_function`.

        Returns
        -------
        GrandPy
            A GrandPy instance with one analysis per contrast. Each analysis
            has two columns named "LFC" and "M".
        """
        from .diffexp import _compute_lfc

        new_gp = _compute_lfc(data=self, prefix=prefix, contrasts=contrasts, mode_slot=mode_slot, lfc_function=lfc_function,
                              normalization=normalization, compute_m=compute_m, genes=genes, verbose=verbose, **kwargs)

        return new_gp

    def pairwise_deseq2(
        self,
        contrasts: pd.DataFrame,
        prefix: str = None,
        separate: bool = False,
        mode_slot: Union[str, ModeSlot] = "count",
        normalization: Union[str, Sequence[float]] = None,
        genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
        verbose: bool = False
    ) -> "GrandPy":
        """
        Run PyDESeq2 for each contrast defined in the contrast matrix.

        Notes
        -----
        Uses fit_type="mean" for compatibility with pydeseq2.
        pydeseq2 does not currently support fit_type="local".

        If the following Error is raised, try setting `seperate` to True.
        `"ValueError: Illegal intersection of contrasts for joint estimation of variance!"`

        See Also
        --------
        GrandPy.get_contrasts
            Get a contrast matrix.

        GrandPy.get_analysis_table
            Retrieves stored analyses.

        GrandPy.pairwise
            Combined log2 fold change and Wald test differential expression analysis.

        GrandPy.compute_lfc
            Estimate log2 fold changes and optional M values for each contrast.

        Parameters
        ----------
        contrasts : pd.DataFrame
            Matrix defining pairwise comparisons (samples x contrasts; 1/-1 values). Obtained via `get_contrasts`.

        prefix : str, optional
            Prefix added to the name of the analyses.

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
            A GrandPy instance containing analysis results.
        """
        from .diffexp import _pairwise_deseq2

        new_gp = _pairwise_deseq2(data=self, contrasts=contrasts, prefix=prefix, separate=separate,
                                  mode_slot=mode_slot, normalization=normalization, genes=genes, verbose=verbose)

        return new_gp

    def pairwise(
        self,
        contrasts: pd.DataFrame,
        prefix: str = None,
        lfc_function: Callable = psi_lfc,
        mode_slot: Union[str, ModeSlot] = "count",
        normalization: Union[str, Sequence[float]] = None,
        separate: bool = False,
        genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
        verbose: bool = False,
        **kwargs
    ) -> "GrandPy":
        """
        Combined log2 fold change and Wald test differential expression analysis.
        This function performs both the LFC computation (via `compute_lfc`) and DESeq2 testing
        (via `pairwise_deseq2`).

        Notes
        -----
        If the following Error is raised, try setting `seperate` to True.
        `"ValueError: Illegal intersection of contrasts for joint estimation of variance!"`

        See Also
        --------
        GrandPy.get_contrasts
            Get a contrast matrix.

        GrandPy.get_analysis_table
            Retrieves stored analyses.

        GrandPy.compute_lfc
            Estimate log2 fold changes and optional M values for each contrast.

        GrandPy.pairwise_deseq2
            Run DESeq2 for each contrast defined in the contrast matrix.

        Parameters
        ----------
        contrasts : pd.DataFrame
            Contrast matrix defining comparisons (samples x contrasts; values 1, -1). Obtained via `get_contrasts`.

        prefix : str, optional
            Prefix added to the name of the analyses.

        lfc_function : Callable, default psi_lfc
            Function to compute the log2 fold changes. Implemented: psi_lfc, norm_lfc.

        mode_slot: str or ModeSlot, default "count"
            The name of the data slot to take data from. Usually 'count', optionally with a mode.
            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'.

        normalization : str or sequence, optional
            Normalization strategy; name of slot or numeric vector.

        separate : bool, default False
            If True, run `pairwise_deseq2` separately for each contrast (two-group comparisons).

        genes : str or int or Sequence[str or int or bool], optional
            Restrict computation to this subset of genes. Either by their index, their symbol, their ensemble ID, or a boolean mask.

        verbose : bool, default False
            Whether to print progress messages.

        **kwargs
            Passed to `lfc_function`.

        Returns
        -------
        GrandPy
            A GrandPy instance with analysis results.
        """
        from .diffexp import _pairwise

        new_gp = _pairwise(data=self, contrasts=contrasts, prefix=prefix, lfc_function=lfc_function,
                           mode_slot=mode_slot, normalization=normalization, separate=separate,genes=genes,
                           verbose=verbose, **kwargs)

        return new_gp





