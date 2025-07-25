import copy
import re
import warnings
from collections.abc import Sequence, Mapping
from typing import Any, Union, Literal, Callable
from os import PathLike
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from .lfc import psi_lfc
from .slot_tool import SlotTool, ModeSlot
from .plot_tool import PlotTool, Plot
from .analysis_tool import AnalysisTool
from .utils import _ensure_list, _make_unique, _reindex_by_index_name, _subset_dense_or_sparse


class GrandPy:
    """
    Create a GrandPy object.

    Data is typically loaded using the `read_grand()` function, which parses preprocessed GrandR-compatible
    data formats into a GrandPy object.

    Notes
    -----
    The Object is designed to be immutable. Changes are made through `with_...` methods.
    Simple getters are implemented as properties, more complex ones through `get_...` methods.

    Examples
    --------
    Read a GrandPy Object from a file.

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

        # obs and var are swapped to allow the data to be gene * sample instead of sample * gene
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
                          "Consider adding one (see GrandPy.with_condition()) or "
                          "adjusting the DataFrame if it should already exist. (see GrandPy.replace())")
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
        return f"<GrandPy object with {self._anndata.n_obs} genes and {self._anndata.n_vars} samples>"

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



    # ----- basic "mutation" functions -----
    def replace(
            self,
            *,
            prefix: str = None,
            gene_info: pd.DataFrame = None,
            coldata: pd.DataFrame = None,
            slots: dict[str, Union[np.ndarray, sp.csr_matrix]] = None,
            metadata: dict[str, Any] = None,
            analyses: dict[str, Any] = None,
            plots: dict[str, Any] = None,
            anndata: ad.AnnData = None
    ) -> "GrandPy":
        """
        Replaces given parameters in a new GrandPy instance.

        This function is useful when you want to modify the GrandPy instance on your own.

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

        See Also
        --------
        GrandPy.to_anndata: Retrieves the internal anndata instance.

        Returns
        -------
        GrandPy
            A GrandPy instance with the given parameters replaced.
        """
        if anndata is None:
            anndata = self._anndata.copy()
        else:
            anndata = anndata.copy()

        return self.__class__(
            prefix = prefix if prefix is not None else anndata.uns.get('prefix'),
            gene_info = gene_info.copy() if gene_info is not None else anndata.obs,
            coldata = coldata.copy() if coldata is not None else anndata.var,
            slots = copy.deepcopy(slots) if slots is not None else anndata.layers,
            metadata = copy.deepcopy(metadata) if metadata is not None else anndata.uns.get("metadata"),
            analyses = copy.deepcopy(analyses) if analyses is not None else anndata.uns.get("analyses"),
            plots = copy.deepcopy(plots) if plots is not None else anndata.uns.get("plots")
        )

    def _dev_replace(
            self,
            *,
            prefix: str = None,
            gene_info: pd.DataFrame = None,
            coldata: pd.DataFrame = None,
            slots: dict[str, Union[np.ndarray, sp.csr_matrix]] = None,
            metadata: dict[str, Any] = None,
            analyses: dict[str, Any] = None,
            plots: dict[str, Any] = None,
            anndata: ad.AnnData = None
    ) -> "GrandPy":
        """
        This function is 'replace' for internal use.

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
            A new GrandPy object with the given parameters replaced.
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
        return self.replace()


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

        compression: Literal["gzip" or "lzf"], optional
            For ['lzf', 'gzip'], see the h5py :ref:`dataset_compression`.
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
        else:
            x = prefix.split('/')
            return x[-1]

    @property
    def shape(self) -> tuple[int]:
        """
        Get the dimension of the slots(data).
        """
        return self._anndata.X.shape

    @property
    def dim_names(self) -> tuple[list[str], list[str]]:
        """
        Get the column and row names of the data.
        """
        row_names = self.gene_info.index.tolist()
        column_names = self.coldata.index.tolist()
        return row_names, column_names

    @property
    def default_slot(self) -> str:
        """
        Get the name of the default slot
        """
        return self.metadata.get('default_slot')

    def with_default_slot(self, name: str) -> "GrandPy":
        """
        Sets the default slot set to `name`.

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
        """
        return self.__slot_tool.slots()

    @property
    def _slot_data(self) -> dict[str, Union[np.ndarray, sp.csr_matrix]]:
        """
        Get the raw data of all available slots as they are stored internally.

        Returns
        -------
        dict[str, Union[np.ndarray, sp.csr_matrix]]
            The data of all available slots.
        """
        return self.__slot_tool.slot_data()

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
        return self.__slot_tool.check_slot(slot, allow_ntr=allow_ntr)

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
        return self.__slot_tool.resolve_mode_slot(mode_slot, allow_ntr=allow_ntr, ntr_nan=ntr_nan)

    def with_dropped_slots(self, slots_to_remove: Union[str, Sequence[str]]) -> "GrandPy":
        """
        Returns a new GrandPy object with specified slot(s) removed.

        Parameters
        ----------
        slots_to_remove: str or Sequence[str]
            One or more slots to remove from the data.

        Returns
        ----------
        GrandPy
            A new GrandPy object with specified slot(s) removed.
        """
        slots_to_remove = _ensure_list(slots_to_remove)

        new_slots, new_metadata = self.__slot_tool.with_dropped_slots(slots_to_remove)

        return self._dev_replace(slots=new_slots, metadata=new_metadata)

    def with_slot(self, name: str, new_slot: Union[np.ndarray, pd.DataFrame, sp.csr_matrix, list], *, set_to_default = False) -> "GrandPy":
        """
        Returns a new GrandPy Object with the new slot added. Will overwrite if the slot already exists and give a warning.

        Recommended: use this function with DataFrames for security.

        It can only check the order of genes and samples/cells if the given matrix is a pandas DataFrame.
        Otherwise, the given matrix is expected to have rows and columns in the same order as existing slots.

        Parameters
        ----------
        name: str
            Name of the new slot.

        new_slot: np.ndarray or pd.DataFrame or sp.csr_matrix
            The data to be added as a new slot.

        set_to_default: bool, default False
            If True, sets the new slot as the default slot.

        Returns
        -------
        GrandPy
            A new GrandPy object with the new slot added.
        """
        new_slots, new_metadata = self.__slot_tool.with_slot(name, new_slot, set_to_default=set_to_default)

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
            A new GrandPy object with a new 'ntr' slot.

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

        Returns
        -------
        list[str]
            A list of analysis names.

        See Also
        --------
        GrandPy.get_analysis:
            Get the names of analyses matching a pattern.

        GrandPy.with_dropped_analyses:
            Remove analyses with a regex pattern.

        GrandPy.with_analysis:
            Add an analysis to the object. Usually not to be used directly.
        """
        return self.__analysis_tool.analyses()

    def get_analyses(self, pattern: Union[str, int, Sequence[Union[str, int, bool]]] = None, regex: bool = True, description: bool = False) -> list[str]:
        """
        Get the names of analyses. Either by a regex, names, indices, or a boolean mask.

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

        See Also
        --------
        GrandPy.analyses:
            Get a list of all available analyses.

        GrandPy.with_dropped_analyses:
            Remove analyses with a regex pattern.

        GrandPy.with_analysis:
            Add an analysis to the object. Usually not to be used directly.
        """
        return self.__analysis_tool.get_analyses(pattern, regex=regex, description=description)

    def with_analysis(self, name: str, table: pd.DataFrame, by: str = None) -> "GrandPy":
        """
        Returns a new GrandPy object with added analyses.

        Not to be used directly in most cases, instead it is called by analysis methods.

        If used directly, the Dataframe has to contain gene names (Ensemble ids) or symbols,
        that are either already the index or the column name is given to the 'by' parameter.

        Parameters
        ----------
        name: str
            The name of the analysis.

        table: pd.DataFrame
            A DataFrame containing the analysis data. Has to contain gene names or symbols.

        by: str, optional
            A column in the table to be used as an index.

        Returns
        -------
        A new GrandPy object with added analyses.

        See Also
        --------
        GrandPy.analyses:
            Get the names of all stored analyses.

        GrandPy.get_analysis:
            Get the names of analyses matching a pattern.

        GrandPy.with_dropped_analyses:
            Remove analyses with a regex pattern.
        """
        new_analyses = self.__analysis_tool.with_analysis(name, table, by=by)

        return self._dev_replace(analyses=new_analyses)

    def with_dropped_analyses(self, pattern: str = None) -> "GrandPy":
        """
        Returns a new GrandPy object with analyses matching the pattern removed.

        If no pattern is given, all analyses will be removed.

        Parameters
        ----------
        pattern: str, optional
            A regex pattern to match analyses.

        Returns
        -------
            A new GrandPy object with removed analyses.

        See Also
        --------
        GrandPy.analyses:
            Get the names of all stored analyses.

        GrandPy.get_analysis:
            Get the names of analyses matching a pattern.

        GrandPy.with_analysis:
            Add an analysis to the object. Usually not to be used directly.
        """
        new_analyses = self.__analysis_tool.drop_analyses(pattern)

        return self._dev_replace(analyses=new_analyses)



    # ----- All plot methods -----
    @property
    def __plot_tool(self):
        return PlotTool(self._anndata)

    @property
    def plots(self) -> dict[str, dict[str, Any]]:
        """
        Get a dictionary of available plot names.

        Returns
        -------
        dict[str, dict[str, Any]]
            A dictionary mapping plot types('gene', 'global') to plot names.

        See Also
        --------
        GrandPy.with_plot
            Add a plot function.

        GrandPy.with_dropped_plots
            Remove plots matching a regex pattern.
        """
        return self.__plot_tool.plots()

    def with_plot(self, name: str, function: Union[Plot, Callable]) -> "GrandPy":
        """
        Returns a new GrandPy object with a plot added. Either a global or gene plot.
        Global plots only take a GrandPy object. Gene plots additionally require a gene.

        Parameters
        ----------
        name: str
            A name for the plot.

        function: Plot or Callable
            A Plot Object, or a funktion, that takes a GrandPy object or a GrandPy object and a gene as input and returns a plot.

        Returns
        -------
        GrandPy
            A new GrandPy object with a plot added.

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
        {'gene': ['scatter']}

        Executing stored plot functions:

        >>> sars.plot_global("scatter")
        >>> sars.plot_gene("old_vs_new", "Mock.1h.A")

        See Also
        --------
        Plot
            A class used to store plot functions as.

        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_dropped_plots
            Remove plots matching a regex.
        """
        new_plots = self.__plot_tool.add_plot(name, function)

        return self._dev_replace(plots=new_plots)

    def plot_gene(self, name: str, gene: str):
        """
        Executes a stored plot function for a given gene.

        Parameters
        ----------
        name: str
            The name of the stored plot.

        gene: str
            The name of a gene.

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
        """
        try:
            return self._anndata.uns["plots"]["gene"][name](self, gene)
        except KeyError:
            raise KeyError(f"No plot named, '{name}' was found. These are all available gene plots: {self.plots.get('gene', None)}")

    def plot_global(self, name: str):
        """
        Executes a stored global plot function.

        Parameters
        ----------
        name: str
            The name of the stored plot.

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
        """
        try:
            return self._anndata.uns["plots"]["global"][name](self)
        except KeyError:
            raise KeyError(f"No plot named '{name}' was found. These are all available global plots: {self.plots.get('global', None)}")

    def with_dropped_plots(self, pattern: str = None) -> "GrandPy":
        """
        Returns a new GrandPy object with plot names matching the regex pattern removed.

        Parameters
        ----------
        pattern: str, optional
            A regular expression matching plot names to be dropped.

        Returns
        -------
        GrandPy
            A new GrandPy object with plot names matching the pattern removed.

        See Also
        --------
        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_plot
            Add a plot function.
        """
        new_plots = self.__plot_tool.drop_plot(pattern)

        return self._dev_replace(plots=new_plots)



    # ----- Methods relating to coldata, gene_info or metadata -----
    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get the metadata about the GrandPy object.
        """
        return self._anndata.uns.get('metadata').copy()


    @property
    def gene_info(self) -> pd.DataFrame:
        """
        Get the gene_info DataFrame.
        """
        return self._anndata.obs.copy()

    def with_gene_info(self, value: Union[Mapping, pd.Series, pd.DataFrame, np.ndarray, Sequence], name:str = None) -> "GrandPy":
        """
        Returns a new instance with modified gene_info. If 'name' does not already exist as a column in `gene_info`, it will be added.

        Otherwise, the column 'name' will be replaced by the given value or updated if a dictionary was given.

        Parameters
        ----------
        value : Mapping or pd.Series or pd.DataFrame or np.ndarray or Sequence
            The values to assign can be any iterable.
            Can also be a dictionary when trying to update a column.

            If 'name' is None, 'value' is expected to be a pandas Series or DataFrame.

        name : str, optional
            The name of the column to be modified.

        See Also
        --------
        GrandPy.replace: Replace whole parts of the instance, such as gene_info.

        Returns
        -------
        GrandPy
            A new GrandPy instance with updated gene_info.
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
        """
        return self.gene_info["Symbol"].tolist()

    def get_genes(self, genes: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, get_gene_symbols: bool = True, regex: bool = False) -> list[str]:
        """
        Get gene names or symbols.

        If no genes are specified, all genes are returned.

        Parameters
        ----------
        genes: str or int or Sequence[str or int or bool], optional
            Genes to be retrieved. Either by their index, their symbol, their ensemble ID, a boolean mask, or a regex.

        get_gene_symbols: bool, default True
            If True, gene symbols will be returned.
            Otherwise, gene names (Ensemble IDs) will be returned.

        regex: bool, default False
            If True, `genes` will be interpreted as a regular expression.

        Returns
        -------
        list[str]
            A list containing the specified genes.

        See Also
        --------
        GrandPy.get_columns
            Get the sample/cell names.

        GrandPy.gene_info
            Get the entire gene_info DataFrame.

        GrandPy.get_index
            Get the index of gene names/symbols.
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
        Get the index of: a gene, a list of genes, or in accordance to a boolean filter.

        Either by gene name or symbol, or by a boolean mask.

        Integers are returned unchanged.

        If names and indices are mixed, only one of them will be used. Chosen by the higher number of matches.

        Parameters
        ----------
        genes: str or int or Sequence[str or int or bool], optional
            Specifies which indices to return.
        regex: bool, default False
            If True, `gene` will be interpreted as a regular expression.

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

        if any(pd.isna(genes)):
            warnings.warn("NaN values removed from gene input.")
            genes = [g for g in genes if pd.notna(g)]

        # Regex matching (only works with str input)
        if regex:
            pattern = genes[0]
            if not isinstance(pattern, str):
                raise ValueError("If 'regex' is True, 'genes' must be a string.")
            mask = gene_col.str.contains(pattern, regex=True) | symbol_col.str.contains(pattern, regex=True)
            return list(np.flatnonzero(mask))

        # Boolean mask
        if all(isinstance(g, (bool, np.bool_)) for g in genes):
            if len(genes) != n:
                raise ValueError("Boolean mask length does not match gene count.")
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
        Symbols are derived from the column 'Gene', containg Ensemble IDs.

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

        new_gene_info = self.gene_info.copy()
        genes = new_gene_info["Gene"].tolist()

        mg = mygene.MyGeneInfo()

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

        result = _make_unique(result["symbol"].dropna())
        result.index.name = "Gene"
        result = _reindex_by_index_name(result, new_gene_info)

        if new_gene_info.get("Symbol", None) is not None:
            new_gene_info.update(result)

            return self._dev_replace(gene_info = new_gene_info)

        return self.with_gene_info(name="Symbol", value=result.values)

    def get_classified_genes(self, classification_label: str) -> list:
        """
        Returns a list of gene names corresponding to the given classification label.

        Parameters
        ----------
        classification_label : str
            The classification label to use.

        Examples
        --------
        Retrieve all genes with the `Type` 'Unknown' from the GrandPy instance 'sars'.

        >>> self.get_classified_genes("Unknown")
        ['ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF10', 'ORF1ab', 'S']

        Returns
        -------
        list:
            A list of gene names corresponding to the given classification label.
        """
        return self.gene_info[self.gene_info["Type"] == classification_label].get("Symbol").tolist()

    # TODO: get_significant_genes() doc string umschreiben
    def get_significant_genes(
            self,
            analysis = None,
            regex = True,
            criteria: str = None,
            as_table: bool = False,
            use_symbols: bool = True,
            gene_info: bool = True
    ) -> Union[list[str], pd.DataFrame]:
        """
        Return significantly regulated genes based on analysis results.

        Parameters
        ----------
        analysis : str or list[str], optional
            Names of the analysis results to evaluate.

        regex : bool, default True
            If True, treat `analysis` as a regular expression.

        criteria : str, optional
            String expression to evaluate significance (e.g. "Q < 0.05 and abs(LFC) >= 1").

        as_table : bool, default False
            If True, return full table instead of a list.

        use_symbols : bool, default True
            Whether to use gene symbols as rownames.

        gene_info : bool, default True
            Whether to include gene info columns in output.
        """
        analyses = self.get_analyses(analysis, regex=regex)
        result = self.gene_info
        result.index = result["Symbol"] if use_symbols else result["Gene"]

        for name in analyses:
            tab = self.get_analysis_table(
                analyses=name,
                regex=False,
                with_gene_info=False
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

        if not as_table:
            result = result.iloc[:, self.gene_info.shape[1]:]

            classes = set(result.dtypes)
            if len(classes) != 1:
                raise ValueError("Output contains mixed data types (logical and numeric).")

            dtype = list(classes)[0]
            if pd.api.types.is_bool_dtype(dtype):
                result = result.any(axis=1)
                return result[result].index.tolist()

            elif pd.api.types.is_numeric_dtype(dtype):
                if result.shape[1] > 1:
                    raise ValueError("Multiple numeric values present, can only return as a table.")
                return result.sort_values(by=result.columns[0], ascending=False).index.tolist()

        if not gene_info:
            result = result.iloc[:, self.gene_info.shape[1]:]

        result = result.sort_values(by=result.columns[-1], ascending=False)
        return result


    @property
    def coldata(self) -> pd.DataFrame:
        """
        Get the coldata DataFrame.
        """
        return self._anndata.var.copy()

    def with_coldata(self, value: Union[Mapping, pd.Series, pd.DataFrame, np.ndarray, Sequence], name: str = None, ) -> "GrandPy":
        """
        Returns a new instance with modified coldata. If 'name' does not already exist as a column in `coldata`, it will be added.

        Otherwise, the column 'name' will be replaced by the given value or updated if a dictionary was given.

        If 'name' is None or not given, 'value' is expected to be a pandas Series or DataFrame.

        Parameters
        ----------
        value : Mapping or pd.Series or pd.DataFrame or np.ndarray or Sequence
            The values to assign can be any iterable or array-like.
            Can also be a dictionary when trying to update a column.

            If 'name' is None, 'value' is expected to be a pandas Series or DataFrame.

        name : str, optional
            The name of the column to be modified.

        See Also
        --------
        GrandPy.replace: Replace whole parts of the instance, such as coldata.

        Returns
        -------
        GrandPy
            A new GrandPy instance with updated coldata.
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
        """
        return self.coldata['Condition'].tolist()

    def with_condition(self, value: Union[str, Sequence[str], pd.Series, Mapping]) -> "GrandPy":
        """
        Set new values for all samples/cells in the coldata.

        Parameters
        ----------
        value: str or Sequence[str] or pd.Series or Mapping
            The conditions to be set for the samples/cells.
            Can also construct the name from other columns in coldata, if their names are given.

        Returns
        -------
        GrandPy
            A new GrandPy object with the specified condition.
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

        if all(isinstance(column, int) for column in columns):
            result = coldata.iloc[columns].index

        else:
            try:
                result = coldata.loc[columns, :].index
            except KeyError as e:
                raise e

        if reorder:
            result = result.reindex(coldata.index).dropna(how="all", axis=0)

        return list(result)

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


    def _apply(self, function: Callable = lambda x: x, *, function_gene_info: Callable = None, function_coldata: Callable = None,
               **kwargs) -> "GrandPy":
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
            genes: Union[str, int, Sequence[Union[str, int]]] = None,
            columns: Union[str, int, Sequence[Union[str, int]]] = None,
            force_numpy: bool = True
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Get the data from a data slot as a numpy array, without row or column names.

        This function is mostly not needed, as get_table() or get_data() are usually better suited and more versitile.

        Parameters
        ----------
        mode_slot: str or ModeSlot
            The name of the data slot. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int]
            The genes to be retrieved. Either by gene symbols, names(Ensembl IDs), indices, or a boolean mask.

        columns: str or int or Sequence[str or int]
            The samples/cells to be retrieved. Either by names, indices, or a boolean mask.

        force_numpy: bool, default True
            If True, return will always be a numpy ndarray, regardless of the type of the slots.
            Otherwise, return the data in their actual type.

        Returns
        -------
        Union[np.ndarray, sp.csr_matrix]
            A data matrix, without column or row names.

        See Also
        --------
        GrandPy.get_table
            Similar to get_matrix, but with row and column names and coldata can be concatenated.

        GrandPy.get_data
            Similar to get_table, but slots are transposed, so gene_info can be concatenated.

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis tables.
        """
        if mode_slot is None:
            mode_slot = self.default_slot

        data = self.__resolve_mode_slot(mode_slot)

        row_indices = self.get_index(genes)
        column_indices = [self.coldata.index.get_loc(column) for column in self.get_columns(columns)]

        data_subset = _subset_dense_or_sparse(data, row_indices, column_indices, force_numpy=force_numpy)

        return data_subset

    def get_data(
            self,
            mode_slot: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
            genes: Union[str, int, Sequence[Union[str, int]]] = None,
            columns: Union[str, int, Sequence[Union[str, int]]] = None,
            *,
            with_coldata: bool = True,
            name_genes_by: str = "Symbol",
            by_rows: bool = False,
            ntr_nan: bool = False
    ) -> pd.DataFrame:
        """
        Get a DataFrame containing the data from data slots, optionally with the corresponding coldata.

        Parameters
        ----------
        mode_slot: str or ModeSlot or Sequence[str or ModeSlot], optional
            The name of the data slots to be retrieved. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int], optional
            The genes to be retrieved. Either by gene symbols, names(Ensembl IDs), or indices.

        columns: str or int or Sequence[str or int], optional
            The samples/cells to be retrieved. Either by names or indices.

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

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified data for the genes and columns.

        See Also
        --------
        GrandPy.get_table
            Similar to get_data, but slots are transposed, so gene_info can be concatenated.

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis tables.

        GrandPy.get_matrix
            Similar to get_data, but transposed and gives numpy arrays without row or column names.
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

            return result_df

        else:
            for slot_name in mode_slot:
                all_data = self.__resolve_mode_slot(slot_name, ntr_nan=ntr_nan).T
                data_subset = _subset_dense_or_sparse(all_data, row_indices, column_indices)

                df = pd.DataFrame(data_subset, index=row_names, columns=column_names)
                df = df.reset_index().melt(id_vars="index", var_name=name_genes_by, value_name="Value")
                df["Slot"] = slot_name
                df.rename(columns={"index": "Name"}, inplace=True)

                result_df = pd.concat([result_df, df], axis=1)

            if with_coldata:
                result_df = coldata.reset_index().merge(
                    result_df,
                    on="Name",
                    how="left"
                )

            result_df = result_df.set_index(["Name"])

            return result_df

    def get_table(
            self,
            mode_slot: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
            genes: Union[str, int, Sequence[Union[str, int]]] = None,
            columns: Union[str, int, Sequence[Union[str, int]]] = None,
            *,
            with_gene_info: bool = False,
            name_genes_by: str = "Symbol",
            summarize: pd.DataFrame = None,
            prefix: str = None,
            ntr_nan: bool = False,
            reorder_columns: bool = False
    ) -> pd.DataFrame:
        """
        Get a DataFrame containing the data from data slots, optionally with the corresponding gene_info.

        See Also
        --------
        GrandPy.get_data:
            Similar to get_table, but slots are transposed, so coldata can be concatenated.

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis tables.

        GrandPy.get_summary_matrix:
            Get a summarization matrix for averaging or aggregation. Can be provided to get_table via `summarize`.

        GrandPy.get_matrix:
            Similar to get_table, but gives numpy array without row or column names.

        Parameters
        ----------
        mode_slot: str or ModeSlot or Sequence[str or ModeSlot], optional
            The name of the data slots to be retrieved. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int], optional
            The genes to be retrieved. Either by gene symbols, names(Ensembl IDs), or indices.

        columns: str or int or Sequence[str or int], optional
            The samples/cells to be retrieved. Either by names or indices.

        with_gene_info: bool, default False
            If True, the gene_info DataFrame will be concatenated to the result.

        name_genes_by: str, default "Symbol"
            A column in the gene_info DataFrame to be used as the name of the genes.
            Usually either 'Symbol'(Symbols) or 'Gene'(Ensembl IDs).

        summarize: pd.DataFrame, default None
            A summary DataFrame. This can be retrieved via GrandPy.get_summary_matrix().
            `columns` will be ignored if provided.

        prefix: str, default None
            Will be prepended to all column names.

        ntr_nan: bool, default False
            If True, ntr values for no4sU will be set to NaN.
            Otherwise, they remain 0.

        reorder_columns: bool, default False
            If True, the columns in the result will be in the same order as in the object.
            Otherwise, they will be in the same order as the input.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified data for the genes and columns.
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
        Get a DataFrame containing analysis tables, optionally with the corresponding gene_info.

        Examples
        --------
        Perform any analysis on the instance.

        >>> sars = sars.fit_kinetics()
        >>> sars.analyses
        ['kinetics_Mock', 'kinetics_SARS']

        Retrieve all analyses.

        >>> sars.get_analysis_table(with_gene_info=False)
                Mock_Synthesis  Mock_Half-life  SARS_Synthesis  SARS_Half-life
        Symbol
        UHMK1       175.303203        7.509571    3.123868e+02        2.813804
        ATF3         34.018585        0.943541    4.843992e+02        0.932378
        ...                ...             ...             ...             ...
        ORF1ab      792.905313        1.241805    1.546125e+06        1.270006
        S           522.247609        1.068520    9.717529e+05        1.262822

        Only retrieve specific results for the first three genes.

        >>> sars.get_analysis_table(analyses="kinetics_Mock", columns="Synthesis",
        ...                         genes=[0,1,2], with_gene_info=False)
                Mock_Synthesis
        Symbol
        UHMK1       146.375312
        ATF3         26.020881
        PABPC4      220.606945

        See Also
        --------
        GrandPy.analyses
            Get the names of all stored analyses.

        GrandPy.with_dropped_analysis
            Drop analyses from the object with a regex.

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
            If True, the columns in the result will be prefixed with the given prefix and the name of the condition.
            Otherwise, they will only be named after the respective analysis.

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
            analysis_data = self._anndata.uns["analyses"][name]

            if columns is not None:
                columns = _ensure_list(columns)

                columns = [
                    col for col in analysis_data.columns
                    if any(re.search(pattern, col) for pattern in columns)
                ]

            if genes is not None:
                analysis_data = analysis_data.loc[gene_info.index]

            if columns is not None:
                if regex:
                    matching_cols = [col for col in analysis_data.columns if any(re.search(pat, col) for pat in columns)]
                else:
                    matching_cols = [col for col in columns if col in analysis_data.columns]
            else:
                matching_cols = analysis_data.columns

            selected_data = analysis_data[matching_cols].copy()

            if not prefix_by_analyses:
                cond = name.rsplit("_", 1)[-1]
                selected_data.columns = [col.rsplit(f"{cond}_", 1)[-1] for col in selected_data.columns]

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


    # TODO Beispiele für get_references schreiben
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

        Parameters
        ----------
        reference : str or Callable[[pd.Series], bool], optional
            A condition to define reference samples. Can be:
            - a string (interpreted via `pandas.query`)
            - a function taking a row and returning True for reference samples, False otherwise.

        reference_function : Callable[[pd.Series], Sequence[str]], optional
            A function that returns a list of references for each sample (row-wise).
            If specified, `reference` is ignored.

        group : str or Sequence[str], optional
            One or more column names in `coldata` used to group samples before applying the condition.

        columns : str or Sequence[str], optional
            Limit the input to specific columns from `coldata` for evaluation.

        as_dict : bool, default False
            If True, return a dictionary mapping each sample to its references.
            If False, return a square boolean DataFrame indicating references.

        Returns
        -------
        Union[pd.DataFrame, dict[str, Sequence[str]]]
            Either a reference matrix or a mapping from sample → list of references.

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



    # ----- Functions on the whole object -----
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
        Analyses are all kept with an added prefix to avoid collisions

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
            By default 'dataset0', 'dataset1'.

            To disable this behavior, set to ("", "").
            Then analyses of the object coming first will be kept in case of a name collision.

        Returns
        -------
        GrandPy
            A new merged GrandPy object.
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

    def to_anndata(self, x: Union[str, ModeSlot] = None, original: bool = False) -> ad.AnnData:
        """
        Extracts an Anndata instance from GrandPy.

        In the unstructured data (uns), `analyses`, `metadata` and the `prefix` are stored.
        Can also be a ModeSlot

        Parameters
        ----------
        x: str or ModeSlot, optional
            The name of the slot to be set as the main data matrix X; by default `default_slot`.
            Can also be a `ModeSlot`.

        original: bool, default False
            If False, AnnData will be returned scanpy compatible.

            Otherwise, AnnData will be returned as stored internally.

        Notes
        -----
        When you want the AnnData instance to be scanpy compatible, use `original` = False.
        For this, the internal AnnData is transposed and plots are removed.
        If this is not desired, use `original` = True.

        See Also
        --------
        anndata_to_grandpy
            Returns a GrandPy instance from a given AnnData.

        Returns
        -------
        ad.AnnData
            An AnnData instance containing the data.
        """
        if x is None:
            x = self.default_slot

        adata = self._anndata.copy()
        adata.X = self.get_matrix(x, force_numpy=False)

        if original:
            return adata

        if self.analyses is not None:
            for name, analysis in adata.uns["analyses"].items():
                adata.uns["analyses"][name] = analysis.copy().T

        adata.uns.pop("plots", None)

        return adata.T


    # ----- Processing functions -----
    def normalize(self, genes: Sequence[str] = None, name: str = "norm", slot: str = "count",
                  set_to_default: bool = True, size_factors=None, return_size_factors: bool = False) -> Union["GrandPy",np.ndarray]:
        """
        Normalize gene expression values across cells.

        Parameters
        ----------
        genes : Sequence[str], optional
            A list of gene names to normalize. If None, all genes are normalized.

        name : str, default "norm"
            Name of the layer where the normalized data will be stored.

        slot : str, default "count"
            The name of the data slot to normalize (e.g., "count").

        set_to_default : bool, default True
            If True, set the normalized layer as the default for downstream analysis.

        size_factors : array-like, optional
            Precomputed size factors to use for normalization.
            If None, size factors are computed automatically.

        return_size_factors : bool, default False
            If True, return the size factors used for normalization.

        Returns
        -------
        Union[GrandPy, np.ndarray]
            The size factors used for normalization if `return_size_factors` is True.
            Otherwise, returns a grandPy object.
        """
        from .processing import _normalize

        return _normalize(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default,
                          size_factors=size_factors, return_size_factors=return_size_factors)

    def normalize_fpkm(self, genes=None, name: str = "norm", slot: str = "count", set_to_default=True, total_len=None) -> "GrandPy":
        """
        Normalize gene expression data using the FPKM method (Fragments Per Kilobase Million).

        This method computes FPKM values by normalizing raw counts by both gene length and
        total library size. It is suitable for comparing expression levels across genes within
        the same sample.

        Parameters
        ----------
        genes : list[str], optional
            A list of gene names to normalize. If None, all genes are included.

        name : str, default "norm"
            The name of the data layer where the normalized values will be stored.

        slot : str, default "count"
            The name of the data slot containing raw counts to normalize.

        set_to_default : bool, default True
            If True, sets the resulting normalized layer as the default for downstream analysis.

        total_len : array-like, optional
            Optional precomputed total transcript lengths per cell. If not provided, gene lengths
            are inferred internally.

        Returns
        -------
        GrandPy
            A grandPy object with added normalize_fpkm slot.
    """

    def normalize_tpm(self, genes=None, name: str = "tpm", slot: str = "count", set_to_default=True, total_len=None) -> "GrandPy":
        from .processing import _normalize_tpm

        return _normalize_tpm(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default,
                              total_len=total_len)

    def normalize_rpm(self, genes=None, name: str = "norm", slot: str = "count", set_to_default=True, factor=1e6):
        from .processing import _normalize_rpm

        # return _normalize_rpm(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default, factor=factor)


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
        data : GrandPy
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
        Get the amount of genes in an unfiltered dataset.

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
            Only these genes will be kept if provided (boolean mask, indices, or names).
            Filtering will not be applied to them. (Basically just subsetting)

        return_genes : bool, default False
            If True, return the list of selected gene indices instead of a filtered GrandPy object.

        Returns
        -------
        list[str] or GrandPy
        """
        from .processing import _filter_genes

        return _filter_genes(self, mode_slot, min_expression=min_expression, min_columns=min_columns,
                             min_condition=min_condition, use=use, keep=keep, return_genes=return_genes)



    # ----- modeling functions -----
    def fit_kinetics(
            self,
            fit_type: Literal["nlls", "ntr", "chase"] = "nlls",
            *,
            slot: str = None,
            time: Union[str, np.ndarray, pd.Series, Sequence] = "duration.4sU",
            name_prefix: Union[str, None] = None,
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
        By default, up to: available CPUs - 1. For more control see the `max_processes` and `exact_processes` parameters.

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

        name_prefix: str, optional
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
            Confidence interval size to use in each fit.

        genes: Union[str or int or Sequence[str or int or bool], optional
            Gene(s) to fit. Uses all by default. Specified either by their index, their symbol, their ensamble id, or a boolean mask.

        show_progress: bool, default True
            If True, a progress bar will be displayed.

        **kwargs
            Additional parameters passed to the model-specific fitting function.

            For `"nlls"`:
                - max_iter: Maximum number of optimization iterations, by default 250.
                - steady_state: Whether to use the steady-state model. It can be set for each condition individually by using a dict. By default, True
                - max_processes: The maximum number of processes this function will use. If None or not provided, it will start up to available cores - 1 processes (e.g. 8 cores -> 7 processes)
                - exact_processes: If True, exactly `max_processes` will be used.

            For `"ntr"`:
                - transformed_ntr_map: If True, use the transformed NTR MAP estimator instead of the MAP of the transformed posterior; by default, True.
                - exact_ci: Whether to use exact confidence intervals; by default False.
                - total_function: Function to reduce total expression across time points (e.g., mean, median); by default `numpy.median`.

            For `"chase"`:
                - max_iter: Maximum number of optimization iterations, by default 250.
                - max_processes: The maximum number of processes this function will use. If None or not provided, it will start up to available cores - 1 processes (e.g. 8 cores -> 7 processes)
                - exact_processes: If True, exactly `max_processes` will be used.

        Returns
        -------
        GrandPy
            A GrandPy instance with analysis results added per condition.
        """
        from .modeling import _fit_kinetics

        kinetics = _fit_kinetics(data=self, fit_type=fit_type, slot=slot, return_fields=return_fields,
                                 name_prefix=name_prefix, time=time, ci_size=ci_size, genes=genes,
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

        n_top_genes : int, default 1000
            Uses the n top expressed genes for calibration.
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



    # ----- Differential Expression functions -----
    def get_summary_matrix(
            self,
            *,
            no4sU: bool = False,
            columns: Union[None, str, list[str]] = None,
            average: bool = True
    ) -> pd.DataFrame:
        """
        Return a summarization matrix for averaging or aggregation.

        If this matrix is multiplied with a count table,
        either the average (average=TRUE) or the sum (average=FALSE) of all columns (samples or cells)
        belonging to the same Condition is computed.

        Parameters
        ----------
        no4sU : bool, default False
            If True, no4sU columns will be included in the summary matrix.
            Otherwise, they will be ignored and returned as zeros.

        columns : str or list of str, optional
            Column names (samples) to include. Can be condition names or column filters.

        average : bool, default True
            If True, normalize columns to sum to 1 (i.e., compute group-wise average).

        Returns
        -------
        pd.DataFrame
            A (samples × conditions) matrix indicating group membership (optionally normalized).
        """
        from .diffexp import _get_summary_matrix

        return _get_summary_matrix(self, no4sU, columns, average)


    def get_contrasts(self, contrast: list = "Condition", columns: Union[Sequence[bool], bool] = None, group: Union[Sequence[str], str] = None, name_format: str = None, no4su: bool = True) -> pd.DataFrame:
        """
        Generate contrast matrix for differential comparisons.

        Parameters
        ----------
        coldata : pd.DataFrame
            DataFrame containing sample metadata (e.g., conditions, groups).

        contrast : list[str]
            Defines the contrast logic:
            - [condition_column] → all pairwise contrasts
            - [condition_column, reference_level] → all vs. reference
            - [condition_column, level_A, level_B] → specific comparison A vs B

        columns : list[str], optional
            Subset of sample IDs to consider in contrast computation. If None, all samples are used.

        group : str, optional
            Column name in `coldata` to compute contrasts within each group level separately.

        name_format : str, optional
            Format string for naming contrast columns.
            Supports placeholders: "$A", "$B", "$COL", "$GRP".
            Default: "$A vs $B" (or "$A vs $B.$GRP" if `group` is set)

        no4sU : bool, default False
            If True, samples where `coldata['no4sU'] == False` are excluded from contrast computation.

        Returns
        -------
        a GrandPy object
            A contrast matrix where rows are samples and columns represent contrasts.
            Values are -1, 0, or 1 depending on contrast design.
        """
        from .diffexp import _get_contrasts

        return _get_contrasts(self, contrast = contrast, columns = columns, group = group, name_format = name_format, no4sU = no4su)


    def compute_lfc(
            self,
            name_prefix: str = None,
            contrasts: pd.DataFrame = None,
            mode_slot: Union[str, ModeSlot] = "count",
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
        GrandPy.get_analysis_table: Retrieves stored analyses.
        GrandPy.pairwise: Combined log2 fold change and Wald test differential expression analysis.
        GrandPy.pairwise_deseq2: Run DESeq2 for each contrast defined in the contrast matrix.

        Parameters
        ----------
        name_prefix : str, optional
            The prefix for the new analysis name; e.g. 'total' or 'new'.

        contrasts : pd.DataFrame, optional
            Contrast matrix defining comparisons (samples x contrasts; values 1, -1).

        mode_slot: str or ModeSlot, default "count"
            The name of the data slot to take data from. Usually 'count', optionally with a mode.
            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'.

        lfc_function : Callable, default psi_lfc
            Function to compute the log2 fold changes.

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
            Passed to LFC_fun.

        Returns
        -------
        GrandPy
            A GrandPy instance with one analysis per contrast. Each analysis
            adds two columns named "{prefix}_{contrast}_LFC" and
            "{prefix}_{contrast}_M".
        """
        from .diffexp import _compute_lfc

        new_gp = _compute_lfc(data=self, name_prefix=name_prefix, contrasts=contrasts, mode_slot=mode_slot, lfc_function=lfc_function,
                              normalization=normalization, compute_m=compute_m, genes=genes, verbose=verbose, **kwargs)

        return new_gp

    def pairwise_deseq2(
        self,
        contrasts: pd.DataFrame,
        name_prefix: str = None,
        separate: bool = False,
        mode_slot: Union[str, ModeSlot] = "count",
        normalization: Union[str, Sequence[float]] = None,
        genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
        verbose: bool = False
    ) -> "GrandPy":
        """
        Run DESeq2 (via pydeseq2) for each contrast defined in the contrast matrix.

        Notes
        -----
        Uses fit_type="mean" for compatibility with pydeseq2.
        pydeseq2 does not currently support fit_type="local".

        See Also
        --------
        GrandPy.get_analysis_table: Retrieves stored analyses.
        GrandPy.pairwise: Combined log2 fold change and Wald test differential expression analysis.
        GrandPy.compute_lfc: Estimate log2 fold changes and optional M values for each contrast.

        Parameters
        ----------
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
            A GrandPy instance containing analysis results.
        """
        from .diffexp import _pairwise_DESeq2

        new_gp = _pairwise_DESeq2(data=self, contrasts=contrasts, name_prefix=name_prefix, separate=separate,
                                  mode_slot=mode_slot, normalization=normalization, genes=genes, verbose=verbose)

        return new_gp

    def pairwise(
        self,
        contrasts: pd.DataFrame,
        name_prefix: str = None,
        lfc_function: Callable = psi_lfc,
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

        See Also
        --------
        GrandPy.get_analysis_table: Retrieves stored analyses.
        GrandPy.compute_lfc: Estimate log2 fold changes and optional M values for each contrast.
        GrandPy.pairwise_deseq2: Run DESeq2 for each contrast defined in the contrast matrix.

        Parameters
        ----------
        contrasts : pd.DataFrame
            Contrast matrix defining comparisons (samples x contrasts; values 1, -1).

        name_prefix : str, optional
            Prefix for naming the output analysis tables.

        lfc_function : Callable, default psi_lfc
            Function to compute the log2 fold changes. Implemented: psi_lfc, norm_lfc.

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
        GrandPy
            A GrandPy instance with analysis results.
        """
        from .diffexp import _pairwise

        new_gp = _pairwise(data=self, contrasts=contrasts, name_prefix=name_prefix, lfc_function=lfc_function,
                           mode_slot=mode_slot, normalization=normalization, genes=genes, verbose=verbose, **kwargs)

        return new_gp




def anndata_to_grandpy(anndata: ad.AnnData, transpose: bool = True) -> GrandPy:
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
        Meaning obs has to relate to the rows of X (coldata) and var to the columns (gene_info).

        See Also
        --------
        GrandPy.to_anndata
            Extract a GrandPy instance from the AnnData.

        Returns
        -------
        GrandPy
            A GrandPy instance built from the AnnData.
        """
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

def read_h5ad(path: Union[PathLike[str], str]) -> GrandPy:
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

    anndata.uns["plots"] = {}

    return anndata_to_grandpy(anndata, transpose = False)



