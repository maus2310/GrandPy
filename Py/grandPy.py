import copy
import re
import warnings
from collections.abc import Sequence, Mapping
from typing import Any, Union, Literal, Callable
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from Py.slot_tool import SlotTool, ModeSlot
from Py.plot_tool import PlotTool, Plot
from Py.analysis_tool import AnalysisTool
from Py.processing import _filter_genes
from Py.utils import _ensure_list, _make_unique, _reindex_by_index_name, _subset_dense_or_sparse


class GrandPy:
    """
    Create a GrandPy object.

    Data is typically loaded using the `read_grand()` function, which parses preprocessed GrandR-compatible
    data formats into a GrandPy object.

    The Object is designed to be immutable. Changes are made through `with_...` methods.

    Simple getters are implemented as properties, more complex ones through `get_...` methods.

    Examples
    --------
    Read a GrandPy Object from a file.

    >>> import grandpy as gp
    >>> sars = gp.read_grand("./data/sars.tsv", design=("Condition", "Time", "Replicate"))
    >>> print(sars)
    GrandPy:
    Read from ./data/sars.tsv
    1045 genes, 12 samples/cells
    Available data slots: count, ntr, alpha, beta
    Available analyses: None
    Available plots: None
    Default data slot: count

    Parameters
    ----------
    prefix: str, optional
        Path to the data file.

    gene_info: pd.DataFrame, optional
        Genes and their metadata.

    coldata: pd.DataFrame, optional
        Samples and their metadata.

    slots: dict[str, Union[np.ndarray, sp.csr_matrix]]], optional
        Name and the corresponding data matrix.

    metadata: dict[str, Any], optional
        Metadata about the data and file.

    analyses: dict[str, pd.DataFrame], optional
        Results from analyzing functions.

    plots: dict[str, dict[str, Plot]], optional
        Plot functions. (global or gene plots)

    See Also
    --------
    read_grand
        Read a file into a GrandPy object.
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
        self._adata = ad.AnnData(
            X = slots["count"],
            obs = gene_info,
            var = coldata,
        )
        self._is_sparse = True if isinstance(slots["count"], sp.csr_matrix) else False

        self._initialize_slots(slots)
        self._initialize_uns_data(prefix, metadata, analyses, plots)
        self._ensure_no4sU_column()
        self._ensure_condition_column()

    def _initialize_slots(self, slots=None):
        for key, matrix in slots.items():
            self._adata.layers[key] = matrix

    def _initialize_uns_data(self, prefix: str, metadata: dict[str, Any], analyses: dict[str, pd.DataFrame], plots: dict[str, dict[str, Plot]]):
        if metadata.get('default_slot') is None:
                metadata["default_slot"] = "count"

        self._adata.uns['prefix'] = prefix if prefix is not None else "Unknown"
        self._adata.uns['metadata'] = metadata
        self._adata.uns['analyses'] = analyses if analyses is not None else {}
        self._adata.uns['plots'] = plots if plots is not None else {}

    def _ensure_no4sU_column(self):
        if 'no4sU' not in self._adata.var.columns:
            warnings.warn("No 'no4sU' entry in coldata, assuming all samples/cells as 4sU treated! "
                          "If the column is supposed to already exist, consider renaming it (see GrandPy.with_coldata())")
            self._adata.var["no4sU"] = False

    def _ensure_condition_column(self):
        if 'Condition' not in self._adata.var.columns:
            warnings.warn("No 'Condition' entry in coldata, assuming all samples/cells as 'Control'! "
                          "Consider changing it (see GrandPy.with_condition()) or "
                          "renaming an existing column if it should already exist. (see GrandPy.with_coldata())")
            self._adata.var["Condition"] = "Control"


    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self._adata.uns['prefix']}\n"
            f"{self._adata.n_obs} genes, {self._adata.n_vars} samples/cells\n"
            f"Available data slots: {self.slots}\n"
            f"Available analyses: {self.analyses}\n"
            f"Available plots: {self.plots}\n"
            f"Default data slot: {self.default_slot}\n"
        )

    def __getitem__(self, items):
        new_adata = self._adata.copy()
        new_adata = new_adata[items]

        # Reorders all existing analyses according to the new genes present in gene_info
        if new_adata.uns.get("analyses") is not None:
            for key in new_adata.uns["analyses"].keys():
                new_adata.uns["analyses"][key] = _reindex_by_index_name(new_adata.uns["analyses"][key], new_adata.obs)

        return self._dev_replace(anndata = new_adata)


    # ----- Replace functions -----
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

        Should be handled with care, as no additional checks are performed.

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

        Returns
        -------
        GrandPy
            A new GrandPy object with the given parameters replaced.

        See Also
        --------
            get_matrix()
                Gives the raw data for a slot, as it is stored internally.

            to_anndata()
                Retrieves the anndata instance.
        """
        if anndata is None:
            anndata = self._adata.copy()

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
        This function is replace() for internal use.

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
            anndata = self._adata.copy()

        return self.__class__(
            prefix = prefix if prefix is not None else anndata.uns.get('prefix'),
            gene_info = gene_info if gene_info is not None else anndata.obs,
            coldata = coldata if coldata is not None else anndata.var,
            slots = slots if slots is not None else anndata.layers,
            metadata = metadata if metadata is not None else anndata.uns.get("metadata"),
            analyses = analyses if analyses is not None else anndata.uns.get("analyses"),
            plots = plots if plots is not None else anndata.uns.get("plots")
        )


    # ----- Basic properties and methods -----
    @property
    def title(self) -> str:
        """
        Get a title for the GrandPy object.
        The title is derived from the prefix.
        """
        prefix = self._adata.uns.get('prefix')
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
        return self._adata.X.shape

    @property
    def dim_names(self) -> tuple[list[str], list[str]]:
        """
        Get the column and row names of the data.
        """
        row_names = self._adata.var_names.tolist()
        column_names = self._adata.obs_names.tolist()
        return row_names, column_names

    @property
    def default_slot(self) -> str:
        """
        Get the name of the default slot
        """
        return self._adata.uns.get('metadata').get('default_slot')

    def with_default_slot(self, value: str) -> "GrandPy":
        """
        Returns a copy of the GrandPy object with the default slot set to `value`.

        Parameters
        ----------
        value: str
            Sets the default slot to this value.

        Returns
        -------
        "GrandPy"
            Returns a new GrandPy object having the new default slot.
        """
        assert value in self._adata.layers.keys(), "Trying to set a default_slot that is not an available slot"

        new_metadata = self._adata.uns.get('metadata', {}).copy()
        new_metadata['default_slot'] = value

        return self._dev_replace(metadata=new_metadata)


    # ----- All slot methods ------
    @property
    def _slot_manager(self) -> SlotTool:
        return SlotTool(self._adata, self._is_sparse)

    @property
    def slots(self) -> list[str]:
        """
        Get the names of all available slots.
        """
        return self._slot_manager.slots()

    @property
    def _slot_data(self) -> dict[str, Union[np.ndarray, sp.csr_matrix]]:
        """
        Get the raw data of all available slots as they are stored internally.

        Returns
        -------
        dict[str, Union[np.ndarray, sp.csr_matrix]]
            The data of all available slots.
        """
        return self._slot_manager.slot_data()

    def _check_slot(self, slot: str, *, allow_ntr: bool = True) -> bool:
        """
        Checks if a given slot exists in the data slots.

        Parameters
        ----------
        slot: str
            The slot to be checked.

        allow_ntr: bool, default True
            If True, the slot "ntr" is allowed as input.

        Returns
        -------
        bool:
            True if the slot exists, False otherwise.
        """
        return self._slot_manager.check_slot(slot, allow_ntr=allow_ntr)

    def _resolve_mode_slot(self, mode_slot: Union[str, ModeSlot], *, allow_ntr: bool = True, ntr_nan: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
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
        return self._slot_manager.resolve_mode_slot(mode_slot, allow_ntr=allow_ntr, ntr_nan=ntr_nan)

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

        new_slots, new_metadata = self._slot_manager.with_dropped_slots(slots_to_remove)

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
        new_slots, new_metadata = self._slot_manager.with_slot(name, new_slot, set_to_default=set_to_default)

        return self._dev_replace(slots=new_slots, metadata=new_metadata)

    def with_ntr_slot(self, as_ntr: str, save_ntr_as: str = None) -> "GrandPy":
        """
        Set a different slot as the new 'ntr' slot. If save_ntr_as is not set, the former ntr slot will be removed.
        The slot 'ntr' will be used for mode_slots and other functions relating to ntr.

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
        new_slots = self._slot_manager.with_ntr_slot(as_ntr, save_ntr_as=save_ntr_as)

        return self._dev_replace(slots=new_slots)


    # ----- All analysis methods -----
    @property
    def _analysis_manager(self):
        return AnalysisTool(self._adata)

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
        return self._analysis_manager.analyses()

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
            A list containing the names of all found analyses.

        Raises
        ------
        ValueError
            Raises an error if any pattern has no matches.

        See Also
        --------
        GrandPy.analyses:
            Get a list of all available analyses.

        GrandPy.with_dropped_analyses:
            Remove analyses with a regex pattern.

        GrandPy.with_analysis:
            Add an analysis to the object. Usually not to be used directly.
        """
        return self._analysis_manager.get_analyses(pattern, regex=regex, description=description)

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
        new_analyses = self._analysis_manager.with_analysis(name, table, by=by)

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
        new_analyses = self._analysis_manager.drop_analyses(pattern)

        return self._dev_replace(analyses=new_analyses)


    # ----- All plot methods -----
    @property
    def _plot_manager(self):
        return PlotTool(self._adata)

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
        return self._plot_manager.plots()

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
        new_plots = self._plot_manager.add_plot(name, function)

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
        return self._adata.uns["plots"]["gene"][name](self, gene)

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
        return self._adata.uns["plots"]["global"][name](self)

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
        new_plots = self._plot_manager.drop_plot(pattern)

        return self._dev_replace(plots=new_plots)



    # ----- Methods relating to coldata, gene_info or metadata -----
    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get the metadata about the GrandPy object.
        """
        return self._adata.uns.get('metadata').copy()


    @property
    def gene_info(self) -> pd.DataFrame:
        """
        Get the gene_info DataFrame.
        """
        return self._adata.obs.copy()

    def with_gene_info(self, column: str, value: Union[Mapping, pd.Series, Sequence[Any]]) -> "GrandPy":
        """
        Returns a new object with modified gene_info. If the column name does not already exist, a new column will be added.

        Otherwise, the column will be replaced by the given value or updated if a dictionary was given.

        Parameters
        ----------
        column : str
            The name of the column to be modified.

        value : Mapping or pd.Series or Sequence[Any]
            The values to assign to the column can be any iterable. Can also be a dictionary when trying to update a column.
            A Series will be matched to gene_info by its index.

        Returns
        -------
        GrandPy
            A new GrandPy object with updated gene_info.
        """
        new_gene_info = self.gene_info

        if isinstance(value, Mapping):
            if isinstance(value, Mapping):
                if all(v in new_gene_info[column].values for v in value.keys()):
                    for match_value, new_val in value.items():
                        new_gene_info.loc[new_gene_info[column] == match_value, column] = new_val
                else:
                    for row_index, new_value in value.items():
                        if row_index in new_gene_info.index:
                            new_gene_info.at[row_index, column] = new_value
            return self._dev_replace(gene_info = new_gene_info)

        if isinstance(value, pd.Series):
            value.index = _make_unique(value.index, warn = False)

        new_gene_info[column] = value

        return self._dev_replace(gene_info = new_gene_info)

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
            mask = gene_col.str.contains(pattern, regex=True) | symbol_col.str.contains(pattern, regex=True)
            return list(np.flatnonzero(mask))

        # Boolean mask
        if all(isinstance(g, (bool, np.bool_)) for g in genes):
            if len(genes) != n:
                raise ValueError("Boolean mask length does not match gene count.")
            return list(np.flatnonzero(genes))

        # Integer index
        if all(isinstance(g, (int, np.integer)) for g in genes):
            if not all(0 <= g < n for g in genes):
                raise IndexError("One or more gene indices out of bounds.")
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

    # TODO with_updated_symbols vervollständigen. Aktuell wird noch nicht unique gemacht und
    #  auch nur die Spalte 'Symbol' wird geändert. None Behandlung ebenfalls fraglich
    def with_updated_symbols(self, species: str = "human") -> "GrandPy":
        import mygene

        new_gene_info = self.gene_info.copy()
        genes = new_gene_info["Gene"].tolist()

        mg = mygene.MyGeneInfo()

        # Query: get the symbols in batches (a single large request is not possible with mygene)
        try:
            result = mg.querymany(
                genes,
                scopes="ensembl.gene",  # Oder "symbol", je nach Inhalt
                fields="symbol",
                species=species,
                as_dataframe=True,
            )
        except Exception as e:
            raise RuntimeError(f"MyGeneInfo request failed: {e}")

        # Turn the query output into a gene_info dataframe.
        result = result.reset_index()
        result = result[["query","symbol"]]
        symbol_map = dict(zip(result["query"], result["symbol"]))
        new_gene_info["Symbol"] = new_gene_info["Gene"].map(symbol_map)

        return self._dev_replace(gene_info=new_gene_info)

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

        Either by their index, their symbol, their ensamble id, a boolean mask, or a regex.

        If no genes are specified, all genes are returned.

        Parameters
        ----------
        genes: str or int or Sequence[str or int or bool], optional
            Genes to be retrieved.
        get_gene_symbols: bool, default True
            If True, gene symbols will be returned. Otherwise, gene names will be returned.
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
        return self._adata.var.copy()

    def with_coldata(self, column: str, value: Union[Mapping, pd.Series, Sequence[Any]]) -> "GrandPy":
        """
        Returns a new object with modified coldata. If the column name does not already exist, a new column will be added.

        Otherwise, the column will be replaced by the given value or updated if a dictionary was given.

        Examples
        --------


        Parameters
        ----------
        column : str
            The name of the column to be modified or added.

        value : Mapping or pd.Series or Sequence[Any]
            The values to assign to the column can be any iterable. Can also be a dictionary when trying to update a column.

        Returns
        -------
        GrandPy
            A new GrandPy object with updated coldata.
        """
        new_coldata = self.coldata

        if column in new_coldata.columns:
            if isinstance(value, Mapping):
                if all(v in new_coldata[column].values for v in value.keys()):
                    for match_value, new_val in value.items():
                        new_coldata.loc[new_coldata[column] == match_value, column] = new_val
                else:
                    for row_index, new_value in value.items():
                        if row_index in new_coldata.index:
                            new_coldata.at[row_index, column] = new_value
                return self._dev_replace(coldata = new_coldata)

        new_coldata[column] = value

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
        Get sample/cell names. Either by their index, their name, or a boolean mask.

        If no columns are specified, all sample/cell names are returned.

        Parameters
        ----------
        columns: str or int or Sequence[str or int or bool], optional
            Samples/cells to be retrieved.

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
            get the gene symbols/names.
        """
        coldata = self.coldata

        if columns is None:
            return self.columns

        columns = _ensure_list(columns)

        if all(isinstance(column, int) for column in columns):
            result = coldata.iloc[columns].index

        elif all(isinstance(column, (str, bool)) for column in columns):
            result = coldata.loc[columns, :].index

        else:
            raise TypeError("The input must be either string, int or a boolean mask. They cannot be mixed")

        if reorder:
            result = result.reindex(self._adata.var.index).dropna()

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

        This is what you call when samples/calls have been mislabeled.

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
                raise TypeError("A Matrix in the GrandPy Object has an unexpected type. Only pd.DataFrame, np.ndarray and scipy.sparse.csr_matrix are supported")

            return matrix

        if isinstance(column1, str):
            column1 = self._adata.var.index.get_loc(column1)
        if isinstance(column2, str):
            column2 = self._adata.var.index.get_loc(column2)

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
        old_obs_index = self._adata.obs.index
        old_var_index = self._adata.var.index

        new_adata = self._adata.copy()

        # Apply function to gene_info
        if function_gene_info is not None:
            new_gene_info = function_gene_info(new_adata.obs, **kwargs)
            new_obs_index = new_gene_info.index
        else:
            new_gene_info = new_adata.obs
            new_obs_index = old_obs_index

        # Apply function to coldata
        if function_coldata is not None:
            new_coldata = function_coldata(new_adata.var, **kwargs)
            new_var_index = new_coldata.index
        else:
            new_coldata = new_adata.var
            new_var_index = old_var_index

        # # Retrieve index for reordering
        # row_reorder = None if new_obs_index.equals(old_obs_index) else new_obs_index.get_indexer(old_obs_index)
        # col_reorder = None if new_var_index.equals(old_var_index) else new_var_index.get_indexer(old_var_index)

        row_indices = old_obs_index.get_indexer_for(new_obs_index)
        column_indices = old_var_index.get_indexer_for(new_var_index)

        new_layers = {}

        for key in self._adata.layers.keys():
            matrix = function(self._adata.layers[key], **kwargs)

            # # Adjust row order if necessary
            # if row_reorder is not None:
            #     matrix = matrix[row_reorder, :]
            #
            # # Adjust column order if necessary
            # if col_reorder is not None:
            #     matrix = matrix[:, col_reorder]

            matrix = matrix[np.ix_(row_indices, column_indices)]

            new_layers[key] = matrix

        new_analyses = {}
        # Also fix analysis reindexing if needed
        if new_adata.uns['analyses'] is not None:
            new_adata.uns['analyses'] = {
                key: _reindex_by_index_name(value, new_gene_info)
                for key, value in new_adata.uns['analyses'].items()
            }

        return self._dev_replace(gene_info=new_gene_info, coldata=new_coldata, slots=new_layers, analyses=new_analyses)


    def get_matrix(
            self,
            mode_slot: Union[str, ModeSlot] = None,
            genes: Union[str, int, Sequence[Union[str, int]]] = None,
            columns: Union[str, int, Sequence[Union[str, int]]] = None,
            force_numpy: bool = True
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Get the raw data from a data slot, without row or column names.

        This function is mostly not needed, as get_table() or get_data() are usually better suited.

        Parameters
        ----------
        mode_slot: str or ModeSlot
            The name of the data slot. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int]
            The genes to be retrieved. Either by gene symbols, names(Ensembl ids), indices, or a boolean mask.

        columns: str or int or Sequence[str or int]
            The samples/cells to be retrieved. Either by names, indices, or a boolean mask.

        force_numpy: bool, default True
            If True, return will always be a numpy ndarray, regardless of the type of the slots.
            Otherwise, return the data in their actual type.

        Returns
        -------
        Union[np.ndarray, sp.csr_matrix]
            A raw data matrix, without column or row names.

        See Also
        --------
        GrandPy.get_table
            Similar to get_matrix(), but with row and column names and coldata can be concatenated.

        GrandPy.get_data
            Similar to get_data(), but slots are transposed, so gene_info can be concatenated.

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis tables.
        """
        if mode_slot is None:
            mode_slot = self.default_slot

        data = self._resolve_mode_slot(mode_slot)

        row_indices = self.get_index(genes)
        column_indices = [self._adata.var.index.get_loc(column) for column in self.get_columns(columns)]

        data_subset = _subset_dense_or_sparse(data, row_indices, column_indices, force_numpy=force_numpy)

        return data_subset

    def get_data(
            self,
            mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
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
        mode_slots: str or ModeSlot or Sequence[str or ModeSlot], optional
            The name of the data slots. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int], optional
            The genes to be retrieved. Either by gene symbols, names(Ensembl ids), or indices.

        columns: str or int or Sequence[str or int], optional
            The samples/cells to be retrieved. Either by names or indices.

        with_coldata: bool, default True
            If True, the coldata DataFrame will be concatenated to the result.

        name_genes_by: str, default "Symbol"
            A column in the gene_info DataFrame to be used as the name of the genes.
            Usually either `Symbol`(Symbols) or 'Gene'(Ensembl IDs).

        by_rows: bool, default False
            If True, add rows if there are multiple genes or mode_slots.
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
            Similar to get_data(), but slots are transposed, so gene_info can be concatenated.

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis tables.

        GrandPy.get_matrix
            Similar to get_data(), but transposed and only gives raw_data without row or column names.
        """
        coldata = self.coldata
        gene_info = self.gene_info

        if mode_slots is None:
            mode_slots = self.default_slot

        mode_slots = _ensure_list(mode_slots)

        row_indices = [coldata.index.get_loc(column) for column in self.get_columns(columns)]
        column_indices = self.get_index(genes)

        row_names = coldata.iloc[row_indices].index.tolist()
        column_names = gene_info.iloc[column_indices][name_genes_by].tolist()

        result_df = pd.DataFrame()

        if not by_rows:
            for slot_name in mode_slots:
                all_data = self._resolve_mode_slot(slot_name, ntr_nan=ntr_nan).T
                data_subset = _subset_dense_or_sparse(all_data, row_indices, column_indices)

                if len(mode_slots) > 1:
                    local_column_names = [f"{name}_{slot_name}" for name in column_names]
                else:
                    local_column_names = column_names

                processed_data = pd.DataFrame(data_subset, index=row_names, columns=local_column_names)
                result_df = pd.concat([result_df, processed_data], axis=1)

            if with_coldata:
                result_df = pd.concat([coldata.iloc[row_indices], result_df], axis=1)

            return result_df

        else:
            for slot_name in mode_slots:
                all_data = self._resolve_mode_slot(slot_name, ntr_nan=ntr_nan).T
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
            mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
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

        Parameters
        ----------
        mode_slots: str or ModeSlot or Sequence[str or ModeSlot], optional
            The name of the data slots to be retrieved. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: str or int or Sequence[str or int], optional
            The genes to be retrieved. Either by gene symbols, names(Ensembl ids), or indices.

        columns: str or int or Sequence[str or int], optional
            The samples/cells to be retrieved. Either by names or indices.

        with_gene_info: bool, default False
            If True, the gene_info DataFrame will be concatenated to the result.

        name_genes_by: str, default "Symbol"
            A column in the gene_info DataFrame to be used as the name of the genes.
            Usually either `Symbol`(Symbols) or `Gene`(Ensembl IDs).

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

        See Also
        --------
        GrandPy.get_data:
            Similar to get_table(), but slots are transposed, so coldata can be concatenated.

        GrandPy.get_analysis_table:
            Get a DataFrame containing analysis tables.

        GrandPy.get_summary_matrix:
            Get a summarization matrix for averaging or aggregation. Can be provided to get_table() via `summarize`.

        GrandPy.get_matrix:
            Similar to get_table(), but gives raw data without row or column names.
        """
        gene_info = self.gene_info
        coldata = self.coldata

        if mode_slots is None:
            mode_slots = self.default_slot
        mode_slots = _ensure_list(mode_slots)

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

        for slot_name in mode_slots:
            all_data = self._resolve_mode_slot(slot_name, ntr_nan=ntr_nan)

            if summarize is not None:
                matrix = all_data @ summarize.values
            else:
                matrix = all_data

            data_subset = _subset_dense_or_sparse(matrix, row_indices=gene_indices, column_indices=column_indices)

            # Column names (add suffix if multiple slots)
            if len(mode_slots) > 1:
                local_colnames = [f"{col}_{slot_name}" for col in column_names]
            else:
                local_colnames = column_names

            slot_df = pd.DataFrame(data_subset, index=gene_names, columns=local_colnames)
            result_df = pd.concat([result_df, slot_df], axis=1)

        if with_gene_info:
            gene_info_block = gene_info.iloc[gene_indices].copy()
            gene_info_block.index = result_df.index  # match exactly
            result_df = pd.concat([gene_info_block, result_df], axis=1)

        if prefix is not None:
            result_df.columns = [f"{prefix}{col}" for col in result_df.columns]

        result_df.index = gene_names
        return result_df

    # TODO: get_analysis_table() add the prefix thingey and by_rows
    def get_analysis_table(
            self,
            analyses: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            genes: Union[str, int, Sequence[Union[str, int]]] = None,
            columns: str = None,
            *,
            regex: bool = True,
            with_gene_info: bool = True,
            name_genes_by: str = "Symbol",
            by_rows: bool = False,
    ) -> pd.DataFrame:
        """
        Get a DataFrame containing analysis tables, optionally with the corresponding gene_info.

        Parameters
        ----------
        analyses: str or int or Sequence[str or int or bool], optional
            The analyses to be retrieved. Either by name, index, or a boolean mask.

        genes: str or int or Sequence[str or int], optional
            The genes for which to retrieve the analysis tables. Either by gene symbols, names(Ensembl ids), or indices.

        columns: str, optional
            A regular expression to match the name of the columns in the analysis tables.

        regex: bool, default True
            If True, 'analyses' will be interpreted as a regular expression.

        with_gene_info: bool, default True
            If True, the gene_info DataFrame will be concatenated to the result.

        name_genes_by: str, default "Symbol"
            The name of the column in the gene_info DataFrame to be used as the name of the genes.
            Usually either `Symbol`(Symbols) or `Gene`(Ensembl IDs).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified analyses for the genes and columns.

        See Also
        --------
        GrandPy.analyses
            Get the names of all stored analyses.

        GrandPy.get_analyses
            Get the names of analyses. Either by a regex, names, indices, or a boolean mask.

        GrandPy.with_analysis
            Add a new analysis to the object.

        GrandPy.with_dropped_analysis
            Drop analyses from the object with a regex.
        """
        analyses = self.get_analyses(analyses, regex = regex)

        row_indices = self.get_index(genes)

        result_df = pd.DataFrame()

        for name in analyses:
            analysis_data = self._adata.uns["analyses"][name]

            analysis_data.index = pd.Index(self._adata.obs[name_genes_by])

            # Only take rows that match 'genes', if specified.
            if genes is not None:
                analysis_data = analysis_data.iloc[row_indices]

            # Only take columns that match 'columns', if specified.
            if columns is not None:
                if regex:
                    matching_cols = [col for col in analysis_data.columns if
                                     any(re.search(pat, col) for pat in columns)]
                else:
                    matching_cols = [col for col in columns if col in analysis_data.columns]
            else:
                matching_cols = analysis_data.columns

            result_df = pd.concat([result_df, analysis_data[matching_cols]], axis=1)

        if with_gene_info:
            result_df = pd.concat([self._adata.obs.iloc[row_indices], result_df], axis=1)

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
                mask_df = df_subset.eval(reference, engine="python") # Alternativ: selected_refs = df_subset.query(reference).index
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
            axis: Literal["gene_info", 0, "coldata", 1] = 1,
            join: Literal["inner", "outer"] = "inner",
            merge: Union[Literal["same", "unique", "first", "only"], Callable] = "unique",
    ) -> "GrandPy":
        """
        Merge the other object with the current instance along a given axis. Uses `unique` for merging metadata and plots.

        Analyses are merged if their names in both objects are identical. Otherwise, they are dropped.

        Parameters
        ----------
        other: GrandPy
            The object to merge with the current instance.

        axis: {"gene_info" or 0 or "coldata" or 1}, default 1
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

        Returns
        -------
        GrandPy
            A new merged GrandPy object.
        """
        from Py.utils import concat

        objects = [self, other]

        return concat(objects, axis=axis, join=join, merge=merge)

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
            subset = self._adata[:, mask]
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

        adata = self._adata.copy()
        adata.X = self.get_matrix(x, force_numpy=False)

        if original:
            return adata

        if self.analyses is not None:
            for name, analysis in adata.uns["analyses"].items():
                adata.uns["analyses"][name] = analysis.copy().T

        adata.uns.pop("plots", None)

        return adata.T


    # ----- Processing functions -----
    def compute_ntr_ci(self, ci_size: float = 0.95, name_lower: str = "lower", name_upper: str = "upper")-> "GrandPy":
        from Py.processing import _compute_ntr_ci

        return _compute_ntr_ci(self, ci_size, name_lower, name_upper)

    def compute_steady_state_half_lives(self, time=None, name="HL", columns=None, max_hl=48.0, ci_size=0.95, compute_ci=False, as_analysis=False) -> "GrandPy":
        from Py.processing import _compute_steady_state_half_lives

        return _compute_steady_state_half_lives(self, time, name=name ,columns=columns, max_hl=max_hl, ci_size=ci_size, compute_ci=compute_ci, as_analysis=as_analysis)

    def normalize(self, genes = None, name: str = "norm", slot: str = "count", set_to_default = True, size_factors = None, return_size_factors = False):
        from Py.processing import _normalize

        return _normalize(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default, size_factors=size_factors, return_size_factors=return_size_factors)

    def normalize_fpkm(self, genes = None, name: str = "norm", slot: str = "count", set_to_default = True, total_len = None):
        from Py.processing import _normalize_fpkm

        return _normalize_fpkm(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default, total_len = total_len)

    def normalize_tpm(self, genes=None, name: str = "tpm", slot: str = "count", set_to_default=True, total_len=None):
        from Py.processing import _normalize_tpm

        return _normalize_tpm(self, genes=genes, name=name, slot=slot, set_to_default=set_to_default, total_len=total_len)

    def normalize_rpm(self, genes= None, name: str = "norm", slot: str = "count", set_to_default = True, factor = 1e6):
        from Py.processing import _normalize_rpm

        return _normalize_rpm(self, genes=genes, name=name, slot=slot, factor=factor)


    def filter_genes(
        self,
        mode_slot: Union[str, "ModeSlot"] = None,
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

        Parameters
        ----------
        mode_slot : str or ModeSlot, optional
            Which data slot to use.

        min_expression : Number, default 100
            Minimum value threshold to consider a gene expressed.

        min_columns : int, optional
            Minimum number of samples the gene must meet `min_expression` in.
            Defaults to half the number of columns in the matrix.
            Will be ignored if `min_condition` is provided

        min_condition : int, optional
            Overrides `min_columns` if set.

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
        return _filter_genes(self, mode_slot, min_expression=min_expression, min_columns=min_columns,
                             min_condition=min_condition, use=use, keep=keep, return_genes=return_genes)


    # ----- modeling functions -----
    def fit_kinetics(
            self,
            fit_type: Literal["nlls", "ntr", "chase"] = "nlls",
            *,
            slot: str = None,
            name_prefix: Union[str, None] = None,
            return_fields: Union[str, Sequence[str]] = None,
            time: Union[str, np.ndarray, pd.Series, Sequence] = "Time",
            ci_size: float = 0.95,
            genes: Union[str, Sequence[str]] = None,
            max_processes: int = None,
            show_progress: bool = True,
            **kwargs
    ) -> "GrandPy":
        """
        Fit kinetic models to gene expression data.

        This method fits mass-action kinetic models of RNA dynamics using one of several approaches.
        Fits are performed separately per condition.

        The `"nlls"` and `"chase"` methods require normalized input.
        The `"ntr"` method is independent of normalization but assumes steady-state kinetics.

        Parameters
        ----------
        fit_type: {'nlls' or 'ntr' or 'chase'}, default 'nlls'
            The type of model to fit:

            - `"nlls"`: Fit synthesis and degradation rates using non-linear least squares.
            - `"ntr"`: Estimate degradation from new-to-total ratios, assuming steady-state.
            - `"chase"`: Fit degradation from decay of labeled RNA only.

        slot: str, optional
            Name of the data slot used to extract old/new RNA expression. Defaults to the default slot.

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

        time: str or array-like, default "Time"
            Either a column name in `coldata` or a list of timepoints.

        ci_size: float, default 0.95
            Confidence interval size to use in each fit.

        genes: Union[str or int or Sequence[str or int or bool], optional
            Gene(s) to fit. Uses all by default. Specified either by their index, their symbol, their ensamble id, or a boolean mask.

        max_processes: int, optional
            This function decides dynamically how many processes to use.
            By default, up to available CPUs - 1 (e.g. 8 cores -> 7 processes).

        show_progress: bool, default True
            If True, a progress bar will be displayed.

        **kwargs: dict
            Additional parameters passed to the model-specific fitting function.

            For `"nlls"`:
                - max_iter: Maximum number of optimization iterations, by default 250.
                - steady_state: Whether to use the steady-state model. Can be set for each condition individually by using a dict. By default True

            For `"ntr"`:
                - transformed_ntr_map: Whether to assume that NTR values are MAP transformed; by default True.
                - exact_ci: Whether to use exact confidence intervals; by default False.
                - total_function: Function to reduce total expression across time points (e.g., mean, median); by default `numpy.median`.

            For `"chase"`:
                - max_iter: Maximum number of optimization iterations, by default 250.

        Notes
        -----
        This function will create as many worker processes as the machine has processors for larger datasets.
        See the `processes` parameter for more control.

        See Also
        --------
        GrandPy.get_analysis_table: Retrieve analyses from the object.
        GrandPy.normalize: Normalizes the expression data.

        Returns
        -------
        GrandPy
            A new GrandPy object with analysis results added per condition.
        """
        from Py.modeling import fit_kinetics

        kinetics = fit_kinetics(data=self, fit_type=fit_type, slot=slot, return_fields=return_fields,
                                name_prefix=name_prefix, time=time, ci_size=ci_size, genes=genes,
                                max_processes=max_processes, show_progress=show_progress, **kwargs)

        new_gp = self
        for name, analysis in kinetics.items():
            new_gp = new_gp.with_analysis(name, analysis)

        return new_gp


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
        from Py.diffexp import _get_summary_matrix

        return _get_summary_matrix(self, no4sU, columns, average)





def anndata_to_grandpy(anndata: ad.AnnData, transpose: bool = True) -> "GrandPy":
        """
        Create a GrandPy instance from an AnnData instance.

        Parameters
        ----------
        anndata:
            The AnnData to convert.

        transpose:
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
