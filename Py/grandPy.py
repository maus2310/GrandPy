import re
import warnings
from typing import Any, Union, Sequence, Literal, Mapping, Callable
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from Py.utils import _ensure_list, _make_unique
from Py.slot_manager import SlotManager, ModeSlot
from Py.plot_manager import PlotManager, Plot
from Py.analysis_manager import AnalysisManager


class GrandPy:
    """
    Create a GrandPy object.

    Data is typically loaded using the `read_grand()` function, which parses preprocessed GrandR-compatible
    data formats into a usable GrandPy object.

    The Object is designed to be immutable. Changes are made through `with_...` methods.

    Simple getters are implemented as properties, more complex ones through `get_...` methods.

    Subsetting is similar to pandas.

    Examples
    --------
    Read a GrandPy Object from a file.

    >>> import GrandPy as gp
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
    prefix: str
        Path to the data file.
    gene_info: pd.DataFrame
        Genes and their metadata.
    coldata: pd.DataFrame
        Samples and their metadata.
    slots: dict[str, Union[np.ndarray, pd.DataFrame, sp.csr_matrix]]]
        Name and the corresponding data matrix.
    metadata: dict[str, Any]
        Metadata about the data and file.
    analyses: dict[str, Any]
        Results from analyzing functions.
    plots: dict[str, Any]
        Plot functions. (global or gene plots)
    """

    def __init__(self,
                 prefix: str = None,
                 gene_info: pd.DataFrame = None,
                 coldata: pd.DataFrame = None,
                 slots: dict[str, Union[np.ndarray, sp.csr_matrix]] = None,
                 metadata: dict[str, Any] = None,
                 analyses: dict[str, pd.DataFrame] = None,
                 plots: dict[str, dict] = None):

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

    def _initialize_slots(self, slots=None):
        for key, matrix in slots.items():
            self._adata.layers[key] = matrix

    def _initialize_uns_data(self, prefix, metadata, analyses, plots):
        self._adata.uns['prefix'] = prefix
        self._adata.uns['metadata'] = metadata
        self._adata.uns['analyses'] = analyses if analyses is not None else {}
        self._adata.uns['plots'] = plots if plots is not None else {}

    def _ensure_no4sU_column(self):
        if 'no4sU' not in self._adata.var.columns:
            warnings.warn("No no4sU entry in coldata, assuming all samples/cells as 4sU treated!")
            self._adata.obs["no4sU"] = False


    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self._adata.uns.get('prefix', 'Unknown')}\n"
            f"{self._adata.n_obs} genes, {self._adata.n_vars} samples/cells\n"
            f"Available data slots: {self.slots}\n"
            f"Available analyses: {self.analyses}\n"
            f"Available plots: {self.plots}\n"
            f"Default data slot: {self.default_slot}\n"
        )

    def __getitem__(self, items):
        new_adata = self._adata.copy()
        new_adata = new_adata[items]
        return self.replace(anndata = new_adata)


    def replace(self,
                *,
                prefix: str = None,
                gene_info: pd.DataFrame = None,
                coldata: pd.DataFrame = None,
                slots: dict[str, Union[np.ndarray, sp.csr_matrix]] = None,
                metadata: dict[str, Any] = None,
                analyses: dict[str, Any] = None,
                plots: dict[str, Any] = None,
                anndata: ad.AnnData = None) -> "GrandPy":
        """
        This function is useful when you want to modify the GrandPy instance on your own.

        USE WITH CAUTION!

        It is not recommended to use this function directly,
        as it will replace the given parameters without sufficient checks or copying.
        This can make parts mutable if not handled correctly.

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

        See Also
        --------
            slot_data: Gives access to the raw data of each slot.
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


    # Basic properties and methods.
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

        return self.replace(metadata=new_metadata)


    # All slot methods.
    @property
    def _slot_manager(self) -> SlotManager:
        return SlotManager(self._adata, self._is_sparse)

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

        It is not recommended to use this function directly,
        as it will return Matrizes without row or column names. (in contrast to get_data() and get_table())

        If you want to use this to modify the data on your own and then add it back to the object with with_slot(),
        it is recommended to use get_table() -> with_slot() instead.

        Returns
        -------
        dict[str, Union[np.ndarray, sp.csr_matrix]]
            The data of all available slots.
        """
        return self._slot_manager.slot_data()

    def _check_slot(self, slot: str, *, allow_ntr = True) -> bool:
        """
        Checks if a given slot exists in the data slots.

        Parameters
        ----------
        slot: str
            The slot to be checked.

        allow_ntr:
            If True, the slot "ntr" is allowed as input.

        Returns
        -------
        bool:
            True if the slot exists, False otherwise.
        """
        return self._slot_manager.check_slot(slot, allow_ntr=allow_ntr)

    def _resolve_mode_slot(self, mode_slot: Union[str, ModeSlot], *, allow_ntr = True) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Checks whether the given slot is valid and computes the resulting mode slot if a mode was specified.

        Mode slots can be specified in the following formats: ModeSlot('<mode>', '<slot>'), '<mode>_<slot>', or '<slot>'.

        Parameters
        ----------
        mode_slot: Union[str, ModeSlot]
            A slot or a mode slot to be resolved.

        allow_ntr: bool
            If True, the slot "ntr" is allowed as input.

        Returns
        -------
        Union[np.ndarray, sp.csr_matrix]
            The resulting slot after the mode has been applied.
        """
        return self._slot_manager.resolve_mode_slot(mode_slot, allow_ntr=allow_ntr)

    def with_dropped_slots(self, slots_to_remove: Union[str, Sequence[str]]) -> "GrandPy":
        """
        Returns a new GrandPy object with specified slot(s) removed.

        Parameters
        ----------
        slots_to_remove: Union[str, Sequence[str]]
            One or more slots to remove from the data.

        Returns
        ----------
        GrandPy
            A new GrandPy object with specified slot(s) removed.
        """
        slots_to_remove = _ensure_list(slots_to_remove)

        new_adata = self._slot_manager.with_dropped_slots(slots_to_remove)

        return self.replace(anndata = new_adata)

    def with_slot(self, name: str, new_slot: Union[np.ndarray, pd.DataFrame, sp.csr_matrix, list], *, set_to_default = False) -> "GrandPy":
        """
        Returns a new GrandPy Object with the new slot added.

        Recommended: use this function with DataFrames for security.

        It can only check the order of genes and samples/cells if the given matrix is a pandas DataFrame.
        Otherwise, the given matrix is expected to have rows and columns in the same order as existing slots.

        Parameters
        ----------
        name: str
            Name of the new slot.
        new_slot: Union[np.ndarray, pd.DataFrame, sp.csr_matrix]
            The data to be added as a new slot.
        set_to_default: bool
            If True, sets the new slot as the default slot.

        Returns
        -------
        GrandPy
            A new GrandPy object with the new slot added.
        """
        new_adata = self._slot_manager.with_slot(name, new_slot, set_to_default=set_to_default)

        return self.replace(anndata = new_adata)

    def with_ntr_slot(self, as_ntr: str, save_ntr_as: str = None) -> "GrandPy":
        """
        Set a different slot as the new 'ntr' slot.
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

        return self.replace(slots=new_slots)


    # All analysis methods.
    @property
    def _analysis_manager(self):
        return AnalysisManager(self._adata)

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
        get_analysis():
            Get the names of analyses matching a pattern.

        with_dropped_analyses():
            Remove analyses with a regex pattern.

        with_analyses():
            Add analyses to the object. Usually not to be used directly.
        """
        return self._analysis_manager.analyses()

    def get_analyses(self, pattern: Union[str, int, Sequence[Union[str, int, bool]]] = None, regex: bool = True) -> list[str]:
        """
        Get the names of analyses. Either by a regex, names, indices, or a boolean mask.

        Parameters
        ----------
        pattern: Union[str, int, Sequence[Union[str, int, bool]]]
            Names of analyses to be retrieved. Can be a regex, names, indices, or a boolean mask.

        regex: bool
            If True, `name` will be interpreted as a regular expression or a list of regular expressions.

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
        analyses:
            Get a list of all available analyses.

        with_dropped_analyses():
            Remove analyses with a regex pattern.

        with_analyses():
            Add analyses to the object. Usually not to be used directly.
        """
        return self._analysis_manager.get_analyses(pattern, regex=regex)

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

        by: str
            A column in the table to be used as index.

        Returns
        -------
        A new GrandPy object with added analyses.

        See Also
        --------
        analyses:
            Get the names of all stored analyses.

        get_analysis():
            Get the names of analyses matching a pattern.

        with_dropped_analyses():
            Remove analyses with a regex pattern.
        """
        new_analyses = self._analysis_manager.with_analysis(name, table, by=by)

        return self.replace(analyses=new_analyses)

    def with_dropped_analyses(self, pattern: str = None) -> "GrandPy":
        """
        Returns a new GrandPy object with analyses matching the pattern removed.

        Parameters
        ----------
        pattern: str
            A regex pattern to match analyses.

        Returns
        -------
            A new GrandPy object with removed analyses.

        See Also
        --------
        analyses:
            Get the names of all stored analyses.

        get_analysis():
            Get the names of analyses matching a pattern.

        with_analyses():
            Add analyses to the object. Usually not to be used directly.
        """
        new_analyses = self._analysis_manager.drop_analyses(pattern)

        return self.replace(analyses=new_analyses)


    # All plot methods.
    @property
    def _plot_manager(self):
        return PlotManager(self._adata)

    @property
    def plots(self) -> dict[str, dict[str, Any]]:
        """
        Get a dictionary of available plot names.

        Returns
        -------
        dict[str, dict[str, Any]]
            A dictionary mapping plot types('gene', 'global') to plot names.
        """
        return self._plot_manager.plots()

    # Beipiel im docstring unvollständig, da wir noch keine global plot funktion haben
    def with_gene_plot(self, name: str, function: Plot) -> "GrandPy":
        """
        Returns a new GrandPy object with a gene plot added.

        Parameters
        ----------
        name: str
            A name for the plot.

        function: Plot
            A funktion, that takes a GrandPy object and a gene name as input and returns a plot.

        Returns
        -------
        GrandPy
            A new GrandPy object with a gene plot added.

        Examples
        --------
        Store the plot function in the object:

        >>> sars.with_gene_plot()

        Compute the plot when needed:

        >>> sars.plot_gene()


        See Also
        --------
        plots
            Get the names of all stored plot functions.

        plot_gene()
            Executes a stored plot function for a given gene.

        with_global_plot()
            Add a global plot.

        with_dropped_plots()
            Remove plots from the object.
        """
        new_plots = self._plot_manager.add_plot(name, "gene", function)

        return self.replace(plots=new_plots)

    # floating fehlt noch
    def with_global_plot(self, name: str, function: Plot, floating: bool = False) -> "GrandPy":
        """
        Returns a new GrandPy object with a global plot added.

        Parameters
        ----------
        name: str
            A name for the plot.

        function: Plot
            A funktion, that takes a GrandPy object as input and returns a plot.

        floating: bool
            If True, the plot will be added as a floating plot.
            Otherwise, the plot will be added as a global plot.

        Returns
        -------
        GrandPy
            A new GrandPy object with a global plot added.

        Examples
        --------
        Store the plot function in the object:

        >>> sars = sars.with_global_plot(
            ...     "scatter",
            ...     Plot(
            ...         function = plot_scatter,
            ...         parameters = {
            ...             "x": "Mock.1h.A",
            ...             "y": "SARS.1h.A",
            ...             "mode_slot": "new_count"
            ...             },
            ...         plot_type = "global"
            ...     )
            ... )

        Compute the plot when needed:

        >>> sars.plot_global("scatter")

        See Also
        --------
        plots
            Get the names of all stored plot functions.

        plot_global()
            Executes a stored global plot function.

        with_gene_plot()
            Add a gene plot.

        with_dropped_plots()
            Remove plots from the object.
        """
        if floating:
            raise NameError("Floating plots are not yet implemented.")
        else:
            new_plots = self._plot_manager.add_plot(name, "global", function)

        return self.replace(plots=new_plots)

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
        plots
            Get the names of all stored plot functions.

        with_gene_plot()
            Add a gene plot.

        plot_global()
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
        plots
            Get the names of all stored plot functions.

        with_global_plot()
            Add a global plot.

        plot_gene()
            Executes a stored plot function for a given gene.
        """
        return self._adata.uns["plots"]["global"][name](self)

    def with_dropped_plot(self, pattern: str = None) -> "GrandPy":
        """
        Returns a new GrandPy object with plot names matching the pattern removed.

        The pattern is interpreted as a regular expression.

        Parameters
        ----------
        pattern: str
            A regular expression matching plot names to be dropped.

        Returns
        -------
        GrandPy
            A new GrandPy object with plot names matching the pattern removed.

        See Also
        --------
        plots
            Get the names of all stored plot functions.

        with_gene_plot()
            Add a gene plot.

        with_global_plot()
            Add a global plot.
        """
        new_plots = self._plot_manager.drop_plot(pattern)

        return self.replace(plots=new_plots)



    # Remaining methods. Mostly methods relating to coldata, gene_info or metadata.
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

        value : Union[Mapping, pd.Series, Sequence[Any]]
            The values to assign to the column can be any iterable. Can also be a dictionary when trying to update a column.
            A Series will be matched to gene_info by its index.

        Returns
        -------
        GrandPy
            A new GrandPy object with updated gene_info.
        """
        new_gene_info = self.gene_info

        if isinstance(value, Mapping):
            new_gene_info.loc[value.keys(), column] = list(value.values())
            return self.replace(gene_info = new_gene_info)

        if isinstance(value, pd.Series):
            value.index = _make_unique(value.index, warn = False)

        new_gene_info[column] = value

        return self.replace(gene_info = new_gene_info)

    def get_index(self, genes: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, regex: bool = False) -> list[int]:
        """
        Get the index of: a gene, a list of genes, or in accordance to a boolean filter.

        Either by gene name or symbol, or by a boolean mask.

        Integers are returned unchanged.

        If names and indices are mixed, only one of them will be used. Chosen by the higher number of matches.

        Parameters
        ----------
        genes: Union[str, int, Sequence[Union[str, int, bool]]]
            Specifies which indices to return.
        regex: bool
            If True, `gene` will be interpreted as a regular expression.

        Returns
        -------
        list[int]
            A list containing the specified indices.
        """
        gene_info = self.gene_info
        index = list(range(len(gene_info.index)))

        if genes is None:
            return index

        gene_column = gene_info.get("Gene")
        symbol_column = gene_info.get("Symbol")

        # Handles regex
        if regex and isinstance(genes, str):
            mask_symbol = gene_column.astype(str).str.contains(genes, regex=True) | \
                   symbol_column.astype(str).str.contains(genes, regex=True)
            return list(np.where(mask_symbol)[0])

        genes = _ensure_list(genes)

        if any(pd.isna(genes)):
            warnings.warn("All None values were removed from the query.")
            genes = [g for g in genes if pd.notna(g)]

        # Handles boolean mask
        if isinstance(genes, list) and all(isinstance(g, (bool, np.bool_)) for g in genes):
            if len(genes) != len(index):
                raise ValueError("Length of boolean filter must match number of genes.")
            return list(np.where(genes)[0])

        # Handles integers
        if isinstance(genes, list) and all(isinstance(g, (int, np.integer)) for g in genes):
            if all(gene < len(gene_info.index) for gene in genes):
                return genes
            else:
                raise IndexError("The given index is out of range.")

        gene_list = pd.Series(genes, dtype=str)

        matches_in_gene = gene_list[gene_list.isin(gene_column)]
        matches_in_symbol = gene_list[gene_list.isin(symbol_column)]

        if len(matches_in_gene) >= len(matches_in_symbol):
            return_column = gene_column
        else:
            return_column = symbol_column

        mapping = pd.Series(index, index=return_column)
        found = gene_list[gene_list.isin(return_column)]
        missing = gene_list[~gene_list.isin(return_column)]

        if not missing.empty:
            preview = ", ".join(missing.head(5))
            more = " ..." if len(missing) > 5 else ""
            warnings.warn(f"Could not find given genes (n={len(missing)}, e.g. {preview}{more})")

        return mapping.loc[found].tolist()

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

        return self.replace(gene_info=new_gene_info)

    @property
    def genes(self) -> list[str]:
        """
        Get the gene symbols.

        These names are used as the row names of the data slots and the row names of gene_info.
        """
        return self.gene_info["Symbol"].tolist()

    def get_genes(self, gene: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, use_gene_symbols: bool = True, regex: bool = False) -> list[str]:
        """
        Get gene names or symbols.

        Either by their index, their name, a boolean mask, or a regex.

        If no genes are specified, all genes are returned.

        Parameters
        ----------
        gene: Union[str, int, Sequence[str|int|bool]]
            Genes to be retrieved.
        use_gene_symbols: bool
            If True, gene symbols will be returned. Otherwise, gene names will be returned.
        regex: bool
            If True, `genes` will be interpreted as a regular expression.

        Returns
        -------
        list[str]
            A list containing the specified genes.

        See Also
        --------
        get_columns()
            get the sample/cell names.

        get_index()
            Get the index of gene names/symbols.
        """
        if use_gene_symbols:
            if gene is None:
                return self.genes

            indices = self.get_index(gene, regex=regex)
            return self.gene_info.iloc[indices]["Symbol"].tolist()

        else:
            if gene is None:
                return self.gene_info["Gene"]

            indices = self.get_index(gene, regex=regex)
            return self.gene_info.iloc[indices]["Gene"].tolist()

    def get_significant_genes(self):
        ...


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

        Parameters
        ----------
        column : str
            The name of the column to be modified.

        value : Union[Mapping, pd.Series, Sequence[Any]]
            The values to assign to the column can be any iterable. Can also be a dictionary when trying to update a column.

        Returns
        -------
        GrandPy
            A new GrandPy object with updated coldata.
        """
        new_coldata = self.coldata

        if column in new_coldata.columns:
            if isinstance(value, Mapping):
                new_coldata.loc[value.keys(), column] = list(value.values())
                return self.replace(coldata = new_coldata)

        new_coldata[column] = value

        return self.replace(coldata = new_coldata)

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
        value: Union[str, Sequence[str], pd.Series, Mapping]
            The conditions to be set for the samples/cells. Can also construct the name from other columns in coldata, if their names are given.

        Returns
        -------
        GrandPy
            A new GrandPy object with the specified condition.
        """
        new_coldata = self._adata.var.copy()

        if isinstance(value, Mapping):
            for k, v in value.items():
                new_coldata.loc[k, "Condition"] = v
            return self.replace(coldata = new_coldata)

        value = _ensure_list(value)

        if all(v in new_coldata.columns for v in value):
            new_coldata['Condition'] = new_coldata[value].astype(str).agg(" ".join, axis=1)
        else:
            if len(value) == 1:
                value = value * 12
            elif len(value) != len(new_coldata.index):
                raise ValueError(
                    f"Number of values ({len(value)}) does not match number of samples/cells ({len(new_coldata.index)})")

            new_coldata['Condition'] = value

        return self.replace(coldata = new_coldata)

    @property
    def columns(self) -> list[str]:
        """
        Get the sample/cell names.

        These names are used as the column names of the data slots and the row names of the coldata.
        """
        return self.coldata["Name"].tolist()

    def get_columns(self, columns: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, reorder: bool = False) -> list[str]:
        """
        Get sample/cell names. Either by their index, their name, or a boolean mask.

        If no columns are specified, all sample/cell names are returned.

        Parameters
        ----------
        columns: Union[str, int, Sequence[str|int|bool]]
            Samples/cells to be retrieved.

        reorder: bool
            If True, the returned list will be in the same order as the original column data.

            Otherwise, the returned list will be in the same order as the input.

        Returns
        -------
        list[str]
            A list containing the specified samples/cells.

        See Also
        --------
        get_genes()
            get the gene symbols/names.
        """
        coldata = self.coldata

        if columns is None:
            return self.columns

        columns = _ensure_list(columns)

        if isinstance(columns[0], int):
            result = coldata.iloc[columns]["Name"]

        elif isinstance(columns[0], (str, bool)):
            result = coldata.loc[columns, "Name"]

        else:
            raise TypeError("The input must be either string, int or a boolean mask. They cannot be mixed")

        if reorder:
            result = result.reindex(self._adata.var.index).dropna()

        return list(result)

    def with_renamed_columns(self, mapping: Mapping) -> "GrandPy":
        """
        Returns a new GrandPy object with columns in all slots and corresponding rows in coldata renamed.

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
        with_swapped_columns()
            Swaps columns in coldata and the corresponding in all slots.

        get_columns()
            Returns the names of columns, either by index, name, or a boolean mask.
        """
        coldata = self.coldata

        missing = [key for key in mapping if key not in coldata.columns]
        if missing:
            warnings.warn(f"Following rows cannot be renamed, as they do not exist in coldata: {missing}")

        new_coldata = coldata.rename(mapping)

        return self.replace(coldata = new_coldata)

    def with_swapped_columns(self, column1: Union[str, int], column2: Union[str, int]) -> "GrandPy":
        """
        Returns a new GrandPy object with the two specified columns in all slots and corresponding rows in coldata swapped.

        Parameters
        ----------
        column1: Union[str, int]
            column to swap with.

        column2: Union[str, int]
            column to swap with.

        Returns
        -------
        GrandPy
            A new GrandPy object with swapped columns.

        See Also
        --------
        with_renamed_columns()
            Rename the columns of all slots.
        """
        def swap(matrix, col1, col2):
            """
            Helper function that swaps columns for ndarray or csr_matrix and rows for a DataFrame.
            """
            if isinstance(matrix, pd.DataFrame):
                rows = list(matrix.index)
                rows[col1], rows[col2] = rows[col2], rows[col1]
                return matrix.loc[rows]

            elif isinstance(matrix, np.ndarray):
                matrix = matrix.copy()
                matrix[:, [col1, col2]] = matrix[:, [col2, col1]]

            elif sp.isspmatrix_csr(matrix):
                # Convert to CSC for efficient column slicing
                csc = matrix.tocsc(copy=True)
                # Swap the columns using slicing
                idx = [csc[:, i] for i in range(csc.shape[1])]
                idx[col1], idx[col2] = idx[col2], idx[col1]
                matrix = sp.hstack(idx).tocsr()

            else:
                raise TypeError("A Matrix in the GrandPy Object has an unexpected type. Only pd.DataFrame, np.ndarray and scipy.sparse.csr_matrix are supported")

            return matrix

        if isinstance(column1, str):
            column1 = self._adata.var.columns.get_loc(column1)
        if isinstance(column2, str):
            column2 = self._adata.var.columns.get_loc(column2)

        return self.apply(swap, function_coldata=swap, col1=column1, col2=column2)

    # Immer noch nicht vollständig.
    def apply(self, function: Callable, *, function_gene_info: Callable = None, function_coldata: Callable = None,
              **kwargs) -> "GrandPy":
        """
        Returns a new GrandPy object with the given function applied to each data slot.

        Can also apply a function to the gene_info and coldata DataFrames.

        To change the order of the matrizes in the object, use subsetting instead.

        It is not advised to use this method for changing the order of columns or rows,
        as slots, gene_info, and coldata are not automatically updated when changing one of them.

        Parameters
        ----------
        function:
            Function to apply to each data slot.
        function_gene_info:
            Function to apply to the gene_info DataFrame.
        function_coldata:
            Function to apply to the coldata DataFrame.
        **kwargs:
            Additional keyword arguments to pass to the function.

        Returns
        -------
        GrandPy
            New GrandPy object with transformed data.

        """
        new_adata = self._adata.copy()
        for key in self._adata.layers.keys():
            new_adata.layers[key] = function(self._adata.layers[key], **kwargs)

        if function_gene_info is not None:
            new_adata.obs = function_gene_info(self._adata.obs, **kwargs)

        if function_coldata is not None:
            new_adata.var = function_coldata(self._adata.var, **kwargs)

        # immer noch nicht perfekt(with_analysis hat den Parameter 'by', das heißt index muss nicht immer 'Symbol')
        if new_adata.uns['analyses'] is not None:
            new_adata.uns['analyses'] = {
                key: value.reindex(index=new_adata.obs.index).dropna() for key, value in new_adata.uns['analyses'].items()
            }

        return self.replace(anndata=new_adata)

    # TODO concat() Verhalten überprüfen(dafür wäre es gut einen gänzlich anderen Datensatz zu haben)
    def concat(self, other: "GrandPy", axis: Literal["gene_info", 0, "coldata", 1] = 1) -> "GrandPy":
        """
        Concatenates the other object with the current instance along a given axis.

        Parameters
        ----------
        other: GrandPy
            The object to concatenate with the current instance.

        axis: Literal["gene_info", 0, "coldata", 1]
            The axis along which to concatenate.

        Returns
        -------
        GrandPy
            A new concatenated GrandPy object.

        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="(Observation|Variable) names are not unique.*",
                                    category=UserWarning, module="anndata")

            if axis == 0 or axis == "gene_info":
                axis = "var"
            elif axis == 1 or axis == "coldata":
                axis = "obs"
            else:
                raise ValueError(f"axis must be either 0, 'gene_info' or 1, 'coldata' not {axis}.")

            new_adata = ad.concat([self._adata, other._adata], axis=axis, merge="unique", uns_merge="unique")

            if axis == "obs":
                new_adata.obs_names_make_unique("_")
            else:
                new_adata.var_names_make_unique("_")

        return self.replace(anndata=new_adata)


    # Doch eher wie slot_data? Anndata Object ist denke ich die Mühe nicht wert.
    def get_matrix(self,
                   mode_slot: Union[str, ModeSlot] = None,
                   genes: Union[str, int, Sequence[Union[str, int]]] = None,
                   columns: Union[str, int, Sequence[Union[str, int]]] = None,) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Get the raw data from a data slot, without row or column names.

        This function is mostly not needed, as get_table(), get_data() or apply() are usually better suited.

        Parameters
        ----------
        mode_slot: Union[str, ModeSlot]
            The name of the data slot. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: Union[str, int, Sequence[Union[str, int]]]
            The genes to be retrieved. Either by gene symbols, names(Ensembl ids), indices or a boolean mask.

        columns: Union[str, int, Sequence[Union[str, int]]]
            The samples/cells to be retrieved. Either by names, indices or a boolean mask.

        Returns
        -------
        Union[np.ndarray, sp.csr_matrix]
            A raw data matrix, without column or row names.

        See Also
        --------
        get_table()
            Similar to get_matrix(), but with row and column names and coldata can be concatenated.

        get_data()
            Similar to get_data(), but slots are transposed, so gene_info can be concatenated.
        """
        if mode_slot is None:
            mode_slot = self.default_slot

        data = self._resolve_mode_slot(mode_slot)

        if genes is None and columns is None:
            return data

        row_indices = self.get_index(genes)
        column_indices = [self._adata.var.index.get_loc(column) for column in self.get_columns(columns)]

        if self._is_sparse:
            data_subset = data[row_indices, :][:, column_indices]

        else:
            data_subset = data[np.ix_(row_indices, column_indices)]

        return data_subset

    # TODO get_data() um die fehlenden Parameter aus R erweitern. (ntr.na, by.rows)
    def get_data(self,
                 mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
                 genes: Union[str, int, Sequence[Union[str, int]]] = None,
                 columns: Union[str, int, Sequence[Union[str, int]]] = None,
                 *,
                 with_coldata: bool = True,
                 name_genes_by = "Symbol") -> pd.DataFrame:
        """
        Get a DataFrame containing the data from data slots, optionally with the corresponding coldata.

        Parameters
        ----------
        mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]]
            The name of the data slots. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: Union[str, int, Sequence[Union[str, int]]]
            The genes to be retrieved. Either by gene symbols, names(Ensembl ids), or indices.

        columns: Union[str, int, Sequence[Union[str, int]]]
            The samples/cells to be retrieved. Either by names or indices.

        with_coldata: bool
            If True, the coldata DataFrame will be concatenated to the result.

        name_genes_by: str
            A column in the gene_info DataFrame to be used as the name of the genes.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified data for the genes and columns.

        See Also
        --------
        get_table():
            Similar to get_data(), but slots are transposed, so gene_info can be concatenated.
        """
        coldata = self.coldata
        gene_info = self.gene_info

        if mode_slots is None:
            mode_slots = self.default_slot

        mode_slots = _ensure_list(mode_slots)

        row_indices = [coldata.index.get_loc(column) for column in self.get_columns(columns)]
        column_indices = self.get_index(genes) if genes is not None else self.get_index(None)

        # row_names is handled like this, so that the index remains unique ('Symbol' was not made unique, but index was)
        row_names = coldata.iloc[row_indices]["Name"].tolist()
        column_names = gene_info.iloc[column_indices].index.tolist() if name_genes_by == "Symbol" else gene_info.iloc[column_indices][name_genes_by].tolist()

        result_df = pd.DataFrame()

        for slot_name in mode_slots:
            all_data = self._resolve_mode_slot(slot_name).T
            data_subset = all_data[np.ix_(row_indices, column_indices)]

            if self._is_sparse:
                data_subset = np.array(data_subset)

            if len(mode_slots) > 1:
                local_column_names = [name + "_" + slot_name.__str__() for name in column_names]
            else:
                local_column_names = column_names

            processed_data = pd.DataFrame(data_subset, index=row_names, columns=local_column_names)

            result_df = pd.concat([result_df, processed_data], axis=1)


        if with_coldata:
            result_df = pd.concat([coldata.iloc[row_indices], result_df], axis=1)

        return result_df

    # TODO get_table() um die fehlenden Parameter aus R erweitern(ntr.na, summarize, prefix, reorder.columns). mode_slot soll auch noch ein regex sein können(der mit analysis names verglichen wird).
    def get_table(self,
                  mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
                  genes: Union[str, int, Sequence[Union[str, int]]] = None,
                  columns: Union[str, int, Sequence[Union[str, int]]] = None,
                  *,
                  with_gene_info: bool = False,
                  name_genes_by = "Symbol") -> pd.DataFrame:
        """
        Get a DataFrame containing the data from data slots, optionally with the corresponding gene_info.

        Parameters
        ----------
        mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]]
            The name of the data slots. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: Union[str, int, Sequence[Union[str, int]]]
            The genes to be retrieved. Either by gene symbols, names(Ensembl ids), or indices.

        columns: Union[str, int, Sequence[Union[str, int]]]
            The samples/cells to be retrieved. Either by names or indices.

        with_gene_info: bool
            If True, the gene_info DataFrame will be concatenated to the result.

        name_genes_by: str
            A column in the gene_info DataFrame to be used as the name of the genes.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified data for the genes and columns.

        See Also
        --------
        get_data():
            Similar to get_table(), but slots are transposed, so coldata can be concatenated.
        """
        coldata = self.coldata
        gene_info = self.gene_info

        if mode_slots is None:
            mode_slots = self.default_slot

        mode_slots = _ensure_list(mode_slots)

        for slot in mode_slots:
            if isinstance(slot, ModeSlot):
                slot = slot.__str__()

        row_indices = self.get_index(genes)
        column_indices = [coldata.index.get_loc(column) for column in self.get_columns(columns)]

        row_names = gene_info.iloc[row_indices][name_genes_by].tolist()
        column_names = coldata.iloc[column_indices]["Name"].tolist()

        result_df = pd.DataFrame()

        for slot_name in mode_slots:
            all_data = self._resolve_mode_slot(slot_name)
            data_subset = all_data[np.ix_(row_indices, column_indices)]

            if self._is_sparse:
                data_subset = data_subset.toarray()

            if len(mode_slots) > 1:
                local_column_names = [name + "_" + slot_name.__str__() for name in column_names]
            else:
                local_column_names = column_names

            processed_data = pd.DataFrame(data_subset, index=row_names, columns=local_column_names)

            result_df = pd.concat([result_df, processed_data], axis=1)

        if with_gene_info:
            result_df = pd.concat([gene_info.iloc[row_indices], result_df], axis=1)

        return result_df

    def get_analysis_table(self,
                           analyses: Union[str, int, Sequence[Union[str, int, bool]]] = None,
                           genes: Union[str, int, Sequence[Union[str, int]]] = None,
                           columns: str = None,
                           *,
                           regex: bool = True,
                           with_gene_info: bool = True,
                           name_genes_by: str = "Symbol") -> pd.DataFrame:
        """
        Get a DataFrame containing analysis tables, optionally with the corresponding gene_info.

        Parameters
        ----------
        analyses: Union[str, int, Sequence[Union[str, int, bool]]]
            The analyses to be retrieved. Either by name, index, or a boolean mask.

        genes: Union[str, int, Sequence[Union[str, int]]]
            The genes for which to retrieve the analysis tables. Either by gene symbols, names(Ensembl ids), or indices.

        columns: str
            A regular expression to match the name of the columns in the analysis tables.

        regex: bool
            If True, 'analyses' will be interpreted as a regular expression.

        with_gene_info: bool
            If True, the gene_info DataFrame will be concatenated to the result.

        name_genes_by: str
            The name of the column in the gene_info DataFrame to be used as the name of the genes.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified analyses for the genes and columns.

        See Also
        --------
        analyses
            Get the names of all stored analyses.

        get_analyses()
            Get the names of analyses. Either by a regex, names, indices, or a boolean mask.

        with_analysis()
            Add a new analysis to the object.

        with_dropped_analysis()
            Drop analyses from the object with a regex.
        """
        analyses = self.get_analyses(analyses, regex = regex)

        row_indices = self.get_index(genes)
        row_names = self._adata.obs.iloc[row_indices][name_genes_by].tolist()

        result_df = pd.DataFrame()

        for name in analyses:
            analysis_data = self._adata.uns["analyses"][name]
            analysis_data_subset = np.array(analysis_data)[row_indices]

            analysis_column_names = [name + "_" + column for column in analysis_data.columns]

            processed_data = pd.DataFrame(analysis_data_subset, index=row_names, columns=analysis_column_names)

            # Only take columns that match to 'columns' if it is specified.
            if columns is not None:
                matching_cols = [col for col in processed_data.columns if re.search(columns, col)]
                result_df = pd.concat([result_df, processed_data[matching_cols]], axis=1)
            else:
                result_df = pd.concat([result_df, processed_data], axis=1)

        if with_gene_info:
            result_df = pd.concat([self._adata.obs.iloc[row_indices], result_df], axis=1)

        return result_df


    def find_references(self):
        ...


    def compute_ntr_ci(self, ci_size: float = 0.95, name_lower: str = "lower", name_upper: str = "upper"):
        from Py.processing import _compute_ntr_ci

        return _compute_ntr_ci(self, ci_size, name_lower, name_upper)
