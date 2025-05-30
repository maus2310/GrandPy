import warnings
from typing import Any, Union, Sequence
import numpy as np
import pandas as pd
import anndata as ad

import scipy.sparse as sp


class ModeSlot:
    """
    Used to store a mode slot.

    Modes can either be 'new', 'old' or 'total'.

    For new the data slot value is multiplied by ntr.
    For old the data slot value is multiplied by 1-ntr.

    Parameters
    ----------
    mode: str
        A mode string. Can either be 'new', 'old' or 'total'.

    slot: str
        An available slot.

    """
    def __init__(self, mode: str, slot: str):
        self._set_mode(mode)
        self.slot = slot

    def __str__(self):
        return f"{self.mode}_{self.slot}"

    def _set_mode(self, mode):
        if mode == "n" or mode == "new":
            self.mode = "new"
        elif mode == "o" or mode == "old":
            self.mode = "old"
        elif mode == "t" or mode == "total" or mode == "" or mode is None:
            self.mode = "total"
        else:
            raise ValueError(f"Invalid mode: {mode}. Can either be 'new', 'old' or 'total'.")


class GrandPy:
    """
    Create a GrandPy object.

    Data is typically loaded using the `read_grand()` function, which parses preprocessed GrandR-compatible
    data formats into a usable GrandPy object.

    Parameters
    ----------
    prefix: str
        Path to the data file.
    gene_info: pandas DataFrame
        Genes and their metadata.
    coldata: pandas DataFrame
        Samples and their metadata.
    slots: dict
        Name and the corresponding data matrix.
    metadata: dict
        Metadata about the data and file.
    analyses:

    plots:

    """

    def __init__(self,
                 prefix: str = None,
                 gene_info: pd.DataFrame = None,
                 coldata: pd.DataFrame = None,
                 slots: dict = None,
                 metadata: dict = None,
                 analyses = None,
                 plots = None):

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
        self._is_sparse = True if isinstance(slots.get("count"), sp.csr_matrix) else False

        self._initialize_slots(slots)
        self._initialize_uns_data(prefix, metadata, analyses, plots)
        self._ensure_no4sU_column()

    def _initialize_slots(self, slots=None):
        if slots is not None:
            for key, matrix in slots.items():
                self._adata.layers[key] = matrix

    def _initialize_uns_data(self, prefix, metadata, analyses, plots):
        self._adata.uns['prefix'] = prefix
        self._adata.uns['metadata'] = metadata
        self._adata.uns['analyses'] = analyses
        self._adata.uns['plots'] = plots

    def _ensure_no4sU_column(self):
        if 'no4sU' not in self._adata.var.columns:
            warnings.warn("No no4sU entry in coldata, assuming all samples/cells as 4sU treated!")
            self._adata.obs["no4sU"] = False


    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self._adata.uns.get('prefix', 'Unknown')}\n"
            f"{self._adata.n_obs} genes, {self._adata.n_vars} samples/cells\n"
            f"Available data slots: {', '.join(self._adata.layers.keys()) or {} or 'None'}\n"
            f"Available analyses: {', '.join(self._adata.uns.get('analyses') or {}) or 'None'}\n"
            f"Available plots: {', '.join(self._adata.uns.get('plots') or {}) or 'None'}\n"
            f"Default data slot: {self._adata.uns['metadata'].get('default_slot', None)}\n"
        )

    def __getitem__(self, items):
        new_adata = self._adata.copy()
        new_adata = new_adata[items]
        return self._replace(new_adata)


    def _replace(self, adata: ad.AnnData = None,
                 *,
                 prefix: str = None,
                 gene_info: pd.DataFrame = None,
                 coldata: pd.DataFrame = None,
                 slots: dict[str, Union[np.ndarray, sp.csr_matrix]] = None,
                 metadata: dict = None) -> "GrandPy":
        """
        Funktion to create a new instance of GrandPy from an already existing one.
        """
        def safe_copy(obj):
            return obj.copy() if obj is not None else None

        if adata is None:
            adata = self._adata

        return self.__class__(
            prefix = prefix if prefix is not None else adata.uns.get('prefix'),
            gene_info = gene_info if gene_info is not None else safe_copy(adata.obs),
            coldata = coldata if coldata is not None else safe_copy(adata.var),
            slots = slots if slots is not None else {**adata.layers},
            metadata = metadata if metadata is not None else (adata.uns.get("metadata",)),
            analyses = safe_copy(adata.uns.get("analyses")),
            plots = safe_copy(adata.uns.get("plots"))
        )


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

        return self._replace(metadata=new_metadata)


    @property
    def slots(self) -> dict[str, Union[np.ndarray, sp.csr_matrix]]:
        """
        Get the names and data of all available slots.
        """
        return self._adata.layers.copy()

    @property
    def slot_names(self) -> list[str]:
        """
        Get the names of all available slots.
        """
        return list(self._adata.layers.keys())

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

        to_remove = _ensure_list(slots_to_remove)
        current_slots = self.slot_names
        remaining = [s for s in current_slots if s not in to_remove]

        if not remaining:
            raise ValueError("Cannot drop all slots - at least one must remain.")

        new_slots = self.slots
        new_slots = {k: self._adata.layers[k] for k in remaining}

        if self.default_slot in to_remove:
            new_metadata = self._adata.uns.get('metadata', {}).copy()
            new_metadata['default_slot'] = remaining[0]
            return self._replace(slots=new_slots, metadata=new_metadata)

        return self._replace(slots=new_slots)

    def with_slot(self, name: str, new_slot: Union[np.ndarray, pd.DataFrame, sp.csr_matrix, list], *, set_to_default = False) -> "GrandPy":
        """
        Returns a new GrandPy Object with the new slot added.

        Can only check the order of genes and samples/cells if the given matrix is a pandas DataFrame.
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
        if name in self._adata.layers.keys():
            raise ValueError(f"Slot '{name}' already exists. Please choose a different name.")

        def _validate_and_convert_new_data(matrix: Union[pd.DataFrame, sp.csr_matrix, np.ndarray]) -> Union[np.ndarray, sp.csr_matrix]:
            # If DataFrame → to NumPy
            if isinstance(matrix, pd.DataFrame):
                matrix.index = _make_unique(pd.Series(matrix.index))
                # Row and column names of the new matrix must be equal to the existing ones.
                try:
                    for i in range(self._adata.n_obs):
                        if matrix.index[i] != self._adata.obs["Symbol"].iloc[i]:
                            raise ValueError(f"Row name mismatch for slot '{name}' at index {i}")
                    for i in range(self._adata.n_vars):
                        if matrix.columns[i] != self._adata.var["Name"].iloc[i]:
                            raise ValueError(f"Column name mismatch for slot '{name}' at index {i}")
                except ValueError as error:
                    warnings.warn(f"The row and column names of the new matrix did not match the existing ones. Data will be saved regardless.\n{error}")

                matrix = matrix.values

            # If sparse, but not csr → to csr
            if self._is_sparse and not isinstance(matrix, sp.csr_matrix):
                matrix = sp.csr_matrix(matrix)

            # If dense, but not ndarray → to ndarray
            if not self._is_sparse and not isinstance(matrix, np.ndarray):
                try:
                    matrix = np.array(matrix)
                except:
                    raise TypeError("Matrix must be ndarray, DataFrame, or scipy sparse matrix")

            return matrix

        new_slot = _validate_and_convert_new_data(new_slot)

        new_slots = self.slots
        new_slots[name] = new_slot

        if set_to_default:
            new_metadata = self._adata.uns.get('metadata', {}).copy()
            new_metadata['default_slot'] = name
            return self._replace(slots = new_slots, metadata = new_metadata)

        return self._replace(slots = new_slots)


    @property
    def condition(self) -> list[str]:
        """
        Get the condition of all samples/cells in the coldata.
        """
        return self.coldata['Condition'].tolist()

    #noch nicht Fertig
    def with_condition(self, value: Union[str, list[str], pd.Series]) -> "GrandPy":
        """

        Parameters
        ----------
        value: Union[str, list[str], pd.Series]
            The condition to be set for the samples/cells.

        Returns
        -------
        GrandPy
            A new GrandPy object with the specified condition.
        """

        value = [value] if isinstance(value, str) else value
        new_adata = self._adata.copy()

        if all(v in self.coldata.columns for v in value):

        #Verhalten momentan noch anders als in GrandR, Name kann nicht benutzt werden, da wir diesen als Index Speichern.

            new_adata.obs['Condition'] = self.coldata[value].astype(str).agg(" ".join, axis=1)
        else:
            #momentan funktioniert die Funktion nur, wenn die Länge von values gleich der Länge des Indexes von coldata ist.
            if len(value) != len(self.coldata.index):
                raise ValueError(
                    f"Number of values ({len(value)}) does not match number of samples/cells ({len(self.coldata.index)})")

            new_adata.obs['Condition'] = pd.Series(value, index=self.coldata.index)

        return self._replace(new_adata)


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

    def with_gene_info(self, column: str, value: Union[dict, pd.Series, pd.DataFrame, Sequence[Any]]) -> "GrandPy":
        """
        Returns a new object with modified gene_info. If the column name does not already exist, a new column will be added.

        Otherwise, the column will be replaced by the given value or updated if a dictionary was given.

        Parameters
        ----------
        column : str
            The name of the column to be modified.

        value : Union[dict, pd.Series, pd.DataFrame, Sequence[Any]]
            The values to assign to the column can be any iterable. Can also be a dictionary when trying to update a column.

        Returns
        -------
        GrandPy
            A new GrandPy object with updated gene_info.
        """
        new_gene_info = self.gene_info

        if isinstance(value, dict):
            new_gene_info.loc[value.keys(), column] = list(value.values())
            return self._replace(gene_info = new_gene_info)

        # Reorders DataFrames and Series to match gene_info.
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.reindex(new_gene_info.index)

        # If the column exists, it will be replaced, otherwise a new one will be added.
        new_gene_info[column] = value

        return self._replace(gene_info = new_gene_info)


    @property
    def coldata(self) -> pd.DataFrame:
        """
        Get the coldata DataFrame.
        """
        return self._adata.var.copy()

    def with_coldata(self, column: str, value: Union[dict, pd.Series, pd.DataFrame, Sequence[Any]]) -> "GrandPy":
        """
        Returns a new object with modified coldata. If the column name does not already exist, a new column will be added.

        Otherwise, the column will be replaced by the given value or updated if a dictionary was given.

        Parameters
        ----------
        column : str
            The name of the column to be modified.

        value : Union[dict, pd.Series, pd.DataFrame, Sequence[Any]]
            The values to assign to the column can be any iterable. Can also be a dictionary when trying to update a column.

        Returns
        -------
        GrandPy
            A new GrandPy object with updated coldata.
        """
        new_coldata = self.coldata

        if column in new_coldata.columns:
            if isinstance(value, dict):
                # updates column in correspondence to the dictionary
                new_coldata.loc[value.keys(), column] = list(value.values())
                return self._replace(coldata = new_coldata)

        # Reorders DataFrames and Series to match the order of coldata.
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.reindex(new_coldata.index)

        # If the column exists, it will be replaced, otherwise a new one will be added.
        new_coldata[column] = value

        return self._replace(coldata = new_coldata)

    # TODO apply() vervollständigen
    def apply(self, function, *, function_gene_info=None, function_coldata=None, **kwargs) -> "GrandPy":
        """
        Returns a new GrandPy object with the given function applied to each data slot.\n
        Can also apply a function to the gene_info and coldata DataFrames.

        Parameters
        ----------
        function:
            Function to apply to each data slat (receives each matrix individually).
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
            new_adata.var = function_gene_info(self._adata.var, **kwargs)
        if function_coldata is not None:
            new_adata.obs = function_coldata(self._adata.obs, **kwargs)

        # Noch nicht vollständig

        if self._adata.uns['analyses'] is not None:
            ...

        return self._replace(new_adata)


    @property
    def genes(self) -> list[str]:
        """
        Get the gene symbols, which are both: the column names of the data slots and the row names of gene_info.
        """
        return self.gene_info["Symbol"]

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
        get_index: Get the index of gene names or symbols.
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


    @property
    def columns(self) -> list[str]:
        """
        Get the sample/cell names

        These names are used as the column names of the data slots and the row names of the coldata.

        Returns
        -------
        list[str]
            A list of sample/cell names

        See Also
        --------
        coldata : get the entire coldata DataFrame
        """
        return self.coldata["Name"].tolist()

    def get_columns(self, samples_or_cells: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, reorder: bool = False) -> list[str]:
        """
        Get sample/cell names. Either by their index, their name, or a boolean mask.

        If no samples_or_cells are specified, all are returned.

        Parameters
        ----------
        samples_or_cells: Union[str, int, Sequence[str|int|bool]]
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
        get_genes: get the gene symbols/names(columns of the data slots)
        """
        coldata = self.coldata

        if samples_or_cells is None:
            return self.columns

        samples_or_cells = _ensure_list(samples_or_cells)

        # Tries to search by index
        try:
            result = list(coldata.iloc[samples_or_cells]["Name"])
        except:
            # Tries to search by name or boolean mask
            try:
                result = list(coldata.loc[samples_or_cells, "Name"])
            except:
                raise TypeError("The input must be either string, int or a boolean mask. They cannot be mixed")

        return result


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

        # Alternative Möglichkeit:

        # # Finds all matches in 'Symbol'
        # result = [gene_info.index.get_loc(idx) for idx in gene_info[gene_info["Symbol"].isin(genes)].index]
        #
        # if not result:
        #     # Finds all matches in 'Gene'
        #     result = [gene_info.index.get_loc(idx) for idx in gene_info[gene_info["Gene"].isin(genes)].index]
        #
        # return result


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
        if not allow_ntr and slot == "ntr":
            return False
        return slot in self.slot_names

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

        def parse_mode_slot(mode_slot_unparsed: str) -> ModeSlot:
            """
            Helper function to parse a mode_slot string.
            """
            mode_slot_candidate = mode_slot_unparsed.split("_", 1)

            if len(mode_slot_candidate) == 1:
                return ModeSlot("total", mode_slot_unparsed)

            if len(mode_slot_candidate) != 2:
                raise ValueError(
                    f"Invalid mode_slot: '{mode_slot_unparsed}'. Expected format: '<mode>_<slot>' or ModeSlot('<mode>', '<slot>').")

            mode, slot = mode_slot_candidate

            return ModeSlot(mode, slot)

        def one_minus_csr_matrix(matrix: sp.csr_matrix) -> sp.csr_matrix:
            """
            Helper funktion to compute one minus a sparse matrix.
            """
            ones = sp.csr_matrix(np.ones(matrix.shape), dtype=matrix.dtype)

            return ones - matrix

        # if mode_slot is a string, it gets parsed into a ModeSlot Object
        if isinstance(mode_slot, str):
            mode_slot = parse_mode_slot(mode_slot)

        if not self._check_slot(mode_slot.slot, allow_ntr=allow_ntr):
            raise ValueError(f"Slot '{mode_slot.slot}' not found in data slots.")

        slot = self._adata.layers[mode_slot.slot]
        ntr = self._adata.layers["ntr"]

        resulting_mode_slot = slot

        # The resulting data is computed, depending on the mode
        if mode_slot.mode != "total":
            if self._is_sparse:
                resulting_mode_slot = slot.multiply(ntr) if mode_slot.mode == "new" else slot.multiply(one_minus_csr_matrix(ntr))
            else:
                resulting_mode_slot = slot * ntr if mode_slot.mode == "new" else slot * (1 - ntr)

        return resulting_mode_slot


    # TODO: get_data() um die fehlenden Parameter aus R erweitern. (ntr.na, by.rows)
    def get_data(self,
                 mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
                 genes: Union[str, int, Sequence[Union[str, int]]] = None,
                 samples_or_cells: Union[str, int, Sequence[Union[str, int]]] = None,
                 *,
                 with_coldata: bool = True,
                 name_genes_by = "Symbol") -> pd.DataFrame:
        """
        Get a DataFrame containing the data from data slots.

        Parameters
        ----------
        mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]]
            The name of the data slots. If None, uses the default slot.

            A mode("new"|"old"|"total") can be specified in the following formats: ModeSlot('<mode>', '<slot>') or '<mode>_<slot>'

        genes: Union[str, int, Sequence[Union[str, int]]]
            The genes to be retrieved. Can be gene symbols, names, or an index.

        samples_or_cells: Union[str, int, Sequence[Union[str, int]]]
            The cells/samples to be retrieved. Either by name or index.

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

        get_analysis_table():

        """
        coldata = self.coldata
        gene_info = self.gene_info

        if mode_slots is None:
            mode_slots = self.default_slot

        # Transforming all parameters into lists
        mode_slots = _ensure_list(mode_slots)
        genes = _ensure_list(genes)
        samples_or_cells = _ensure_list(samples_or_cells)

        # Retrieving the indices of the selected genes and columns
        row_indices = [coldata.index.get_loc(column) for column in self.get_columns(samples_or_cells)] if samples_or_cells != [None] else range(len(coldata))
        column_indices = self.get_index(genes) if genes != [None] else self.get_index(None)

        # Retrieving the names of the selected genes and columns
        row_names = coldata.iloc[row_indices]["Name"].tolist()
        column_names = gene_info.iloc[column_indices][name_genes_by].tolist()

        result_df = pd.DataFrame()

        for slot_name in mode_slots:
            all_data = self._resolve_mode_slot(slot_name)

            data_subset = all_data[np.ix_(column_indices, row_indices)].T
            local_column_names = [name +  "_" + slot_name for name in column_names]
            processed_data = pd.DataFrame(data_subset, index=row_names, columns=local_column_names)

            result_df = pd.concat([result_df, processed_data], axis=1)


        if with_coldata:
            result_df = pd.concat([coldata.iloc[row_indices], result_df], axis=1)

        return result_df

    # TODO: get_table() um die fehlenden Parameter aus R erweitern. mode_slot soll auch noch ein regex sein können(der mit analysis names verglichen wird).
    def get_table(self,
                  mode_slots: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
                  genes: Union[str, int, Sequence[Union[str, int]]] = None,
                  samples_or_cells: Union[str, int, Sequence[Union[str, int]]] = None,
                  *,
                  with_gene_info: bool = False,
                  name_genes_by = "Symbol") -> pd.DataFrame:
        """

        """
        coldata = self.coldata
        gene_info = self.gene_info

        if mode_slots is None:
            mode_slots = self.default_slot

        mode_slots = _ensure_list(mode_slots)
        genes = _ensure_list(genes)
        samples_or_cells = _ensure_list(samples_or_cells)

        row_indices = self.get_index(genes) if genes != [None] else self.get_index(None)
        column_indices = [coldata.index.get_loc(column) for column in self.get_columns(samples_or_cells)] if samples_or_cells != [None] else range(len(coldata))

        row_names = gene_info.iloc[row_indices][name_genes_by].tolist()
        column_names = coldata.iloc[column_indices]["Name"].tolist()

        result_df = pd.DataFrame()

        for slot_name in mode_slots:
            all_data = self._resolve_mode_slot(slot_name)

            data_subset = all_data[np.ix_(row_indices, column_indices)]
            local_column_names = [name + "_" + slot_name for name in column_names]
            processed_data = pd.DataFrame(data_subset, index=row_names, columns=local_column_names)

            result_df = pd.concat([result_df, processed_data], axis=1)

        if with_gene_info:
            result_df = pd.concat([gene_info.iloc[row_indices], result_df], axis=1)

        return result_df



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


def _make_unique(series: pd.Series) -> pd.Series:
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

    for val in series:
        if val not in counts:
            counts[val] = 0
            result.append(val)
        else:
            counts[val] += 1
            result.append(f"{val}_{counts[val]}")
    return pd.Series(result, index=series.index)


def _ensure_list(x):
    if isinstance(x, (str, int, bool)) or x is None:
        return [x]
    return list(x)

