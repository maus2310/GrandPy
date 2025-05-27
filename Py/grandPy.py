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
    GrandPy: A Python implementation of the GrandR data model for RNA labeling analysis.

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

        # obs and var are swapped to allow the data to be gene * sample
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


    def _replace(self, adata: ad.AnnData) -> "GrandPy":
        def _safe_copy(obj):
            return obj.copy() if obj is not None else None

        return self.__class__(
            prefix = adata.uns.get('prefix'),
            gene_info = _safe_copy(adata.obs),
            coldata = _safe_copy(adata.var),
            slots = {**adata.layers},
            metadata = _safe_copy(adata.uns.get("metadata",)),
            analyses = _safe_copy(adata.uns.get("analyses")),
            plots = _safe_copy(adata.uns.get("plots"))
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

        new_adata = self._adata.copy()
        new_adata.uns['metadata']['default_slot'] = value

        return self._replace(new_adata)


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

        to_remove = [slots_to_remove] if isinstance(slots_to_remove, str) else slots_to_remove
        current_slots = list(self._adata.layers.keys())
        remaining = [s for s in current_slots if s not in to_remove]

        if not remaining:
            raise ValueError("Cannot drop all slots - at least one must remain.")

        new_adata = self._adata.copy()
        new_adata.layers = {k: self._adata.layers[k] for k in remaining}

        if self.default_slot in to_remove:
            new_adata.uns['metadata']['default_slot'] = remaining[0]

        return self._replace(new_adata)

    def with_slot(self, name: str, matrix: Union[np.ndarray, pd.DataFrame, sp.csr_matrix], *, set_to_default = False) -> "GrandPy":
        """
        Returns a new GrandPy Object with the new slot added.

        The matrix given is expected to have rows and columns in the same order as existing slots.

        Parameters
        ----------
        name: str
            Name of the new slot.
        matrix: Union[np.ndarray, pd.DataFrame, sp.csr_matrix]
            The data to be added as a new slot.
        set_to_default: bool
            If True, sets the new slot as the default slot.

        Returns
        -------

        """
        if name in self._adata.layers.keys():
            raise ValueError(f"Slot '{name}' already exists. Please choose a different name.")

        if self._is_sparse:
            matrix = _to_sparse(matrix)
        else:
            matrix = _validate_and_convert_new_data(matrix)

        new_adata = self._adata.copy()

        new_adata.layers[name] = matrix

        if set_to_default:
            new_adata.uns['metadata']['default_slot'] = name

        return self._replace(new_adata)


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

    def get_gene_info(self, columns: Union[str, Sequence[str]] = None) -> pd.DataFrame:
        """
        Get a subset of the gene_info DataFrame.

        Parameters
        ----------
        columns: Union[str, Sequence[str]]
            A column name or a list of column names to be retrieved from the gene_info DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified columns from the gene_info DataFrame.

        See Also
        --------
        with_gene_info: Set a subset of the gene_info DataFrame
        """
        df_gene_info = self.gene_info

        if columns is None:
            return df_gene_info

        if isinstance(columns, str):
            columns = [columns]

        return df_gene_info[list(columns)]

    # ist noch nicht vollständig/fehlerhaft
    def with_gene_info(self, column: str, value: Any) -> "GrandPy":
        """
        Returns a new GrandPy object with the specified column in gene_info set to the specified value.

        Examples
        --------
        gp.with_gene_info("Condition", {"gene_1": "Control", "gene_2": "Treatment"})

        Parameters
        ----------
        column: Any
            The column to be modified.

        value: Any
            The value to be set for the specified column. Can be a single value, a dictionary, a Series, or a DataFrame.

        Returns
        -------
        GrandPy
            A new GrandPy object with the specified column in gene_info set to the specified value.

        See Also
        --------
        get_gene_info: Get a subset of the gene_info DataFrame
        """
        new_adata = self._adata.copy()
        df_gene_info = new_adata.obs
        column_to_change = new_adata.obs.columns.get_loc(column)

        if isinstance(value, dict):
            indices = self.get_index(value.keys())
            new_adata.obs.iloc[indices, column_to_change] = list(value.values())
            return self._replace(new_adata)

        if isinstance(value, (pd.Series, pd.DataFrame)):
            indices = self.get_index(value.index)
            new_adata.obs.iloc[indices, column_to_change] = value
            return self._replace(new_adata)

        if not (np.isscalar(value) or len(value) == len(df_gene_info)):
            raise ValueError(f"Value has wrong length: {len(value)} vs {len(df_gene_info)} rows in gene_info")

        df_gene_info[column] = value

        return self._replace(new_adata)


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
    def coldata(self) -> pd.DataFrame:
        """
        Get the coldata DataFrame.
        """
        return self._adata.var.copy()

    # aktuell extrem schlecht
    def with_coldata(self, column: Union[str, pd.Series, pd.DataFrame], value: Sequence[Any] = None) -> "GrandPy":
        """
        Return a new object with modified coldata.

        Parameters
        ----------
        column : Union[str, pd.Series, pd.DataFrame]
            If string, sets or updates the column with the given values.

            If DataFrame or Series, concatenates the new columns to the existing coldata.(value is ignored)

        value : Sequence[Any]
            The values to assign to the column can be list, array, or Series.

        Returns
        -------
        GrandPy
            A new GrandPy object with updated coldata.
        """
        obs = self.coldata
        new_adata = self._adata.copy()

        if isinstance(column, (pd.DataFrame, pd.Series)):
            try:
                new_obs = pd.concat([obs, column], axis=1)
                new_adata.obs = new_obs
            except ValueError as e:
                raise ValueError(f"Error concatenating column to coldata: {str(e)}")

        elif isinstance(column, str) and value is not None:
            if isinstance(value, (list, np.ndarray)) and len(value) == len(obs):
                value = pd.Series(value, index=obs.index)

            if isinstance(value, pd.Series):
                # Effizientere Überprüfung der Indices
                missing_indices = set(value.index) - set(obs.index)
                if missing_indices:
                    raise ValueError(f"Missing indices coldata: {', '.join(map(str, list(missing_indices)[:5]))}"
                                     f"{' ...' if len(missing_indices) > 5 else ''}")

                try:
                    if value.index.equals(obs.index):
                        obs[column] = value
                    else:
                        obs.loc[value.index, column] = value
                except Exception as e:
                    raise ValueError(f"Error setting the values: {str(e)}")
            else:
                obs[column] = value

            new_adata.obs = obs
        else:
            raise ValueError("Argument combination not valid for coldata()")

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

    def get_columns(self, sample_or_cell_name: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, reorder: bool = False) -> list[str]:
        """
        Get sample/cell names. Either by their index, their name, or a boolean mask.

        If no columns are specified, all columns are returned.

        Parameters
        ----------
        sample_or_cell_name: Union[str, int, Sequence[str|int|bool]]
            Samples/cell to be retrieved.

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
        all_names = self.columns

        if sample_or_cell_name is None:
            return all_names

        # Single-value handling(str, int)
        elif isinstance(sample_or_cell_name, (int, np.integer)):
            return [all_names[sample_or_cell_name]]
        elif isinstance(sample_or_cell_name, (str, np.str_)):
            return [sample_or_cell_name]

        if not isinstance(sample_or_cell_name, (list, tuple, np.ndarray, pd.Series)):
            raise TypeError("Invalid input type for sample_or_cell_names. Must be int, str, list, tuple, np.ndarray, or pd.Series.")

        # list handling(list, tuple)
        if all(isinstance(i, (int, np.integer)) for i in sample_or_cell_name):
            result = [all_names[i] for i in sample_or_cell_name]
        if all(isinstance(i, (str, np.str_)) for i in sample_or_cell_name):
            result = sample_or_cell_name
        elif all(isinstance(i, (bool, np.bool_)) for i in sample_or_cell_name):
            if len(sample_or_cell_name) != len(all_names):
                raise ValueError("Length of boolean filter must match number of samples/cells.")
            result = list(all_names[sample_or_cell_name])
        else:
            raise TypeError("Inkonsistent input types for sample_or_cell_names. All values in the iterable must have the same type(int, str or bool).)")

        return result if reorder else [name for name in all_names if name in result]


    def get_index(self, gene: Union[str, int, Sequence[Union[str, int, bool]]] = None, *, regex: bool = False) -> list[int]:
        """
        Get the index of: a gene, a list of genes, or in accordance to a boolean filter.

        Either by gene name or symbol, or by a boolean mask.

        Integers are returned unchanged.

        If names and indices are mixed, only one of them will be used. Chosen by the higher number of matches.

        Parameters
        ----------
        gene: Union[str, int, Sequence[Union[str, int, bool]]]
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

        if gene is None:
            return index

        gene_column = gene_info.get("Gene")
        symbol_column = gene_info.get("Symbol")

        if regex and isinstance(gene, str):
            mask = gene_column.astype(str).str.contains(gene, regex=True) | \
                   symbol_column.astype(str).str.contains(gene, regex=True)
            return list(np.where(mask)[0])

        # gene zu einer Liste konvertieren
        gene = _ensure_list(gene)

        if isinstance(gene, list) and any(pd.isna(gene)):
            warnings.warn("All None values were removed from the query.")
            gene = [g for g in gene if pd.notna(g)]

        if isinstance(gene, list) and all(isinstance(g, (int, np.integer)) for g in gene):
            return gene

        if isinstance(gene, list) and all(isinstance(g, (bool, np.bool_)) for g in gene):
            if len(gene) != len(index):
                raise ValueError("Length of boolean filter must match number of genes.")
            return list(np.where(gene)[0])

        gene_list = pd.Series(gene, dtype=str)

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

        def _parse_mode_slot(mode_slot_unparsed: str) -> ModeSlot:
            """
            Parse a mode_slot string.
            """
            mode_slot_candidate = mode_slot_unparsed.split("_", 1)

            if len(mode_slot_candidate) == 1:
                return ModeSlot("total", mode_slot_unparsed)

            if len(mode_slot_candidate) != 2:
                raise ValueError(
                    f"Invalid mode_slot: '{mode_slot_unparsed}'. Expected format: '<mode>_<slot>' or ModeSlot('<mode>', '<slot>').")

            mode, slot = mode_slot_candidate

            return ModeSlot(mode, slot)

        # if mode_slot is a string, it gets parsed into a ModeSlot Object
        if isinstance(mode_slot, str):
            mode_slot = _parse_mode_slot(mode_slot)


        if not self._check_slot(mode_slot.slot, allow_ntr=allow_ntr):
            raise ValueError(f"Slot '{mode_slot.slot}' not found in data slots.")

        slot = self._adata.layers[mode_slot.slot]
        ntr = self._adata.layers["ntr"]

        resulting_mode_slot = slot

        # The resulting data is computed, depending on the mode
        if mode_slot.mode != "total":
            if self._is_sparse:
                resulting_mode_slot = slot.multiply(ntr) if mode_slot.mode == "new" else slot.multiply(_one_minus(ntr))
            else:
                resulting_mode_slot = slot * ntr if mode_slot.mode == "new" else slot * (1 - ntr)

        return resulting_mode_slot


    # TODO: get_data() um die fehlenden Parameter aus R erweitern und eingabe mehrerer slots ermöglichen.
    def get_data(self,
                 mode_slot: Union[str, ModeSlot, Sequence[Union[str, ModeSlot]]] = None,
                 gene: Union[str, Sequence[str]] = None,
                 columns: Union[str, Sequence[str]] = None,
                 *,
                 with_coldata: bool = True) -> pd.DataFrame:
        """
        Get a subset of on or multiple data slots.

        Parameters
        ----------
        mode_slot: Union[str, ModeSlot, Sequence[str|ModeSlot]]]
            The name of the desired data slot. If None, uses the default slot.

        gene: Union[str, Sequence[str]]
            The genes to be retrieved. Can be gene symbols or names.

        columns: Union[str, Sequence[str]]
            The cells/samples to be retrieved.

        with_coldata: bool
            If True, the coldata DataFrame will be concatenated to the result.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the data for the specified genes and columns.

        See Also
        --------
        ModeSlot: Represents a mode slot.

        """
        if mode_slot is None:
            mode_slot = self.default_slot

        # Mode slot muss noch verarbeitet werden.
        # Vorübergehender Ersatz:
        data = self._adata.layers[mode_slot]

        if isinstance(columns, str):
            columns = [columns]

        row_indices = [self.coldata.index.get_loc(column) for column in columns] if columns is not None else range(len(self.coldata))
        column_indices = self.get_index(gene)

        result_rows = self.coldata.iloc[row_indices]["Name"].tolist()
        result_columns = self.gene_info.iloc[column_indices]["Symbol"].tolist()

        data_subset = data[np.ix_(row_indices, column_indices)]


        result_df = pd.DataFrame(data_subset, index = result_rows, columns = result_columns)

        if with_coldata:
            result_df = pd.concat([self.coldata.iloc[row_indices], result_df], axis=1)

        return result_df


def _validate_and_convert_new_data(matrix) -> Union[np.ndarray, sp.csr_matrix]:
    # Falls DataFrame → zu NumPy
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    # Falls Liste → zu NumPy
    if isinstance(matrix, list):
        matrix = np.array(matrix)

    # Falls sparse, aber nicht csr → zu csr
    if sp.issparse(matrix) and not isinstance(matrix, sp.csr_matrix):
        matrix = sp.csr_matrix(matrix)

    # Falls dicht → alles okay
    elif not sp.issparse(matrix):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be ndarray, DataFrame, or scipy sparse matrix")

    shape = matrix.shape
    # Shape prüfen
    if matrix.shape != shape:
        raise ValueError(f"Matrix shape {matrix.shape} does not match expected shape {shape}")

    return matrix

def _to_sparse(matrix: Union[pd.DataFrame, np.ndarray]) -> sp.csr_matrix:
    """
    Convert a dense NumPy array or Pandas DataFrame to a csr_matrix.

    Parameters
    ----------
    matrix: Union[pd.DataFrame, np.ndarray]
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

def _one_minus(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """
    Helper funktion to compute one minus a sparse matrix.
    """
    ones = sp.csr_matrix(np.ones(matrix.shape), dtype=matrix.dtype)

    return ones - matrix

def _ensure_list(x):
    if isinstance(x, (str, int, bool)) or x is None:
        return [x]
    return list(x)

