import warnings
from typing import Any
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


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

    Returns
    -------

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

        self._adata = ad.AnnData(
            X = slots["count"],
            obs = coldata,
            var = gene_info,
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
        if 'no4sU' not in self._adata.obs.columns:
            warnings.warn("No no4sU entry in coldata, assuming all samples/cells as 4sU treated!")
            self._adata.obs["no4sU"] = False


    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self._adata.uns.get('prefix', 'Unknown')}\n"
            f"{self._adata.n_vars} genes, {self._adata.n_obs} samples/cells\n"
            f"Available data slots: {', '.join(self._adata.layers.keys()) or {} or 'None'}\n"
            f"Available analyses: {', '.join(self._adata.uns.get('analyses') or {}) or 'None'}\n"
            f"Available plots: {', '.join(self._adata.uns.get('plots') or {}) or 'None'}\n"
            f"Default data slot: {self._adata.uns['metadata'].get('default_slot', None)}\n"
        )


    def _replace(self, adata: ad.AnnData) -> 'GrandPy':
        def _safe_copy(obj):
            return obj.copy() if obj is not None else None

        return self.__class__(
            prefix = adata.uns.get('prefix'),
            gene_info = _safe_copy(adata.var),
            coldata = _safe_copy(adata.obs),
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
        row_names = self._adata.obs_names.tolist()
        column_names = self._adata.var_names.tolist()
        return row_names, column_names


    @property
    def default_slot(self) -> str:
        """
        Get the name of the default slot
        """
        return self._adata.uns.get('metadata').get('default_slot')

    def with_default_slot(self, value) -> "GrandPy":
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
    def slots(self) -> list[str]:
        """
        Get the names of all available data slots.
        """
        return list(self._adata.layers.keys())

    def with_dropped_slots(self, slots_to_remove: str | list[str]) -> "GrandPy":
        """
        Return a new GrandPy object with specified slot(s) removed.

        Parameters
        ----------
        slots_to_remove: str or list of str
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

    def with_slot(self, name, matrix, *, set_to_default = False) -> "GrandPy":
        """
        Returns a new GrandPy Object with the new slot added.

        Parameters
        ----------
        name: str
            Name of the new slot.
        matrix: numpy.ndarray or pandas.DataFrame or scipy sparse matrix
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
    def with_condition(self, value: str | list[str]) -> Any:
        """

        Parameters
        ----------

        Returns
        -------

        """

        value = [value] if isinstance(value, str) else value
        new_adata = self._adata.copy()

        if all(v in self.coldata.columns for v in value):

        #Verhalten momentan noch anders als i n GrandR, Name kann nicht benutzt werden, da wir diesen als Index Speichern.

            new_adata.obs['Condition'] = self.coldata[value].astype(str).agg(" ".join, axis=1)
        else:
            #momentan funktioniert die Funktion nur, wenn die Länge von values gleich der Länge des Indexes von coldata ist.
            if len(value) != len(self.coldata.index):
                raise ValueError(
                    f"Number of values ({len(value)}) does not match number of samples/cells ({len(self.coldata.index)})")

            new_adata.obs['Condition'] = pd.Series(value, index=self.coldata.index)

        return self._replace(new_adata)


    @property
    def metadata(self) -> dict:
        """
        Get the metadata about the GrandPy object.
        """
        return self._adata.uns.get('metadata').copy()


    # TODO gene_info() vervollständigen
    @property
    def gene_info(self) -> pd.DataFrame:
        """
        Get the gene info DataFrame.
        """

        return self._adata.var.copy()

    # ist noch nicht vollständig/fehlerhaft
    def with_gene_info(self, column=None, value=None):
        """

        Parameters
        ----------

        Returns
        -------

        """
        if column is None:
            return self._adata.var
        elif value is None:
            return pd.Series(self._adata.var[column].values, index=self._adata.var["Symbol"])
        else:
            ...


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
        return self._adata.obs.copy()

    def with_coldata(self, column, value=None) -> "GrandPy":
        """
            Return a new object with modified coldata.

            Parameters
            ----------
            column : str, Series, or DataFrame
                If str and value is None, returns the specified column.
                If str and value is given, sets or updates the column with the given values.
                If DataFrame or Series, concatenates the new columns to the existing coldata.
            value : optional
                The values to assign to the column, can be list, array, or Series.

            Returns
            -------
            GrandPy
                A new GrandPy object with updated coldata.
        """
        obs = self._adata.obs.copy()
        new_adata = self._adata.copy()

        if isinstance(column, (pd.DataFrame, pd.Series)):
            try:
                new_obs = pd.concat([obs, column], axis=1)
                new_adata.obs = new_obs
            except ValueError as e:
                raise ValueError(f"Error concatenating column to coldata: {str(e)}")

        elif isinstance(column, str) and value is None:
            if column not in obs:
                raise KeyError(f"Column '{column}' not found in coldata")
            return obs[column]

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
        Get the gene symbols contained in gene info.
        """
        return self._adata.var.index.tolist()

    def get_genes(self, genes: str|list[str]|int|list[int]|list[bool] = None,*, use_gene_symbols: bool = True, regex: bool = False) -> list[str]:
        """
        Get gene names or symbols. Either by their index, their name, a boolean mask, or a regex.\n
        If no genes are specified, all genes are returned.

        Parameters
        ----------
        genes: str|list[str]|int|list[int]|list[bool]
            Genes to be retrieved.
        use_gene_symbols: bool
            If True, gene symbols will be returned. Otherwise, gene names will be returned.
        regex: bool
            If True, `genes` will be interpreted as a regular expression.

        Returns
        -------
        list[str]
            A list containing the specified genes.

        """
        if use_gene_symbols:
            if genes is None:
                return self._adata.var.index.tolist()

            indices = self.get_index(genes, regex=regex)
            return self._adata.var.iloc[indices].index.tolist()

        else:
            if genes is None:
                return self._adata.var.get("Gene").tolist()

            indices = self.get_index(genes, regex=regex)
            return self._adata.var.iloc[indices]["Gene"].tolist()

    @property
    def columns(self):
        return list(self._adata.obs.index)

    def get_columns(self, columns=None, reorder=False):
        """

        Parameters
        ----------

        Returns
        -------

        """
        column_data = self._adata.obs
        if columns is None:  # Wenn keine Auswahl angegeben ist: alle Zellnamen zurückgeben
            result = list(column_data["Name"])

        elif isinstance(columns, str):  # Wenn eine Bedingung als String angegeben ist (wie "condition == 'Mock'")
            try:
                result = list(column_data.query(columns)["Name"])
            except Exception as e:
                raise ValueError(f"Invalid query string for columns: {e}")

        elif isinstance(columns, (list, tuple, np.ndarray, pd.Index)):  # Wenn direkt eine Liste oder ein Array mit Zellnamen übergeben wird
            selected_names = list(map(str, columns))
            result = [name for name in selected_names if name in column_data["Name"].values]
        else:
            raise ValueError("Invalid argument combination for columns.")

        result = result if reorder else [name for name in column_data["Name"] if name in result]  # Gib Zellnamen in Originalreihenfolge zurück (wie in column_data.index), wenn reorder False

        return result


    #funkitoniert momentan nicht, muss noch angepasst werden
    def get_index(self, gene: str|list[str]|list[bool]|int|list[int] = None,* , regex: bool = False) -> list[int]:
        """
        Get the index of: a gene, a list of genes, or in accordance to a boolean filter.\n
        Integers are returned unchanged.

        Parameters
        ----------
        gene: str or list of str or list of bool or int or list of int
            Specifies which indices to return.
        regex: bool
            If True, `gene` will be interpreted as a regular expression.

        Returns
        -------
        list[int]
            A list containing the specified indices.
        """
        gene_info = self._adata.var.copy()
        index = list(range(len(gene_info.index)))

        if isinstance(gene, (list, tuple, pd.Series, np.ndarray)) and any(pd.isna(gene)):
            warnings.warn("All None values were removed from the query.")
            gene = [g for g in gene if pd.notna(g)]

        if gene is None:
            return list(index)

        if isinstance(gene, int):
            return [gene]

        if isinstance(gene, (list, tuple, np.ndarray)) and all(isinstance(g, (int, np.integer)) for g in gene):
            return gene

        gene_column = gene_info.get("Gene")
        symbol_column = gene_info.index

        if regex and isinstance(gene, str):
            mask = gene_column.astype(str).str.contains(gene, regex=True) | \
                   symbol_column.astype(str).str.contains(gene, regex=True)
            return list(np.where(mask)[0])

        if isinstance(gene, (list, tuple, np.ndarray)) and all(isinstance(g, (bool, np.bool_)) for g in gene):
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


    def _check_slot(self, slot_name) -> bool:
        """
        Check wether a given slot exists in the GrandPy object.
        Parameters
        ----------
        slot_name : str
            The name of the slot to check.

        Returns
        ----------
        bool
            True if the slot exists, False otherwise.
        """
        if slot_name not in self._adata.layers:
            raise KeyError(f"Slot '{slot_name}' not found.")
        return True

    def _check_mode_slot(self, slot_name, mode):
        slot = self._adata.uns.get("mode_layers", {}).get(slot_name)
        if isinstance(slot, dict) and mode in slot:
            return True
        raise KeyError(f"Mode-Slot '{slot_name}' with mode '{mode}' not found.")

    def _parse_mode_slot(self, slot, mode=None, check=True) -> str:
        """
        Resolve a mode-specific slot name.

        Parameters
        -----------
        slot : str
            Base slot name (e.g. "ntr").
        mode : str
            Mode name (e.g. "MAP"). If None, returns the base slot name.
        check : bool
            If True, raises KeyError if the resolved slot does not exist.

        Returns
        -----------
        str
            The resolved slot name (e.g. "ntr_MAP" or "ntr").
        """

        if mode is None:
            if check:
                self._check_slot(slot)
            return slot

        mode_slots = self._adata.uns.get("mode_layers", {})

        if slot not in mode_slots or not isinstance(mode_slots[slot], dict):
            raise KeyError(f"Slot '{slot}' does not support modes.")

        resolved = mode_slots[slot].get(mode)
        if resolved is None:
            raise KeyError(f"Mode '{mode}' does not resolve to '{slot}'.")

        if check:
            self._check_slot(resolved)

        return resolved

def _validate_and_convert_new_data(matrix) -> "np.ndarray" or "sp.csr_matrix":
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

def _to_sparse(matrix):
    """
    Convert a dense NumPy array or Pandas DataFrame to a csr_matrix.

    Parameters
    ----------
    matrix: pandas.DataFrame or numpy.ndarray
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
        raise ValueError("Matrix could not be converted to a sparse matrix. Use numpy.ndarray or a pandas.DataFrame only containing numbers")

    return sparse_matrix

