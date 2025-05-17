import warnings
from typing import Any
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pandas import Series


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
                 plots = None,
                 parent: ad.AnnData = None):

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

    def _initialize_uns_data(self, prefix, metadata, analyses, plots, parent=None):
        self._adata.uns['prefix'] = prefix if prefix is not None else parent.adata.uns.get('prefix') if parent is not None else None
        self._adata.uns['metadata'] = metadata if metadata is not None else parent.adata.uns.get('metadata') if parent is not None else None
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
        Get a title for the GrandPy object

        Parameters
        ----------

        Returns
        -------
        str
            A title consisting of the filename in `prefix`.
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
        Get the dimension of the data(slots).

        Parameters
        ----------

        Returns
        -------
        tuple
            Dimension of the data.
        """
        return self._adata.X.shape

    @property
    def dim_names(self) -> tuple[list[str], list[str]]:
        """
        Get the column and row names of the data.

        Parameters
        ----------

        Returns
        -------
        tuple
            Two lists containing the column and row names.

        """
        row_names = self._adata.obs_names.tolist()
        column_names = self._adata.var_names.tolist()
        return row_names, column_names

    @property
    def default_slot(self) -> str:
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
            Returns the GrandPy object with the new default slot.

        """
        assert value in self._adata.layers.keys(), "Trying to set a default_slot that is not an available slot"

        new_adata = self._adata.copy()
        new_adata.uns['metadata']['default_slot'] = value

        return self._replace(new_adata)

    @property
    def slots(self) -> list[str]:
        """
        Get the names of available data slots.

        Parameters
        ----------

        Returns
        -------
        list of str
            Names of all slots
        """
        return list(self._adata.layers.keys())

    def drop_slot(self, slot_pattern) -> "GrandPy":
        """
        Remove slots matching a pattern.

        Parameters
        ----------
        slot_pattern: str or list of str
            All slots matching this pattern will be removed.

        Returns
        -------
        "GrandPy"
            Returns the GrandPy object with slots matching the pattern removed.
        """
        pattern = [slot_pattern] if isinstance(slot_pattern, str) else slot_pattern

        keep_keys = [key for key in self._adata.layers.keys() if key not in pattern]
        if not keep_keys:
            raise ValueError("Cannot drop all slots!")

        if self.default_slot not in keep_keys:
            self._adata.uns['metadata']['default_slot'] = keep_keys[0]

        self._adata.layers = {k: self._adata.layers[k] for k in keep_keys}
        return self

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

    def add_slot(self, name, matrix, set_to_default = False):
        """
        Add a new data slot to the GrandPy object.

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

        if self.is_sparse():
            matrix = _to_sparse(matrix)
        else:
            matrix = self._validate_and_convert_new_data(matrix)

        self._check_shape(name, matrix)

        self._adata.layers[name] = matrix

        if set_to_default:
            self._adata.uns['metadata']['default_slot'] = name

        return self

    def _validate_and_convert_new_data(self, matrix) -> "np.ndarray" or "sp.csr_matrix":
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

    # TODO condition(): Sich für das genaue Verhalten der Funktion entscheiden
    def condition(self, value=None) -> Any:
        """

        Parameters
        ----------

        Returns
        -------

        """
        coldata = self._adata.obs
        if value is None:
            return coldata['Condition'].tolist()
        else:
            value = [value] if isinstance(value, str) else value
            if all(v in coldata.columns for v in value):
                # Warum existiert diese if Klausel?
                self._adata.obs['Condition'] = coldata[value].astype(str).agg(" ".join, axis=1)
            else:
                # if len(value) == 1:
                #     value = value * len(coldata.index)
                # if len(value) == 2:
                #     value = value * (len(coldata.index) // 2) + value[:len(coldata.index) % 2]
                if len(value) != len(coldata.index):
                    raise ValueError(
                        f"Number of values ({len(value)}) does not match number of samples ({len(coldata.index)})")

                self._adata.obs['Condition'] = pd.Series(value, index=coldata.index)

            return self

    def metadata(self) -> dict:
        """
        Get the metadata.

        Parameters
        ----------

        Returns
        -------
        dict
            Metadata
        """
        return self._adata.uns.get('metadata')

    # TODO gene_info() vervollständigen
    def gene_info(self, column=None, value=None):
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
    def apply(self, function, function_gene_info=None, function_coldata=None, **kwargs) -> "GrandPy":
        """
        Apply a function to all slots, gene_info and/or coldata.
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
        new_slots = {}
        for key in self._adata.layers.keys():
            new_slots[key] = function(self._adata.layers[key], **kwargs)

        new_gene_info = function_gene_info(self._adata.var, **kwargs) if function_gene_info is not None else None
        new_coldata = function_coldata(self._adata.obs, **kwargs) if function_coldata is not None else None
        new_analyses = None

        if self._adata.uns['analyses'] is not None:
            ...

        return GrandPy(gene_info=new_gene_info, slots=new_slots, coldata=new_coldata, parent=self)

    def coldata(self, column=None, value=None):
        """

        Parameters
        ----------

        Returns
        -------

        """
        obs = self._adata.obs
        if column is None:
            return obs

        elif isinstance(column, (pd.DataFrame, pd.Series)):
            try:
                self._adata.obs = pd.concat([obs, column], axis=1)
                return self
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
                        self._adata.obs[column] = value
                    else:
                        self._adata.obs.loc[value.index, column] = value
                except Exception as e:
                    raise ValueError(f"Error setting the values: {str(e)}")
            else:
                self._adata.obs[column] = value

            return self
        else:
            raise ValueError("Argument combination not valid for coldata()")

    def genes(self, genes: str|list[str] = None, use_gene_symbols: bool = True, regex: bool = False) -> Series:
        """
        Get gene names or symbols. If no genes are specified, all genes are returned.

        Parameters
        ----------
        genes: str or list of str
            Genes to be retrieved.
        use_gene_symbols: bool
            If True, gene symbols will be returned. Otherwise, gene names will be returned.
        regex: bool
            If True, `genes` will be interpreted as a regular expression.

        Returns
        -------
        Series
            All genes or the specified genes.

        """
        column = "Symbol" if use_gene_symbols else "Gene"

        if genes is None:
            return self._adata.var[column]

        indices = self.get_index(genes, regex=regex)
        return self._adata.var.iloc[indices][column]

    def columns(self, columns=None, reorder=False):
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

    # Braucht die Methode "warn" überhaupt?
    def get_index(self, gene: str|list[str]|list[bool]|int|list[int] = None, regex: bool = False, warn: bool = True) -> list[int]:
        """
        Get the index of a gene, a list of genes or in accordance to a boolean filter.\n
        Integers are returned unchanged.

        Parameters
        ----------
        gene: str or list of str or list of bool or int or list of int
            Specifies which indices to return.
        regex: bool
            If True, `gene` will be interpreted as a regular expression.
        warn: bool
            If True, a warning will be displayed if a gene wasn't found or `gene` contains None values.

        Returns
        -------
        list[int]
            All indices or the specified indices.

        """
        gene_info = self._adata.var
        index = gene_info.index

        if isinstance(gene, (list, tuple, pd.Series, np.ndarray)) and any(pd.isna(gene)):
            if warn:
                warnings.warn("All None values were removed from the query.")
            gene = [g for g in gene if pd.notna(g)]

        if gene is None:
            return list(index)

        if isinstance(gene, int):
            return [gene]

        if isinstance(gene, (list, tuple, np.ndarray)) and all(isinstance(g, (int, np.integer)) for g in gene):
            return gene

        gene_col = gene_info.get("Gene")
        symbol_col = gene_info.get("Symbol")

        if regex and isinstance(gene, str):
            mask = gene_col.astype(str).str.contains(gene, regex=True) | \
                   symbol_col.astype(str).str.contains(gene, regex=True)
            return index[mask].tolist()

        if isinstance(gene, (list, tuple, np.ndarray)) and all(isinstance(g, (int, np.integer)) for g in gene):
            return list(gene)

        if isinstance(gene, (list, tuple, np.ndarray)) and all(isinstance(g, (bool, np.bool_)) for g in gene):
            if len(gene) != len(index):
                raise ValueError("Length of boolean filter must match number of genes.")
            return list(index[np.where(gene)[0]])

        gene_list = pd.Series(gene, dtype=str)

        matches_in_gene = gene_list[gene_list.isin(gene_col)]
        matches_in_symbol = gene_list[gene_list.isin(symbol_col)]

        if len(matches_in_gene) >= len(matches_in_symbol):
            ref_col = gene_col
        else:
            ref_col = symbol_col

        mapping = pd.Series(index, index=ref_col)
        found = gene_list[gene_list.isin(ref_col)]
        missing = gene_list[~gene_list.isin(ref_col)]

        if warn and not missing.empty:
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