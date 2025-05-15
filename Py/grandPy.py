import warnings
from typing import Any
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pandas import Series
from scipy.sparse import csr_matrix


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

    parent: GrandPy
        Child will inherit prefix, gene_info, coldata, slots and metadata from the parent if not specified.

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
                 parent: 'GrandPy' = None):

        gene_info = gene_info if gene_info is not None else parent.adata.var if parent is not None else None
        coldata = coldata if coldata is not None else parent.adata.obs if parent is not None else None
        slots = slots if slots is not None else parent.adata.layers if parent is not None else None

        if gene_info is None:
            raise ValueError("GrandPy object must have gene_info.")
        if coldata is None:
            raise ValueError("GrandPy object must have coldata.")
        if slots is None:
            raise ValueError("GrandPy object must have slots (data).")

        self.adata = ad.AnnData(
            X = sp.csr_matrix(np.zeros(shape=(coldata.shape[0], gene_info.shape[0]))),
            obs = pd.DataFrame(coldata),
            var = pd.DataFrame(gene_info),
        )

        self._initialize_slots(slots)
        self._initialize_uns_data(prefix, metadata, analyses, plots, parent)
        self._ensure_no4sU_column()

    def _initialize_slots(self, slots=None):
        if slots is not None:                                                                                           # musste es etwas anpassen, damit check_mode_slot() funktionieren kann, denn ntr etc. werden eig als dict übergeben
            for key, matrix in slots.items():
                self._check_shape(key, matrix)
                self.adata.layers[key] = matrix

    # Namen werden und können aktuell nicht überprüft werden
    def _check_shape(self, name, slot):
        row_is, column_is = slot.shape
        row_should, column_should = self.adata.shape
        if row_is != row_should:
            raise ValueError(f"Number of rows do not match the data for the {name} Matrix!")
        if column_is != column_should:
            raise ValueError(f"Number of columns do not match the data for the {name} Matrix!")

    def _initialize_uns_data(self, prefix, metadata, analyses, plots, parent=None):
        self.adata.uns['prefix'] = prefix if prefix is not None else parent.adata.uns.get('prefix') if parent is not None else None
        self.adata.uns['metadata'] = metadata if metadata is not None else parent.adata.uns.get('metadata') if parent is not None else None
        self.adata.uns['analyses'] = analyses
        self.adata.uns['plots'] = plots

    def _ensure_no4sU_column(self):
        if 'no4sU' not in self.adata.obs.columns:
            warnings.warn("No no4sU entry in coldata, assuming all samples/cells as 4sU treated!")
            self.adata.obs["no4sU"] = False


    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self.adata.uns.get('prefix', 'Unknown')}\n"
            f"{self.adata.n_vars} genes, {self.adata.n_obs} samples/cells\n"
            f"Available data slots: {', '.join(self.adata.layers.keys()) or {} or 'None'}\n"
            f"Available analyses: {', '.join(self.adata.uns.get('analyses') or {}) or 'None'}\n"
            f"Available plots: {', '.join(self.adata.uns.get('plots') or {}) or 'None'}\n"
            f"Default data slot: {self.adata.uns['metadata'].get('default_slot', None)}\n"
        )


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
        prefix = self.adata.uns.get('prefix')
        if prefix is None:
            raise KeyError("Title not available. Please specify a prefix when initializing the GrandPy object")
        else:
            x = prefix.split('/')
            return x[-1]

    def is_sparse(self) -> bool:
        """
        Checks if the data is stored in sparse format.

        Parameters
        ----------

        Returns
        -------
        bool
            True if the data is stored in sparse format, False otherwise.
        """
        return isinstance(self.adata.layers["count"], sp.csr_matrix)

    def dim(self) -> tuple[int]:
        """
        Get the dimension of the data.

        Parameters
        ----------

        Returns
        -------
        tuple
            Dimension of the data.
        """
        return self.adata.X.shape

    def dim_names(self) -> tuple[list[str]]:
        """
        Get the column and row names of the data.

        Parameters
        ----------

        Returns
        -------
        tuple
            Two lists containing the column and row names.

        """
        column_names = self.adata.obs["Name"].tolist()
        row_names = self.adata.var["Gene"].tolist()
        return column_names, row_names

    def default_slot(self, value=None) -> Any:
        """
        Get or set the default slot.

        Parameters
        ----------
        value: str
            If provided, sets the default slot to this value.

        Returns
        -------
        str or "GrandPy"
            If `value` is None, returns the default slot. Otherwise, returns the GrandPy object with the new default slot.

        """
        if value is None:
            return self.adata.uns.get('metadata').get('default_slot')
        else:
            assert value in self.adata.layers.keys(), "Trying to set a default_slot that is not an available slot"
            self.adata.uns['metadata']['default_slot'] = value
            return self

    def slots(self) -> list[str]:
        """
        Get the names of available data slots.

        Parameters
        ----------

        Returns
        -------
        list
            Names of all slots
        """
        return list(self.adata.layers.keys())

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

        keep_keys = [key for key in self.adata.layers.keys() if key not in pattern]
        if not keep_keys:
            raise ValueError("Cannot drop all slots!")

        if self.default_slot() not in keep_keys:
            self.adata.uns['metadata']['default_slot'] = keep_keys[0]

        self.adata.layers = {k: self.adata.layers[k] for k in keep_keys}
        return self

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
        if name in self.adata.layers.keys():
            raise ValueError(f"Slot '{name}' already exists. Please choose a different name.")

        if self.is_sparse():
            matrix = _to_sparse(matrix)
        else:
            matrix = self._validate_and_convert_new_data(matrix)

        self._check_shape(name, matrix)

        self.adata.layers[name] = matrix

        if set_to_default:
            self.adata.uns['metadata']['default_slot'] = name

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

    # Hab mich noch nicht entschieden, wie sich die Funktion verhalten soll
    def condition(self, value=None) -> Any:
        """

        Parameters
        ----------

        Returns
        -------

        """
        coldata = self.adata.obs
        if value is None:
            return coldata['Condition'].tolist()
        else:
            value = [value] if isinstance(value, str) else value
            if all(v in coldata.columns for v in value):
                # Warum existiert diese if Klausel?
                self.adata.obs['Condition'] = coldata[value].astype(str).agg(" ".join, axis=1)
            else:
                # if len(value) == 1:
                #     value = value * len(coldata.index)
                # if len(value) == 2:
                #     value = value * (len(coldata.index) // 2) + value[:len(coldata.index) % 2]
                if len(value) != len(coldata.index):
                    raise ValueError(
                        f"Number of values ({len(value)}) does not match number of samples ({len(coldata.index)})")

                self.adata.obs['Condition'] = pd.Series(value, index=coldata.index)

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
        return self.adata.uns.get('metadata')

    def gene_info(self, column=None, value=None):
        """

        Parameters
        ----------

        Returns
        -------

        """
        if column is None:
            return self.adata.var
        elif value is None:
            return pd.Series(self.adata.var[column].values, index=self.adata.var["Symbol"])
        else:
            ...

    def apply(self, function, function_gene_info=None, function_coldata=None, **kwargs):
        """

        Parameters
        ----------

        Returns
        -------

        """
        new_slots = {}
        for key in self.adata.layers.keys():
            new_slots[key] = function(self.adata.layers[key], **kwargs)

        new_gene_info = function_gene_info(self.adata.var, **kwargs) if function_gene_info is not None else None
        new_coldata = function_coldata(self.adata.obs, **kwargs) if function_coldata is not None else None
        new_analyses = None

        if self.adata.uns['analyses'] is not None:
            ...

        return GrandPy(gene_info=new_gene_info, slots=new_slots, coldata=new_coldata, parent=self)

    def coldata(self, column=None, value=None):
        """

        Parameters
        ----------

        Returns
        -------

        """
        obs = self.adata.obs
        if column is None:
            return obs

        elif isinstance(column, (pd.DataFrame, pd.Series)):
            try:
                self.adata.obs = pd.concat([obs, column], axis=1)
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
                        self.adata.obs[column] = value
                    else:
                        self.adata.obs.loc[value.index, column] = value
                except Exception as e:
                    raise ValueError(f"Error setting the values: {str(e)}")
            else:
                self.adata.obs[column] = value
            
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
            return self.adata.var[column]

        indices = self.get_index(genes, regex=regex)
        return self.adata.var.iloc[indices][column]

    def columns(self, columns=None, reorder=False):
        """

        Parameters
        ----------

        Returns
        -------

        """
        column_data = self.adata.obs
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
        gene_info = self.adata.var
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

    def _check_slot(self, slot_name):
        if slot_name in self.adata.layers and slot_name not in self.adata.uns.get("mode_layers", {}):
            return True
        raise KeyError(f"Slot '{slot_name}' not found.")

    def _check_mode_slot(self, slot_name, mode):
        slot = self.adata.uns.get("mode_layers", {}).get(slot_name)
        if isinstance(slot, dict) and mode in slot:
            return True
        raise KeyError(f"Mode-Slot '{slot_name}' with mode '{mode}' not found.")

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