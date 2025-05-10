# rpy2 anschauen

import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


class GrandPy:
    def __init__(self, prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
        gene_info = gene_info if gene_info is not None else parent.adata.var if parent is not None else None
        coldata = coldata if coldata is not None else parent.adata.obs if parent is not None else None
        slots = slots if slots is not None else parent.adata.layers if parent is not None else None

        # Hauptmatrix von anndata noch fraglich.
        # Passt eine leere Matrix als X?

        self.adata = ad.AnnData(
            X = np.zeros(shape=(coldata.shape[0], gene_info.shape[0])),
            obs = pd.DataFrame(coldata),
            var = pd.DataFrame(gene_info),
        )

        def _check_names(self, name, matrix):
            n_obs, n_vars = matrix.shape
            if n_obs != self.adata.n_obs:
                raise ValueError(f"Number of rows do not match the data for the {name} Matrix!")
            if n_vars != self.adata.n_vars:
                raise ValueError(f"Number of columns do not match the data for the {name} Matrix!")

            # Namen werden und können aktuell nicht überprüft werden (dafür müsste man die layers mit pandas statt numpy speichern)

        if slots is not None:                                                                                           # musste es etwas anpassen, damit check_mode_slot() funktionieren kann, denn ntr etc. werden eig als dict übergeben
            for key, matrix in slots.items():
                if isinstance(matrix, dict):
                    for mode_key, submatrix in matrix.items():
                        _check_names(self, f"{key}:{mode_key}", submatrix)
                    self.adata.uns.setdefault("mode_layers", {})[key] = matrix
                else:
                    _check_names(self, key, matrix)
                    self.adata.layers[key] = matrix

        self.adata.uns['prefix'] = prefix if prefix is not None else parent.adata.uns.get('prefix') if parent is not None else None
        self.adata.uns['metadata'] = metadata if metadata is not None else parent.adata.uns.get('metadata') if parent is not None else None
        self.adata.uns['analyses'] = analyses
        self.adata.uns['plots'] = plots

        if 'no4sU' not in self.adata.obs.columns:
            warnings.warn("No no4sU entry in coldata, assuming all samples/cells as 4sU treated!")
            self.adata.obs["no4sU"] = False

    def __str__(self):
        normal_slots = list(self.adata.layers.keys())
        mode_slots = list(self.adata.uns.get("mode_layers", {}).keys())
        all_slots = sorted(set(normal_slots + mode_slots))

        return (
            f"GrandPy:\n"
            f"Read from {self.adata.uns.get('prefix', 'Unknown')}\n"
            f"{self.adata.n_vars} genes, {self.adata.n_obs} samples/cells\n"
            f"Available data slots: {', '.join(all_slots) if all_slots else 'None'}\n"
            f"Available analyses: {', '.join(self.adata.uns.get('analyses') or {}) or 'None'}\n"
            f"Available plots: {', '.join(self.adata.uns.get('plots') or {}) or 'None'}\n"
            f"Default data slot: {self.adata.uns['metadata'].get('default_slot', None)}\n"
        )


    def title(self):
        prefix = self.adata.uns.get('prefix')
        if prefix is None:
            return None
        else:
            x = prefix.split('/')
            return x[-1]

    def is_sparse(self):
        return isinstance(self.adata.layers["count"], sp.csr_matrix)

    def dim(self):
        return self.adata.X.shape

    def default_slot(self, value=None):
        if value is None:
            return self.adata.uns.get('metadata').get('default_slot')
        else:
            assert value in self.adata.layers, "Trying to set a default_slot that is not an available slot"
            self.adata.uns['metadata']['default_slot'] = value
            return self

    def slots(self, include_mode_slots = True):
        normal_slots = list(self.adata.layers.keys())
        if not include_mode_slots:
            return normal_slots

        mode_slots = list(self.adata.uns.get("mode_layers", {}).keys())
        return sorted(set(normal_slots + mode_slots))

    def drop_slot(self, pattern: str):
        keep_keys = [key for key in self.adata.layers.keys() if key not in pattern]
        if not keep_keys:
            raise ValueError("Cannot drop all slots!")
        else:
            if self.default_slot() not in keep_keys:
                self.adata.uns['metadata']['default_slot'] = keep_keys[0]
            self.adata.layers = {k: self.adata.layers[k] for k in keep_keys}
        return self

    def condition(self, value=None):
        if value is None:
            return self.adata.obs['Condition'].tolist()
        else:
            ...

    def metadata(self):
        return self.adata.uns.get('metadata')

    def gene_info(self, column=None, value=None):
        if column is None:
            return self.adata.var
        elif value is None:
            return pd.Series(self.adata.var[column].values, index=self.adata.var["Symbol"])
        else:
            ...


    def apply(self, function, function_gene_info=None, function_coldata=None, **kwargs):
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


    def columns(self, columns=None, reorder=False):
        column_data = self.adata.obs
        if columns is None:  # Wenn keine Auswahl angegeben ist: alle Zellnamen zurückgeben
            selected = list(column_data.index)

        elif isinstance(columns, str):  # Wenn eine Bedingung als String angegeben ist (wie "condition == 'A'")
            try:
                selected = list(column_data.query(columns).index)
            except Exception as e:
                raise ValueError(f"Invalid query string for columns: {e}")

        elif isinstance(columns, (list, tuple, np.ndarray, pd.Index)):  # Wenn direkt eine Liste oder ein Array mit Zellnamen übergeben wird
            selected = list(map(str, columns))
        else:
            raise ValueError("Invalid argument combination for columns.")

        if reorder:
            return selected
        else:
            return [idx for idx in column_data.index if idx in selected]  # Gib Zellnamen in Originalreihenfolge zurück (wie in column_data.index)

    def to_index(self, genes, remove_missing=True, warn=True):
        gene_info = self.adata.var.reset_index(drop=True)

        # Einzelnes Gen als String → in Liste umwandeln
        if isinstance(genes, str):
            genes = [genes]

        # Entscheide, ob "Gene" oder "Symbol" genommen wird
        if 'Gene' in gene_info.columns and all(g in gene_info['Gene'].values for g in genes):
            use_col = 'Gene'
        elif 'Symbol' in gene_info.columns and all(g in gene_info['Symbol'].values for g in genes):
            use_col = 'Symbol'
        else:
            gene_hits = gene_info['Gene'].isin(genes) if 'Gene' in gene_info.columns else pd.Series(False,
                                                                                                    index=gene_info.index)
            symbol_hits = gene_info['Symbol'].isin(genes) if 'Symbol' in gene_info.columns else pd.Series(False,
                                                                                                          index=gene_info.index)
            use_col = 'Gene' if gene_hits.sum() > symbol_hits.sum() else 'Symbol'

        # Mapping: Name → 1-basierter Index
        gene_to_index = {name: i + 1 for i, name in enumerate(gene_info[use_col])}

        result = {}
        missing = []
        for gene in genes:
            if gene in gene_to_index:
                result[gene] = gene_to_index[gene]
            else:
                missing.append(gene)
                if not remove_missing:
                    result[gene] = None

        if warn and missing:
            warnings.warn(f"Could not find given genes (n={len(missing)} missing, e.g. {', '.join(missing[:5])})!")

        return result

    # Wegen mode.slot fragen wir nochmal nach

    def check_slot(self, slot_name):
        if slot_name in self.adata.layers and slot_name not in self.adata.uns.get("mode_layers", {}):
            return True
        raise KeyError(f"Slot '{slot_name}' not found.")

    def check_mode_slot(self, slot_name, mode):
        slot = self.adata.uns.get("mode_layers", {}).get(slot_name)
        if isinstance(slot, dict) and mode in slot:
            return True
        raise KeyError(f"Mode-Slot '{slot_name}' with mode '{mode}' not found.")