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

        def checknames(self, name, matrix):
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
                        checknames(self, f"{key}:{mode_key}", submatrix)
                    self.adata.uns.setdefault("mode_layers", {})[key] = matrix
                else:
                    checknames(self, key, matrix)
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
            f"Available analyses: {', '.join(self.adata.uns.get('analysis') or {}) or 'None'}\n"
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
        return isinstance(self.adata.X, sp.csr_matrix)

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


    def apply(self, function, function_gene_info=None, function_coldata=None, **kwargs):
        new_slots = {}
        for key in self.adata.layers.keys():
            new_slots[key] = function(self.adata.layers[key], **kwargs)

        new_gene_info = function_gene_info(self.adata.var, **kwargs) if function_gene_info is not None else None
        new_coldata = function_coldata(self.adata.obs, **kwargs) if function_coldata is not None else None
        new_analyses = None

        if self.adata.uns['analyses'] is not None:
            ...
        # Muss später noch ergänzt werden

        return GrandPy(gene_info=new_gene_info, slots=new_slots, coldata=new_coldata, parent=self)

    def coldata(self, column=None, value=None):
        obs = self.adata.obs
        if column is None:  # Kein Argument → ganze coldata zurückgeben
            return obs

        elif isinstance(column, (pd.DataFrame, pd.Series)):  # DataFrame oder Series übergeben → an bestehende coldata anhängen
            self.adata.obs = pd.concat([obs, pd.DataFrame(column)], axis=1)
            return self

        elif isinstance(column, str) and value is None:  # Spaltenname übergeben, aber kein Wert → gib einzelne Spalte zurück
            return obs[column]

        elif isinstance(column, str) and value is not None:  # Spaltenname + Wert → neue Spalte setzen oder bestehende überschreiben
            if isinstance(value, (list, np.ndarray)) and len(value) == len(obs):  # Liste oder Array → in Series mit dem passenden Index umwandeln
                value = pd.Series(value, index=obs.index)

            if isinstance(value, pd.Series):  # Series mit benanntem Index
                if not value.index.equals(obs.index):
                    if not all(name in obs.index for name in value.index):  # Prüfe, ob alle Namen des Index in obs vorhanden sind
                        raise ValueError("Series index does not match obs index!")
                    self.adata.obs.loc[value.index, column] = value

                else:
                    self.adata.obs[column] = value

            else:  # entspricht in R dem Fall: length(value) == 1 oder direkter Spaltenzuweisung
                self.adata.obs[column] = value

            return self
        else:
            raise ValueError("Invalid argument combination for coldata.")

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

    def check_slot(self, slot_name):                                                                                    # Bsp: new_gp_object.check_slot("ntr")  # Wenn Slot nicht existiert - Fehlermeldung
        in_layers = slot_name in self.adata.layers
        in_mode_layers = slot_name in self.adata.uns.get("mode_layers", {})

        if not (in_layers or in_mode_layers):
            raise ValueError(f"Slot '{slot_name}' does not exist in '{self}'.")


    def check_mode_slot(self, slot_name, mode):
        mode_layers = self.adata.uns.get("mode_layers", {})
        slot = mode_layers.get(slot_name, None)
        if slot is None:
            raise ValueError(f"Slot '{slot_name}' does not exist.")

        if not isinstance(slot, dict):
            raise TypeError(f"Slot '{slot_name}' is not stored in the mode-shape (no dict).")

        if mode not in slot:
            raise ValueError(f"Mode '{mode}' us not in Slot '{slot_name}'.")