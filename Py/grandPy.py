# rpy2 anschauen

import numpy as np
import pandas as pd
import anndata as ad


# Am Beispieldatensatz sars: (Keine sparse, nur dense Matrizen)

# prefix = String   (Dateipfad)
# gene_info = Matrix[size = Anzahl Gene * 4(Gene(String, Gen ID), Symbol(String, Gen Symbol), Length(int, Länge in Basenpaaren), Typ(Categorical, "Cellular" oder "Unknown"))]    (Informationen zu den Genen in den Daten)
# slots = List(Matrizen)   (Hier stehen die tatsächlichen Daten(z.B: count: Reads pro Gen und Sample[size = Anzahl Gene * Anzahl Samples], ntr: new to total ratio))
# coldata = Matrix[size = Anzahl Samples * 6(Name(Categorical), Condition(Categorical), Replicate(Categorical), duration.4sU(int, Zeit seit 4sU Behandlung), duration.4sU.original(Categorical), no4sU(boolean))]  (Metadaten zu den Samples)
# metadata = List(Description(String), default.slot(String), GRAND-SLAM version(int), Output(String))   (Zusatzinformationen über das Object)
# analyses =
# plots =
# parent =


class GrandPy:
    def __init__(self, prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
        gene_info = gene_info if gene_info is not None else parent.adata.var if parent is not None else None
        coldata = coldata if coldata is not None else parent.adata.obs if parent is not None else None
        slots = slots if slots is not None else parent.adata.layers if parent is not None else None

        #Damit __str__ mit analyses, plots und metadata so arbeiten kann, wie es gerade tut und keinen Fehler bei None gibt
        analyses = {} if analyses is None else analyses
        plots = {} if plots is None else plots
        metadata = {} if metadata is None else metadata

        # Hauptmatrix von anndata noch fraglich.
        # Passt eine leere Matrix als X?

        self.adata = ad.AnnData(
            X = np.zeros((coldata.shape[0] if coldata is not None else 0,gene_info.shape[0] if gene_info is not None else 0)),
            obs = pd.DataFrame(coldata),
            var = pd.DataFrame(gene_info),
        )

        def checknames(self, name, matrix):
            n_obs, n_vars = matrix.shape

            # print(f"Shape der geladenen Matrix: {matrix.shape}")         # Test für load - bitte erstmal lassen
            # print(f"Shape von adata (Gene, Sample): {self.adata.shape}") # Test für load - bitte erstmal lassen

            if n_obs != self.adata.n_obs:
                raise ValueError(f"Number of rows do not match for {name}!")
            if n_vars != self.adata.n_vars:
                raise ValueError(f"Number of columns do not match for {name}!")

            # Namen werden nicht überprüft(anders als in R)


        if slots is not None:
            for key, matrix in slots.items():
                checknames(self, key, matrix)
                self.adata.layers[key] = matrix

        self.adata.uns['prefix'] = prefix if prefix is not None else parent.adata.uns.get('prefix', None) if parent is not None else None
        self.adata.uns['metadata'] = metadata if metadata is not None else parent.adata.uns.get('metadata', None) if parent is not None else None
        self.adata.uns['analyses'] = analyses
        self.adata.uns['plots'] = plots


    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self.adata.uns['prefix']}\n"
            f"{self.adata.n_vars} genes, {self.adata.n_obs} samples/cells\n"
            f"Available data slots: {', '.join(self.adata.layers) if self.adata.layers else 'None'}\n"
            f"Available analyses: {', '.join(self.adata.uns.get('analysis', {}).keys()) or 'None'}\n"
            f"Available plots: {', '.join(self.adata.uns.get('plots', {}).keys()) or 'None'}\n"
            f"Default data slot: {self.adata.uns['metadata'].get('default_slot', None)}\n"
        )


    # Funktionen in der Klasse?

    def default_slot(self, value=None):
        if value is None:
            return self.adata.uns['metadata'].get('default_slot', None)
        else:
            self.adata.uns['metadata']['default_slot'] = value
            return self


    def slots(self):
        return list(self.adata.layers.keys())

    # Coldata: Zugriff und Modifikation
    def coldata(self, column=None, value=None):
        obs = self.adata.obs
        if column is None:                                                              #Kein Argument → ganze coldata zurückgeben
            return obs

        elif isinstance(column, (pd.DataFrame, pd.Series)):                             #DataFrame oder Series übergeben → an bestehende coldata anhängen
            self.adata.obs = pd.concat([obs, pd.DataFrame(column)], axis=1)
            return self

        elif isinstance(column, str) and value is None:                                 #Spaltenname übergeben, aber kein Wert → gib einzelne Spalte zurück
            return obs[column]

        elif isinstance(column, str) and value is not None:                             #Spaltenname + Wert → neue Spalte setzen oder bestehende überschreiben
            if isinstance(value, (list, np.ndarray)) and len(value) == len(obs):        #Liste oder Array → in Series mit dem passenden Index umwandeln
                value = pd.Series(value, index=obs.index)

            if isinstance(value, pd.Series):                                            #Series mit benanntem Index
                if not value.index.equals(obs.index):
                    if not all(name in obs.index for name in value.index):              #Prüfe, ob alle Namen des Index in obs vorhanden sind
                        raise ValueError("Series index does not match obs index!")
                    self.adata.obs.loc[value.index, column] = value

                else:
                    self.adata.obs[column] = value

            else:                                                                       # entspricht in R dem Fall: length(value) == 1 oder direkter Spaltenzuweisung
                self.adata.obs[column] = value

            return self
        else:
            raise ValueError("Invalid argument combination for coldata.")



# alte Implementierung
#
# class GrandPy:
#     def __init__(self, prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
#         self.prefix = prefix if prefix is not None else getattr(parent, 'prefix', None)
#         self.gene_info = gene_info if gene_info is not None else getattr(parent, 'gene_info', None)
#         self.slots = slots if slots is not None else getattr(parent, 'data', None)
#         self.coldata = coldata if coldata is not None else getattr(parent, 'coldata', None)
#         self.metadata = metadata if metadata is not None else getattr(parent, 'metadata', None)
#         self.analyses = analyses
#         self.plots = plots
#
#
#     def checknames(self, name, a):
#         ...
