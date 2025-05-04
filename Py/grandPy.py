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

        if slots is not None:
            for key, matrix in slots.items():
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
