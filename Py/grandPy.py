# Fragen nach Library Wahl. numpy + scipy / numpy + scipy + pandas / numpy + pandas ?
# rpy2 anschauen

import numpy as np
import pandas as pd
import anndata as ad

# Wie ich es verstehe, sollte das GrandR Object konzeptionell ca. so aussehen
# (damit der Code Sinn ergibt, parallel die Funktion grandR aus grandR.R anschauen):

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
        gene_info = gene_info if gene_info is not None else getattr(parent, 'adata', None).var if parent else None
        coldata = coldata if coldata is not None else getattr(parent, 'adata', None).obs if parent else None

        # Wählt die erste Matrix in slots als Hauptmatrix(adata.X)
        # temporäre Lösung!

        default_key = next(iter(slots))
        X = slots[default_key]

        self.adata = ad.AnnData(
            X = None,
            obs = pd.DataFrame(coldata),
            var = pd.DataFrame(gene_info),
        )

        # Speichere alle Datenmatrizen aus slots in adata.layers
        for key, matrix in slots.items():
            self.adata.layers[key] = matrix

        self.adata.uns['prefix'] = prefix if prefix is not None else getattr(parent, 'prefix', None)
        self.adata.uns['metadata'] = metadata if metadata is not None else getattr(parent, 'metadata', None)
        self.adata.uns['analyses'] = analyses
        self.adata.uns['plots'] = plots

        #
        #
        # Default data slot wird nur übergangsweise so gehandhabt
        #
        #

    def __str__(self):
        return (
            f"GrandPy:\n"
            f"Read from {self.adata.uns['prefix']}\n"
            f"{self.adata.n_vars} genes, {self.adata.n_obs} samples/cells\n"
            f"Available data slots: {', '.join(slots.keys()) if slots else 'None'}\n"
            f"Available analyses: {', '.join(self.adata.uns.get('analysis', {}).keys()) or 'None'}\n"
            f"Available plots: {', '.join(self.adata.uns.get('plots', {}).keys()) or 'None'}\n"
            f"Default data slot: { ... }\n"
        )




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
