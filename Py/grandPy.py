# Fragen nach Library Wahl. numpy + scipy / numpy + scipy + pandas / numpy + pandas ?
# rpy2 anschauen

import numpy as np
import pandas as pd
#  import scipy.sparse

# Wie ich es verstehe, sollte das GrandR Object konzeptionell ca. so aussehen
# (damit der Code Sinn ergibt, parallel die Funktion grandR aus grandR.R anschauen):

# Am Beispieldatensatz sars: (Keine sparse, nur dense Matrizen)

# prefix = String   (Dateipfad)
# gene_info = Matrix{pandas?}[size = Anzahl Gene * 4(Gene(String, Gen ID), Symbol(String, Gen Symbol), Length(int, Länge in Basenpaaren), Typ(Categorical, "Cellular" oder "Unknown"))]    (Informationen zu den Genen in den Daten)
# slots = List(Matrizen{numpy/scipy?})   (Hier stehen die tatsächlichen Daten(z.B: count: Reads pro Gen und Sample[size = Anzahl Gene * Anzahl Samples], ntr: new to total ratio))
# coldata = Matrix{pandas?}[size = Anzahl Samples * 6(Name(Categorical), Condition(Categorical), Replicate(Categorical), duration.4sU(int, Zeit seit 4sU Behandlung), duration.4sU.original(Categorical), no4sU(boolean))]  (Metadaten zu den Samples)
# metadata = List(Description(String), default.slot(String), GRAND-SLAM version(int), Output(String))   (Zusatzinformationen über das Object)
# analyses =
# plots =
# parent =


class GrandPy:
    def __init__(self, prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
        self.prefix = prefix if prefix is not None else getattr(parent, 'prefix', None)
        self.gene_info = gene_info if gene_info is not None else getattr(parent, 'gene_info', None)
        self.slots = slots if slots is not None else getattr(parent, 'data', None)
        self.coldata = coldata if coldata is not None else getattr(parent, 'coldata', None)
        self.metadata = metadata if metadata is not None else getattr(parent, 'metadata', None)
        self.analysis = analyses
        self.plots = plots

    def checknames(self, name, a):
        ...


    # Hier fehlt noch: checknames; 1 for und 3 ifs



