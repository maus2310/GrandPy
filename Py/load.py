import pandas as pd
import numpy as np
import anndata as ad
from grandPy import GrandPy

# IN BEARBEITUNG !!!!!! Ausgabe noch nicht richtig
# bisher nur für dense Matrix, keine sparse Matrix!
# orientiert an dem tutorial: https://grandr.erhard-lab.de/articles/web/loading-data.html

def load_data(file_path, slot_name = "count", default_slot = "count"):
    """
    Mit der Funktion wird eine TSV-Datei eingelesen,
    im Anschluss wird ein GrandPy-Objekt erstellt.

    Parameter
    file_path: Pfad zur gewünschten TSV-Datei
    slot_name: Name des Daten-Slots
    default_slot: Standard-Daten-Slot

    Ausgabe: GrandPy-Objekt
    """
    data = pd.read_csv(file_path, sep="\t", index_col = 0).T  # Einlesen

    # gene_info beinhaltet Metadaten über die Gene, anders wie bei coldata
    # sind hier die Zeilen der Hauptmatrix beschrieben
    # Anforderung laut Website: "must contain 4 columns"
    gene_info = pd.DataFrame({                              # DataFrame gene_Info erstellen
        "Gene": data.columns,                               # gene ID
        "ID": data.columns,
        "Symbol": data.columns,
        "Length": [1000] * len(data.columns),               # Platzhalter, wird später durch die echten Werte ersetzt
        "Type": ["Cellular"] * len(data.columns)            # Platzhalter, s.o.
    }, index = data.columns)

    # coldata beschreibt die Eigenschaften der Spalten des gegebenen Datensatzes
    # bspw. Samples, celltypes, time stencils, conditions etc.
    # ergo ein DataFrame, das Infos über jedes Sample (jede Spalte der Hauptmatrix) beinhaltet
    coldata = pd.DataFrame({                                # DataFrame coldata erstellen
        "Name": data.index,
        "Condition": ["A"] * len(data.index),
        "Replicate": ["rep1"] * len(data.index),
        "duration.4sU": [0] * len(data.index),
        "duration.4sU.original": ["0min"] * len(data.index),
        "not4sU": [False] * len(data.index)
    }, index = data.index)

    slots = {slot_name: data.values}                        # slots Dictionary erstellen

    # metadata beinhaltet allgemeine Infos über den Datensatz
    metadata = {"default_slot": default_slot}               # metadata Dictionary erstellen

    gp = GrandPy(
        prefix = file_path,
        gene_info = gene_info,
        slots = slots,
        coldata = coldata,
        metadata = metadata
    )

    return gp

# Beispielanwendung:
file_path = "data/sars.tsv"
new_gp_object = load_data(file_path)
print(new_gp_object)

# Was wir damit anstellen können
print("Titel des Datensatzes:", new_gp_object.adata.uns["prefix"])
print("Anzahl der Samples (Zeilen):", new_gp_object.adata.n_obs)    # hier falsch ...
print("Anzahl der Gene (Spalten):", new_gp_object.adata.n_vars)     # Anzahl der Gene (Spalten) FALSCH

# Ausgabe der ersten Zeilen von obs und var zur Kontrolle
# print("Daten der Proben (obs):\n", new_gp_object.adata.obs.head())
# print("Daten der Gene (var):\n", new_gp_object.adata.var.head())


# data = pd.read_csv(file_path, sep="\t", index_col=0)  # Einlesen der TSV-Datei
# print(data.shape)  # Gibt die Dimensionen der Matrix aus
# print(data.head())  # Zeigt die ersten paar Zeilen an