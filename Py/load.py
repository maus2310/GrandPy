import pandas as pd
import numpy as np
import anndata as ad

from Py.grandPy import GrandPy


# Version bisher nur für dense Matrix möglich
# kann sars.tsv Datei laden im Format GRAND-SLAM 2.0
# orientiert an dem Tutorial: https://grandr.erhard-lab.de/articles/web/loading-data.html
# slots: count - ntr, alpha, beta durch künstliche correct_matrix() ERGÄNZT, es sind nur 10 samples und wir haben sie dadurch auf 12 "erweitert"

def read_grand(file_path, design = ("Condition", "Time", "Replicate"), default_slot = "count"):
    """
    Mit der Funktion wird eine TSV-Datei eingelesen,
    im Anschluss wird ein GrandPy-Objekt erstellt.

    Parameter
    file_path: Pfad zur gewünschten TSV-Datei
    design entspricht dem Design-Vektor aus dem Tutorial
    default_slot: Standard-Daten-Slot

    Ausgabe: GrandPy-Objekt
    """
    data = pd.read_csv(file_path, sep = "\t")                                                                           # Einlesen & Trennung mit Tab


    slot_suffix = {"count": " Readcount", "ntr": " MAP", "alpha": " alpha", "beta": " beta"}

    slot_columns = {
        key: [col for col in data.columns if col.endswith(suffix)]
        for key, suffix in slot_suffix.items()
    }

    gene_info = data[["Gene", "Symbol", "Length"]].copy()
    # gene_info["Type"] = np.where(gene_info["Symbol"].str.startswith("MT-"), "mito", "Cellular")                       # Ermöglicht eine grobe Einteilung, uns fehlt die classify_genes Funktion.
    # Habe es mal formal wie in R doch erweitert: teil die Zelltypen genauer ein:
    gene_info["Type"] = "Unknown"
    gene_info.loc[gene_info["Symbol"].str.startswith("MT-"), "Type"] = "mito"
    gene_info.loc[gene_info["Gene"].str.contains("ERCC-"), "Type"] = "ERCC"
    gene_info.loc[gene_info["Gene"].str.match(r"^ENS.*G\d+$"), "Type"] = "Cellular"

    matrices = {}
    for key in slot_suffix.keys():
        matrix = data[slot_columns[key]].to_numpy().T
        nan_mask = np.isnan(matrix)
        matrices[key] = np.where(nan_mask, 0, matrix)

    def correct_matrix(mat):                                                                                            # fügt Nullen an die Matrix, um die Dimension zu korrigieren
        zeros = np.zeros((2, mat.shape[1]))
        return np.vstack([mat, zeros])

    for key in ["ntr", "alpha", "beta"]:
        matrices[key] = correct_matrix(matrices[key])

    slots = {
        "count": matrices["count"],
        "ntr": matrices["ntr"],
        "alpha": matrices["alpha"],
        "beta": matrices["beta"]
    }


    sample_names = [col.replace(" Readcount", "") for col in slot_columns["count"]]
    design_data = pd.DataFrame([name.split(".") for name in sample_names],
                               columns=design,
                               index=np.arange(len(sample_names))
                               )
    design_data.insert(0, "Name",sample_names)
    design_data["no4sU"] = design_data["Time"].isin(["no4sU", "no4sU", "-"])                                            # Default

    coldata = design_data#

    # adata will Index als Strings(sonst kommt die Warnung: Transforming to str index)
    gene_info.index = gene_info.index.astype(str)
    coldata.index = coldata.index.astype(str)

    metadata = {
        "Description": "count data",
        "default_slot": default_slot,
        "GRAND-SLAM version": 2,
        "Output": "dense"
    }

    return GrandPy(
        prefix = file_path,
        gene_info = gene_info,
        slots = slots,
        coldata = coldata,
        metadata = metadata
    )

# Beispielanwendung:
# file_path = "data/sars.tsv"
# new_gp_object = read_grand(file_path)
# print(new_gp_object)

# new_gp_object.check_mode_slot("ntr", "raw")
# new_gp_object.check_slot("count")

# Ausgabe: Überblick über Gene, Samples, Slots

# Was wir damit anstellen können
# print("Titel des Datensatzes:", new_gp_object.adata.uns["prefix"])                                                    # Ausgabe: "data/sars.tsv
# print("Anzahl der Samples (Zeilen):", new_gp_object.adata.n_obs)                                                      # Ausgabe: "12"
# print("Anzahl der Gene (Spalten):", new_gp_object.adata.n_vars)                                                       # Ausgabe: "19659"

# print(new_gp_object.slots())                                                                                          # Liste aller geladenen Datenslots (bisher nur count)
# print(new_gp_object.coldata())                                                                                        # sollte die Metadaten-Tab zu den Samples zeigen (bin mir unsicher, ob ich das richtig verstanden habe - bitte gegenchecken :D)
# print(new_gp_object.adata.var.head())                                                                                 # Ausgabe der "vereinfachten" Gen-Info-Table mit den Spalten "Gene, Symbol, Length, Type"

# type_counts = new_gp_object.adata.var["Type"].value_counts()                                                          # gruppiert und zählt die existierenden Zelltypen
# print(type_counts)                                                                                                    # Ausgabe

def gene_type_counts(grandpy_obj):                                                                                      # Funktion für Gruppierung nach Zelltypen (brauchen wir die noch? Ich hab etwas den Überblick verloren)
    return grandpy_obj.adata.var["Type"].value_counts()

# Beispielanwendung für die Verwendung von gene_type_counts - Gruppieren und zählen nach Zelltypen
# grandpy_obj = new_gp_object
# gene_type_counts(grandpy_obj)

# Ausgabe der Original-Datei (Bei Bedarf)
# data = pd.read_csv(file_path, sep = "\t")
# print(data.head())

# counts = new_gp_object.adata.layers["count"]
# print(counts.shape)                                                                                                   # bei sars Ausgabe: (12, 19659) - 12 Samples, 19659 Gene
# gene_names = new_gp_object.adata.var["Symbol"].values
# print(gene_names)                                                                                                     # Ausgabe: Namen von Genen
# sample_names = new_gp_object.adata.obs["Name"].values
# print(sample_names)                                                                                                   # Ausgabe: Name von Samples
# gene_idx = list(gene_names).index("GAPDH")                                                                            # Gen-Name hier ersetzbar
# print(counts[:, gene_idx])                                                                                            # Aussage wie stark ein bestimmtes Gen in den einzelnen Proben/Samples exprimiert wird

# So und nun evt. eine erste Visualisierung aber bitte mit Vorsicht genießen
# import matplotlib.pyplot as plt
# import seaborn as sns
# gene = "GAPDH"
# counts = new_gp_object.adata.layers["count"]
# gene_idx = list(new_gp_object.adata.var["Symbol"]).index(gene)
# df_plot = new_gp_object.coldata()
# df_plot["expression"] = counts[:, gene_idx]
# sns.boxplot(x="Condition", y="expression", data=df_plot)
# plt.title(f"Expression of {gene}")
# plt.show()