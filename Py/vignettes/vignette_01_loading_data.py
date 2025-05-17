"""
Vignette: Loading Data into GrandPy

This tutorial shows how to load count data (from GRAND-SLAM 2.0) into the GrandPy framework.
We follow the example from the original grandR tutorial: https://grandr.erhard-lab.de/articles/web/loading-data.html
"""

from Py.load import read_grand

# ------------------------------------------------------------
# Step 1: Define file path to the count matrix
# ------------------------------------------------------------

# Example file: download this from https://zenodo.org/record/5834034
file_path = "../data/sars.tsv"

# ------------------------------------------------------------
# Step 2: Load the data using read_grand()
# ------------------------------------------------------------

# sars.tsv contains data from 10 samples, split by Condition, Time, and Replicate
gp = read_grand(file_path, design=("Condition", "Time", "Replicate"))

# ------------------------------------------------------------
# Step 3: Overview of the loaded GrandPy object
# ------------------------------------------------------------

print(gp)

# ------------------------------------------------------------
# Step 4: Inspect coldata (sample metadata)
# ------------------------------------------------------------

print("Sample Metadata (coldata):")
print(gp.coldata().head())

# Example: list all samples treated with 4sU
print("Samples with 4sU treatment:")
print(gp.coldata().index[~gp.coldata()["no4sU"]].tolist())

# ------------------------------------------------------------
# Step 5: Inspect gene_info (gene metadata)
# ------------------------------------------------------------

print("Gene Metadata (gene_info):")
print(gp.gene_info().head())

# Count gene types
print("Distribution of gene types:")
print(gp.gene_info()["Type"].value_counts())

# ------------------------------------------------------------
# Step 6: Inspect data slots
# ------------------------------------------------------------

print("Available data slots:")
print(gp.slots)

# Check shape of count matrix
print("Shape of count matrix:")
print(gp.shape)                             # (samples, genes)

# ------------------------------------------------------------
# Step 7: Expression of a specific gene
# ------------------------------------------------------------

gene = "GAPDH"                              # random
gene_idx = list(gp.gene_info()["Symbol"]).index(gene)
counts = gp._adata.layers["count"]

print(f'Expression values for {gene}:')
print(counts[:, gene_idx])

# ------------------------------------------------------------
# Step 8: Visualize expression across conditions
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

df_plot = gp.coldata().copy()
df_plot["expression"] = counts[:, gene_idx]

plt.figure(figsize=(6, 4))
sns.boxplot(x="Condition", y="expression", data=df_plot)
plt.title(f"Expression of {gene}")
plt.show()

# the boxplot shows the expression of the gene "GAPH"
# comparison between the conditions "Mock" and "Sars"