from Py.load import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde



#Zum testen: plot_scatter_simple(sars, x="Mock.no4sU.A", y="SARS.no4sU.A", slot="count")
def plot_scatter_simple(data, x: str, y: str, slot: str = "count"):

    # Matrix holen
    matrix = data._adata.layers[slot]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    # Spaltenindizes holen
    x_idx = list(data.coldata["Name"]).index(x)
    y_idx = list(data.coldata["Name"]).index(y)

    # Daten extrahieren
    x_vals = matrix[:, x_idx]
    y_vals = matrix[:, y_idx]

    # Scatterplot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, s=10, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} ({slot})")
    plt.grid(False)
    plt.show()