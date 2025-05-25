from Py.load import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_scatter_simple(data, x: str, y: str, slot: str = "count", remove_outlier: bool = False):
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

    # Outlier entfernen (IQR-Methode)
    if remove_outlier:
        def filter_iqr(vals):
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (vals >= lower) & (vals <= upper)

        mask_x = filter_iqr(x_vals)
        mask_y = filter_iqr(y_vals)
        mask = mask_x & mask_y
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

    # Dichte berechnen (2D KDE)
    xy = np.vstack([x_vals, y_vals])
    kde = gaussian_kde(xy)(xy)  # Dichtewerte für jeden Punkt

    # Sortiere Punkte nach Dichte, damit hohe Dichte oben liegt (bessere Sichtbarkeit)
    idx = kde.argsort()
    x_vals, y_vals, kde = x_vals[idx], y_vals[idx], kde[idx]

    # Plot
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x_vals, y_vals, c=kde, s=10, cmap='viridis', alpha=0.8)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} ({slot})")
    plt.colorbar(scatter, label='Density')
    plt.grid(False)
    plt.show()
