from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
from Py.load import *
from scipy.stats import gaussian_kde

def plot_scatter(data,
                        x: str,
                        y: str,
                        mode_slot: str | ModeSlot = None,
                        remove_outlier: bool = False,
                        xlim: Optional[tuple[float, float]] = None,
                        ylim: Optional[tuple[float, float]] = None):
    """
        ScatterPlot

        Parameters
        ----------
        x: str
            an expression to compute the x value or a character corresponding to a sample (or cell) name or a fully qualified analysis result name
        y: str
            an expression to compute the y value or a character corresponding to a sample (or cell) name or a fully qualified analysis result name
        mode_slot: str | ModeSLot
            Count, Ntr ...
        remove_outlier: bool
            Detects and removes outliers

        Returns
        -------
        Plot:
            Plot object containing the scatter plot
        """
    matrix = data._resolve_mode_slot(mode_slot)

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    x_idx = list(data.coldata["Name"]).index(x)
    y_idx = list(data.coldata["Name"]).index(y)
    x_vals = matrix[:, x_idx]
    y_vals = matrix[:, y_idx]

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

    xy = np.vstack([x_vals, y_vals])
    kde = gaussian_kde(xy)(xy)
    idx = kde.argsort()
    x_vals, y_vals, kde = x_vals[idx], y_vals[idx], kde[idx]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x_vals, y_vals, c=kde, s=1, cmap='viridis', alpha=0.8)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} ({mode_slot})")
    plt.colorbar(scatter, label='Density')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(False)
    plt.show()