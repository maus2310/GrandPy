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
                        show_outlier: bool = True,
                        xlim: Optional[tuple[float, float]] = None,
                        ylim: Optional[tuple[float, float]] = None,):
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
        xlim: tuple[float, float]
            define the x axis limits (defining the lower and upper bound)
        ylim: tuple[float, float]
            define the y axis limits (defining the lower and upper bound)

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
    x_vals_all = matrix[:, x_idx]
    y_vals_all = matrix[:, y_idx]

    mask_keep = np.ones_like(x_vals_all, dtype=bool)

    if remove_outlier:
        def get_bounds(vals):
            q1, q3 = np.percentile(vals[np.isfinite(vals)], [25, 75])
            iqr = q3 - q1
            return q1 - 1.5 * iqr, q3 + 1.5 * iqr

        x_lower, x_upper = get_bounds(x_vals_all)
        y_lower, y_upper = get_bounds(y_vals_all)

        mask_x = (x_vals_all >= x_lower) & (x_vals_all <= x_upper)
        mask_y = (y_vals_all >= y_lower) & (y_vals_all <= y_upper)
        mask_keep = mask_x & mask_y

        if xlim is None:
            xlim = (
                x_vals_all[mask_x].min(),
                x_vals_all[mask_x].max()
            )
        if ylim is None:
            ylim = (
                y_vals_all[mask_y].min(),
                y_vals_all[mask_y].max()
            )

    x_vals = x_vals_all[mask_keep]
    y_vals = y_vals_all[mask_keep]

    xy = np.vstack([x_vals, y_vals])
    kde = gaussian_kde(xy)(xy)
    idx = kde.argsort()
    x_vals, y_vals, kde = x_vals[idx], y_vals[idx], kde[idx]

    plt.figure(figsize=(6, 6))
    if remove_outlier and show_outlier:
        x_out = x_vals_all[~mask_keep]
        y_out = y_vals_all[~mask_keep]
        plt.scatter(x_out, y_out, color="lightgray", s=1, alpha=0.5, label="Outliers")
    scatter = plt.scatter(x_vals, y_vals, c=kde, s=5, cmap='viridis', alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} ({mode_slot})")
    plt.colorbar(scatter, label='Density')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(False)
    #plt.tight_layout()
    plt.show()