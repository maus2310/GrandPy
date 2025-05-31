#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
from Py.grandPy import GrandPy, ModeSlot
from scipy.stats import gaussian_kde





#Parameter die noch fehlen:
    # 1. analysis
    # 2. xcol/ycol
    # 3. log, log_x, log_y
    # 4. axis, axis_x, axis_y
    # 5. lim
    # 6. filter
    # 7. genes
    # 8. highlight_label
    # 9. label, label_repel
    # 10. facet
    # 11. color, collorpalette, colorbreaks, color_label, na_color
    # 12. density_margin, density_n
    # 13. rastersize
    # 14. correlation, correlation_x/_y/_hjust/_vjust
    # 15. layers.below


def plot_scatter(data: GrandPy,
                 x: str,
                 y: str,
                 mode_slot: str | ModeSlot = None,
                 remove_outlier: bool = True,
                 show_outlier: bool = True,  #Funktioniert noch nicht richtig
                 path_for_save: str = None,
                 lim: Optional[tuple[float, float]] = None,
                 x_lim: Optional[tuple[float, float]] = None,
                 y_lim: Optional[tuple[float, float]] = None,
                 size: float = 5,
                 cross: Optional[bool] = None,
                 diag: Optional[bool | float | tuple] = None,
                 highlight: Optional[Union[list[str], dict[str, list[str]]]] = None):
    """
        ScatterPlot

        Parameters
        ----------
        data: GrandPy
            Object of GrandPy class
        x: str
            An expression to compute the x value or a character corresponding to a sample (or cell) name or a fully qualified analysis result name
        y: str
            An expression to compute the y value or a character corresponding to a sample (or cell) name or a fully qualified analysis result name
        mode_slot: str | ModeSlot
            Specifies which data slot to use (e.g., "count", "norm")
        remove_outlier: bool
            Whether to detect and remove outliers using IQR filtering
        show_outlier: bool
            If True, outliers will be plotted in light gray
        path_for_save: str
            Saves the plot as a PNG to the specified directory (filename is auto-generated)
        lim: tuple[float, float]
            Defines both xlim and ylim if they are not set explicitly
        x_lim: tuple[float, float]
            Define the x-axis limits (lower and upper bounds)
        y_lim: tuple[float, float]
            Define the y-axis limits (lower and upper bounds)
        size: float
            Size of each point in the scatter plot
        diag: bool | float | list[float]
            If True, draws the identity line (y = x).
            If float, draws one line: y = x + diag.
            If list of floats, draws multiple lines: y = x + offset for each value.
        cross: bool
            If True, draws horizontal and vertical dashed lines at x = 0 and y = 0
        highlight: list[str] | dict[str, list[str]]
            A list of gene names or a dictionary mapping colors to gene lists.
            Genes will be highlighted in the plot with size 3× the default.

        Returns
        -------
        None
            The function creates and optionally saves a matplotlib plot.
        """

    # error messages
    if x not in data.coldata["Name"].tolist():
        raise ValueError(f"x is not a valid expression.")
    if y not in data.coldata["Name"].tolist():
        raise ValueError(f"y is not a valid expression.")



    matrix = data._resolve_mode_slot(mode_slot)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    x_idx = list(data.coldata["Name"]).index(x)
    y_idx = list(data.coldata["Name"]).index(y)
    x_vals_all = matrix[:, x_idx]
    y_vals_all = matrix[:, y_idx]

    mask_keep = np.ones_like(x_vals_all, dtype=bool)

    if lim is not None:
        if x_lim is None:
            x_lim = lim
        if y_lim is None:
            y_lim = lim


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

        if x_lim is None:
            x_lim = (
                x_vals_all[mask_x].min(),
                x_vals_all[mask_x].max()
            )
        if y_lim is None:
            y_lim = (
                y_vals_all[mask_y].min(),
                y_vals_all[mask_y].max()
            )

    x_vals = x_vals_all[mask_keep]
    y_vals = y_vals_all[mask_keep]

    xy = np.vstack([x_vals, y_vals])
    kde = gaussian_kde(xy)(xy)
    idx = kde.argsort()
    x_vals, y_vals, kde = x_vals[idx], y_vals[idx], kde[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    if remove_outlier and show_outlier:
        x_out = x_vals_all[~mask_keep]
        y_out = y_vals_all[~mask_keep]
        ax.scatter(x_out, y_out, color="lightgray", s=size, alpha=0.5, label="Outliers")

    scatter = ax.scatter(x_vals, y_vals, c=kde, s=size, cmap='viridis', alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} ({mode_slot})")
    fig.colorbar(scatter, label='Density')

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    if diag is not None:
        x_plot = np.linspace(*plt.xlim(), 100)
        if isinstance(diag, bool) and diag:
            plt.plot(x_plot, x_plot, linestyle="--", color="gray", label="y = x")
        elif isinstance(diag, (int, float)):
            plt.plot(x_plot, x_plot + diag, linestyle="--", color="gray", label=f"y = x + {diag}")
        elif isinstance(diag, (list, tuple)):
            for offset in diag:
                plt.plot(x_plot, x_plot + offset, linestyle="--", color="gray", label=f"y = x + {offset}")


    if cross:
        plt.axvline(0, linestyle="--", color="gray")
        plt.axhline(0, linestyle="--", color="gray")

    if highlight is not None:
        def get_indices(genes):
            return data.get_index(genes, regex=False)

        if isinstance(highlight, dict):
            for color, genes in highlight.items():
                idxs = get_indices(genes)
                if idxs:
                    plt.scatter(x_vals_all[idxs], y_vals_all[idxs], color=color, s=size * 3)
        else:
            idxs = get_indices(highlight)
            if idxs:
                plt.scatter(x_vals_all[idxs], y_vals_all[idxs], color="red", s=size * 3)

    plt.grid(False)
    #offsets = scatter.get_offsets()
    #print(len(offsets))
    if path_for_save is not None :
        fig.savefig(f"{path_for_save}{x}_{y}_{mode_slot}.png", format="png", dpi=300)
    plt.show()
    plt.close(fig)