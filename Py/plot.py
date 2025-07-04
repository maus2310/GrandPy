import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors
from IPython.core.pylabtools import figsize
from matplotlib import cm
import numpy as np
import pandas as pd
from typing import Optional, Union
import re

from Py.analysis_tool import AnalysisTool
from Py.grandPy import GrandPy, ModeSlot, _parse_as_mode_slot
from scipy.stats import gaussian_kde, iqr
from scipy.ndimage import gaussian_filter
import seaborn as sns
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
import warnings
from scipy.sparse import issparse

from Py.modeling import fit_kinetics


def _is_sparse_matrix(mat):
    return issparse(mat)


# TODO Plots: PlotExpressionTest, PlotAnalyses, PlotTypeDistribution, FormatCorrelation

def _get_plot_limits(vals, override_lim=None):
    """Compute IQR-based limits if not overridden."""
    if override_lim is not None:
        return override_lim
    q1, q3 = np.percentile(vals[np.isfinite(vals)], [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def parse_time_to_float(time_str):
    match = re.match(r"(\d+(\.\d+)?)h", time_str)
    if match:
        return float(match.group(1))
    else:
        return 0.0

def _apply_outlier_filter(x_vals, y_vals, remove):
    """Return filtered masks and updated axis limits based on actual min/max."""
    mask = np.ones_like(x_vals, dtype=bool)

    if not remove:
        return mask, None, None

    # IQR-basierte Filtergrenzen
    def get_bounds(vals, min_iqr=1e-6):
        q1, q3 = np.percentile(vals[np.isfinite(vals)], [25, 75])
        iqr = max(q3 - q1, min_iqr)
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    x_lower, x_upper = get_bounds(x_vals)
    y_lower, y_upper = get_bounds(y_vals)

    mask_x = (x_vals >= x_lower) & (x_vals <= x_upper)
    mask_y = (y_vals >= y_lower) & (y_vals <= y_upper)
    mask = mask_x & mask_y

    x_auto_lim = (x_vals[mask_x].min(), x_vals[mask_x].max())
    y_auto_lim = (y_vals[mask_y].min(), y_vals[mask_y].max())

    return mask, x_auto_lim, y_auto_lim


def _plot_diagonal(ax, diag, x_range):
    """Draw identity or offset lines."""
    if diag is True:
        ax.plot(x_range, x_range, linestyle="--", color="gray", label="y = x")
    elif isinstance(diag, (int, float)):
        ax.plot(x_range, x_range + diag, linestyle="--", color="gray", label=f"y = x + {diag}")
    elif isinstance(diag, (list, tuple)):
        for offset in diag:
            ax.plot(x_range, x_range + offset, linestyle="--", color="gray", label=f"y = x + {offset}")


def _highlight_points(ax, data, x_vals, y_vals, highlight, size):
    def get_indices(genes):
        return data.get_index(genes, regex=False)
    if isinstance(highlight, dict):
        for color, genes in highlight.items():
            idxs = get_indices(genes)
            if idxs:
                ax.scatter(x_vals[idxs], y_vals[idxs], color=color, s=size * 3)
    else:
        idxs = get_indices(highlight)
        if idxs:
            ax.scatter(x_vals[idxs], y_vals[idxs], color="red", s=size * 3)


def _setup_default_aes(data: GrandPy, aest: dict | None = None) -> dict:
    if aest is None:
        aest = {}

    coldata = data.coldata

    if not any(k in aest for k in ["color", "colour"]):
        if "Condition" in coldata.columns:
            aest["color"] = "Condition"
        else:
            warnings.warn("Column 'Condition' not found in coldata. Please add a 'Condition' column to get the right colors.")

    if "Replicate" in coldata.columns and "shape" not in aest:
        aest["shape"] = "Replicate"

    return aest


def _density2d(x, y, n=100, margin='n'):
    x = np.asarray(x)
    y = np.asarray(y)
    xy = np.vstack([x, y])

    mask = np.isfinite(x + y)
    if not np.any(mask):
        return np.full_like(x, np.nan, dtype=float)

    # small spread for modeslot = "ntr"
    if np.all(x[mask] == x[mask][0]):
        x[mask] = np.array([x[mask][0] - 0.5, x[mask][0] + 0.5] + [x[mask][0]] * (np.sum(mask) - 2))
    if np.all(y[mask] == y[mask][0]):
        y[mask] = np.array([y[mask][0] - 0.5, y[mask][0] + 0.5] + [y[mask][0]] * (np.sum(mask) - 2))

    xy = np.vstack([x, y])

    kde = gaussian_kde(xy[:, mask])

    density = np.full(x.shape, np.nan)
    density[mask] = kde(xy[:, mask])

    if margin == 'x':
        for xi in np.unique(x[mask]):
            sel = (x == xi)
            density[sel] /= np.nanmax(density[sel])
    elif margin == 'y':
        for yi in np.unique(y[mask]):
            sel = (y == yi)
            density[sel] /= np.nanmax(density[sel])
    else:
        density /= np.nanmax(density)

    return density

#TODO bei lim farbe checken
# TODO Log noch weiter testen
# parameter die noch fehlen:
    # 2. xcol/ycol
    # 3. log_x, log_y
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
#Beispielaufruf: plot_scatter(sars, mode_slot="count", remove_outlier=True, show_outlier=True, highlight="UHMK1")
def plot_scatter(
    data: GrandPy,
    x: Optional[str] = None,
    y: Optional[str] = None,
    log: bool = False,
    mode_slot: str | ModeSlot = None,
    remove_outlier: bool = True,
    show_outlier: bool = True,
    size: float = 5,
    limit: Optional[tuple[float, float]] = None,
    x_limit: Optional[tuple[float, float]] = None,
    y_limit: Optional[tuple[float, float]] = None,
    cross: Optional[bool] = None,
    diagonal: Optional[bool | float | tuple] = None,
    highlight: Optional[Union[list[str], dict[str, list[str]]]] = None,
    path_for_save: Optional[str] = None,
    analysis: str = None,
):
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
    log: bool
        Whether to plot the logarithmic scale
    mode_slot: str | ModeSlot
        Specifies which data slot to use (e.g., "count", "norm")
    remove_outlier: bool
        Whether to detect and remove outliers using IQR filtering
    show_outlier: bool
        If True, outliers will be plotted in light gray
    path_for_save: str
        Saves the plot as a PNG to the specified directory (must end with \\ or \\\\. e.g. "C:\\\\Users\\\\user\\\\Desktop\\\\")
    limit: tuple[float, float]
        Defines both xlim and ylim if they are not set explicitly
    x_limit: tuple[float, float]
        Define the x-axis limits (lower and upper bounds)
    y_limit: tuple[float, float]
        Define the y-axis limits (lower and upper bounds)
    size: float
        Size of each point in the scatter plot
    diagonal: bool | float | list[float]
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

    if data.analyses:
        if analysis is None:
            analysis = data.analyses[0]
        else:
            analysis = analysis
        names = data.get_analysis_table(with_gene_info=False).keys().tolist()
    else:
        names = list(data.coldata["Name"])
    print("Names:", names)
    print("Analysis:", analysis)
    x = x or names[0]
    y = y or names[1]
    if x not in names:
        raise ValueError(f"x is not a valid expression.")
    if y not in names:
        raise ValueError(f"y is not a valid expression.")

    if mode_slot is None:
        mode_slot = data.default_slot

    raw_matrix = data._resolve_mode_slot(mode_slot)
    print("Mode Slot:", mode_slot)
    print("Raw Matrix:", raw_matrix)
    print("x", x)
    print("y", y)
    if _is_sparse_matrix(raw_matrix):
        df = data.get_analysis_table()
        print("DEBUG DF", df)
        print("DEBUG: Use sparse plot")
    elif analysis:
        df = data.get_analysis_table(with_gene_info=False)
        print("DEBUG: Use analysis plot")
    else:
        df = data.get_table(mode_slots=mode_slot)
        print("DEBUG: Use dense plot")

    if x in df.columns:
        x_vals_all = df[x].to_numpy()
    elif analysis:
        x_vals_all = df.T.iloc[0].to_numpy()
    else:
        col_index = list(data.coldata["Name"]).index(x)
        matrix = data._resolve_mode_slot(mode_slot)
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        x_vals_all = matrix[:, col_index]
    print("DEBUG", x_vals_all)

    if y in df.columns:
        y_vals_all = df[y].to_numpy()
    elif analysis:
        y_vals_all = df.T.iloc[1].to_numpy()
    else:
        col_index = list(data.coldata["Name"]).index(y)
        matrix = data._resolve_mode_slot(mode_slot)
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        y_vals_all = matrix[:, col_index]
    print("DEBUG", y_vals_all)
    if np.all(np.isnan(x_vals_all)):
        raise ValueError(f"All Values for '{x}' in slot '{mode_slot}' are NaN. - Plot not possible!")
    if np.all(np.isnan(y_vals_all)):
        raise ValueError(f"All Values for '{x}' in slot '{mode_slot}' are NaN. - Plot not possible!")

    if limit:
        x_limit = x_limit or limit
        y_limit = y_limit or limit

    if log:
        mask_pos = (x_vals_all > 0) & (y_vals_all > 0)
        if not np.any(mask_pos):
            raise ValueError("No positive values. Log transform not possible!")
        x_vals_log = np.log10(x_vals_all[mask_pos])
        y_vals_log = np.log10(y_vals_all[mask_pos])

        mask_keep, auto_x_lim, auto_y_lim = _apply_outlier_filter(x_vals_log, y_vals_log, remove_outlier)

        x_vals = x_vals_log[mask_keep]
        y_vals = y_vals_log[mask_keep]

        outlier_x = x_vals_log[~mask_keep]
        outlier_y = y_vals_log[~mask_keep]

        x_limit = x_limit or auto_x_lim
        y_limit = y_limit or auto_y_lim

    else:
        mask_keep, auto_x_lim, auto_y_lim = _apply_outlier_filter(x_vals_all, y_vals_all, remove_outlier)

        x_vals = x_vals_all[mask_keep]
        y_vals = y_vals_all[mask_keep]

        outlier_x = x_vals_all[~mask_keep]
        outlier_y = y_vals_all[~mask_keep]

        x_limit = x_limit or auto_x_lim
        y_limit = y_limit or auto_y_lim

    # Compute Density
    density = _density2d(x_vals, y_vals, n=100, margin='n')
    idx = density.argsort()
    x_vals, y_vals, density = x_vals[idx], y_vals[idx], density[idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot outliers
    if remove_outlier and show_outlier:
        margin = 0.01
        clipped_x = np.clip(outlier_x, x_limit[0] + margin, x_limit[1] - margin)
        clipped_y = np.clip(outlier_y, y_limit[0] + margin, y_limit[1] - margin)
        ax.scatter(clipped_x, clipped_y, color="grey", s=size + 10, alpha=1, label="Outliers")

    # Main scatter
    scatter = ax.scatter(x_vals, y_vals, c=density, s=size, cmap="viridis", alpha=1, rasterized=True, antialiased=True)
    fig.colorbar(scatter, ax=ax, label="Density")

    # Axis labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y} ({mode_slot})")

    if x_limit:
        ax.set_xlim(x_limit)
    if y_limit:
        ax.set_ylim(y_limit)

    # Diagonal
    if diagonal:
        _plot_diagonal(ax, diagonal, np.linspace(*ax.get_xlim(), 100))

    # Cross lines
    if cross:
        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")

    # Highlight
    if highlight:
        _highlight_points(ax, data, x_vals_all, y_vals_all, highlight, size)

    ax.grid(False)

    if path_for_save:
        fig.savefig(f"{path_for_save}{x}_{y}_{mode_slot}.png", format="png", dpi=300)
    plt.tight_layout()
    plt.show()
    plt.close()


def _transform_no(matrix: np.ndarray) -> np.ndarray:
    return matrix

def _transform_z(matrix: np.ndarray, center: bool = True, scale: bool = True) -> np.ndarray:
    if not center and not scale:
        return matrix.astype(np.float64)

    from scipy.stats import zscore
    print(matrix)
    return zscore(matrix, axis=1, ddof=1, nan_policy='omit' if (center or scale) else 'propagate')

def _transform_vst(data, selected_columns: list, mode_slot, genes) -> pd.DataFrame:
    mat = data.get_table(mode_slots=mode_slot, columns=selected_columns, genes=genes)
    mat = mat.loc[:, mat.notna().any(axis=0)]

    selected_columns_valid = mat.columns.tolist()

    coldata = data.coldata.loc[selected_columns_valid]
    coldata["condition"] = coldata["Condition"]

    if str(mode_slot).lower() == "count":
        slotmat = mat.T.round().astype(int)
    else:
        slotmat = mat.T

    slotmat = slotmat.loc[coldata.index]
    try:
        dds = DeseqDataSet(counts=slotmat, metadata=coldata, design_factors="Condition")

    except Exception:
        warnings.warn("Column 'Condition' not found in coldata. Please add a 'Condition' column to use this function.")

    dds.deseq2()
    dds.vst_fit()
    vst_array = dds.vst_transform()
    vst_df = pd.DataFrame(vst_array, index=slotmat.index, columns=slotmat.columns)
    return vst_df

def _transform_logfc() -> np.ndarray:
    ... #TODO LFC funktion in diffexpr fehlt noch dafür

def _make_continuous_colors(values, colors=None, breaks=None):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    def quantile(arr, q):
        return np.nanpercentile(arr, q*100)

    if quantile(values, 0.25) < 0:
        quant = [50, 95]

        if breaks == "minmax":
            ll = np.nanmax(np.abs(values))
            breaks = np.linspace(-ll, ll, 5)
        elif isinstance(breaks, int):
            quant = [q * 100 for q in np.linspace(0, 1, breaks + 1)[1:-1]] + [95]
            breaks = None

        if breaks is None:
            upper = np.nanpercentile(values[values > 0], quant)
            lower = np.nanpercentile(np.abs(values[values < 0]), quant)
            pm = np.maximum(upper, lower)
            breaks = [-b for b in reversed(pm)] + [0] + list(pm)

        if colors is None:
            colors = ["#CA0020", "#F4A582", "#F7F7F7", "#92C5DE", "#0571B0"]

    else:
        quant = [5, 25, 50, 75, 95]

        if breaks == "minmax":
            breaks = np.linspace(np.nanmin(values), np.nanmax(values), 5)
        elif isinstance(breaks, int):
            quant = [5] + [q * 100 for q in np.linspace(0, 1, breaks)[1:-1]] + [95]
            breaks = None

        if breaks is None:
            breaks = np.nanpercentile(values, quant)

        if colors is None:
            colors = ["#FFFFB2", "#FECC5C", "#FD8D3C", "#F03B20", "#BD0026"]
    reverse = False
    if isinstance(colors, str):
        if colors.startswith("rev"):
            reverse = True
            colors = colors[3:]
        try:
            cmap = cm.get_cmap(colors, len(breaks))
            color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
        except ValueError:
            cmap = cm.get_cmap("viridis", len(breaks))
            color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]

        if reverse:
            color_list = color_list[::-1]

        colors = color_list
    else:
        colors = list(colors)
    return {"breaks": breaks, "colors": colors}

def plot_heatmap(
    data,
    mode_slot: Union[str, list, None] = None,
    columns: Optional[Union[str, list]] = None,
    genes: Optional[list] = None,
    transform: Union[str, callable] = "Z",
    cluster_genes: bool = True,
    cluster_columns: bool = False,
    label_genes: Optional[bool] = None,
    xlabels: Optional[list] = None,
    breaks: Optional[list] = None,
    colors: Optional[Union[list, str]] = None,
    title: Optional[str] = None,
    return_matrix: bool = False,
    na_to: Optional[float] = None,
):
    """
     Create heatmaps from grandR objects.

     Convenience method to compare among more two variables (slot data or analyses results).

     Parameters
     ----------
     data : GrandPy
         The GrandPy object containing the data to visualize.

     mode_slot : str or list of str, optional
         Either one or more mode.slot specifications (e.g., "count", "new_count"),
         or names matching analysis results. If None, uses the default slot.

     columns : str or list, optional
         The columns (samples/cells) to include. Can be a logical expression
         over the sample metadata, a list of column names, or None for all columns.

     genes : list, optional
         A list of gene names to restrict the plot to. If None, uses all genes.

     transform : str or callable, default="Z"
         Transformation to apply to the data matrix. Possible string values are
         "Z" (z-score per row), "vst" (variance stabilizing transform),
         "logFC" (log2 fold change), or "none".

     cluster_genes : bool, default=True
         Whether to cluster genes (rows) hierarchically.

     cluster_columns : bool, default=False
         Whether to cluster samples/cells (columns) hierarchically.

     label_genes : bool, optional
         Whether to show gene names on the y-axis. Defaults to True if number
         of genes is <=50, otherwise False.

     xlabels : list, optional
         Custom labels for the x-axis. Only valid if a single mode_slot is specified.

     breaks : list, optional
         Numeric vector specifying color breaks for the heatmap.

     colors : list or str, optional
         A color palette or a list of colors for the heatmap gradient.

     title : str, optional
         The title for the heatmap.

     return_matrix : bool, default=False
         If True, prints the matrix used for plotting.

     na_to : float, optional
         Value to substitute for missing (NA) values before plotting.

     See Also
        --------
        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_plot
            Add a plot function.

        GrandPy.with_dropped_plots
            Remove plots matching a regex.

        GrandPy.plot_global
            Executes a stored global plot function.


     """

    if mode_slot is None:
        mode_slot = data.default_slot

    mode_slots = (
        [_parse_as_mode_slot(t) for t in mode_slot]
        if isinstance(mode_slot, list)
        else [_parse_as_mode_slot(mode_slot)]
    )

    is_slot = all(m.mode is not None for m in mode_slots)
    is_analysis = all(m.mode is None for m in mode_slots)

    if not (is_slot or is_analysis):
        raise ValueError("Cannot mix data slot and analysis in 'type'!")

    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    if is_slot:
        if len(mode_slots) > 1 and xlabels is not None:
            raise ValueError("Cannot use 'xlabels' with multiple slots")

        table = data.get_table(mode_slots=mode_slot, genes=genes, columns=selected_columns)
    else:
        table = data.get_analysis_table(names=[ms.slot for ms in mode_slots], genes=genes)
        table = table[selected_columns]
    mat = table.to_numpy(dtype=np.float64)
    gene_names = table.index.to_list()
    sample_names = table.columns.to_list()

    if isinstance(transform, str):
        print(mat)
        transform = transform.lower()
        if transform == "z":
            mat = _transform_z(mat)
            label = "z score"
        elif transform in ["no", "none"]:
            mat = _transform_no(mat)
            label = " "
        elif transform == "vst":
            df_vst = _transform_vst(data, selected_columns, mode_slot=mode_slot, genes=genes)

            if genes is not None:
                df_vst = df_vst[genes]
            sample_names = df_vst.index.to_list()
            gene_names = df_vst.columns.to_list()
            mat = df_vst.to_numpy()
            mat = mat.T
            label = "VST"
        elif transform == "logfc":
            if selected_columns is None or len(selected_columns) == 0:
                raise ValueError("Need columns=... to compute logFC reference")
            ref_cols = list(range(len(selected_columns)))
            mat = _transform_logfc(mat, ref_columns=ref_cols)
            label = "log2 FC"
        else:
            raise ValueError(f"Unknown transform: {transform}")

    if na_to is not None:
        mat = np.where(np.isnan(mat), na_to, mat)

    if xlabels is not None and len(xlabels) == len(sample_names):
        sample_names = xlabels
    if label_genes is None:
        label_genes = len(gene_names) <= 50

    df = pd.DataFrame(mat, index=gene_names, columns=sample_names)
    color_df = _make_continuous_colors(mat, colors=colors, breaks=breaks)

    breaks = color_df["breaks"]
    colors = color_df["colors"]

    min_break, max_break = breaks[0], breaks[-1]
    scaled_breaks = [(b - min_break) / (max_break - min_break) for b in breaks]
    color_list = list(zip(scaled_breaks, colors))

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import Normalize
    cmap = LinearSegmentedColormap.from_list("custom", color_list)
    norm = Normalize(vmin=min_break, vmax=max_break)


    sns.clustermap(
        df,
        figsize=(10, 6),
        cmap=cmap,
        norm = norm,
        row_cluster=cluster_genes,
        col_cluster=cluster_columns,
        yticklabels=label_genes,
        xticklabels=True,
        cbar_kws={"label": label},

    )

    if return_matrix:
        print(df)
    if title:
        plt.title(title, y=1.05)
    plt.tight_layout()
    plt.show()
    plt.close()


#Beispielaufruf: plot_pca(sars)
def plot_pca(
    data: GrandPy,
    mode_slot: str | ModeSlot = None,
    path_for_save: Optional[str] = None,
    ntop: int = 500,
    aest: Optional[dict] = None,
    x: int = 1,
    y: int = 2,
    columns: Union[str, list, None] = None,
    do_vst: bool = True):

    if mode_slot is None:
        mode_slot = data.default_slot

    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    mat = data.get_table(mode_slots=mode_slot, columns=selected_columns)
    coldata = data.coldata.loc[selected_columns]

    mat = mat.loc[:, mat.notna().any(axis=0)]
    coldata = coldata.loc[mat.columns]

    if str(mode_slot).lower() == "count":
        slotmat = mat.T.round().astype(int)
    else:
        slotmat = mat.T

    if do_vst and str(mode_slot).lower() == "count":
        try:
            dds = DeseqDataSet(counts=slotmat, metadata=coldata, design_factors="Condition")

        except Exception:
            warnings.warn(
                "Column 'Condition' not found in coldata. Please add a 'Condition' column to use this function.")

        dds.deseq2()
        dds.vst_fit()
        vst_array = dds.vst_transform()

        vst_df = pd.DataFrame(vst_array, index=slotmat.index, columns=slotmat.columns)
        top_genes = vst_df.var().nlargest(min(ntop, vst_df.shape[1])).index
        mat_for_pca = vst_df[top_genes]
    else:
        top_genes = slotmat.var().nlargest(min(ntop, slotmat.shape[1])).index
        mat_for_pca = slotmat[top_genes]

    # PCA
    pca = PCA()
    pcs = pca.fit_transform(mat_for_pca)
    percent_var = pca.explained_variance_ratio_
    pc_df = pd.DataFrame(pcs, index=mat_for_pca.index, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])])
    df = pd.concat([pc_df, coldata], axis=1)

    aest = _setup_default_aes(data, aest)
    style = aest.get("shape")
    hue = aest.get("color")


    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=f"PC{x}", y=f"PC{y}", style=style, hue=hue, s=50)
    plt.xlabel(f"PC{x}: {percent_var[x - 1] * 100:.1f}% variance")
    plt.ylabel(f"PC{y}: {percent_var[y - 1] * 100:.1f}% variance")
    plt.title(f"PCA_({mode_slot})")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    if path_for_save:
        plt.savefig(f"{path_for_save}/PCA_{mode_slot}.png", dpi=300)
    plt.show()

#Beispielaufruf: plot_gene_old_vs_new(sars, "UHMK1", show_ci=True)
def plot_gene_old_vs_new(
    data: GrandPy,
    gene: str,
    slot: Optional[str] = None,
    columns: Optional[Union[list, str]] = None,
    log: bool = True,
    show_ci: bool = False,
    aest: Optional[dict] = None,
    size: float = 50
):

    if slot is None:
        slot = data.default_slot


    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    coldata = data.coldata.loc[selected_columns]

    plot_df = data.get_data(mode_slots=[ModeSlot("old", slot), ModeSlot("new", slot)], genes=gene, columns=selected_columns, with_coldata=True)

    new_names = {plot_df.columns[-2]: "old", plot_df.columns[-1]: "new"}
    plot_df = plot_df.rename(columns=new_names)

    aest = _setup_default_aes(data, aest)
    style = aest.get("shape")
    hue = aest.get("color")
    if hue not in plot_df.columns:
        hue = None
    if style not in plot_df.columns:
        style = None

    fig, ax = plt.subplots(figsize=(10, 6))

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel(f"Old RNA ({slot})")
    ax.set_ylabel(f"New RNA ({slot})")
    ax.set_title(f"Gene: {gene}")

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")

        ci_lower = data.get_matrix(mode_slot="lower", genes=gene, columns=selected_columns)
        ci_upper = data.get_matrix(mode_slot="upper", genes=gene, columns=selected_columns)
        total = data.get_matrix(mode_slot=slot, genes=gene, columns=selected_columns)

        plot_df["ci_lower"] = ci_lower
        plot_df["ci_upper"] = ci_upper
        plot_df["total"] = total

        valid_mask = (
                (plot_df["old"] > 0) &
                (plot_df["new"] > 0) &
                (plot_df["ci_lower"] >= 0) &
                (plot_df["ci_upper"] >= 0) &
                (plot_df["ci_lower"] <= plot_df["ci_upper"])
        )
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            warnings.warn(f"{n_invalid} data points with invalid CI were excluded from error bars.", UserWarning)

        df_ci = plot_df[valid_mask]

        ymin = df_ci["ci_lower"] * df_ci["total"]
        ymax = df_ci["ci_upper"] * df_ci["total"]
        yerr = [df_ci["new"] - ymin, ymax - df_ci["new"]]

        xmin = (1 - df_ci["ci_upper"]) * df_ci["total"]
        xmax = (1 - df_ci["ci_lower"]) * df_ci["total"]
        xerr = [df_ci["old"] - xmin, xmax - df_ci["old"]]

        if hue and hue in df_ci.columns:
            unique_groups = df_ci[hue].dropna().unique()
            try:
                unique_groups = sorted(unique_groups)
            except TypeError:
                unique_groups = list(unique_groups)
            palette = sns.color_palette(n_colors=len(unique_groups))
            color_map = {grp: col for grp, col in zip(unique_groups, palette)}

            for grp in unique_groups:
                grp_mask = df_ci[hue] == grp
                ax.errorbar(
                    df_ci.loc[grp_mask, "old"],
                    df_ci.loc[grp_mask, "new"],
                    xerr=[
                        df_ci.loc[grp_mask, "old"] - xmin[grp_mask],
                        xmax[grp_mask] - df_ci.loc[grp_mask, "old"]
                    ],
                    yerr=[
                        df_ci.loc[grp_mask, "new"] - ymin[grp_mask],
                        ymax[grp_mask] - df_ci.loc[grp_mask, "new"]
                    ],
                    fmt='none',
                    ecolor=color_map[grp],
                    capsize=3,
                    linewidth=1,
                )
        else:
            ax.errorbar(
                df_ci["old"],
                df_ci["new"],
                xerr=xerr,
                yerr=yerr,
                fmt='none',
                ecolor='grey',
                capsize=3,
                linewidth=1,
            )

    sns.scatterplot(
        data=plot_df,
        x="old",
        y="new",
        hue=hue,
        style=style,
        s=size,
        ax=ax
    )
    plt.tight_layout()
    plt.show()
    plt.close()

#Beispielaufruf: plot_gene_total_vs_ntr(sars, "UHMK1")
def plot_gene_total_vs_ntr(
    data: GrandPy,
    gene: str,
    slot: Optional[str] = None,
    columns: Optional[Union[list, str]] = None,
    log: bool = True,
    show_ci: bool = False,
    aest: Optional[dict] = None,
    size: float = 50
):
    if slot is None:
        slot = data.default_slot

    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    coldata = data.coldata.loc[selected_columns]

    plot_df = data.get_data(mode_slots=[slot, "ntr"], genes=gene, columns=selected_columns, with_coldata=True)

    new_names = {plot_df.columns[-2]: "total", plot_df.columns[-1]: "ntr"}
    plot_df = plot_df.rename(columns=new_names)

    aest = _setup_default_aes(data, aest)
    hue = aest.get("color")
    style = aest.get("shape")

    if hue not in plot_df.columns:
        hue = None
    if style not in plot_df.columns:
        style = None

    if hue and hue in plot_df.columns:
        unique_groups = plot_df[hue].dropna().unique()
        palette = sns.color_palette("Set2", n_colors=len(unique_groups))
        color_map = {grp: col for grp, col in zip(unique_groups, palette)}
    else:
        palette = None
        color_map = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())


    ax.set_xlabel(f"Total RNA ({slot})")
    ax.set_ylabel("NTR")
    ax.set_title(f"Gene: {gene}")

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")

        ci_lower = data.get_matrix(mode_slot="lower", genes=gene, columns=selected_columns)
        ci_upper = data.get_matrix(mode_slot="upper", genes=gene, columns=selected_columns)

        plot_ci = plot_df.assign(ci_lower=ci_lower, ci_upper=ci_upper)

        if hue and hue in coldata.columns:
            plot_ci[hue] = coldata[hue]

        valid_mask = (
            (plot_ci["total"] > 0) &
            (plot_ci["ntr"] >= 0) &
            (plot_ci["ci_lower"] >= 0) &
            (plot_ci["ci_upper"] >= 0) &
            (plot_ci["ci_lower"] <= plot_ci["ntr"]) &
            (plot_ci["ci_upper"] >= plot_ci["ntr"])
        )

        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            warnings.warn(f"{n_invalid} data points with invalid CI were excluded from error bars.", UserWarning)

        plot_ci = plot_ci[valid_mask]

        unique_groups = plot_ci[hue].dropna().unique()
        palette = sns.color_palette(n_colors=len(unique_groups))
        color_map = {grp: col for grp, col in zip(sorted(unique_groups), palette)}
        if hue and hue in plot_ci.columns:
            for grp in plot_ci[hue].dropna().unique():
                grp_df = plot_ci[plot_ci[hue] == grp]
                x = grp_df["total"].to_numpy()
                y = grp_df["ntr"].to_numpy()
                yerr = [
                    y - grp_df["ci_lower"].to_numpy(),
                    grp_df["ci_upper"].to_numpy() - y
                ]
                ax.errorbar(
                    x, y, yerr=yerr,
                    fmt='none',
                    ecolor=color_map[grp],
                    capsize=3
                )
        else:
            x = plot_ci["total"].to_numpy()
            y = plot_ci["ntr"].to_numpy()
            yerr = [
                y - plot_ci["ci_lower"].to_numpy(),
                plot_ci["ci_upper"].to_numpy() - y
            ]
            ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='gray', capsize=3)

    sns.scatterplot(
        data=plot_df,
        x="total",
        y="ntr",
        hue=hue,
        style=style,
        s=size,
        ax=ax
    )

    plt.tight_layout()
    plt.show()
    plt.close()

#Beispielaufruf: plot_gene_groups_points(sars, "UHMK1", group="Time")
def plot_gene_groups_points(
    data: GrandPy,
    gene: str,
    group: str = "Condition",
    mode_slot: Union[str, ModeSlot] = None,
    columns: Optional[Union[list, str]] = None,
    log: bool = True,
    show_ci: bool = False,
    aest: Optional[dict] = None,
    size: float = 50,
    transform: Optional[callable] = None,
    dodge: bool = False,
):

    if mode_slot is None:
        mode_slot = data.default_slot

    mode_slot = _parse_as_mode_slot(mode_slot)
    slot = mode_slot.slot
    mode = mode_slot.mode

    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    coldata = data.coldata.loc[selected_columns]

    if slot == "ntr":
        plot_df = data.get_data(mode_slots="ntr", genes=gene, columns=selected_columns, with_coldata=True)
        plot_df["ntr"] = plot_df[gene]
        plot_df["value"] = plot_df[gene]
        log = False
    else:
        plot_df = data.get_data(mode_slots=[slot, "ntr"], genes=gene, columns=selected_columns, with_coldata=True)
        new_names = {plot_df.columns[-2]: "slot_val", plot_df.columns[-1]: "ntr"}
        plot_df = plot_df.rename(columns=new_names)

        if mode == "total":
            plot_df["value"] = plot_df["slot_val"]
        elif mode == "new":
            plot_df["value"] = plot_df["slot_val"] * plot_df["ntr"]
        elif mode == "old":
            plot_df["value"] = plot_df["slot_val"] * (1 - plot_df["ntr"])
        else:
            raise ValueError(f"Unknown mode '{mode}' in mode_slot '{mode_slot}'")
    plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]

    if group not in plot_df.columns:
        raise ValueError(f"Group column '{group}' not found in coldata!")

    if transform is not None: #TODO Was tut das? In R: if (!is.null(transform)) df=transform(df)
        plot_df = transform(plot_df)

    aest = _setup_default_aes(data, aest)
    hue = aest.get("color")
    style = aest.get("shape")

    fig, ax = plt.subplots(figsize=(10, 6))
    group_order = plot_df[group].unique()
    group_to_num = {g: i for i, g in enumerate(group_order)}
    plot_df["_xbase"] = plot_df[group].map(group_to_num)

    if dodge and "Replicate" in plot_df.columns:
        replicates = sorted(plot_df["Replicate"].unique())
        rep_to_offset = {r: (i - (len(replicates) - 1) / 2) * 0.15 for i, r in enumerate(replicates)}
        plot_df["_xpos"] = plot_df["_xbase"] + plot_df["Replicate"].map(rep_to_offset)
    else:
        plot_df["_xpos"] = plot_df["_xbase"]


    sns.scatterplot(
        data=plot_df,
        x="_xpos",
        y="value",
        hue=hue,
        style=style,
        s=size,
        ax=ax
    )
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order)

    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ylabel = "NTR" if slot == "ntr" else f"{mode.capitalize()} RNA ({slot})"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(f"{gene} by {group}")
    plt.xticks(rotation=90)

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")

        ci_lower = data.get_matrix(mode_slot="lower", genes=gene, columns=selected_columns)
        ci_upper = data.get_matrix(mode_slot="upper", genes=gene, columns=selected_columns)

        plot_df = plot_df.assign(lower=ci_lower, upper=ci_upper)

        if slot == "ntr":
            ymin = plot_df["lower"]
            ymax = plot_df["upper"]
        elif mode == "new":
            ymin = plot_df["lower"] * plot_df["slot_val"]
            ymax = plot_df["upper"] * plot_df["slot_val"]
        elif mode == "old":
            ymin = (1 - plot_df["upper"]) * plot_df["slot_val"]
            ymax = (1 - plot_df["lower"]) * plot_df["slot_val"]
        else:
            ymin = ymax = None

        if ymin is not None:
            mask = (
                    (plot_df["value"] >= 0) &
                    (ymin >= 0) & (ymax >= 0) &
                    (ymin <= plot_df["value"]) & (ymax >= plot_df["value"])
            )
            df_ci = plot_df[mask]
            yerr = [
                df_ci["value"] - ymin[mask],
                ymax[mask] - df_ci["value"]
            ]

            if hue and hue in plot_df.columns:
                unique_groups = df_ci[hue].unique()
                palette = sns.color_palette(n_colors=len(unique_groups))
                color_map = {grp: col for grp, col in zip(sorted(unique_groups), palette)}

                for grp in unique_groups:
                    grp_mask = df_ci[hue] == grp
                    ax.errorbar(
                        x=df_ci.loc[grp_mask, "_xpos"],
                        y=df_ci.loc[grp_mask, "value"],
                        yerr=[
                            (df_ci.loc[grp_mask, "value"] - ymin[mask][grp_mask]),
                            (ymax[mask][grp_mask] - df_ci.loc[grp_mask, "value"])
                        ],
                        fmt='none',
                        ecolor=color_map[grp],
                        capsize=3,
                        linewidth=1,
                    )
            else:
                ax.errorbar(
                    x=df_ci["_xpos"],
                    y=df_ci["value"],
                    yerr=yerr,
                    fmt='none',
                    ecolor='gray',
                    capsize=3,
                    linewidth=1,
                )

    if hue:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.close()

#Beispielaufruf: plot_gene_groups_bars(sars, "UHMK1", xlabels="Condition + '.' + Replicate")
def plot_gene_groups_bars(
    data: GrandPy,
    gene: str,
    slot: Optional[str] = None,
    columns: Optional[Union[str, list]] = None,
    show_ci: bool = False,
    xlabels: Optional[Union[str, list]] = None,
    transform: Optional[callable] = None
):
    if slot is None:
        slot = data.default_slot

    mode_slot_old = ModeSlot("old", slot)
    mode_slot_new = ModeSlot("new", slot)

    if columns is None:
        selected_columns = data.columns
    else:
        selected_columns = data.get_columns(columns)

    coldata = data.coldata.loc[selected_columns]

    if isinstance(xlabels, list):
        xlabels = xlabels
    elif isinstance(xlabels, str):
        local_vars = {col: coldata[col].astype(str) for col in coldata.columns}
        try:
            xlabels = eval(xlabels, {}, local_vars).tolist()
        except Exception as e:
            raise ValueError(f"xlab expression could not be evaluated: {e}")
    else:
        xlabels = coldata["Name"].tolist()

    plot_df = data.get_data(mode_slots=[mode_slot_old, mode_slot_new], genes=gene, columns=selected_columns)

    new_names = {plot_df.columns[-2]: "old", plot_df.columns[-1]: "new"}
    plot_df = plot_df.rename(columns=new_names)

    plot_df["xlab"] = xlabels

    if isinstance(transform, str):
        mat = plot_df[["old", "new"]]
        transform = transform.lower()
        if transform == "z":
            mat = _transform_z(mat)
            label = "z score"
        elif transform in ["no", "none"]:
            mat = _transform_no(mat)
            label = " "
        elif transform == "vst":
            df_vst = _transform_vst(data, selected_columns, mode_slot=mode_slot, genes=genes)

            if genes is not None:
                df_vst = df_vst[genes]
            sample_names = df_vst.index.to_list()
            gene_names = df_vst.columns.to_list()
            mat = df_vst.to_numpy()
            mat = mat.T
            label = "VST"
        elif transform == "logfc":
            if selected_columns is None or len(selected_columns) == 0:
                raise ValueError("Need columns=... to compute logFC reference")
            ref_cols = list(range(len(selected_columns)))
            mat = _transform_logfc(mat, ref_columns=ref_cols)
            label = "log2 FC"
        else:
            raise ValueError(f"Unknown transform: {transform}")

    x = np.arange(len(plot_df))
    width = 0.8

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x, plot_df["old"], width, label="Old", color="lightgray")
    bar2 = ax.bar(x, plot_df["new"], width, bottom=plot_df["old"], label="New", color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["xlab"], rotation=90)
    ax.set_ylabel(f"Total RNA ({slot})")
    ax.set_xlabel("")
    ax.set_title(gene)

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")

        total = data.get_matrix(mode_slot=slot, genes=gene, columns=selected_columns)
        lower = data.get_matrix(mode_slot="lower", genes=gene, columns=selected_columns)
        upper = data.get_matrix(mode_slot="upper", genes=gene, columns=selected_columns)


        ymin = (1 - upper) * total
        ymax = (1 - lower) * total

        err_low = total - ymin
        err_high = ymax - total

        mask = (
                np.isfinite(total) &
                np.isfinite(lower) & np.isfinite(upper) &
                np.isfinite(err_low) & np.isfinite(err_high) &
                (lower <= upper)
        )

        if np.any(mask):
            x_valid = np.arange(len(total))[mask]
            total_valid = total[mask]
            err_valid = [err_low[mask], err_high[mask]]

            for i, (xi, y0, y1, valid) in enumerate(zip(x, ymin, ymax, mask)):
                if valid:
                    ax.plot([xi, xi], [y0, y1], color="black", linewidth=1)

    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# Beispielsaufruf: plot_gene_snapshot_timecourse(sars, "UHMK1")
def plot_gene_snapshot_timecourse(
    data: GrandPy,
    gene: str,
    time: str = "Time.original",
    mode_slot: Union[str, ModeSlot, None] = None,
    columns: Optional[Union[str, list]] = None,
    average_lines: bool = True,
    exact_tics: bool = True,
    log: bool = True,
    show_ci: bool = False,
    aest: Optional[dict] = None,
    size: float = 50,
    dodge: bool = False
):


    if mode_slot is None:
        mode_slot = data.default_slot

    mode_slot = _parse_as_mode_slot(mode_slot)
    slot = mode_slot.slot
    mode = mode_slot.mode
    log = False if slot == "ntr" else log

    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    df = data.get_data(mode_slots=mode_slot, genes=gene, columns=selected_columns, with_coldata=True, ntr_nan=False)

    if time not in df.columns:
        raise ValueError(f"Column '{time}' not found in coldata!")
    x_vals_numeric = df[time].apply(parse_time_to_float)

    if not exact_tics:
        x_breaks = sorted(df[time].unique())
    else:
        x_breaks_numeric = np.linspace(x_vals_numeric.min(), x_vals_numeric.max(), 5)
        x_breaks = [f"{int(round(x))}h" for x in x_breaks_numeric]

    aest = _setup_default_aes(data, aest)
    hue = aest.get("color")
    style = aest.get("shape")

    if hue not in df.columns:
        hue = None
    if style not in df.columns:
        style = None
    df["Time_float"] = x_vals_numeric

    df["Time_float_dodged"] = df["Time_float"]

    if dodge and hue and hue in df.columns:
        hue_values = sorted(df[hue].unique())
        n = len(hue_values)
        if n > 1:
            spread = min(0.2, 0.1 * (n - 1))
            offset_map = {
                val: (-spread + 2 * spread * i / (n - 1)) for i, val in enumerate(hue_values)
            }
        else:
            offset_map = {hue_values[0]: 0}

        df["Time_float_dodged"] = df.apply(
            lambda row: row["Time_float"] + offset_map.get(row[hue], 0),
            axis=1
        )

    x = "Time_float"
    y = gene
    ylabel = "NTR" if slot == "ntr" else f"{mode.capitalize()} RNA ({slot})"
    fig, ax = plt.subplots(figsize=(10, 6))

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")


        dfslot = data.get_matrix(mode_slot=slot, genes=gene, columns=selected_columns)
        dfmode_slot = data.get_matrix(mode_slot=mode_slot, genes=gene, columns=selected_columns)
        lower = data.get_matrix(mode_slot="lower", genes=gene, columns=selected_columns)
        upper = data.get_matrix(mode_slot="upper", genes=gene, columns=selected_columns)


        if mode_slot.slot == "ntr":
            ymin = lower
            ymax = upper
        elif mode_slot.mode == "new":
            ymin = lower * dfslot
            ymax = upper * dfslot
        elif mode_slot.mode == "old":
            ymin = (1 - upper) * dfslot
            ymax = (1 - lower) * dfslot
        elif mode_slot.mode == "total":
            ymin = dfmode_slot
            ymax = dfmode_slot
        else:
            raise ValueError(f"Unknown mode: {mode_slot.mode}")

        err_low = dfmode_slot - ymin
        err_high = ymax - dfmode_slot

        x_all = df[time].apply(parse_time_to_float).to_numpy()

        if dodge and hue and hue in df.columns:
            hue_vals = sorted(df[hue].unique())
            n = len(hue_vals)
            if n > 1:
                spread = min(0.2, 0.1 * (n - 1))
                offsets = {val: (-spread + 2 * spread * i / (n - 1)) for i, val in enumerate(hue_vals)}
            else:
                offsets = {hue_vals[0]: 0.0}
            hue_col = df[hue].to_numpy()
            x_dodged = np.array([x + offsets.get(h, 0.0) for x, h in zip(x_all, hue_col)])
        else:
            x_dodged = x_all

        mask = (
                np.isfinite(dfmode_slot) & np.isfinite(lower) & np.isfinite(upper) &
                np.isfinite(err_low) & np.isfinite(err_high) &
                (dfmode_slot >= 0) & (lower >= 0) & (upper >= 0) &
                (lower <= upper) & (err_low >= 0) & (err_high >= 0)
        )

        n_invalid = np.sum(~mask)
        if n_invalid > 0:
            warnings.warn(f"{n_invalid} data points with invalid CI were excluded from error bars.", UserWarning)

        if np.any(mask):
            x_valid_all = x_dodged[mask]
            y_valid_all = dfmode_slot[mask]
            err_low_valid = err_low[mask]
            err_high_valid = err_high[mask]

            if hue and hue in df.columns:
                hue_col = df[hue].to_numpy()[mask]
                palette = sns.color_palette(n_colors=len(np.unique(hue_col)))
                hue_to_color = {val: palette[i] for i, val in enumerate(sorted(np.unique(hue_col)))}

                for val in sorted(np.unique(hue_col)):
                    group_mask = hue_col == val
                    ax.errorbar(
                        x=x_valid_all[group_mask],
                        y=y_valid_all[group_mask],
                        yerr=[err_low_valid[group_mask], err_high_valid[group_mask]],
                        fmt='none',
                        ecolor=hue_to_color[val],
                        capsize=3
                    )
            else:
                ax.errorbar(
                    x=x_valid_all,
                    y=y_valid_all,
                    yerr=[err_low_valid, err_high_valid],
                    fmt='none',
                    ecolor='black',
                    capsize=3
                )


    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xticks(x_breaks if not exact_tics else x_breaks_numeric)
    ax.set_xticklabels(x_breaks)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(gene)

    if average_lines:
        avg_input_df = df.drop(columns=["Replicate"], errors="ignore")
        group_cols = [x]
        if hue and hue in avg_input_df.columns:
            group_cols.append(hue)
        if style and style in avg_input_df.columns and style != hue:
            group_cols.append(style)
        avg_df = avg_input_df.groupby(group_cols, observed=True)[y].mean().reset_index()

        sns.lineplot(
            data=avg_df,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            legend=False,
            linewidth=1
        )
    sns.scatterplot(data=df, x="Time_float_dodged", y=y, hue=hue, style=style, s=size, ax=ax)
    plt.tight_layout()
    plt.show()
    plt.close()

#TODO Noch mehr testen mit analysen die wir aber noch nicht haben
def plot_vulcano(
    data: GrandPy,
    analyses = None,
    p_cutoff: float = 0.05,
    lfc_cutoff: float = 1,
    annotate_numbers=False #TODO Default ist True aber noch nicht implementiert
):
    if analyses is None:
        analyses = data.analyses[0]
    df = data.get_analysis_table(analyses=analyses, regex=False, columns=["LFC", "Q"], with_gene_info=False)
    df.columns = [col.split('.')[-1] for col in df.columns]
    df['neg_log10_Q'] = -np.log10(df['Q'])

    x_vals_all = df['LFC'].to_numpy()
    y_vals_all = df['neg_log10_Q'].to_numpy()

    if np.all(np.isnan(x_vals_all)):
        raise ValueError(f"Alle LFC Werte sind NaN. Plot nicht möglich.")
    if np.all(np.isnan(y_vals_all)):
        raise ValueError(f"Alle Q Werte sind NaN. Plot nicht möglich.")

    mask_keep, _, _ = _apply_outlier_filter(x_vals_all, y_vals_all, remove=True)
    x_vals = x_vals_all[mask_keep]
    y_vals = y_vals_all[mask_keep]

    try:
        density = _density2d(x_vals, y_vals, n=100, margin='n')
    except NameError:
        density = None

    fig, ax = plt.subplots(figsize=(10, 6))

    if density is not None:
        idx = density.argsort()
        x_vals, y_vals, density = x_vals[idx], y_vals[idx], density[idx]
        scatter = ax.scatter(x_vals, y_vals, c=density, s=10, cmap="viridis", alpha=0.7)
        fig.colorbar(scatter, ax=ax, label="Density")
    else:
        scatter = ax.scatter(x_vals, y_vals, c='blue', s=10, alpha=0.7)

    ax.axhline(-np.log10(p_cutoff), linestyle='--', color='grey')
    if lfc_cutoff != 0:
        ax.axvline(lfc_cutoff, linestyle='--', color='grey')
        ax.axvline(-lfc_cutoff, linestyle='--', color='grey')
    else:
        ax.axvline(0, linestyle='--', color='grey')

    ax.set_xlabel(r'$\log_2$ Fold Change (LFC)')
    ax.set_ylabel(r'$-\log_{10}$ FDR (Q)')
    ax.set_title(analyses)

    plt.tight_layout()
    plt.show()

def f_old_nonequi(t, f0, ks, kd):
    return f0 * np.exp(-t * kd)

def f_new(t, ks, kd):
    return ks / kd * (1 - np.exp(-t * kd))

def parse_time_str(t):
    match = re.match(r"(\d+(?:\.\d+)?)", str(t))
    return float(match.group(1)) if match else 0

# Beispielaufruf: plot_gene_progressive_timecourse(sars, "UHMK1")
def plot_gene_progressive_timecourse( # TODO docstring mit see also fit kinets wegen kwargs
    data: GrandPy,
    gene: str,
    slot: Optional[str] = None,
    time: str = "Time.original",
    fit_type: str = "nlls",
    size: float = 50,
    exact_tics: bool = True,
    **kwargs
):

    if slot is None:
        slot = data.default_slot

    timepoints = data.coldata[time].values
    time_numeric = np.array([parse_time_str(t) for t in timepoints])

    condition = (
        data.coldata["Condition"]
        if "Condition" in data.coldata.columns
        else pd.Series([gene] * len(timepoints))
    )

    total = data.get_matrix(mode_slot=slot, genes=gene)
    new = data.get_matrix(mode_slot=ModeSlot("new", slot), genes=gene)
    old = data.get_matrix(mode_slot=ModeSlot("old", slot), genes=gene)

    df = pd.DataFrame({
        "time": timepoints,
        "time_numeric": time_numeric,
        "total": total.flatten(),
        "new": new.flatten(),
        "old": old.flatten(),
        "condition": condition
    })


    from Py.utils import _get_kinetics_data
    fit_results = _get_kinetics_data(
        data,
        genes=gene,
        fit_type=fit_type,
        return_fields=["Synthesis", "Half-life", "f0", "Degradation"],
        show_progress=True,
        **kwargs
    )
    if condition.nunique() == 1:
        fit_results = {gene: fit_results}

    tt = np.linspace(0, df["time_numeric"].max(), 200)
    fitted = []
    for cond in df["condition"].unique():
        fit = fit_results.get(f"kinetics_{cond}")
        if fit is None:
            continue

        f0 = fit.loc[gene, f"{cond}_f0"]
        ks = fit.loc[gene, f"{cond}_Synthesis"]
        kd = fit.loc[gene, f"{cond}_Degradation"]

        fitted.append(pd.DataFrame({
            "time_numeric": tt,
            "Expression": f_old_nonequi(tt, f0, ks, kd),
            "Type": "old",
            "condition": cond
        }))
        fitted.append(pd.DataFrame({
            "time_numeric": tt,
            "Expression": f_new(tt, ks, kd),
            "Type": "new",
            "condition": cond
        }))
    df_fitted = pd.concat(fitted, ignore_index=True)

    df_long = pd.melt(
        df,
        id_vars=["time_numeric", "condition"],
        value_vars=["total", "new", "old"],
        var_name="Type",
        value_name="Expression"
    )

    g = sns.FacetGrid(
        df_long,
        col="condition",
        sharey=True,
        sharex=True,
    )

    # Plot total, new, old points
    g.map_dataframe(
        sns.scatterplot,
        x="time_numeric",
        y="Expression",
        hue="Type",
        palette={"total": "gray", "new": "#e34a33", "old": "#2b8cbe"},
        s=size
    )

    # Plot total curve
    g.map_dataframe(
        lambda data, color, **kws: sns.lineplot(
            data=data[data["Type"] == "total"]
            .groupby(["time_numeric", "condition"], as_index=False)
            .mean(numeric_only=True),
            x="time_numeric",
            y="Expression",
            color="gray",
            linestyle="solid",
            linewidth=1, antialiased=True
        )
    )

    # Plot fitted curves
    for cond, ax in zip(df["condition"].unique(), g.axes.flat):
        for line_type in ["new", "old"]:
            df_fit = df_fitted[
                (df_fitted["condition"] == cond) &
                (df_fitted["Type"] == line_type)
                ]
            ax.plot(
                df_fit["time_numeric"],
                df_fit["Expression"],
                linestyle="dashed",
                linewidth=1,
                color={"new": "#e34a33", "old": "#2b8cbe"}[line_type],
                label=f"{line_type} (fit)"
            )

    g.set_ylabels("Expression")
    g.set_titles("{col_name}")
    g.add_legend(title="RNA")
    # Exact tics
    if exact_tics:
        time_original = data.coldata["Time.original"]
        time_numeric = np.array([parse_time_str(t) for t in time_original])
        brdf = pd.DataFrame({
            "time_numeric": time_numeric,
            "time_original": time_original.str.replace("_", ".", regex=False)
        }).drop_duplicates().sort_values("time_numeric")
        for ax in g.axes.flat:
            ax.set_xticks(brdf["time_numeric"])
            ax.set_xticklabels(brdf["time_original"], rotation=45)
            ax.set_xlabel("")
    else:
        unique_times = np.sort(df["time_numeric"].unique())
        for ax in g.axes.flat:
            ax.set_xticks(unique_times)
            ax.set_xticklabels([f"{t:.2f}" for t in unique_times], rotation=45)
            ax.set_xlabel("4sU labeling [h]")
    plt.show()
    plt.close()

#TODO noch nicht einmal getestet
def plot_ma(
    data: GrandPy,
    analysis: Optional[str] = None,
    p_cutoff: float = 0.05,
    lfc_cutoff: float = 1.0,
    annotate_numbers: bool = True,
    path_for_save: Optional[str] = None
):
    if analysis is None:
        analysis = data.analyses[0]

    df = data.get_analysis_table(
        analyses=analysis,
        regex=False,
        columns=["M", "LFC", "Q"],
        with_gene_info=False
    )

    if "Q" not in df.columns:
        df["Q"] = 1.0
    df["Q"] = df["Q"].fillna(1.0)

    x_vals = df["M"].to_numpy() + 1
    y_vals = df["LFC"].to_numpy()
    q_vals = df["Q"].to_numpy()

    colors = np.where(q_vals < p_cutoff, "black", "gray")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_vals, y_vals, c=colors, s=10, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Total expression")
    ax.set_ylabel(r"$\log_2$ Fold Change (LFC)")
    ax.set_title(analysis)

    if lfc_cutoff != 0:
        ax.axhline(y=lfc_cutoff, linestyle="--", color="gray")
        ax.axhline(y=-lfc_cutoff, linestyle="--", color="gray")
    else:
        ax.axhline(y=0, linestyle="--", color="gray")

    if annotate_numbers:
        up = np.sum((y_vals > lfc_cutoff) & (q_vals < p_cutoff))
        down = np.sum((y_vals < -lfc_cutoff) & (q_vals < p_cutoff))
        ax.annotate(f"n={up}", xy=(x_vals.max(), y_vals.max()), xycoords="data",
                    ha="right", va="top")
        ax.annotate(f"n={down}", xy=(x_vals.max(), y_vals.min()), xycoords="data",
                    ha="right", va="bottom")

    plt.tight_layout()
    plt.show()
    plt.close()
