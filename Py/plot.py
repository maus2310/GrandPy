import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Union
from Py.grandPy import GrandPy, ModeSlot
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.preprocessing import scale, StandardScaler
from sklearn.decomposition import PCA
from types import SimpleNamespace
from pydeseq2.dds import DeseqDataSet


def _get_plot_limits(vals, override_lim=None):
    """Compute IQR-based limits if not overridden."""
    if override_lim is not None:
        return override_lim
    q1, q3 = np.percentile(vals[np.isfinite(vals)], [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _apply_outlier_filter(x_vals, y_vals, remove):
    """Return filtered masks and updated axis limits based on actual min/max."""
    mask = np.ones_like(x_vals, dtype=bool)

    if not remove:
        return mask, None, None

    # IQR-basierte Filtergrenzen
    def get_bounds(vals):
        q1, q3 = np.percentile(vals[np.isfinite(vals)], [25, 75])
        iqr = q3 - q1
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

def plot_scatter(
    data: GrandPy,
    x: Optional[str] = None,
    y: Optional[str] = None,
    mode_slot: str | ModeSlot = None,
    path_for_save: Optional[str] = None,
    remove_outlier: bool = True,
    show_outlier: bool = True,
    size: float = 5,
    lim: Optional[tuple[float, float]] = None,
    x_lim: Optional[tuple[float, float]] = None,
    y_lim: Optional[tuple[float, float]] = None,
    cross: Optional[bool] = None,
    diag: Optional[bool | float | tuple] = None,
    highlight: Optional[Union[list[str], dict[str, list[str]]]] = None
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
    mode_slot: str | ModeSlot
        Specifies which data slot to use (e.g., "count", "norm")
    remove_outlier: bool
        Whether to detect and remove outliers using IQR filtering
    show_outlier: bool
        If True, outliers will be plotted in light gray
    path_for_save: str
        Saves the plot as a PNG to the specified directory (must end with \\ or \\\\. e.g. "C:\\\\Users\\\\user\\\\Desktop\\\\")
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

    #Default expressions
    names = list(data.coldata["Name"])
    x = x or names[0]
    y = y or names[1]
    if x not in names:
        raise ValueError(f"x is not a valid expression.")
    if y not in names:
        raise ValueError(f"y is not a valid expression.")
    #Default slot
    if mode_slot is None:
        mode_slot = data.default_slot


    matrix = data._resolve_mode_slot(mode_slot)
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix

    x_idx = list(data.coldata["Name"]).index(x)
    y_idx = list(data.coldata["Name"]).index(y)
    x_vals_all = matrix[:, x_idx]
    y_vals_all = matrix[:, y_idx]

    if np.all(np.isnan(x_vals_all)):
        raise ValueError(f"All Values for '{x}' in slot '{mode_slot}' are NaN. - Plot not possible!")
    if np.all(np.isnan(y_vals_all)):
        raise ValueError(f"All Values for '{x}' in slot '{mode_slot}' are NaN. - Plot not possible!")

    if lim:
        x_lim = x_lim or lim
        y_lim = y_lim or lim

    # Filter outliers
    mask_keep, auto_x_lim, auto_y_lim = _apply_outlier_filter(x_vals_all, y_vals_all, remove_outlier)
    x_vals, y_vals = x_vals_all[mask_keep], y_vals_all[mask_keep]
    x_lim = x_lim or auto_x_lim
    y_lim = y_lim or auto_y_lim

    # KDE for density color
    if np.allclose(x_vals, y_vals):
        epsilon = np.random.normal(0, 1e-4, size=x_vals.shape)
        y_vals = y_vals + epsilon
    kde = gaussian_kde(np.vstack([x_vals, y_vals]))(np.vstack([x_vals, y_vals]))
    idx = kde.argsort()
    x_vals, y_vals, kde = x_vals[idx], y_vals[idx], kde[idx]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot outliers
    if remove_outlier and show_outlier:
        out_x = x_vals_all[~mask_keep]
        out_y = y_vals_all[~mask_keep]

        margin = 0.01
        clipped_x = np.clip(out_x, x_lim[0] + margin, x_lim[1] - margin)
        clipped_y = np.clip(out_y, y_lim[0] + margin, y_lim[1] - margin)
        ax.scatter(clipped_x, clipped_y, color="grey", s=size+10, alpha=1, label="Outliers")

    # Main scatter
    scatter = ax.scatter(x_vals, y_vals, c=kde, s=size, cmap="viridis", alpha=1)
    fig.colorbar(scatter, ax=ax, label="Density")

    # Axis labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y} ({mode_slot})")
    if x_lim: ax.set_xlim(x_lim)
    if y_lim: ax.set_ylim(y_lim)

    # Diagonal
    if diag:
        _plot_diagonal(ax, diag, np.linspace(*ax.get_xlim(), 100))

    # Cross lines
    if cross:
        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")

    # Highlight
    if highlight:
        _highlight_points(ax, data, x_vals_all, y_vals_all, highlight, size)

    ax.grid(False)
    #offsets = scatter.get_offsets()
    #print(len(offsets))
    # Save / Show

    if path_for_save:
        fig.savefig(f"{path_for_save}{x}_{y}_{mode_slot}.png", format="png", dpi=300)
    plt.show(block=True)
    plt.close(fig)


# noch nix gut aber gibt eine heatmap aus, aber nicht exact die gleiche
def plot_heatmap(
    data: GrandPy,
    mode_slot="count",
    genes=None,
    columns=None,
    transform="Z",
    cluster_genes=True,
    cluster_columns=False,
    title=None,
    na_to=np.nan):

    def _z_score_rows(matrix):
        mean = np.nanmean(matrix, axis=1, keepdims=True)
        std = np.nanstd(matrix, axis=1, keepdims=True)
        std[std == 0] = 1
        print(matrix - mean / std)
        return (matrix - mean) / std

    matrix = data._resolve_mode_slot(mode_slot)
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix

    if genes is not None:
        gene_idx = data.get_index(genes)
        matrix = matrix[gene_idx, :]
    else:
        gene_idx = range(matrix.shape[0])

    if columns is not None:
        col_idx = [data.columns.index(c) for c in columns]
        matrix = matrix[:, col_idx]
    else:
        col_idx = range(matrix.shape[1])

    # Transformation
    if transform == "Z":
        matrix = _z_score_rows(matrix)
        label = "z score"
    elif transform == "logFC":
        ref = np.mean(matrix[:, :2], axis=1, keepdims=True)
        matrix = np.log2((matrix + 1e-8) / (ref + 1e-8))
        label = "log2 FC"
    elif transform == "none":
        label = ""

    # NA ersetzen
    if not np.isnan(na_to):
        matrix = np.nan_to_num(matrix, nan=na_to)

    sns.set(context="notebook")
    g = sns.clustermap(
        matrix,
        cmap="RdBu",
        row_cluster=cluster_genes,
        col_cluster=cluster_columns,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": label}
    )

    if title:
        plt.title(title)
    plt.show(block=True)


def setup_default_aes(data: GrandPy, aest: dict | None = None) -> dict:
    if aest is None:
        aest = {}

    coldata = data.coldata

    if "Condition" in coldata.columns and not any(k in aest for k in ["color", "colour"]):
        aest["color"] = "Condition"

    if "Replicate" in coldata.columns and "shape" not in aest:
        aest["shape"] = "Replicate"

    return aest

# gibt warum auch immer einen plot zwischen 0 und 1 aus. muss ich nochmal drüber schauen :)
#
def plot_pca(
    data: GrandPy,
    mode_slot: str | ModeSlot = None,
    path_for_save: Optional[str] = None,
    ntop: int = 500,
    aest: Optional[dict] = None,
    x: int = 1,
    y: int = 2,
    columns: Union[str, list, None] = None,
    do_vst: bool = True
    ):
    if mode_slot is None:
        mode_slot = data.default_slot

    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    mat = data.get_matrix(mode_slot=mode_slot, columns=selected_columns)

    coldata = data.coldata.loc[selected_columns].copy()

    mat = mat.loc[:, mat.notna().any(axis=0)]
    coldata = coldata.loc[mat.columns]

    if do_vst:
        mat = np.log2(mat + 1)

    variances = mat.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(min(ntop, len(variances))).index
    mat = mat.loc[top_genes]

    scaled = StandardScaler().fit_transform(mat.T)
    pca = PCA()
    pcs = pca.fit_transform(scaled)
    percent_var = pca.explained_variance_ratio_


    df = pd.DataFrame(pcs, index=mat.columns, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])])
    df = pd.concat([df, coldata.reset_index(drop=True)], axis=1)

    plt.figure(figsize=(6, 6))
    xlab = f"PC{x}: {percent_var[x - 1] * 100:.1f}% variance"
    ylab = f"PC{y}: {percent_var[y - 1] * 100:.1f}% variance"

    aest = setup_default_aes(data, aest)
    hue = aest.get("color")
    style = aest.get("shape")

    sns.scatterplot(data=df, x=f"PC{x}", y=f"PC{y}", hue=hue, style=style, size=10, s=80)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title("PCA")

    if path_for_save:
        plt.savefig(f"{path_for_save}/PCA_{mode_slot}.png", dpi=300)

    plt.show()