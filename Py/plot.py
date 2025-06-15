import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from typing import Optional, Union
from Py.grandPy import GrandPy, ModeSlot, _parse_as_mode_slot
from scipy.stats import gaussian_kde, iqr
from scipy.ndimage import gaussian_filter
import seaborn as sns
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
import warnings
# TODO Plots: PlotExpressionTest, PlotAnalyses, VulcanoPlot, MAPlot, PlotTypeDistribution, FormatCorrelation
#      Helper: Transform, Transform.no, Transform.Z, Transform.vst, Transform.logFC

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


def _setup_default_aes(data: GrandPy, aest: dict | None = None) -> dict:
    if aest is None:
        aest = {}

    coldata = data.coldata

    if "Condition" in coldata.columns and not any(k in aest for k in ["color", "colour"]):
        aest["color"] = "Condition"

    if "Replicate" in coldata.columns and "shape" not in aest:
        aest["shape"] = "Replicate"

    return aest


def _density2d(x, y, n=100, bw_x=None, bw_y=None, margin='n'):
    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.isfinite(x + y)
    if not np.any(finite_mask):
        return np.full_like(x, np.nan, dtype=float)

    x = x[finite_mask]
    y = y[finite_mask]

    if np.all(x == x[0]):
        x = np.array([x[0] - 0.5, x[0] + 0.5])
    if np.all(y == y[0]):
        y = np.array([y[0] - 0.5, y[0] + 0.5])

    def _bandwidth_nrd(v):
        h = (np.max(v) - np.min(v)) / 1.34
        return 1.06 * min(np.std(v, ddof=1), h) * len(v) ** (-1 / 5)

    bw_x = bw_x or _bandwidth_nrd(x)
    bw_y = bw_y or _bandwidth_nrd(y)

    xbins = np.linspace(np.min(x), np.max(x), n)
    ybins = np.linspace(np.min(y), np.max(y), n)

    H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    sigma_x = bw_x / dx
    sigma_y = bw_y / dy

    H_smooth = gaussian_filter(H, sigma=[sigma_x, sigma_y])

    if margin == 'x':
        H_smooth = H_smooth / np.max(H_smooth, axis=1, keepdims=True)
    elif margin == 'y':
        H_smooth = H_smooth / np.max(H_smooth, axis=0, keepdims=True)
    else:
        H_smooth /= np.max(H_smooth)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((xbins[:-1], ybins[:-1]), H_smooth.T, bounds_error=False, fill_value=0)

    result = np.full(len(finite_mask), np.nan)
    result[finite_mask] = interp(np.vstack([x, y]).T)
    result /= np.nanmax(result)

    return result

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
#Beispielaufruf: plot_scatter(sars, mode_slot="count", remove_outlier=True, show_outlier=True, highlight="UHMK1")
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

    # Compute Density
    z = _density2d(x_vals, y_vals, n=100, margin='n')
    idx = z.argsort()
    x_vals, y_vals, z = x_vals[idx], y_vals[idx], z[idx]


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
    scatter = ax.scatter(x_vals, y_vals, c=z, s=size, cmap="viridis", alpha=1, rasterized=False)
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


# für mode_slot = "counts" und do.vst=True alles gut
# für mode_slot = "alpha" und do.vst=False sind die werte irgendwie an der y achse gespiegelt???
# für mode_slot = "ntr" und do.vst=False alles komisch
# für mode_slot = "beta" und do.vst=False alles einwandfrei
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
    coldata = data.coldata.loc[selected_columns].copy()

    # Drop columns (samples) that contain only NaN values
    mat = mat.loc[:, mat.notna().any(axis=0)]
    coldata = coldata.loc[mat.columns]

    if str(mode_slot).lower() == "count":
        slotmat = mat.T.round().astype(int)
    else:
        slotmat = mat.T

    metadata_df = coldata.copy()
    metadata_df["condition"] = metadata_df["Condition"]

    if do_vst and str(mode_slot).lower() == "count":
        metadata_df = coldata.copy()
        metadata_df["condition"] = metadata_df["Condition"]

        dds = DeseqDataSet(counts=slotmat, metadata=metadata_df, design_factors="condition")
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


    plt.figure(figsize=(8, 6))
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

    coldata = data.coldata.loc[selected_columns].copy()

    x = data.get_table(mode_slots=ModeSlot("old", slot), genes=gene, columns=selected_columns).iloc[0].to_numpy()
    y = data.get_table(mode_slots=ModeSlot("new", slot), genes=gene, columns=selected_columns).iloc[0].to_numpy()

    plot_df = pd.DataFrame({
        "old": x,
        "new": y
    }, index=coldata.index)
    #print("Old", x)
    #print("New", y)

    plot_df = pd.concat([plot_df, coldata], axis=1)
    aest = _setup_default_aes(data, aest)
    style = aest.get("shape")
    hue = aest.get("color")
    if hue not in plot_df.columns:
        hue = None
    if style not in plot_df.columns:
        style = None

    fig, ax = plt.subplots(figsize=(6, 6))

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

        ci_lower = data.get_table(mode_slots="lower", genes=gene, columns=selected_columns).iloc[0].to_numpy()
        ci_upper = data.get_table(mode_slots="upper", genes=gene, columns=selected_columns).iloc[0].to_numpy()
        #print("Lower", ci_lower)
        #print("Upper", ci_upper)
        total = data.get_table(mode_slots=slot, genes=gene, columns=selected_columns).iloc[0].to_numpy()

        plot_ci = pd.DataFrame({
            "old": x,
            "new": y,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "total": total
        })

        valid_mask = (
                (plot_ci["old"] > 0) &
                (plot_ci["new"] > 0) &
                (plot_ci["ci_lower"] >= 0) &
                (plot_ci["ci_upper"] >= 0) &
                (plot_ci["ci_lower"] <= plot_ci["ci_upper"])
        )
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            warnings.warn(f"{n_invalid} data points with invalid CI were excluded from error bars.", UserWarning)

        plot_ci = plot_ci[valid_mask]

        x = plot_ci["old"].to_numpy()
        y = plot_ci["new"].to_numpy()

        ymin = plot_ci["ci_lower"] * plot_ci["total"]
        ymax = plot_ci["ci_upper"] * plot_ci["total"]
        yerr = [y - ymin, ymax - y]

        xmin = (1 - plot_ci["ci_upper"]) * plot_ci["total"]
        xmax = (1 - plot_ci["ci_lower"]) * plot_ci["total"]
        xerr = [x - xmin, xmax - x]

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor='grey')

    sns.scatterplot(
        data=plot_df,
        x="old",
        y="new",
        hue=hue,
        style=style,
        s=size,
        ax=ax,
        palette=sns.color_palette("Set2", n_colors=2)
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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

    coldata = data.coldata.loc[selected_columns].copy()

    total = data.get_table(mode_slots=slot, genes=gene, columns=selected_columns).iloc[0].to_numpy()
    ntr = data.get_table(mode_slots="ntr", genes=gene, columns=selected_columns).iloc[0].to_numpy()

    plot_df = pd.DataFrame({
        "total": total,
        "ntr": ntr
    }, index=coldata.index)

    plot_df = pd.concat([plot_df, coldata], axis=1)

    aest = _setup_default_aes(data, aest)
    hue = aest.get("color")
    style = aest.get("shape")

    if hue not in plot_df.columns:
        hue = None
    if style not in plot_df.columns:
        style = None

    fig, ax = plt.subplots(figsize=(6, 6))

    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        ax.set_xscale("linear")

    ax.set_xlabel(f"Total RNA ({slot})")
    ax.set_ylabel("NTR")
    ax.set_title(f"Gene: {gene}")

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")

        ci_lower = data.get_table(mode_slots="lower", genes=gene, columns=selected_columns).iloc[0].to_numpy()
        ci_upper = data.get_table(mode_slots="upper", genes=gene, columns=selected_columns).iloc[0].to_numpy()

        plot_ci = pd.DataFrame({
            "total": total,
            "ntr": ntr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

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

        x = plot_ci["total"].to_numpy()
        y = plot_ci["ntr"].to_numpy()

        yerr = [
            y - plot_ci["ci_lower"].to_numpy(),
            plot_ci["ci_upper"].to_numpy() - y
        ]

        ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='gray')

    sns.scatterplot(
        data=plot_df,
        x="total",
        y="ntr",
        hue=hue,
        style=style,
        s=size,
        ax=ax,
        palette=sns.color_palette("Set2", n_colors=2)
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    transform: Optional[callable] = None
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

    coldata = data.coldata.loc[selected_columns].copy()

    if slot == "ntr":
        df = data.get_data(mode_slots="ntr", genes=gene, columns=selected_columns, with_coldata=True)
        df["ntr"] = df[gene]
        df["value"] = df[gene]
        log = False
    else:
        df = data.get_data(mode_slots=[slot, "ntr"], genes=gene, columns=selected_columns, with_coldata=True)
        df_slot = data.get_table(mode_slots=slot, genes=gene, columns=selected_columns).iloc[0].to_numpy()
        df_ntr = data.get_table(mode_slots="ntr", genes=gene, columns=selected_columns).iloc[0].to_numpy()

        df["slot_val"] = df_slot
        df["ntr"] = df_ntr

        if mode == "total":
            df["value"] = df["slot_val"]
        elif mode == "new":
            df["value"] = df["slot_val"] * df["ntr"]
        elif mode == "old":
            df["value"] = df["slot_val"] * (1 - df["ntr"])
        else:
            raise ValueError(f"Unknown mode '{mode}' in mode_slot '{mode_slot}'")
    df = df.loc[:, ~df.columns.duplicated()]

    if group not in df.columns:
        raise ValueError(f"Group column '{group}' not found in coldata!")

    if transform is not None:
        df = transform(df)

    aest = _setup_default_aes(data, aest)
    hue = aest.get("color")
    style = aest.get("shape")

    fig, ax = plt.subplots(figsize=(8, 6))
    group_order = df[group].unique()
    group_to_num = {g: i for i, g in enumerate(group_order)}
    df["_xpos"] = df[group].map(group_to_num)


    sns.scatterplot(
        data=df,
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

        ci_lower = data.get_table(mode_slots="lower", genes=gene, columns=selected_columns).iloc[0].to_numpy()
        ci_upper = data.get_table(mode_slots="upper", genes=gene, columns=selected_columns).iloc[0].to_numpy()

        df["lower"] = ci_lower
        df["upper"] = ci_upper

        if slot == "ntr":
            ymin = df["lower"]
            ymax = df["upper"]
        elif mode == "new":
            ymin = df["lower"] * df["slot_val"]
            ymax = df["upper"] * df["slot_val"]
        elif mode == "old":
            ymin = (1 - df["upper"]) * df["slot_val"]
            ymax = (1 - df["lower"]) * df["slot_val"]
        else:
            ymin = ymax = None

        if ymin is not None:
            mask = (
                    (df["value"] >= 0) &
                    (ymin >= 0) & (ymax >= 0) &
                    (ymin <= df["value"]) & (ymax >= df["value"])
            )
            df_ci = df[mask].copy()
            yerr = [
                df_ci["value"] - ymin[mask],
                ymax[mask] - df_ci["value"]
            ]

            ax.errorbar(
                x=df_ci["_xpos"],
                y=df_ci["value"],
                yerr=yerr,
                fmt='none',
                ecolor='gray',
                capsize=3,
            )

    if hue:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.close()

#TODO show_ci muss noch überarbeitet werden
#Beispielaufruf: plot_gene_groups_bars(sars, "UHMK1", xlab="Condition + '.' + Replicate")
def plot_gene_groups_bars(
    data: GrandPy,
    gene: str,
    slot: Optional[str] = None,
    columns: Optional[Union[str, list]] = None,
    show_ci: bool = False,
    xlab: Optional[Union[str, list]] = None,
    transform: Optional[callable] = None
):
    if slot is None:
        slot = data.default_slot

    mode_slot_old = ModeSlot("old", slot)
    mode_slot_new = ModeSlot("new", slot)

    if columns is None:
        selected_columns = data.columns
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    else:
        selected_columns = data.get_columns(columns)

    coldata = data.coldata.loc[selected_columns].copy()
    old_vals = data.get_table(mode_slots=mode_slot_old, genes=gene, columns=selected_columns).iloc[0].to_numpy()
    new_vals = data.get_table(mode_slots=mode_slot_new, genes=gene, columns=selected_columns).iloc[0].to_numpy()

    if isinstance(xlab, list):
        xlabels = xlab
    elif isinstance(xlab, str):
        local_vars = {col: coldata[col].astype(str) for col in coldata.columns}
        try:
            xlabels = eval(xlab, {}, local_vars).tolist()
        except Exception as e:
            raise ValueError(f"xlab expression could not be evaluated: {e}")
    else:
        xlabels = coldata["Name"].tolist()

    df = pd.DataFrame({
        "sample": selected_columns,
        "xlab": xlabels,
        "old": old_vals,
        "new": new_vals
    })

    if transform is not None:
        df = transform(df)

    x = np.arange(len(df))
    width = 0.8

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x, df["old"], width, label="Old", color="lightgray")
    bar2 = ax.bar(x, df["new"], width, bottom=df["old"], label="New", color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(df["xlab"], rotation=90)
    ax.set_ylabel(f"Total RNA ({slot})")
    ax.set_xlabel("")
    ax.set_title(gene)

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")

        total = data.get_table(mode_slots=slot, genes=gene, columns=selected_columns).iloc[0].to_numpy()
        lower = data.get_table(mode_slots="lower", genes=gene, columns=selected_columns).iloc[0].to_numpy()
        upper = data.get_table(mode_slots="upper", genes=gene, columns=selected_columns).iloc[0].to_numpy()

        ymin = (1 - upper) * total
        ymax = (1 - lower) * total

        err_low = total - ymin
        err_high = (ymax - total)

        mask = (
                np.isfinite(total) &
                np.isfinite(lower) & np.isfinite(upper) &
                np.isfinite(err_low) & np.isfinite(err_high) &
                (total >= 0) &
                (lower >= 0) & (upper >= 0) &
                (lower <= upper) &
                (err_low >= 0) & (err_high >= 0)
        )

        n_invalid = np.sum(~mask)
        if n_invalid > 0:
            warnings.warn(f"{n_invalid} data points with invalid CI were excluded from error bars.", UserWarning)

        if np.any(mask):
            x_valid = np.arange(len(total))[mask]
            total_valid = total[mask]
            err_valid = [err_low[mask], err_high[mask]]

            ax.errorbar(
                x=x_valid,
                y=total_valid,
                yerr=err_valid,
                fmt='none',
                ecolor='black',
                capsize=3

        )

    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# Beispielsaufruf: plot_gene_snapshot_timecourse(sars, "UHMK1")
def plot_gene_snapshot_timecourse(
    data: GrandPy,
    gene: str,
    time: str = "Time",
    mode_slot: Union[str, ModeSlot, None] = None,
    columns: Optional[Union[str, list]] = None,
    average_lines: bool = True,
    exact_tics: bool = True,
    log: bool = True,
    show_ci: bool = False,
    aest: Optional[dict] = None,
    size: float = 50
):
    import re
    def parse_time_to_float(time_str):
        match = re.match(r"(\d+(\.\d+)?)h", time_str)
        if match:
            return float(match.group(1))
        else:
            return 0.0


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

    df = data.get_data(mode_slots=[slot, "ntr"], genes=gene, columns=selected_columns, with_coldata=True)
    slot_val = data.get_table(mode_slots=slot, genes=gene, columns=selected_columns).iloc[0].to_numpy()
    ntr_val = data.get_table(mode_slots="ntr", genes=gene, columns=selected_columns).iloc[0].to_numpy()

    if mode == "total":
        df["value"] = slot_val
    elif mode == "new":
        df["value"] = slot_val * ntr_val
    elif mode == "old":
        df["value"] = slot_val * (1 - ntr_val)
    else:
        raise ValueError(f"Unknown mode '{mode}' in mode_slot '{mode_slot}'")

    df["ntr"] = ntr_val
    df[slot] = slot_val

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

    y = "ntr" if slot == "ntr" else "value"
    ylabel = "NTR" if slot == "ntr" else f"{mode.capitalize()} RNA ({slot})"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=time, y=y, hue=hue, style=style, s=size, ax=ax)

    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xticks(x_breaks if not exact_tics else x_breaks_numeric)
    ax.set_xticklabels(x_breaks)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(gene)

    if average_lines:
        avg_input_df = df.drop(columns=["Replicate"], errors="ignore")
        group_cols = [time]
        if hue and hue in avg_input_df.columns:
            group_cols.append(hue)
        if style and style in avg_input_df.columns and style != hue:
            group_cols.append(style)

        avg_df = avg_input_df.groupby(group_cols, observed=True)[y].mean().reset_index()

        sns.lineplot(
            data=avg_df,
            x=time,
            y=y,
            hue=hue if hue in avg_df.columns else None,
            style=style if style in avg_df.columns else None,
            ax=ax,
            legend=False
        )

    if show_ci:
        if "lower" not in data.slots or "upper" not in data.slots:
            raise ValueError("CI slots ('lower' and 'upper') are missing. Run compute_ntr_ci() first.")

        lower = data.get_table(mode_slots="lower", genes=gene, columns=selected_columns).iloc[0].to_numpy()
        upper = data.get_table(mode_slots="upper", genes=gene, columns=selected_columns).iloc[0].to_numpy()

        if slot == "ntr":
            ymin = lower
            ymax = upper
            center = df["ntr"].to_numpy()
        elif mode == "new":
            ymin = lower * slot_val
            ymax = upper * slot_val
            center = df["value"].to_numpy()
        elif mode == "old":
            ymin = (1 - upper) * slot_val
            ymax = (1 - lower) * slot_val
            center = df["value"].to_numpy()
        else:
            ymin = ymax = center = None

        if ymin is not None:
            err_low = center - ymin
            err_high = ymax - center

            mask = (
                    np.isfinite(center) & np.isfinite(lower) & np.isfinite(upper) &
                    np.isfinite(err_low) & np.isfinite(err_high) &
                    (center >= 0) & (lower >= 0) & (upper >= 0) &
                    (lower <= upper) & (err_low >= 0) & (err_high >= 0)
            )

            n_invalid = np.sum(~mask)
            if n_invalid > 0:
                warnings.warn(f"{n_invalid} data points with invalid CI were excluded from error bars.", UserWarning)

            if np.any(mask):
                x_vals = df[time].to_numpy()
                x_valid = x_vals[mask]
                center_valid = center[mask]
                err_valid = [err_low[mask], err_high[mask]]

                ax.errorbar(
                    x=x_valid,
                    y=center_valid,
                    yerr=err_valid,
                    fmt='none',
                    ecolor='gray',
                    capsize=3
                )

    plt.tight_layout()
    plt.show()
    plt.close()