import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors

from pathlib import Path
from matplotlib import cm
from typing import Optional, Union, Callable
from scipy.stats import gaussian_kde, iqr, pearsonr, spearmanr, kendalltau
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from IPython.core.pylabtools import figsize
from scipy.sparse import issparse


from .grandPy import GrandPy
from .slot_tool import ModeSlot, _parse_as_mode_slot

def _is_sparse_matrix(mat: any)-> bool:
    """
        Check whether a matrix is a SciPy sparse matrix.

        Parameters
        ----------
        mat : any
            Object to check for sparse matrix type.

        Returns
        -------
        bool
            True if `mat` is a SciPy sparse matrix, False otherwise.
        """
    return issparse(mat)

def _parse_time_to_float(t: str)-> float:
    """
        Convert a time string like '24h' or '1.5h' to a float (in hours).

        Parameters
        ----------
        t : str
            Time string expected to end with 'h', e.g. '12h' or '0.5h'.

        Returns
        -------
        float
            Parsed numeric value in hours. Returns 0.0 if the format is invalid.

        Examples
        --------
        >>> _parse_time_to_float("24h")
        24.0
        >>> _parse_time_to_float("1.5h")
        1.5
        >>> _parse_time_to_float("invalid")
        0.0
        """
    match = re.match(r"(\d+(\.\d+)?)h", t)
    if match:
        return float(match.group(1))
    else:
        return 0.0

def _apply_outlier_filter(x_vals: np.ndarray, y_vals: np.ndarray, remove_outlier: bool)-> tuple[np.ndarray, tuple | None, tuple | None]:
    """
    Apply IQR-based outlier filtering to x and y values.

    Parameters
    ----------
    x_vals : np.ndarray
        Array of x-values.
    y_vals : np.ndarray
        Array of y-values.
    remove : bool
        Whether to apply outlier filtering. If False, all data points are kept.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask indicating which data points are kept after filtering.
    x_auto_lim : tuple of float or None
        Auto-scaled x-axis limits (min, max) based on non-outlier x-values.
        Returns None if `remove` is False.
    y_auto_lim : tuple of float or None
        Auto-scaled y-axis limits (min, max) based on non-outlier y-values.
        Returns None if `remove` is False.

    Notes
    -----
    Outliers are identified using the interquartile range (IQR) method:
    values outside [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] are considered outliers.
    """
    mask = np.ones_like(x_vals, dtype=bool)

    if remove_outlier == 0:
        return mask, None, None

    def get_bounds(vals, min_iqr=1e-6):
        q1, q3 = np.percentile(vals[np.isfinite(vals)], [25, 75])
        iqr = max(q3 - q1, min_iqr)
        return q1 - remove_outlier * iqr, q3 + remove_outlier * iqr

    x_lower, x_upper = get_bounds(x_vals)
    y_lower, y_upper = get_bounds(y_vals)

    mask_x = (x_vals >= x_lower) & (x_vals <= x_upper)
    mask_y = (y_vals >= y_lower) & (y_vals <= y_upper)
    mask = mask_x & mask_y

    x_auto_lim = (x_vals[mask_x].min(), x_vals[mask_x].max())
    y_auto_lim = (y_vals[mask_y].min(), y_vals[mask_y].max())

    return mask, x_auto_lim, y_auto_lim

def _plot_diagonal(ax, diag, x_range):
    """
    Draw one or more diagonal reference lines on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The matplotlib axis to draw on.
        diag (bool | int | float | list | tuple): If True, draw y = x. If numeric, draw y = x + offset.
                                                  If list/tuple, draw multiple y = x + offset lines.
        x_range (array-like): The x-values over which to draw the lines.
    """
    if diag is True:
        ax.plot(x_range, x_range, linestyle="--", color="gray", label="y = x")
    elif isinstance(diag, (int, float)):
        ax.plot(x_range, x_range + diag, linestyle="--", color="gray", label=f"y = x + {diag}")
    elif isinstance(diag, (list, tuple)):
        for offset in diag:
            ax.plot(x_range, x_range + offset, linestyle="--", color="gray", label=f"y = x + {offset}")

def _highlight_points(ax, data, x_vals, y_vals, highlight, size, highlight_size):
    """
        Highlight specific points on a scatter plot.

        Parameters:
            ax (matplotlib.axes.Axes): The axis object to draw on.
            data (GrandPy): The data object providing index lookup via `get_index()`.
            x_vals (np.ndarray): X-values of the points.
            y_vals (np.ndarray): Y-values of the points.
            highlight (list[str] | dict[str, list[str]] | None): List of gene names to highlight in red,
                                                                  or a dict mapping colors to gene lists.
            size (float): Base size of the scatter points.
            highlight_size (float): Scaling factor for highlighted points.
        """
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
            ax.scatter(x_vals[idxs], y_vals[idxs], color="red", s=size * highlight_size)

def _label_points(ax, data, x_vals, y_vals, label, size, highlight_size, y_label_offset):
    """
        Label specific points on a scatter plot.

        Parameters:
            ax (matplotlib.axes.Axes): The axis object to draw on.
            data (GrandPy): The data object providing index lookup via `get_index()`.
            x_vals (np.ndarray): X-values of the points.
            y_vals (np.ndarray): Y-values of the points.
            label (list[str] | dict[str, list[str]] | None): List of gene names to label
            size (float): Base size of the scatter points.
            highlight_size (float): Scaling factor for labels.
        """
    def get_indices(genes):
        return data.get_index(genes, regex=False)

    idxs = get_indices(label)
    if isinstance(label, list) and all(isinstance(g, str) for g in label):
        gene_names = label
    else:
        gene_names = [data.genes[i] for i in idxs]
    y_offset = (np.max(y_vals) - np.min(y_vals)) * y_label_offset
    for x_, y_, name in zip(x_vals[idxs], y_vals[idxs], gene_names):
        ax.text(x_, y_ + y_offset, name, fontsize=size * highlight_size/1.8, ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=0.5))

def _setup_default_aes(data: GrandPy, aest: dict | None = None) -> dict:
    """
       Set up default aesthetics dictionary for plotting based on data coldata.

       Parameters:
           data (GrandPy): The data object containing metadata in `coldata`.
           aest (dict | None): Optional initial aesthetics dictionary to update.

       Returns:
           dict: Updated aesthetics dictionary with default keys added if missing.
       """
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

def _density2d(x: np.ndarray, y: np.ndarray, n: int = 100, margin: str ='n')-> np.ndarray:
    """
        Compute a 2D kernel density estimate (KDE) for points (x, y).

        Parameters:
            x (array-like): 1D array of x coordinates.
            y (array-like): 1D array of y coordinates.
            n (int): Number of points for KDE grid (not used directly here but kept for API compatibility).
            margin (str): Margin normalization, options:
                          'x' - normalize densities along unique x values,
                          'y' - normalize densities along unique y values,
                          'n' - normalize overall density.

        Returns:
            np.ndarray: Density values for each (x, y) point, same shape as input.
                        NaN where input is invalid or KDE could not be computed.
        """
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

def _make_continuous_colors(values, colors: str = None, breaks: int | list = None)-> dict:
    """
       Generate color breaks and corresponding colors for continuous values.

       The function determines appropriate quantiles or breaks for the given values,
       distinguishes between values that contain negatives and positives, and chooses
       a diverging or sequential color scheme accordingly. It can also handle color maps
       by name and optionally reverse them.

       Parameters:
           values (array-like): Numeric values for which colors should be generated.
           colors (str or sequence of str, optional): Color map name (e.g. 'viridis', 'revRdBu') or
               list of hex colors. If a string starts with 'rev', the color map will be reversed.
           breaks (str, int, or array-like, optional): Specifies how to calculate breaks:
               - 'minmax': use min/max for breaks evenly spaced,
               - int: number of breaks to generate,
               - array-like: explicit break points.

       Returns:
           dict: Dictionary with keys:
               'breaks' : array of break points for the color scale,
               'colors' : list of colors corresponding to the breaks.
       """
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

def _transform_no(matrix: np.ndarray) -> np.ndarray:
    """
        Identity transform: returns the input matrix unchanged.

        Parameters:
            matrix (np.ndarray): Input matrix.

        Returns:
            np.ndarray: The same input matrix without any modifications.
        """
    return matrix

def _transform_z(matrix: np.ndarray, center: bool = True, scale: bool = True) -> np.ndarray:
    """
        Apply z-score transformation to each row of the input matrix.

        Parameters:
            matrix (np.ndarray): Input 2D array (samples x features).
            center (bool): If True, subtract the mean (center the data).
            scale (bool): If True, divide by the standard deviation (scale the data).

        Returns:
            np.ndarray: Transformed matrix with z-score normalization applied row-wise.
        """
    if not center and not scale:
        return matrix.astype(np.float64)

    from scipy.stats import zscore
    return zscore(matrix, axis=1, ddof=1, nan_policy='omit' if (center or scale) else 'propagate')

def _transform_vst(data: np.ndarray, selected_columns: list, mode_slot, genes) -> pd.DataFrame:
    """
        Perform variance stabilizing transformation (VST) on selected gene expression data.

        Parameters:
            data: GrandPy-like object with `get_table` and `coldata`.
            selected_columns (list): Columns to select from the data.
            mode_slot (str): Mode slot, e.g., "count" or others.
            genes (list): Genes to include in the transformation.

        Returns:
            pd.DataFrame: VST-transformed data frame (samples x genes).
        """
    mat = data.get_table(mode_slots=mode_slot, columns=selected_columns, genes=genes)
    mat = mat.loc[:, mat.notna().any(axis=0)]

    selected_columns_valid = mat.columns.tolist()

    coldata = data.coldata
    coldata["condition"] = coldata["Condition"]

    slotmat = mat.T.round().astype(int)

    slotmat = slotmat.loc[coldata.index]

    dds = DeseqDataSet(counts=slotmat, metadata=coldata, design_factors="Condition", low_memory=True, quiet=True)

    dds.deseq2()
    dds.vst_fit()
    vst_array = dds.vst_transform()
    vst_df = pd.DataFrame(vst_array, index=slotmat.index, columns=slotmat.columns)
    return vst_df

def _transform_logFC(m: np.ndarray, reference_columns: Optional[list[int]] = None, lfc_fun: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """
    Compute log2 fold changes for a matrix against a reference (e.g., mean of columns).

    Parameters
    ----------
    m : np.ndarray
        Expression matrix (genes × samples).
    reference_columns : list of int, optional
        Columns used to compute reference expression per gene.
        Defaults to all columns.
    lfc_fun : function, optional
        A custom logFC function: takes (x, ref) and returns logFCs.
        If None, uses log2((x+1)/(ref+1)).

    Returns
    -------
    np.ndarray
        Transformed matrix of log fold changes (same shape as input).
    """
    if reference_columns is None:
        reference_columns = list(range(m.shape[1]))

    ref = np.nanmean(m[:, reference_columns], axis=1)

    if lfc_fun is None:
        lfc_fun = lambda x, r: np.log2((x + 1) / (r + 1))

    result = np.apply_along_axis(lambda v: lfc_fun(v, ref), axis=0, arr=m)
    return result

def _f_old_nonequi(t: float, f0: float, s: float, d: float)-> (float | np.ndarray):
    """
        Calculate concentration decay over time from an initial amount without synthesis.

        Models exponential decay of a substance starting at initial concentration `f0`
        with degradation rate `kd`. The synthesis rate `ks` is not used in this model.

        Parameters
        ----------
        t : float or np.ndarray
            Time point(s) at which to evaluate the function.
        f0 : float
            Initial concentration at time zero.
        ks : float
            Synthesis rate (not used in this function).
        kd : float
            Degradation rate constant.

        Returns
        -------
        float or np.ndarray
            Concentration at time `t`, calculated as `f0 * exp(-kd * t)`.
        """
    return f0 * np.exp(-t * d)

def _f_old_equi(t: float, s: float, d: float) -> float:
    """
    Computes the expected amount of old RNA under steady-state assumptions.
    """
    return s / d * np.exp(-t * d)

def _f_new(t: float, s: float, d: float)-> float | np.ndarray:
    """
        Calculate concentration over time with synthesis and degradation reaching equilibrium.

        Models the concentration change over time given a synthesis rate `ks` and degradation
        rate `kd`, converging to the steady-state level `ks / kd`.

        Parameters
        ----------
        t : float or np.ndarray
            Time point(s) at which to evaluate the function.
        ks : float
            Synthesis rate.
        kd : float
            Degradation rate constant.

        Returns
        -------
        float or np.ndarray
            Concentration at time `t`, calculated as `ks / kd * (1 - exp(-kd * t))`.
        """
    return s / d * (1 - np.exp(-t * d))

def _format_correlation(method: str = "pearson", n_format: str | None = None, coeff_format: str | None = ".2f", p_format: str | None = ".2g", slope_format: str | None = None, rmsd_format: str | None = None, min_obs: int = 5) -> callable:
    """
        Create a function to compute and format correlation statistics between two arrays.

        Parameters
        ----------
        method : str, default "pearson"
            Correlation method to use. One of "pearson", "spearman", or "kendall".
        n_format : str or None, optional
            Format string (without '%') for the number of observations (e.g., "d"). If None, omit this from output.
        coeff_format : str or None, optional
            Format string for the correlation coefficient (e.g., ".2f"). If None, omit this from output.
        p_format : str or None, optional
            Format string for the p-value (e.g., ".2g" or ".2e"). If None, omit this from output.
        slope_format : str or None, optional
            Format string for the PCA-based slope between x and y (not from linear regression). If None, omit this.
        rmsd_format : str or None, optional
            Format string for the root mean square deviation (RMSD). If None, omit this.
        min_obs : int, default 5
            Minimum number of valid observations required to compute and return results.

        Returns
        -------
        callable
            A function that takes two numeric arrays (x, y) and returns a formatted string describing
            the correlation statistics, or None if not enough valid data points.

        Notes
        -----
        - NaN or infinite values are automatically excluded before computing correlations.
        - The slope is computed from the first principal component and reflects direction, not regression.
        """
    def formatted_correlation(x, y):
        if len(x) != len(y):
            raise ValueError("Cannot compute correlation, unequal lengths!")

        use = np.isfinite(x) & np.isfinite(y)
        if np.sum(use) < len(x):
            print(f"Removed {np.sum(~use)}/{len(x)} non-finite values while computing correlation!")

        x = x[use]
        y = y[use]

        if len(x) < min_obs:
            return None

        if method == "pearson":
            cc, p_value = pearsonr(x, y)
            p_name = "R"
        elif method == "spearman":
            cc, p_value = spearmanr(x, y)
            p_name = "\u03C1"
        elif method == "kendall":
            cc, p_value = kendalltau(x, y)
            p_name = "\u03C4"
        else:
            raise ValueError("Invalid correlation method. Choose from 'pearson', 'spearman', or 'kendall'.")

        formatted_n = f"n={len(x)}" if n_format else ""
        formatted_p = f"p{'<' if p_value < 2.2e-16 else '='}{p_format}" if p_format else ""
        formatted_coeff = f"{p_name}={cc:{coeff_format}}" if coeff_format else ""

        if slope_format:
            pca = np.linalg.svd(np.cov(x, y))[0][:, 0]
            formatted_slope = f"s={pca[1] / pca[0]:{slope_format}}"
        else:
            formatted_slope = ""

        if rmsd_format:
            rmsd = np.sqrt(np.mean((x - y) ** 2))
            formatted_rmsd = f"rmsd={rmsd:{rmsd_format}}"
        else:
            formatted_rmsd = ""

        return "\n".join(filter(None, [formatted_n, formatted_coeff, formatted_p, formatted_slope, formatted_rmsd]))

    return formatted_correlation

#Beispielaufruf: plot_scatter(sars, mode_slot="count", remove_outlier=True, show_outlier=True, highlight="UHMK1")
def plot_scatter(
    data: GrandPy,
    x: Optional[str] = None,
    y: Optional[str] = None,
    genes: Optional[list[str]] = None,
    filter: Optional[Union[slice, tuple[int, int],list[int], list[tuple[int, int]], np.ndarray, pd.Series]] = None,
    log: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    axis: bool = False,
    axis_x: bool = False,
    axis_y: bool = False,
    mode_slot: str | ModeSlot = None,
    remove_outlier: bool = True,
    show_outlier: float = 1.5,
    size: float = 5,
    limit: Optional[tuple[float, float]] = None,
    x_limit: Optional[tuple[float, float]] = None,
    y_limit: Optional[tuple[float, float]] = None,
    color: str = None,
    color_palette: str = None,
    cross: Optional[bool] = None,
    diagonal: Optional[bool | float | tuple] = None,
    highlight: Optional[Union[list[str], dict[str, list[str]]]] = None,
    highlight_size: float = 3,
    label: Optional[Union[list[int], list[str]]] = None,
    y_label_offset: float = 0.001,
    analysis: str = None,
    rasterized: bool = False,
    density_margin: str = "n",
    density_n: int = 100,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    figsize: tuple[float, float] = (10, 6),
    show_plot: bool = True,
):
    """
        Plot a scatter plot of expression values from a GrandPy object.

        This function visualizes values associated with two variables (x and y), which can be either sample names,
        analysis result names, or expressions. It supports various transformations, highlighting, axis styling, and
        density coloring.
        Parameters
        ----------
        data : GrandPy
            GrandPy object containing expression data and metadata.

        x : str, optional
            Sample name, analysis result name, or expression used for the x-axis. Defaults to the first available.

        y : str, optional
            Sample name, analysis result name, or expression used for the y-axis. Defaults to the second available.

        genes : list of str, optional
            Subset of genes to include in the plot.

        filter : slice | tuple[int, int] | list[int] | list[tuple[int, int]] | np.ndarray | pd.Series, optional
            Filters the data to a subset of rows (genes) before plotting.

            Can be:
                - a slice (e.g., slice(0, 100))
                - a tuple specifying a range (e.g., (0, 100))
                - a list of indices
                - a list of (start, stop) tuples
                - a boolean mask (Series or array)

        log : bool, default=False
            If True, apply log10 transform to both x and y values (ignores zeros and negatives).

        log_x : bool, default=False
            If True, apply log10 transform to x-axis values.

        log_y : bool, default=False
            If True, apply log10 transform to y-axis values.

        axis : bool, default=False
            If True, remove both axes (ticks, labels, spines).

        axis_x : bool, default=False
            If True, remove x-axis only.

        axis_y : bool, default=False
            If True, remove y-axis only.

        mode_slot : str or ModeSlot, optional
            The data slot to use, e.g., "count", "norm", or a ModeSlot instance.

        remove_outlier : float, default=1.5
            Whether to detect and remove outliers using IQR-based filtering.

        show_outlier : bool, default=True
            If True, plot filtered outliers in gray behind the main scatter plot.

        size : float, default=5
            Size of each point in the scatter plot.

        limit : tuple[float, float], optional
            Sets both x and y limits to the same value, unless x_limit or y_limit are specified.

        x_limit : tuple[float, float], optional
            Explicitly set the x-axis limits.

        y_limit : tuple[float, float], optional
            Explicitly set the y-axis limits.

        color : str, optional
            Variable name from DataFrame to color points by. If None, use density-based coloring.

        color_palette : str, default="viridis"
            Name of the matplotlib colormap to use for coloring.

        cross : bool, optional
            If True, draw dashed lines at x=0 and y=0.

        diagonal : bool | float | tuple, optional
            If True, draw identity line (y = x).
            If float or tuple of float(s), draw y = x + offset(s).

        highlight : list[str] | dict[str, list[str]], optional
            Genes to highlight. Either:
                - list of gene names (default color used)
                - dict mapping color → list of genes

        highlight_size : float, default=3
            Size of highlighted points (multiplied with base size).

        label : list[int] | list[str], optional
            Genes to label in the plot (by name or index).

        y_label_offset : float, default=0.001
            Vertical offset to apply when rendering gene labels.

        analysis : str, optional
            Analysis name to use when extracting data from analysis tables.

        rasterized : bool, default=False
            Whether to rasterize scatter plot (useful for large plots with many points).

        density_margin : str, default="n"
            Defines density estimation behavior (passed to internal function).

        density_n : int, default=100
            Number of bins or grid size for density computation.

        path_for_save : str, optional
            If given, save the figure as a PNG to this directory with auto-generated filename.

        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.

        figsize : tuple[float, float], default=(10, 6)
            Size of the figure in inches (width, height).

        show_plot : bool, default=True
            Show the plot.
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

    if data.analyses:
        if analysis is None:
            analysis = data.analyses[0]
        else:
            analysis = analysis
        names = data.get_analysis_table(with_gene_info=False).keys().tolist()
    else:
        names = list(data.coldata["Name"])
    x = x or names[0]
    y = y or names[1]
    if x not in names:
        raise ValueError(f"x is not a valid expression.")
    if y not in names:
        raise ValueError(f"y is not a valid expression.")

    if mode_slot is None:
        mode_slot = data.default_slot

    raw_matrix = data.get_matrix(mode_slot, force_numpy=False)
    if _is_sparse_matrix(raw_matrix) or analysis:
        df = data.get_analysis_table(genes=genes, with_gene_info=False)
    else:
        df = data.get_table(mode_slots=mode_slot, genes=genes)

    if filter is not None:
        if isinstance(filter, (tuple, slice)):
            df = df.iloc[slice(*filter) if isinstance(filter, tuple) else filter]
        elif isinstance(filter, list):
            indices = []
            for item in filter:
                if isinstance(item, tuple):
                    indices.extend(range(*item))
                elif isinstance(item, slice):
                    indices.extend(range(item.start, item.stop))
                elif isinstance(item, int):
                    indices.append(item)
                else:
                    raise TypeError(f"Unsupported filter element: {item}")
            df = df.iloc[indices]
        elif isinstance(filter, (np.ndarray, pd.Series)):
            df = df.iloc[filter]
        else:
            raise TypeError("Invalid filter type")

    if x in df.columns:
        x_vals_all = df[x].to_numpy()
    elif analysis:
        x_vals_all = df.T.iloc[0].to_numpy()
    else:
        col_index = list(data.coldata["Name"]).index(x)
        x_vals_all = data.get_matrix(mode_slot, columns=col_index, force_numpy=True)

    if y in df.columns:
        y_vals_all = df[y].to_numpy()
    elif analysis:
        y_vals_all = df.T.iloc[1].to_numpy()
    else:
        col_index = list(data.coldata["Name"]).index(y)
        y_vals_all = data.get_matrix(mode_slot, columns=col_index, force_numpy=True)
    if np.all(np.isnan(x_vals_all)):
        raise ValueError(f"All Values for '{x}' in slot '{mode_slot}' are NaN. - Plot not possible!")
    if np.all(np.isnan(y_vals_all)):
        raise ValueError(f"All Values for '{x}' in slot '{mode_slot}' are NaN. - Plot not possible!")

    if limit:
        x_limit = x_limit or limit
        y_limit = y_limit or limit

    if log:
        log_x = True
        log_y = True

    mask = np.ones_like(x_vals_all, dtype=bool)
    if log_x:
        mask &= x_vals_all > 0
    if log_y:
        mask &= y_vals_all > 0

    if not np.any(mask):
        raise ValueError("No positive values for selected log transform. Log transform not possible!")

    x_vals_trans = np.log10(x_vals_all[mask]) if log_x else x_vals_all[mask]
    y_vals_trans = np.log10(y_vals_all[mask]) if log_y else y_vals_all[mask]

    mask_keep, auto_x_lim, auto_y_lim = _apply_outlier_filter(x_vals_trans, y_vals_trans, remove_outlier)

    x_vals = x_vals_trans[mask_keep]
    y_vals = y_vals_trans[mask_keep]

    outlier_x = x_vals_trans[~mask_keep]
    outlier_y = y_vals_trans[~mask_keep]

    x_limit = x_limit or auto_x_lim
    y_limit = y_limit or auto_y_lim

    if not color:
        color = _density2d(x_vals, y_vals, n=density_n, margin=density_margin)
        idx = color.argsort()
        x_vals, y_vals, color = x_vals[idx], y_vals[idx], color[idx]
    else:
        color = color

    fig, ax = plt.subplots(figsize=figsize)

    # Plot outliers
    if remove_outlier and show_outlier:
        margin = 0.01
        clipped_x = np.clip(outlier_x, x_limit[0] + margin, x_limit[1] - margin)
        clipped_y = np.clip(outlier_y, y_limit[0] + margin, y_limit[1] - margin)
        ax.scatter(clipped_x, clipped_y, color="grey", s=size + 10, alpha=1, label="Outliers")

    # Main scatter
    scatter = ax.scatter(x_vals, y_vals, c=color, s=size, cmap=color_palette, alpha=1, rasterized=rasterized, antialiased=True)
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
        if log:
            _highlight_points(ax, data, x_vals, y_vals, highlight, size, highlight_size)
        else:
            _highlight_points(ax, data, x_vals_all, y_vals_all, highlight, size, highlight_size)

    if label:
        if log:
            _label_points(ax, data, x_vals, y_vals, label, size, highlight_size, y_label_offset)
        else:
            _label_points(ax, data, x_vals_all, y_vals_all, label, size, highlight_size, y_label_offset)

    ax.grid(False)
    if axis:
        ax.set_axis_off()
    if axis_x:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
    if axis_y:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if path_for_save:
        fig.savefig(f"{path_for_save}/{x}_{y}_{mode_slot}.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()


def plot_heatmap(
    data,
    mode_slot: Union[str, list, None] = None,
    columns: Optional[Union[str, list]] = None,
    genes: Optional[list] = None,
    summarize: pd.DataFrame = None,
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
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
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

        summarize: pd.DataFrame, optional
            A summary DataFrame. This can be retrieved via GrandPy.get_summary_matrix(). columns will be ignored if provided

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

        path_for_save : str, optional
            Saves the plot as a PNG to the specified directory

        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.

        See Also
        --------
        GrandPy.plots : Get the names of all stored plot functions.
        GrandPy.with_plot : Add a plot function.
        GrandPy.with_dropped_plots : Remove plots matching a regex.
        GrandPy.plot_global : Executes a stored global plot function.
        GrandPy.get_summary_matrix : Get a summarization matrix for averaging or aggregation.
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

        table = data.get_table(mode_slots=mode_slot, genes=genes, columns=selected_columns, summarize=summarize)
    else:
        table = data.get_analysis_table(names=[ms.slot for ms in mode_slots], genes=genes)
        table = table[selected_columns]
    mat = table.to_numpy(dtype=np.float64)
    gene_names = table.index.to_list()
    sample_names = table.columns.to_list()

    if isinstance(transform, str):
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
            mat = _transform_logFC(mat)
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

    g = sns.clustermap(
        df,
        figsize=(10, 6),
        cmap=cmap,
        norm = norm,
        row_cluster=cluster_genes,
        col_cluster=cluster_columns,
        yticklabels=label_genes,
        xticklabels=True,
        cbar_kws={"label": label},
        cbar_pos=(0.000002, 0.08, 0.02, 0.84),
    )
    # Write the number of genes on the y-axis
    g.ax_heatmap.set_ylabel(f"n = {df.shape[0]}")

    if return_matrix:
        print(df, table)
    if title:
        plt.title(title, y=1.05)
    #plt.tight_layout()     # Makes cbar way to big :(
    if path_for_save:
        g.savefig(f"{path_for_save}/Heatmap_{mode_slot}.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()


#Beispielaufruf: plot_pca(sars)
def plot_pca(
    data: GrandPy,
    mode_slot: str | ModeSlot = None,
    ntop: int = 500,
    aest: Optional[dict] = None,
    x: int = 1,
    y: int = 2,
    columns: Union[str, list, None] = None,
    do_vst: bool = True,
    show_progress: bool = True,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
        Perform a principal component analysis (PCA) on a GrandPy dataset and visualize the results.

        The function extracts the specified mode slot, optionally applies
        variance-stabilizing transformation (VST) on raw counts using PyDESeq2-like methods, selects
        the top most variable features, and then computes and plots a PCA biplot colored by sample
        metadata.

        Parameters
        ----------
        data : GrandPy
            A GrandPy object containing the data matrix and metadata.
        mode_slot : str or ModeSlot, optional
            The slot or mode to use for data retrieval.
        ntop : int, default=500
            Number of top most variable genes/features to include in the PCA.
        aest : dict, optional
            A dictionary defining aesthetic mappings for plotting, e.g., color or shape.
        x : int, default=1
            Principal component to use for the x-axis (e.g., 1 = PC1).
        y : int, default=2
            Principal component to use for the y-axis (e.g., 2 = PC2).
        columns : str, list, or None, optional
            Column selection filter: if str, interpreted as a pandas query on coldata;
            if list, interpreted as a list of sample names to include; if None, all samples.
        do_vst : bool, default=True
            Whether to apply variance-stabilizing transformation on raw counts before PCA
            (only if mode_slot is 'count').
        show_progress: bool, default=True
            Shows progress for the PCA. (Only for vst = True)
        path_for_save : str or None, optional
            If given, saves the PCA plot as a PNG in the specified directory.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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

    slotmat = mat.T.round().astype(int)

    if do_vst:
        try:
            dds = DeseqDataSet(counts=slotmat, metadata=coldata, design_factors="Condition", low_memory=True, quiet=show_progress)

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
    plt.title(f"PCA({mode_slot})")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if path_for_save:
        plt.savefig(f"{path_for_save}/PCA_{mode_slot}.{save_fig_format}", format = save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()

#Beispielaufruf: plot_gene_old_vs_new(sars, "UHMK1", show_ci=True)
def plot_gene_old_vs_new(
    data: GrandPy,
    gene: str,
    slot: Optional[str] = None,
    columns: Optional[Union[list, str]] = None,
    log: bool = True,
    show_ci: bool = False,
    aest: Optional[dict] = None,
    size: float = 50,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
        Plot old versus new RNA for a single gene, optionally including confidence intervals.

        This function visualizes the ratio of old and new RNA (based on a labeled pulse-chase experiment)
        for a given gene across samples. Optionally, confidence intervals from credible interval slots
        can be added as error bars. Supports aesthetic grouping via color/shape, log-scaling, and export
        of the figure to disk.

        Parameters
        ----------
        data : GrandPy
            A GrandPy object containing the data matrix and associated metadata.
        gene : str
            The gene to plot.
        slot : str, optional
            The data slot to use (default: data.default_slot).
        columns : str, list, or None, optional
            Column selection filter: if str, interpreted as pandas query on coldata;
            if list, interpreted as a list of sample names to include; if None, all samples.
        log : bool, default=True
            Whether to use logarithmic axes for plotting.
        show_ci : bool, default=False
            If True, adds error bars for credible intervals using slots 'lower' and 'upper'.
        aest : dict, optional
            Dictionary defining aesthetic mappings for plotting (e.g. color, shape).
        size : float, default=50
            Size of scatter points.
        path_for_save : str or None, optional
            If provided, saves the resulting plot as a PNG in the given directory.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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

        ci_lower = data.get_data(mode_slots="lower", genes=gene, columns=selected_columns, with_coldata=False)
        ci_upper = data.get_data(mode_slots="upper", genes=gene, columns=selected_columns, with_coldata=False)
        total = data.get_data(mode_slots=slot, genes=gene, columns=selected_columns, with_coldata=False)

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
    if path_for_save:
        fig.savefig(f"{path_for_save}/{gene}_Old_vs_New.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
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
    size: float = 50,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
        Plot total RNA versus newly transcribed RNA ratio (NTR) for a single gene.

        This function visualizes, for a specified gene, the total RNA abundance against its
        NTR (newly transcribed RNA ratio) across samples, with optional confidence intervals.
        Supports aesthetic grouping by color and shape, log-scaling, and saving the figure.

        Parameters
        ----------
        data : GrandPy
            A GrandPy object containing the data matrix and sample metadata.
        gene : str
            The gene to plot.
        slot : str, optional
            Data slot to use for the total RNA (default: data.default_slot).
        columns : str, list, or None, optional
            Column selection filter: if str, interpreted as pandas query on coldata;
            if list, interpreted as sample names to include; if None, all samples.
        log : bool, default=True
            Whether to use logarithmic scaling for the x-axis.
        show_ci : bool, default=False
            If True, adds error bars using the `lower` and `upper` credible interval slots.
        aest : dict, optional
            Dictionary defining aesthetic mappings (e.g., color, shape).
        size : float, default=50
            Size of scatter points.
        path_for_save : str or None, optional
            If provided, saves the resulting plot as a PNG in the given directory.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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

        ci_lower = data.get_data(mode_slots="lower", genes=gene, columns=selected_columns, with_coldata=False)
        ci_upper = data.get_data(mode_slots="upper", genes=gene, columns=selected_columns, with_coldata=False)

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
    if path_for_save:
        fig.savefig(f"{path_for_save}/{gene}_Total_vs_Ntr.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
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
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
        Plot RNA values of a single gene grouped by a specified metadata category.

        This function visualizes RNA values for a chosen gene across groups (e.g. conditions)
        defined by a metadata column. It supports different RNA “modes”,
        confidence intervals, replicate dodging, custom aesthetics, and transformation
        functions for advanced data manipulation.

        Parameters
        ----------
        data : GrandPy
            A GrandPy object containing data matrices and metadata.
        gene : str
            The gene to plot.
        group : str, default="Condition"
            The column name in `coldata` used for grouping samples along the x-axis.
        mode_slot : str or ModeSlot, optional
            Defines which slot and which mode (total/new/old/ntr) to plot. If None, uses
            `data.default_slot`.
        columns : str, list, or None, optional
            Column selection: a pandas query string, a list of sample names, or None to use all samples.
        log : bool, default=True
            Whether to use logarithmic scaling for the y-axis.
        show_ci : bool, default=False
            If True, displays error bars using the `lower` and `upper` credible interval slots.
        aest : dict, optional
            A dictionary defining aesthetic mappings (e.g., color or shape columns in coldata).
        size : float, default=50
            Point size for the scatterplot.
        transform : callable, optional
            A function to transform the `plot_df` DataFrame before plotting
            (e.g., for normalization or filtering).
        dodge : bool, default=False
            Whether to dodge replicates along the x-axis to reduce overlap.
        path_for_save : str or None, optional
            If provided, saves the plot as a PNG file in the specified path.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.

        Notes
        -----
        - If `show_ci=True`, confidence interval slots (`lower` and `upper`) must be available.
        - If `dodge=True`, replicate samples (by `Replicate` column) are slightly shifted
          to avoid overplotting within the same group.
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

    if transform is not None:
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

        ci_lower = data.get_data(mode_slots="lower", genes=gene, columns=selected_columns, with_coldata=False)
        ci_upper = data.get_data(mode_slots="upper", genes=gene, columns=selected_columns, with_coldata=False)

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
    if path_for_save:
        fig.savefig(f"{path_for_save}/{gene}_Groups_Points.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
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
    transform: Optional[callable] = None,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
        Plot stacked bar plots of “new” vs. “old” RNA fractions for a single gene across groups.

        This function visualizes how the total RNA of a gene is composed of “new” and “old” fractions
        across samples or groups, based on the specified slot. Optionally, confidence intervals
        can be drawn, and user-defined transformations applied.

        Parameters
        ----------
        data : GrandPy
            A GrandPy object containing data matrices and metadata.
        gene : str
            The gene to plot.
        slot : str, optional
            The GrandPy slot from which to take expression values. If None, uses `data.default_slot`.
        columns : str, list, or None, optional
            Which samples to include: a pandas query string, a list of sample names,
            or None to use all samples.
        show_ci : bool, default=False
            If True, shows confidence intervals based on slots `lower` and `upper`.
        xlabels : str, list, or None, optional
            Custom x-axis labels. If a string, it will be evaluated as a Python expression
            within the sample metadata. If a list, directly used as labels.
        transform : callable or str, optional
            Transformation to apply to the data before plotting:
            * `'z'` for z-score
            * `'vst'` for variance-stabilizing transform
            * `'logfc'` for log fold change
            * `'none'` for no transform
            or a custom Python function for advanced users.
        path_for_save : str or None, optional
            If provided, saves the plot as PNG in the given path.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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
            df_vst = _transform_vst(data, selected_columns, mode_slot=slot, genes=gene)

            if gene is not None:
                df_vst = df_vst[gene]
            sample_names = df_vst.index.to_list()
            gene_names = df_vst.columns.to_list()
            mat = df_vst.to_numpy()
            mat = mat.T
            label = "VST"
        elif transform == "logfc":
            if selected_columns is None or len(selected_columns) == 0:
                raise ValueError("Need columns=... to compute logFC reference")
            ref_cols = list(range(len(selected_columns)))
            mat = _transform_logFC(mat, ref_columns=ref_cols)
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

        total = data.get_matrix(mode_slot=slot, genes=gene, columns=selected_columns).squeeze()
        lower = data.get_matrix(mode_slot="lower", genes=gene, columns=selected_columns).squeeze()
        upper = data.get_matrix(mode_slot="upper", genes=gene, columns=selected_columns).squeeze()

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
    if path_for_save:
        fig.savefig(f"{path_for_save}/{gene}_Groups_Bars.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()

# Beispielsaufruf: plot_gene_snapshot_timecourse(sars, "UHMK1")
def plot_gene_snapshot_timecourse(
    data: GrandPy,
    gene: str,
    time: str = "duration.4sU",
    mode_slot: Union[str, ModeSlot, None] = None,
    columns: Optional[Union[str, list]] = None,
    average_lines: bool = True,
    exact_tics: bool = True,
    log: bool = True,
    show_ci: bool = False,
    aest: Optional[dict] = None,
    size: float = 50,
    dodge: bool = False,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
        Plot timecourse snapshot of gene expression for specified mode and samples.

        Visualizes the expression of a single gene over time, optionally showing confidence
        intervals, applying log-scale, and grouping by metadata variables. Supports jittering
        points to avoid overlap (dodge), averaging replicates, and custom aesthetics.

        Parameters
        ----------
        data : GrandPy
            GrandPy object containing gene expression data and sample metadata.
        gene : str
            Gene symbol/name to plot.
        time : str, default='Time.original'
            Column name in sample metadata used as the time variable.
        mode_slot : str, ModeSlot or None, optional
            Specifies the mode and slot of data to plot (e.g., "new", "old", or total RNA).
            If None, uses the default slot from `data`.
        columns : str, list or None, optional
            Which samples to include. Can be:
            - None: all samples
            - string: pandas query to select samples by metadata
            - list: explicit list of sample names
        average_lines : bool, default=True
            Whether to add lines showing mean expression per group.
        exact_tics : bool, default=True
            Whether to use exact numeric time ticks or approximate categorical breaks.
        log : bool, default=True
            Whether to apply log-scale to y-axis (disabled if slot is "ntr").
        show_ci : bool, default=False
            Whether to plot confidence intervals. Requires precomputed slots 'lower' and 'upper'.
        aest : dict or None, optional
            Aesthetic mapping for color and style. Keys typically include "color" and "shape"
            corresponding to metadata columns.
        size : float, default=50
            Marker size for scatter plot points.
        dodge : bool, default=False
            Whether to horizontally jitter points by hue to reduce overlap.
        path_for_save : str or None, optional
            Directory path to save the plot image. If None, does not save.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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

    if isinstance(df[time][0], np.float64):
        x_vals_numeric = df[time]
    else:
        x_vals_numeric = df[time].apply(_parse_time_to_float)

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


        dfslot = data.get_matrix(mode_slot=slot, genes=gene, columns=selected_columns).squeeze()
        dfmode_slot = data.get_matrix(mode_slot=mode_slot, genes=gene, columns=selected_columns).squeeze()
        lower = data.get_matrix(mode_slot="lower", genes=gene, columns=selected_columns).squeeze()
        upper = data.get_matrix(mode_slot="upper", genes=gene, columns=selected_columns).squeeze()


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


        if dodge and hue and hue in df.columns:
            hue_vals = sorted(df[hue].unique())
            n = len(hue_vals)
            if n > 1:
                spread = min(0.2, 0.1 * (n - 1))
                offsets = {val: (-spread + 2 * spread * i / (n - 1)) for i, val in enumerate(hue_vals)}
            else:
                offsets = {hue_vals[0]: 0.0}
            hue_col = df[hue].to_numpy()
            x_dodged = np.array([x + offsets.get(h, 0.0) for x, h in zip(x_vals_numeric, hue_col)])
        else:
            x_dodged = x_vals_numeric

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
    if path_for_save:
        fig.savefig(f"{path_for_save}/{gene}_Snapshot_Timecourse.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()


def plot_vulcano(
    data: GrandPy,
    analyses = None,
    p_cutoff: float = 0.05,
    lfc_cutoff: float = 1,
    annotate_numbers=False,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
#TODO Docstring
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
    if path_for_save:
        fig.savefig(f"{path_for_save}/Vulcano.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()


# Beispielaufruf: plot_gene_progressive_timecourse(sars, "UHMK1")
def plot_gene_progressive_timecourse(
    data: GrandPy,
    gene: str,
    slot: Optional[str] = None,
    time: str = "duration.4sU",
    fit_type: str = "nlls",
    size: float = 25,
    exact_tics: bool = True,
    show_ci: bool = False,
    rescale: bool = True,
    return_tables: bool = False,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
    **kwargs
):
    """
        Plot progressive time course of gene expression including fits for synthesis and degradation.

        Visualizes the total, new, and old RNA expression of a gene over time with fitted kinetic curves
        based on synthesis and degradation parameters. Supports multiple conditions if available.

        Parameters
        ----------
        data : GrandPy
            The GrandPy object containing the expression and metadata.
        gene : str
            Gene name to plot.
        slot : str, optional
            Data slot to use for expression values. Defaults to the data's default slot.
        time : str, default="Time.original"
            Column name in coldata containing time points.
        fit_type : str, default="nlls"
            Type of fit to perform. Passed to the kinetics fitting function.
        size : float, default=50
            Marker size for scatter points.
        exact_tics : bool, default=True
            Whether to use exact original time labels on the x-axis ticks.
        path_for_save : str, optional
            Directory path to save the plot PNG. If None, plot is not saved.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
        **kwargs
            Additional keyword arguments passed to the kinetics fitting function.

        See Also
        --------
        :func:`GrandPy.fit_kinetics`
            Function used to fit synthesis and degradation kinetics to the data.

        GrandPy.plots
            Get the names of all stored plot functions.

        GrandPy.with_plot
            Add a plot function.

        GrandPy.with_dropped_plots
            Remove plots matching a regex.

        GrandPy.plot_global
            Executes a stored global plot function.
        """
    if slot is None:
        slot = data.default_slot

    if f"{time}.original" in data.coldata.columns:
        time_original = data.coldata[f"{time}.original"]
        time_original = np.array(time_original, dtype=str)
    time_values = data.coldata[time].values

    condition = (
        data.coldata["Condition"]
        if "Condition" in data.coldata.columns
        else pd.Series([gene] * len(time))
    )

    total = data.get_matrix(mode_slot=slot, genes=gene).squeeze()
    new = data.get_matrix(mode_slot=ModeSlot("new", slot), genes=gene).squeeze()
    old = data.get_matrix(mode_slot=ModeSlot("old", slot), genes=gene).squeeze()

    if f"{time}.original" in data.coldata.columns:
        df = pd.DataFrame({
            "time_original": time_original,
            "time_values": time_values,
            "total": total,
            "new": new,
            "old": old,
            "condition": condition
        })
    else:
        df = pd.DataFrame({
            "time_values": time_values,
            "total": total,
            "new": new,
            "old": old,
            "condition": condition
        })



    from .utils import _get_kinetics_data
    fit_results = _get_kinetics_data(
        data,
        genes=gene,
        fit_type=fit_type,
        return_fields=["Synthesis", "Half-life", "f0", "Degradation"],
        time=time,
        show_progress=False,
        **kwargs
    )

    if show_ci:
        required_slots = ["lower", "upper"]
        available_slots = data.slots
        if not all(slot in available_slots for slot in required_slots):
            raise ValueError("Compute lower and upper slots first! (ComputeNtrCI)")

        total = data.get_matrix(mode_slot=slot, genes=gene).squeeze()
        lower_mult = data.get_matrix(mode_slot="lower", genes=gene).squeeze()
        upper_mult = data.get_matrix(mode_slot="upper", genes=gene).squeeze()

        lower_total = total
        upper_total = total

        lower_new = total * lower_mult
        upper_new = total * upper_mult

        lower_old = total * (1 - upper_mult)
        upper_old = total * (1 - lower_mult)

        df["lower_total"] = lower_total
        df["upper_total"] = upper_total
        df["lower_new"] = lower_new
        df["upper_new"] = upper_new
        df["lower_old"] = lower_old
        df["upper_old"] = upper_old

    if rescale and fit_type.lower() in ["ntr", "chase"]:
        fac = []
        for i in range(len(df)):
            cond = str(df["condition"].iloc[i])
            t = df["time_values"].iloc[i]

            fit = fit_results.get(f"kinetics_{cond}", fit_results.get(gene))
            if fit is None or gene not in fit.index:
                fac.append(1.0)
                continue

            f0 = fit.loc[gene, f"{cond}_f0"]
            ks = fit.loc[gene, f"{cond}_Synthesis"]
            kd = fit.loc[gene, f"{cond}_Degradation"]
            model_total = _f_old_nonequi(t, f0, ks, kd) + _f_new(t, ks, kd)
            measured_total = df["total"].iloc[i]

            factor = model_total / measured_total if measured_total != 0 else 1.0
            fac.append(factor)

        fac = np.array(fac)
        df["total"] *= fac
        df["new"] *= fac
        df["old"] *= fac

        if show_ci:
            df["lower_total"] *= fac
            df["upper_total"] *= fac
            df["lower_new"] *= fac
            df["upper_new"] *= fac
            df["lower_old"] *= fac
            df["upper_old"] *= fac


    if fit_type == "chase" and "no4sU" in data.coldata.columns:
        mask = ~data.coldata["no4sU"].values
        df = df[mask]


    tt = np.linspace(0, df["time_values"].max(), 100)
    fitted = []
    for cond in df["condition"].unique():
        fit = fit_results.get(f"kinetics_{cond}")
        if fit is None:
            continue
        f0 = fit.loc[gene, f"{cond}_f0"]
        ks = fit.loc[gene, f"{cond}_Synthesis"]
        kd = fit.loc[gene, f"{cond}_Degradation"]

        if fit_type == "chase":
            expr_old = ks / kd - _f_old_equi(tt, ks, kd)
            expr_new = _f_old_equi(tt, ks, kd)
        else:
            expr_old = _f_old_nonequi(tt, f0, ks, kd)
            expr_new = _f_new(tt, ks, kd)

        fitted.append(pd.DataFrame({
            "time_values": tt,
            "Expression": expr_old,
            "Type": "old",
            "condition": cond
        }))
        fitted.append(pd.DataFrame({
            "time_values": tt,
            "Expression": expr_new,
            "Type": "new",
            "condition": cond
        }))
    df_fitted = pd.concat(fitted, ignore_index=True)

    df_long = pd.melt(
        df,
        id_vars=["time_values", "condition"],
        value_vars=["total", "new", "old"],
        var_name="Type",
        value_name="Expression"
    )

    g = sns.FacetGrid(
        df_long,
        col="condition",
        sharey=True,
        sharex=True,
        height=6,
        aspect=1
    )

    # Plot total, new, old points
    g.map_dataframe(
        sns.scatterplot,
        x="time_values",
        y="Expression",
        hue="Type",
        palette={"total": "gray", "new": "#e34a33", "old": "#2b8cbe"},
        s=size,
        antialiased=True
    )

    # Plot total curve
    g.map_dataframe(
        lambda data, color, **kws: sns.lineplot(
            data[data["Type"] == "total"]
            .groupby(["time_values", "condition"], as_index=False)
            .median(numeric_only=True),
            x="time_values",
            y="Expression",
            color="gray",
            linestyle="solid",
            linewidth=1,
            antialiased=True
        )
    )

    # Plot Confidence Intervals
    if show_ci:
        for cond, cond_df in df.groupby("condition"):
            ax = g.axes_dict[cond]

            for typ, color in zip(["total", "new", "old"], ["gray", "#e34a33", "#2b8cbe"]):
                y = cond_df[typ]
                ymin = cond_df[f"lower_{typ}"]
                ymax = cond_df[f"upper_{typ}"]

                yerr = np.array([y - ymin, ymax - y])

                mask = np.isfinite(y) & np.isfinite(ymin) & np.isfinite(ymax)
                err_low = y - ymin
                err_high = ymax - y
                mask_positive = mask & (err_low >= 0) & (err_high >= 0)

                if not np.any(mask_positive):
                    continue

                ax.errorbar(
                    cond_df["time_values"][mask_positive],
                    y[mask_positive],
                    yerr=np.array([err_low[mask_positive], err_high[mask_positive]]),
                    fmt='none',
                    ecolor=color,
                    elinewidth=1,
                    capsize=2,
                    alpha=0.6,
                    antialiased=True
                )

    # Plot fitted curves
    for cond, ax in zip(df["condition"].unique(), g.axes.flat):
        for line_type in ["new", "old"]:
            df_fit = df_fitted[
                (df_fitted["condition"] == cond) &
                (df_fitted["Type"] == line_type)
                ]
            ax.plot(
                df_fit["time_values"],
                df_fit["Expression"],
                linestyle="dashed",
                linewidth=1,
                color={"new": "#e34a33", "old": "#2b8cbe"}[line_type],
                label=f"{line_type} (fit)",
                antialiased=True
            )

    g.set_ylabels("Expression")
    g.set_titles("{col_name}")
    g.add_legend(title="RNA")

    # Exact tics
    if exact_tics:
        if "time_original" in df.columns:
            brdf = df[["time_values", "time_original"]].drop_duplicates().sort_values("time_values")
            brdf["time_original"] = brdf["time_original"].astype(str).str.replace("_", ".", regex=False)
            for ax in g.axes.flat:
                ax.set_xticks(brdf["time_values"])
                ax.set_xticklabels(brdf["time_original"], rotation=45)
                ax.set_xlabel("")
        else:
            unique_times = sorted(df["time_values"].unique())
            for ax in g.axes.flat:
                ax.set_xticks(unique_times)
                ax.set_xticklabels([f"{x:.2f}" for x in unique_times], rotation=0)
                ax.set_xlabel("4sU labeling [h]")
    else:
        from matplotlib.ticker import MaxNLocator
        for ax in g.axes.flat:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel("4sU labeling [h]")
            ax.tick_params(axis='x', rotation=0)
    if path_for_save:
        g.savefig(f"{path_for_save}/{gene}_Progressive_Timecourse.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()
    if return_tables:
        print(df)


def plot_ma(
    data: GrandPy,
    analysis: Optional[str] = None,
    p_cutoff: float = 0.05,
    lfc_cutoff: float = 1.0,
    annotate_numbers: bool = True,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
    Create an MA plot from a GrandPy analysis result.

    The MA plot shows the log2 fold changes (LFC) versus total expression levels,
    highlighting genes with significant differential expression.

    Parameters
    ----------
    data : GrandPy
        The GrandPy object containing analysis results and expression data.

    analysis : str, optional
        Name of the analysis to use for plotting. Defaults to the first available analysis.

    p_cutoff : float, default=0.05
        Significance cutoff for adjusted p-values (Q-values). Genes with Q < p_cutoff are highlighted.

    lfc_cutoff : float, default=1.0
        Fold-change cutoff for highlighting genes with substantial up/down regulation.
        Horizontal reference lines are drawn at ±lfc_cutoff.

    annotate_numbers : bool, default=True
        Whether to annotate the plot with counts of significantly up- and down-regulated genes.

    path_for_save : str, optional
        If specified, saves the plot as a PNG file to this directory.
    save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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
    if path_for_save:
        fig.savefig(f"{path_for_save}/MAPlot.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()


def plot_expression_test(
    data: GrandPy,
    w4sU: str,
    no4sU: Union[str, int],
    ylim: tuple = (-1, 1),
    hl_quantile: float = 0.8,
    size: float = 10,
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
        Generate a scatter plot comparing expression between 4sU-labeled and non-labeled samples.

        The plot shows log2 fold changes (4sU vs no4sU) versus mean log10 expression,
        colored by point density to visualize expression distribution and fold-change relationships.

        Parameters
        ----------
        data : GrandPy
            The GrandPy object containing expression data.

        w4sU : str
            Column name or identifier(s) for 4sU-labeled samples.

        no4sU : str or int
            Column name(s) or a constant value representing non-4sU samples.
            If an int/float, treated as a constant expression value.

        ylim : tuple of float, default=(-1, 1)
            Y-axis limits for log2 fold change values.

        hl_quantile : float, default=0.8
            Quantile to determine half-life cutoff. (Currently unused in the plot)

        size : float, default=10
            Size of scatter points.

        path_for_save : str, optional
            Path to save the plot as PNG, if provided.
        save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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

    w = data.get_matrix(mode_slot="count", columns=w4sU)
    if isinstance(no4sU, (int, float)):
        n = np.full_like(w, fill_value=no4sU)
    else:
        n = data.get_matrix(mode_slot="count", columns=no4sU)
    valid = np.isfinite(w + n)
    w = w[valid]
    n = n[valid]

    m = np.vstack([w, n]).T
    lfc_mat = _transform_logFC(m, reference_columns=[1])
    lfc = lfc_mat[:, 0]
    M = (np.log10(w + 1) + np.log10(n + 1)) / 2
    xy = np.vstack([M, lfc])
    density = _density2d(M, lfc, n=100, margin='n')

    idx = np.argsort(density)
    M, lfc, density = M[idx], lfc[idx], density[idx]
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(M, lfc, c=density, cmap="viridis", s=size, alpha=1)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel("Mean expression (log10)")
    ax.set_ylabel("log2 FC (4sU / no4sU)")
    ax.set_ylim(ylim)
    fig.colorbar(sc, ax=ax, label="Density")
    ax.set_title("Expression Test: 4sU vs no4sU")
    plt.tight_layout()
    if path_for_save:
        fig.savefig(f"{path_for_save}/Expression_Test.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()


def plot_type_distribution(
    data,
    mode_slot: Optional[str] = None,
    relative: bool = False,
    palette: str = "Dark2",
    path_for_save: Optional[str] | Path = None,
    save_fig_format: str = "svg",
    show_plot: bool = True,
):
    """
    Plot the distribution of gene types across conditions.

    Displays a barplot showing the sum (or relative percentage) of expression values
    for each gene type grouped by condition.

    Parameters
    ----------
    data : object
        Object containing expression data and gene information.
        Expected to have methods:
          - get_table(mode_slot) -> pd.DataFrame (genes × conditions)
        and attribute:
          - gene_info : pd.DataFrame with a 'Type' column specifying gene types.

    mode_slot : str, optional
        Data slot to use for expression matrix retrieval.
        If None, uses the default slot defined on the data object.

    relative : bool, default=False
        If True, plots relative percentages per condition instead of absolute sums.

    palette : str, default="Dark2"
        Color palette for the gene types.

    path_for_save : str, optional
        If provided, saves the plot as a PNG file to this path.
    save_fig_format: str, default="svg"
            The format ti save the figure. Can be "png", "svg", or any other format supported by matplotlib.
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

    df = data.get_table(mode_slot)
    gene_types = data.gene_info['Type'].unique()

    sums = pd.DataFrame({
        t: df.loc[data.gene_info['Type'] == t].sum(axis=0)
        for t in gene_types
    })

    sums = sums.loc[:, sums.sum(axis=0) > 0]

    if relative:
        sums = sums.div(sums.sum(axis=1), axis=0) * 100
        mode_slot = f"{mode_slot} [%]"

    df_long = sums.reset_index().melt(id_vars='index', var_name='Type', value_name='value')
    df_long.rename(columns={'index': 'Condition'}, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_long, x='Condition', y='value', hue='Type', palette=palette, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(mode_slot)
    ax.tick_params(axis='x', rotation=90)
    if relative:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0,
            title="Type"
        )
        plt.subplots_adjust(right=0.75)
    else:
        ax.legend(title="Type")
        plt.tight_layout()
    if path_for_save:
        fig.savefig(f"{path_for_save}/Type_Distribution.{save_fig_format}", format=save_fig_format, dpi=300)
    if show_plot:
        plt.show()
        plt.close()