import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from typing import Union, TYPE_CHECKING, Optional, Callable, Sequence
from Py.utils import _ensure_list
from Py.grandPy import GrandPy
from Py.analysis_tool import AnalysisTool
from Py.slot_tool import ModeSlot

from scipy.special import digamma, polygamma
from scipy.stats import norm
from scipy.optimize import minimize, root_scalar

try:
    from pydeseq2.ds import DeseqDataSet, DeseqStats
except ImportError:
    DeseqDataSet = DeseqStats = None

if TYPE_CHECKING:
    from Py.grandPy import GrandPy

# TODO: DELETE EXAMPLES IN THE LAST SPRINT!!!!!

def _get_summary_matrix(
        data: "GrandPy",
        no4sU: bool = False,
        columns: Union[None, str, list[str]] = None,
        average: bool = True) -> pd.DataFrame:
    coldata = data.coldata
    sample_names = coldata.index.tolist()

    if "Condition" not in coldata.columns:
        raise ValueError("Object does not have 'Condition' information!")

    if columns is None:
        columns = sample_names
    else:
        columns = _ensure_list(columns)
        columns = [c for c in columns if c in sample_names]

    # Exclude 4sU-marked samples if requested
    if not no4sU:
        no4su_samples = coldata.index[coldata["no4sU"]]
        columns = list(set(columns) - set(no4su_samples))

    # Mapping: sample_name → condition
    condition_series = coldata.loc[columns, "Condition"]

    # Create indicator matrix
    unique_conditions = condition_series.unique()
    matrix = pd.DataFrame(
        {
            cond: (condition_series == cond).astype(float)
            for cond in unique_conditions
        },
        index=condition_series.index
    )

    # Ensure all samples included, fill 0 if not matched
    all_samples = pd.Index(sample_names)
    matrix = matrix.reindex(index=all_samples).fillna(0)

    # Drop columns with all zeros
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]

    if average:
        matrix = matrix.div(matrix.sum(axis=0), axis=1).fillna(0)

    return matrix


# from psi_lfc.R - waiting for feedback
def empirical_bayes_prior(A: np.ndarray, B: np.ndarray, min_sd: float = 0.0) -> tuple[float, float]:
    """
    Estimates Empirical Bayes prior parameters (a, b) for LFC shrinkage.

    Parameters
    ----------
    A : np.ndarray
        Vector A of counts from condition A.

    B : np.ndarray
        Vector B of counts from condition B.

    min_sd : float
        Minimal standard deviation of the prior (see also Psi_LFC-function below).

    Returns
    -------
    a, b : float
        Estimated prior pseudocounts for A and B, respectively.
    """

    mask = (A > 0) | (B > 0)
    A0, B0 = A[mask], B[mask]

    diff = np.log(A0) - np.log(B0)
    x = np.median(diff)
    q_up = np.quantile(diff, norm.cdf(1))
    q_low = np.quantile(diff, norm.cdf(-1))
    y = max((q_up - x)**2, (-q_low + x)**2)

    if np.isinf(x) or np.isinf(y):
        A1, B1 = A + 1, B + 1
        diff1 = np.log(A1) - np.log(B1)
        x = np.mean(diff1)
        y = np.var(diff1)

    def obj(v):
        return (digamma(v[0]) - digamma(v[1]) - x)**2 + (polygamma(1, v[0]) + polygamma(1, v[1]) - y)**2

    result = minimize(obj, x0=[1.0, 1.0], method="Nelder-Mead")
    a, b = result.x

    sd = np.sqrt((polygamma(1, a) + polygamma(1, b)) / (np.log(2)**2))
    if sd < min_sd:

        def f_fun(f):
            return np.sqrt((polygamma(1, f*a) + polygamma(1, f*b)) / (np.log(2)**2)) - min_sd

        root = root_scalar(f_fun, bracket=[1e-6, 1.0])

        if not root.converged:
            raise RuntimeError("Could not inflate prior SD to min_sd.")

        else:
            f = root.root
            warnings.warn(f"Inflated prior by a factor of {1 / f:.2f}", RuntimeWarning)

        a *= f
        b *= f

    return a, b


# waiting for feedback
def center_median(l: np.ndarray) -> np.ndarray:
    """
    Subtracts the median of the given vector (for normalizing log2 fold changes).

    Parameters
    ----------
    l : np.ndarray
        l Vector of effect sizes (see also Psi_LFC-function below).

    Returns
    -------
    np.ndarray
        A vector of length 2 containing the two parameters.
    """

    return l - np.nanmedian(l)


# waiting for feedback
def norm_lfc(A: np.ndarray,
            B: np.ndarray,
            pseudo: tuple[float, float] = (1.0, 1.0),
            normalize_fun: Callable[[np.ndarray], np.ndarray] = center_median
             ) -> np.ndarray:

    """
    Computes the standard, normalized log2 fold change with given pseudocounts.

    Parameters
    ----------
    A : np.ndarray
        Vector A of counts from condition A.
    B : np.ndarray
        Vector B of counts from condition B.
    pseudo : tuple[float, float]
        Vector of length 2 of the pseudo counts.
    normalize_fun : Callable[[np.ndarray], np.ndarray]
        Function to normalize the obtained effect sizes.

    Returns
    -------
        Normalized LFCs.

    """

    lfc = np.log2(A + pseudo[0]) - np.log2(B + pseudo[1])

    return normalize_fun(lfc)


# waiting for feedback
def Psi_LFC(A: np.ndarray,
            B: np.ndarray,
            prior: Optional[tuple[float, float] | None] = None,
            normalize_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            cre: Union[bool, list[float]] = False,
            verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:

    """
    Computes the optimal effect size estimate and credible intervals if needed.

    Parameters
    ----------
    A : np.ndarray
        Vector A of counts from condition A.

    B : np.ndarray
        Vector B of counts from condition B.

    prior : tuple[float, float] | None
        Prior pseudocounts (a, b). If None, estimated via empirical_bayes_prior(A, B).

    normalize_fun : callable, optional
        Function to normalize raw LFCs (default: median centering).

    cre : bool
        Compute credible intervals.

    verbose : bool
        If True status updates will be provided.

    Returns
    -------
    lfc_centered : np.ndarray
        Shrunk, median‑centered log2 fold‑changes (length = #genes).

    qlfc : np.ndarray, optional
        Credible interval matrix (genes x len(cre)); only if cre is not False.
    """

    if prior is None:
        a, b = empirical_bayes_prior(A, B)
    else:
        a, b = prior

    if verbose:
        print(f"Using prior pseudocounts: a={a:.2f}, b={b:.2f}")

    lfc = (digamma(A + a) - digamma(B + b)) / np.log(2)

    norm_fun = normalize_fun or center_median
    lfc_centered = norm_fun(lfc)

    if verbose:
        print(f"Normalization, median before: {np.nanmedian(lfc):.4f}, after: {np.nanmedian(lfc_centered):.4f}")

    if cre is True:
        cre = [0.05, 0.95]

    if cre is not False:
        qs = cre if isinstance(cre, list) else [cre]
        var = (polygamma(1, A + a) + polygamma(1, B + b)) / (np.log(2) ** 2)
        sd = np.sqrt(var)

        raw_qlfc = np.vstack([norm.ppf(q, loc=lfc, scale=sd) for q in qs]).T
        centered_qlfc = np.apply_along_axis(norm_fun, 0, raw_qlfc)
        return lfc_centered, centered_qlfc

    return lfc_centered


# kinda done - not quite sure about the output yet
def compute_lfc(data: GrandPy,
                name_prefix: str,
                contrasts: Optional[pd.DataFrame] = None,
                slot: str = "count",
                LFC_fun: Callable = Psi_LFC,
                mode: str = "total",
                normalization: Optional[Union[str, Sequence[float]]] = None,
                compute_M: bool = True,
                genes: Optional[list[str]] = None,
                verbose: bool = False,
                **kwargs) -> GrandPy:

    """
    Estimate log2 fold changes and optional M values for each contrast and store analyses.

    Parameters
    ----------
    data : GrandPy
        The grandPy object. Must contain a 'Condition' column in its coldata. See Notes for more.

    name_prefix : str
        The prefix for the new analysis name;
        a "_" and the column names of the contrast matrix are appended;
        can be None (then only the contrast matrix names are used)

    contrasts : pd.DataFrame
        Contrast matrix defining comparisons (samples x contrasts; values 1, -1).

    slot : str
        The slot of the grandPy object to take the data from;
        for Psi_LFC-functions this really should be "count"!

    LFC_fun : function
        Function to compute the log2 fold changes.
        (default Psi_LFC, other viable option: "Norm_LFC") #TODO Norm_LFC()

    mode : str
        Computes LFCs for "total", "new" or "old" RNA.

    normalization :  str or sequence, optional
        If str: use the <normalization>_<slot> slot of normalized counts;
        if sequence: divide each sample by the provided size factor.

    compute_M : bool, default True
        If True and LFC_fun returns M, include the "M" column.

    genes : list of str, optional
        Restrict analysis to these genes; None means all genes.

    verbose : bool
        If True status updates will be provided.

    **kwargs
        Additional keyword arguments are passed to LFC_fun.

    Returns
    -------
    GrandPy
        New GrandPy object with one analysis per contrast. Columns are
        "LFC" (log2 fold change) and optionally "M".

    Notes
    -----
    This functino uses the 'Condition' column in data.coldata to define
    groups for comparison.
    If you need to use a different column, you can rename it to 'Condition'
    before calling this function.
    """
    # TODO @Marius?
    # dadurch dass _get_summary_matrix() die Condition-Spalte festlegt, kann
    # kein anderes Argument verwendet werden.
    # müsste abgeändert werden bei Bedarf, um näher an grandR dran zu sein

    # Filtern der Contrasts
    if contrasts is None:
        contrasts = data.get_contrasts()
    if isinstance(contrasts, dict):
        contrasts = pd.DataFrame(contrasts)

    valid = [col for col in contrasts.columns
        if (1 in contrasts[col].values and -1 in contrasts[col].values)]
    contrasts = contrasts.loc[:, valid]
    if contrasts.shape[1] == 0:
        raise ValueError("Contrasts do not define any comparison!")

    # Roh-Counts aus mode_slot
    mode_slot_obj = ModeSlot(mode, slot)
    try:
        raw_mat = data.get_matrix(mode_slot=str(mode_slot_obj))
    except Exception:
        raise ValueError(f"Invalid mode slot: '{mode_slot_obj}'")
    raw_expr = pd.DataFrame(raw_mat, index=data.genes, columns=data.coldata.index)

    # Thema Genes
    if genes is not None:
        raw_expr = raw_expr.loc[genes]

    # Normalisierungspart
    if isinstance(normalization, (list, np.ndarray, pd.Series)):
        sf = np.array(normalization)
        if sf.shape[0] != raw_expr.shape[1]:
            raise ValueError("Invalid numeric normalization: length mismatch.")
    elif isinstance(normalization, str):
        norm_slot_obj = ModeSlot(normalization, slot)
        try:
            norm_mat = data.get_matrix(mode_slot=str(norm_slot_obj))
        except Exception:
            raise ValueError(f"Invalid normalization slot: '{norm_slot_obj}'")
        norm_expr = pd.DataFrame(norm_mat, index=data.genes, columns=data.coldata.index)
        if genes is not None:
            norm_expr = norm_expr.loc[genes]

    new_data = data

    for contrast in contrasts.columns:
        c = contrasts[contrast]
        A_idx = c[c == 1].index
        B_idx = c[c == -1].index

        sumA = raw_expr[A_idx].sum(axis=1)
        sumB = raw_expr[B_idx].sum(axis=1)

        if isinstance(normalization, (list, np.ndarray, pd.Series)):
            # Shift berechnen
            sf_series = pd.Series(sf, index=raw_expr.columns)
            shift = np.log2(sf_series.loc[A_idx].sum() / sf_series.loc[B_idx].sum())
            lfc_vec = LFC_fun(
                sumA.values, sumB.values,
                normalize_fun=lambda x: x - shift,
                verbose=verbose, **kwargs)
        elif isinstance(normalization, str):
            # erst normalized counts ohne Verschiebung
            nA = norm_expr[A_idx].sum(axis=1).values
            nB = norm_expr[B_idx].sum(axis=1).values
            nlfc = LFC_fun(nA, nB, normalize_fun=lambda x: x, verbose=verbose, **kwargs)
            med = np.median(nlfc)
            # dann raw counts mit Verschiebung um median
            lfc_vec = LFC_fun(
                sumA.values, sumB.values,
                normalize_fun=lambda x: x - med,
                verbose=verbose, **kwargs)
        else:
            # ohne Normalisierung
            lfc_vec = LFC_fun(
                sumA.values, sumB.values,
                verbose=verbose, **kwargs)

        # DataFrame bauen
        df = pd.DataFrame({"LFC": lfc_vec}, index=sumA.index)

        # M‑Value
        if compute_M:
            M = 10 ** (0.5 * (np.log10(sumA + 0.5) + np.log10(sumB + 0.5)))
            df["M"] = M

        df.index.name = "Symbol"

        # Analyse speichern
        name = f"{name_prefix}_{contrast}"
        adata = new_data._adata
        analyses = adata.uns.setdefault("analyses", {})
        analyses[name] = df

    return new_data


# TODO: REWORK!
def pairwise_DEseq2(data: GrandPy,
                    name_prefix : str,
                    contrasts: pd.DataFrame,
                    separate: bool = False,
                    mode: str = "total",
                    slot: str = "count",
                    normalization: Optional[Union[str, Sequence[float]]] = None,
                    genes: Optional[list[str]] = None,
                    verbose: bool = False) -> GrandPy:

    """
    Perform Wald tests for differential expression.
    Apply DESeq2 for comparisons defined in a contrast matrix, requires the DESeq2 package.

    Parameters
    ----------
    data : GrandPy
        The grandPy object.

    name_prefix : str
        The prefix for the new analysis name;
        a "_" and the column names of the contrast matrix are appended;
        can be None (then only the contrast matrix names are used)

    contrasts : pd.DataFrame
        Contrast matrix that defines all pairwise comparisons, generated using get_contrast() #TODO get_contrast() einbauen

    separate : bool
        Model overdispersion separately for all pairwise comparison (TRUE),
        or fit a single model per gene, and extract contrasts (FALSE).

    mode : str
        Compute LFCs for "total", "new", or "old" RNA.

    slot : str
        Which slot to use (should be a count slot, not normalized values).

    normalization : str
         Normalize on "total", "new", or "old".

    # TODO logFC ???

    genes : list
        Restrict analysis to these genes; None means all genes.

    verbose : bool
        If True status updates will be provided.

    Returns
    -------
    GrandPy
        A new GrandPy object including a new analysis table.
        The columns of the new analysis table are
        "M"     - the base mean
        "S"     - the log2FoldChange divided by lfcSE
        "P"     - the Wald test P value
        "Q"     - same as P but Benjamini-Hochberg multiple testing corrected
        "LFC"   - the log2 fold change (only with the logFC parameter set to TRUE) # TODO
    """

    # ' sars <- ReadGRAND(system.file("extdata", "sars.tsv.gz", package = "grandR"),
    # '                   design=c(Design$Condition,Design$dur.4sU,Design$Replicate))
    # ' sars <- subset(sars,Coldata(sars,Design$dur.4sU)==2)
    # ' sars<-PairwiseDESeq2(sars,mode="total",
    # '                               contrasts=GetContrasts(sars,contrast=c("Condition","Mock")))
    # ' sars<-PairwiseDESeq2(sars,mode="new",normalization="total",
    # '                               contrasts=GetContrasts(sars,contrast=c("Condition","Mock")))
    # ' head(GetAnalysisTable(sars,column="Q"))

    if DeseqDataSet is None:
        raise ImportError("pydeseq2 is not installed!")

    tool = AnalysisTool(data.to_anndata())
    new_data = data

    for contrast in contrasts.columns:
        c = contrasts[contrast]
        samples = c[c != 0].index.tolist()
        cols = [data.coldata.index.get_loc(s) for s in samples]
        mat = data.get_matrix(mode_slot=f"{mode}_{slot}")
        sub_counts = pd.DataFrame(mat[:, cols], index=data.genes, columns=samples)
        sub_coldata = data.coldata.loc[samples].copy()
        sub_coldata["Condition"] = np.where(
            c.loc[samples] == 1,
            f"{contrast}_grp1",
            f"{contrast}_grp2")

        dds = DeseqDataSet(counts=sub_counts.T, colData=sub_coldata, design_factors="Condition")
        dds = DeseqStats(dds)
        res = dds.get_results()

        mean_expr = sub_counts.mean(axis=1) + 1
        res["M"] = np.log10(mean_expr)

        table = res[["log2FoldChange", "stat", "pvalue", "padj", "M"]].copy()
        table.columns = ["LFC", "S", "P", "Q", "M"]
        name = f"{name_prefix}_{contrast}"
        new_analyses = tool.with_analysis(name=name, table=table, by="Symbol")
        new_data = new_data.replace(analyses=new_analyses)
    return new_data

#TODO: REWORK!
def pairwise(data: GrandPy,
             name_prefix: str,
             contrasts: pd.DataFrame,
             slot: str = "count",
             mode: str = "total",
             normalization: Optional[Union[str, Sequence[float]]] = None,
             genes: Optional[list[str]] = None,
             verbose: bool = False) -> GrandPy:
    """
    Log2 fold changes and Wald tests for differential expression.
    This function is a shortcut for first calling pairwise_DESeq2 and then compute_LFC.

    Parameters
    ----------
    data : GrandPy
        The grandPy object.

    name_prefix : str
         The prefix for the new analysis name;
         a "_" and the column names of the contrast matrix are appended;
         can be None (then only the contrast matrix names are used).

    contrasts : pd.DataFrame
        Contrast matrix that defines all pairwise comparisons, generated using get_contrasts() #TODO get_contrast() einbauen

    #TODO LFC_fun

    slot : str
        The slot of the grandR object to take the data from; should contain counts!

    mode : str
        Compute LFCs for "total", "new", or "old" RNA.

    normalization : str
        Normalize on "total", "new", or "old".

    genes : list
        Restrict analysis to these genes; None means all genes.

    verbose : bool
        If True status updates will be provided.

    Returns
    -------
    GrandPy
    A new GrandPy object including a new analysis table.
    The columns of the new analysis table are
        "M"     - the base mean
        "S"     - the log2FoldChange divided by lfcSE
        "P"     - the Wald test P value
        "Q"     - same as P but Benjamini-Hochberg multiple testing corrected
        "LFC"   - the log2 fold change (only with the logFC parameter set to TRUE) # TODO
    """

    # contrasts?
    # normalization=mode?

    data2 = pairwise_DEseq2(data,
                            name_prefix=name_prefix, contrasts=contrasts,
                            separate=False, mode=mode, slot=slot,
                            normalization=normalization, genes=genes, verbose=verbose)

    data3 = compute_lfc(data2, name_prefix=name_prefix, contrasts=contrasts,
                        slot=slot, mode=mode, normalization=normalization,
                        genes=genes, verbose=verbose)

    return data3


def _get_contrasts(
        data: "GrandPy",
        contrast: list = ["Condition"],
        columns: Union[Sequence[str], str] = None,
        group: Union[Sequence[str], str] = None,
        name_format: str = None,
        no4sU: bool = False,
        ) -> Union["GrandPy", pd.DataFrame]:

    our_coldata = data.coldata

    if len(contrast) not in [1, 2, 3]:
        raise ValueError("Contrast must be of length 1, 2, or 3.")

    if contrast[0] not in our_coldata.columns:
        raise ValueError(f"Column {contrast[0]} not found in coldata.")
    col = contrast[0]

    # Untermenge der Daten, falls columns gesetzt ist
    use_mask = np.ones(len(our_coldata), dtype=bool)
    if columns is not None:
        use_mask = our_coldata.index.isin(columns) if isinstance(columns, list) else columns

    if name_format is None:
        name_format = "$A vs $B" if group is None else "$A vs $B.$GRP"

    if not no4sU and "no4sU" in our_coldata.columns:
        use_mask &= ~our_coldata["no4sU"].fillna(False)

    def make_name(a, b, grp=""):
        return (name_format
                .replace("$A", str(a))
                .replace("$B", str(b))
                .replace("$COL", col)
                .replace("$GRP", grp))

    def make_vector(a, b, use):
        re = np.zeros(len(our_coldata))
        re[(our_coldata[col] == a) & use] = 1
        re[(our_coldata[col] == b) & use] = -1
        return re

    def contrast_df_for_level_pairs(level_pairs, group_name=""):
        df = {}
        for a, b in level_pairs:
            vec = make_vector(a, b, use_mask)
            name = make_name(a, b, group_name)
            df[name] = vec
        return pd.DataFrame(df, index=data.coldata.index)

    # Hauptlogik für Kontrastgenerierung
    match len(contrast):
        case 1:
            levels = our_coldata.loc[use_mask, col].dropna().unique().tolist()
            level_pairs = list(combinations(levels, 2))
            contrast_df = contrast_df_for_level_pairs(level_pairs)
        case 2:
            levels = our_coldata.loc[use_mask, col].dropna().unique().tolist()
            other_levels = [l for l in levels if l != contrast[1]]
            contrast_df = contrast_df_for_level_pairs([(l, contrast[1]) for l in other_levels])
        case 3:
            contrast_df = contrast_df_for_level_pairs([(contrast[1], contrast[2])])
        case _: raise ValueError("Illegal contrasts (either a name from design (all pairwise comparisons), a name and a reference level (all comparisons vs. the reference), or a name and two levels (exactly this comparison))")


    # Falls Gruppierung gewünscht → innerhalb jeder Gruppe Kontraste berechnen
    if group is not None:
        all_dfs = []
        for grp_val in data.coldata[group].dropna().unique():
            group_mask = (data.coldata[group] == grp_val)
            use_mask_group = use_mask & group_mask
            if len(our_coldata.loc[use_mask_group]) < 2:
                continue
            levels = our_coldata.loc[use_mask_group, col].dropna().unique().tolist()
            if len(levels) < 2:
                continue
            level_pairs = list(combinations(levels, 2)) if len(contrast) == 1 else \
                [(l, contrast[1]) for l in levels if l != contrast[1]] if len(contrast) == 2 else \
                    [(contrast[1], contrast[2])]
            df = contrast_df_for_level_pairs(level_pairs, group_name=str(grp_val))
            all_dfs.append(df)
        contrast_df = pd.concat(all_dfs, axis=1)

    # Entferne Spalten mit nur 0 (irrelevant)
    contrast_df = contrast_df.loc[:, ~(contrast_df == 0).all(axis=0)]

    # Entferne Kontraste mit nur +1 oder nur -1 (nicht sinnvoll)
    remove_mask = ((contrast_df >= 0).all(axis=0)) | ((contrast_df <= 0).all(axis=0))
    if remove_mask.any():
        removed_cols = contrast_df.columns[remove_mask].tolist()
        print(f"Removed uninformative contrasts: {', '.join(removed_cols)}")
        contrast_df = contrast_df.loc[:, ~remove_mask]

    return contrast_df