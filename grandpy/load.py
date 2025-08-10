import gzip
import re
import shutil
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Optional, Callable
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

from grandpy.core_grandpy import GrandPy
from grandpy.utils import _to_sparse, _make_unique

# Predefined design variable names for harmonized analysis (mirrors R's Design list)
DESIGN_KEYS = {
    "conc.4sU": "concentration.4sU",
    "has.4sU": "has.4sU",
    "dur.4sU": "duration.4sU",
    "Replicate": "Replicate",
    "Condition": "Condition",
    "hpi": "hpi",
    "hps": "hps",
    "Library": "Library",
    "Sample": "Sample",
    "Barcode": "Barcode",
    "Origin": "Origin"
}


SEMANTICS = {
    "dur.4sU" : "time",
    "duration.4sU": "time",
    "Experimental.time": "time",
    "Time": "time",
    "concentration.4sU": "concentration"
}


def _local_filename_from_url(url: str, default: str = "download") -> str:
    """
    Return a clean filename extracted from an HTTP(S) URL.

    The query string and fragment are stripped so that the result can be
    used safely as a local temporary file name (e.g. on Windows, where “?”
    is invalid in filenames).

    Parameters
    ----------
    url : str
        Remote URL pointing to a downloadable file. e.g. ".../download=1"
    default : str, optional
        Fallback name if the URL does not contain a path component (default is "download").

    Returns
    -------
    str
        The basename of the URL path *without* query or fragment, e.g. "data.tsv.gz".

    Notes
    -----
    This helper is used internally by 'read_grand' before the file is downloaded into
    a temporary directory.
    """

    parts = urlparse(url)
    name = Path(unquote(parts.path)).name
    return name or default


def _infer_suffixes_from_df(df, known_suffixes=None, estimator="Binom", sparse=False) -> dict:
    """
    Automatically tries to recognize slots (count, ntr, alpha, beta, ...) and their suffixes from column names.

    Parameters
    ----------
    df : pd.DataFrame
         Input data with column names such as “Mock.1h.1 alpha” or “WT.2 NTR MAP”

    known_suffixes : dict[str, list[str]]
        Optional: known suffix suggestions for known slots (e.g. taken from grandR)

    estimator : str, optional
        If specified, will restrict slot recognition to suffixes associated with this estimator
        (e.g. "MAP" -> " MAP NTR", " MAP alpha", etc.)

    sparse : bool, default=False
        If True, only estimator-based slots (ntr, alpha, beta, shape) will be considered

    Returns
    -------
    dict[str, str]
        Slot names with recognized (single) suffix
    """

    if known_suffixes is None:
        if sparse and estimator:
            suffix_map = {
                "ntr":   [f" {estimator} NTR"],
                "alpha": [f" {estimator} alpha"],
                "beta":  [f" {estimator} beta"],
                "shape": [f" {estimator} shape"]
            }
        else:
            suffix_map = {
                "ntr":   [" MAP", " NTR MAP", " Binom NTR MAP", " TbBinom NTR MAP", " TbBinomShape NTR MAP"],
                "alpha": [" alpha", " Binom alpha", " TbBinom alpha", " TbBinomShape alpha"],
                "beta":  [" beta", " Binom beta", " TbBinom beta", " TbBinomShape beta"],
                "shape": [" shape"]
            }

        if not sparse:
            suffix_map.update({
                "count": [" Readcount", " Read count", "Readcount", "Read count"],
                "ll": [" ll"],
                "llr": [" llr"]
            })

        known_suffixes = suffix_map

    result = {}
    for slot, suffix_list in known_suffixes.items():
        for suffix in suffix_list:
            matching = [col for col in df.columns if col.lower().endswith(suffix.lower())]
            if matching:
                result[slot] = suffix
                break

    return result


def _remove_suffixes(name, suffixes):
    """
    Remove specified suffix(es) from a string if present.

    Parameters
    ----------
    name : str
        Input string to process.

    suffixes : str or tuple of str
        Suffix or suffixes to remove.

    Returns
    -------
    str
        The string with the suffix removed if a match was found;
        otherwise, returns the original string unchanged.
    """

    if isinstance(suffixes, str):
        if name.endswith(suffixes):
            return name[:-len(suffixes)]
        else:
            return name
    else:  # tuple of suffixes
        for suf in suffixes:
            if name.endswith(suf):
                return name[:-len(suf)]
        return name  # no suffix matched


def _parse_slots(df, suffixes, sparse):
    """
    Extracts expression matrices from the input DataFrame based on known slot suffixes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from a GRAND-SLAM result file.

    suffixes : dict[str, str]
        Dictionary mapping slot names (e.g. "count", "ntr") to their column suffixes.

    sparse : bool
        Whether to return the matrices in sparse format.

    Returns
    -------
    slots : dict[str, ndarray or csr_matrix]
        Expression matrices keyed by slot.
    sample_names : list[str]
        Unique sample names present in **at least one** slot.
    slot_sample_names : dict[str, list[str]]
        Per-slot sample names (before padding).
    """

    slots = {}
    sample_names = None
    slot_sample_names = {}

    for slot, suffix in suffixes.items():
        cols = [c for c in df.columns if c.endswith(suffix)]

        if not cols:
            continue

        mat = df[cols].to_numpy()
        mat = np.where(np.isnan(mat), 0, mat)
        if sparse:
            mat = _to_sparse(mat)
        slots[slot] = mat

        sample_names_this_slot = [_remove_suffixes(c, suffix) for c in cols]
        slot_sample_names[slot] = sample_names_this_slot

        if sample_names is None:
            sample_names = sample_names_this_slot

    return slots, sample_names, slot_sample_names


def _build_gene_info(df, classify_func):
    """
    Extracts gene metadata and assigns a gene type to each entry.

    Parameters
    ----------
    df : pd.DataFrame
        The full GRAND-SLAM input DataFrame.

    classify_func : callable
        Function to classify genes into types, taking a DataFrame and returning a Series

    Returns
    -------
    pd.DataFrame
        Gene annotation DataFrame with columns: Symbol, Gene, Length Type.
    """

    _validate_input(df, ["Gene", "Symbol", "Length"], context="gene_info")
    gene_info = df[["Gene", "Symbol", "Length"]].copy()

    gene_info["Type"] = classify_func(gene_info)
    gene_info["Symbol"] = _make_unique(gene_info["Symbol"], warn = True)
    gene_info.index = gene_info["Symbol"]
    return gene_info[["Symbol", "Gene", "Length", "Type"]]


def parse_time_string(s):
    """
    Convert textual time strings such as '90min' or '1h' to hours.
    Parameters
    ----------
    s : str | int | float | pandas.NA
        Time specification.

    Returns
    -------
    float or None
        Numeric value in hours or 'None' if the string could not be parsed.
    """
    if pd.isna(s) or s in ["-", "no4sU", "nos4U"]:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)

    s = str(s).strip().lower().replace("_", ".") # Problematik mit 0_5 hiermit versucht zu umgehen?
    if s.endswith("min"):
        return float(s.replace("min", "")) / 60
    elif s.endswith("h"):
        return float(s.replace("h", ""))
    elif re.fullmatch(r"\d+", s):
        return float(s) / 60
    else:
        return None


def _design_semantics(coldata: pd.DataFrame) -> pd.DataFrame:
    """
    Add semantic hints to coldata (time, concentration, ...).

    Parameters
    ----------
    coldata : pd.DataFrame
        Design/metadata table with one row per sample.

    Returns
    -------
    pd.DataFrame
        Same object, but with an '_semantics' attribute that maps selected columns
        to semantic keywords (e.g. {'Time': 'time', 'Concentration': 'concentration'}).
    """

    semantics = {}

    for key, kind in SEMANTICS.items():
        if key in coldata.columns:
            semantics[key] = kind

    coldata.attrs["_semantics"] = semantics
    return coldata


def semantics_time(values, name):
    """
    Convert a sequence of time strings to a numeric column with semantics.

    Parameters
    ----------
    values : Sequence[str]
        Raw time strings (e.g. '90min', '1h', '-').
    name : str
        Column name to use for the parsed numeric values.

    Returns
    -------
    pd.DataFrame
        Two-column dataframe + attached '_semantics' attribute.
    """

    df = pd.DataFrame({name: values})
    df[f"{name}.original"] = df[name]
    df[name] = df[name].map(parse_time_string)
    df = _design_semantics(df)
    return df


def build_coldata(names, design=None, semantics: Optional[dict[str, Callable]] = None):
    """
    Builds sample metadata (coldata) from sample names and an optional experimental design.

    Parameters
    ----------
    names : list[str]
        List of sample identifiers extracted from column names (e.g. ['Mock.90min.A', ...].

    design : tuples[str] or None
        Tuple of design variables to extract from sample names via splitting on '.'.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sample, containing design info and a 'no4sU' flag (if applicable).
    """

    if isinstance(design, pd.DataFrame):
        df = design.copy()
        if df.index.name != "Name":
            if "Name" in df.columns:
                df = df.set_index("Name")
            else:
                raise ValueError("Design DataFrame must have a 'Name' column or a 'Name' index.")
        df.index = df.index.map(str)
        df = df.reindex(names)
        df = df.reset_index().rename(columns={"index": "Name"})

        if "no4sU" not in df.columns:
            df["no4sU"] = False

        return _design_semantics(df)

    if callable(design):
        return design(names)

    split_names = [name.split(".") for name in names]
    max_fields = max(len(parts) for parts in split_names)

    if design is None:
        raise ValueError("Design must be specified.")
    elif len(design) < max_fields:
        design += tuple(f"Extra_{i+1}" for i in range(max_fields - len(design)))

    design = design[:max_fields]

    aligned_rows = [parts + [None] * (max_fields - len(parts)) for parts in split_names]
    coldata = pd.DataFrame(aligned_rows, columns=design, index=pd.Index(names, name="Name"))
    coldata["Name"] = coldata.index

    time_cols = []
    for col in design:
        if col in SEMANTICS:
            orig = f"{col}.original"
            coldata[orig] = coldata[col]
            coldata[col] = coldata[col].map(parse_time_string)
            time_cols.append(orig)

    if semantics:
        for key, func in semantics.items():
            if key in coldata.columns:
                sem_df = func(coldata[f"{key}.original"] if f"{key}.original" in coldata.columns else coldata[key], key)
                for col in sem_df.columns:
                    if col != key and col not in coldata.columns:
                        coldata[col] = sem_df[col]
                if "_semantics" in sem_df.attrs:
                    coldata.attrs.setdefault("_semantics", {}).update(sem_df.attrs["_semantics"])


    if "has.4sU" in coldata.columns:
        coldata["no4sU"] = coldata["has.4sU"].astype(str).str.lower() == "no4su"
    elif "no4sU" not in coldata.columns:
        coldata["no4sU"] = False
        coldata["no4sU"] = (pd.Series(coldata.index, index=coldata.index).str.lower().str.contains("no4su"))

    # für geordnete Ausgabe:
    pairs = []
    for c in design:
        pairs.append(c)
        orig = f"{c}.original"
        if orig in coldata.columns:
            pairs.append(orig)

    ordered = ["Name"] + pairs + ["no4sU"]
    extra = [c for c in coldata.columns if c not in ordered and c != "Name"]
    coldata = coldata[ordered + extra]

    return _design_semantics(coldata)


def _pad_slots(slots, sparse, coldata, slot_sample_names) -> dict:
    """
    Pads all slot matrices to have the same columns (samples), based on coldata["Name"].

    For samples missing in a given slot, a zero-filled column is inserted (dense or sparse).
    This applies to both 4sU-treated and no4sU samples, matching grandR behavior.
    Samples are aligned to match the order in coldata["Name"].

    Parameters
    ----------
    slots : dict[str, np.ndarray or sparse matrix]
        Dictionary of data matrices, e.g. count, ntr, alpha, beta

    sparse : bool
        If True, output matrices are in sparse format. Otherwise, dense NumPy arrays are used.

    coldata : pd.DataFrame
        Sample metadata. Must include a "Name" column, and optionally a "no4sU" flag.

    slot_sample_names : dict[str, list[str]]
        Original sample names per slot (column names without suffix), used for alignment.

    Returns
    -------
    dict[str, np.ndarray or sparse matrix]
        Updated slots with padded sample columns.
    """

    # Liste aller erwarteten Samples aus coldata:
    all_samples = coldata["Name"].tolist()

    for slot_name, matrix in slots.items():
        # Liste der Samples, die im aktuellen Slot tatsächlich vorhanden sind
        slot_samples = slot_sample_names[slot_name]
        n_genes, n_existing_samples = matrix.shape
        new_matrix = []

        # Iteration über alle gewünschten Samples
        for sample in all_samples:
            if sample in slot_samples:
                col_idx = slot_samples.index(sample)
                # Wenn das Sample im Slot vorhanden ist, wird die Spalte übernommen
                col = matrix[:, col_idx]
                # Bei Sparse-Matrix -> Umwandlung in einen dichten Vektor
                if sp.issparse(matrix):
                    col = col.toarray().ravel()
                else:
                    col = col.ravel()
            else:
                # Sample fehlt im Slot
                col = np.zeros(n_genes)

            # Hinzufügen der Spalte
            new_matrix.append(col)

        # Neue Matrix wird zusammengesetzt aus Gene x Sample
        stacked = np.stack(new_matrix, axis=1)

        # Optional als Sparse-Matrix zurückgeben
        if sparse:
            stacked = _to_sparse(stacked)

        # Slot ersetzen durch padded Matrix
        slots[slot_name] = stacked

    return slots


def _validate_input(df, required_columns: list[str], context: str = "", warn_only: bool = False):
    """
    Validates that required columns exist in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate (e.g. full input table or gene metadata).

    required_columns : list[str]
        List of column names that must be present in 'df'.

    context : str
        Optional label to indicate which part of the input is being validated (e.g. 'gene_info').

    warn_only : bool, default=False
        If True, only emit a warning instead of raising an error.

    Returns
    -------
    ValueError
        If one or more required columns are missing (unless 'warn_only=True').
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        message = f"Missing required column(s) in {context or 'input'}: {missing}"
        if warn_only:
            warnings.warn(message)
        else:
            raise ValueError(message)


def classify_genes(gene_info: pd.DataFrame,
                   custom_classes: dict[str, Any] = None,
                   use_default: bool = True,
                   cg_name: str = "Unknown") -> pd.Series:
    """
    Assigns a type to each gene (e.g. "Cellular", "mito", or custom classes).

    Parameters
    ----------
    gene_info : pd.DataFrame
        Gene metadata with at least columns "Gene" and "Symbol".

    custom_classes : dict[str, Any], optional
        Custom class definitions as {class_name: function}, where each function returns a boolean mask.

    use_default : bool, default=True
        Whether to include default classes ("mito", "ERCC", "Cellular").

    cg_name : str, default="Unknown"
        Label for unmatched genes.

    Returns
    -------
    pd.Series
        A categorical Series assigning each gene to a type.

    Notes
    -----
    - Classification is applied in the following priority: custom_classes first,
      then “mito”, then “ERCC”, then “Cellular”, and finally the “Unknown” fallback.
    - The returned Series is of pandas “category” dtype, with categories in that same order.
    """

    classes = {}

    if custom_classes:
        classes.update(custom_classes)

    if use_default:
        classes.update({
            "mito": lambda df: df["Symbol"].str.startswith("MT-", na=False),
            "ERCC": lambda df: df["Gene"].str.contains(r"ERCC-\d{5}", na=False),
            "Cellular": lambda df: df["Gene"].str.match(r"^ENS.*G\d+$", na=False)
        })

    classes[cg_name] = lambda df: pd.Series([True] * len(df), index=df.index)

    gene_type = pd.Series(index=gene_info.index, dtype="object")

    for name, func in reversed(list(classes.items())):
        matches = func(gene_info)
        gene_type[matches] = name

    return gene_type.astype("category")


def _resolve_prefix_path(prefix: str | Path,
                        pseudobulk: Optional[str] = None,
                        targets: Optional[str] = None) -> Path:
    """
    Resolve the path to data.tsv.gz (or .tsv) for a GRAND-SLAM run.

    1. Directly specified file
    2. *prefix*/data.tsv(.gz)
    3. *prefix*.pseudobulk.<targets>.<pseudobulk>/data.tsv.gz
    4. *prefix*.pseudobulk.targets.<pseudobulk>/data.tsv.gz
    5. *prefix*.pseudobulk.<pseudobulk>/data.tsv.gz
    6. *prefix*.pseudobulk.<targets>.*
    7. Fallback: *prefix*.tsv(.gz)

    Parameters
    ----------
    prefix : str or Path
        Base path or prefix to GRAND-SLAM result files.
    pseudobulk : str, optional
        Pseudobulk identifier used in file path pattern.
    targets : str, optional
        Target identifier used in file path pattern.

    Returns
    -------
    pathlib.Path
        Resolved Path to the 'data.tsv.gz' file.
    """
    base = Path(prefix)

    # 1.
    if base.is_file():
        return base

    # 2.
    for fname in ("data.tsv", "data.tsv.gz"):
        cand = base / fname
        if cand.exists() and cand.is_file():
            return cand

    # 2b. Special case: *prefix*/data.tsv/data (data.tsv is a directory) had to insert this for load_test.py
    dt_dir = base / "data.tsv"
    if dt_dir.is_dir():
        for inner_name in ("data", "data.tsv", "data.tsv.gz"):
            inner = dt_dir / inner_name
            if inner.exists() and inner.is_file():
                return inner

    # 3. - 5.
    def _try(path: Path) -> Optional[Path]:
        for fname in ("data.tsv.gz", "data.tsv"):
            cand = path / fname
            if cand.exists():
                return cand
        return None

    if pseudobulk and targets:
        hit = _try(base.parent / f"{base.name}.pseudobulk.{targets}.{pseudobulk}")
        if hit:
            return hit

    if pseudobulk:
        hit = _try(base.parent / f"{base.name}.pseudobulk.targets.{pseudobulk}")
        if hit:
            return hit

    # 6.
    if targets:
        pattern = re.compile(
            rf"^{re.escape(base.name)}\.pseudobulk\.{re.escape(targets)}\..+$"
        )
        for sub in base.parent.iterdir():
            if sub.is_dir() and pattern.match(sub.name):
                hit = _try(sub)
                if hit:
                    return hit

    # 7.
    for ext in (".tsv.gz", ".tsv"):
        cand = Path(f"{base}{ext}")
        if cand.exists():
            return cand

    if pseudobulk and targets:
        swapped_path = prefix.parent / f"{prefix.name}.pseudobulk.{pseudobulk}.{targets}"
        if swapped_path.exists():
            raise FileNotFoundError(
                f"No data found for pseudobulk='{pseudobulk}', targets='{targets}'. Are the arguments swapped?")

    raise FileNotFoundError(f"No data.tsv(.gz) found for prefix='{prefix}' "
        f"(pseudobulk='{pseudobulk}', targets='{targets}')."
    )


def _is_sparse_file(path) -> bool:
    """
    Determines whether the input represents a sparse GRAND-SLAM file by checking for the presence of a 'data.tsv' (or 'data.tsv.gz') file.

    Parameters
    ----------
    path : str or Path
        Directory or file path to check

    Returns
    -------
    bool
        True if the data is considered sparse (no data.tsv present),
        False if data.tsv or data.tsv.gz exists (dense).
    """

    path = Path(path)

    if path.is_file():
        return False
    def exists_any(name, extensions):
        return any((path / f"{name}{ext}").exists() for ext in extensions)

    has_matrix = exists_any("matrix.mtx", [".gz", ""])
    has_barcodes = exists_any("barcodes", [".tsv.gz", ".tsv", ""])
    has_features = exists_any("features", [".tsv.gz", ".tsv", ""])

    return has_matrix and has_barcodes and has_features


def _read_dense(file_path: str,
    default_slot: str = "count",
    design=None,
    *,
    semantics=None,
    classification_genes=None,
    classification_genes_label: str = "Unknown",
    classify_genes_func=None,
    pseudobulk: Optional[str] = None,
    targets: Optional[str] = None,
    estimator: str = "Binom",
    rename_sample: Optional[Callable[[str], str]] = None) -> GrandPy:

    """
    Reads a GRAND-SLAM TSV file into dense NumPy arrays and return GrandPy.

    Parameters
    ----------
    file_path : str
        Path to the input TSV (compressed or uncompressed).

    default_slot : str, default "count"
        Slot to set as the default for downstream operations.

    design : tuple[str] or DataFrame or callable, optional
        Design variables or DataFrame for sample metadata.

    semantics : dict[str, Callable], optional
        Map of column -> function for derived metadata (e.g. time).
        See notebook 03 for an example.

    classification_genes : list[str], optional
        Gene symbols to assign the special label.

    classification_genes_label : str, default "Unknown"
        Label for classification_genes.

    classify_genes_func : callable, optional
        Custom function for gene type classification.

    estimator : str, default "Binom"
        Keyword to infer NTR/alpha/beta suffixes.

    pseudobulk : str, optional
        Pseudobulk tag, propagated into metadata.

    targets : str, optional
        Targets tag, propagated into metadata.

    rename_sample : Callable[[str], str], optional
        Function to rename sample names before coldata is built
        (e.g. regex or string replacements).

    Returns
    -------
    GrandPy
        A GrandPy object containing dense slot arrays.
    """

    if callable(design) or isinstance(design, pd.DataFrame):
        design_arg = design
    elif design is not None:
        if not isinstance(design, (list, tuple)):
            raise TypeError("design must be a tuple/list of strings, a DataFrame, or a callable")
        design_arg = tuple(DESIGN_KEYS.get(d, d) for d in design)
    else:
        design_arg = None

    return _read(file_path, sparse=False, default_slot=default_slot, design=design_arg,
                 semantics=semantics,
                 classification_genes=classification_genes,
                 classification_genes_label=classification_genes_label,
                 classify_genes_func=classify_genes_func,
                 estimator=estimator,
                 rename_sample=rename_sample,
                 pseudobulk=pseudobulk,
                 targets=targets)


def _read_sparse(folder_path,
    default_slot: str = "count",
    design=None,
    semantics=None,
    classification_genes=None,
    classification_genes_label: str = "Unknown",
    classify_genes_func=None,
    pseudobulk: Optional[str] = None,
    targets: Optional[str] = None,
    estimator: str = "Binom",
    rename_sample: Optional[Callable[[str], str]] = None) -> GrandPy:

    """
    Reads a directory of GRAND-SLAM Matrix Market files into sparse CSR matrices.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing matrix.mtx(.gz), features(.tsv), and barcodes(.tsv).

    default_slot : str, default "count"
        Slot to set as the default for downstream operations.

    design :or DataFrame or callable, optional
        Design variables or DataFrame for sample metadata.

    semantics : dict[str, Callable], optional
        Map of column -> function for derived metadata (e.g. time).
        See notebook 03 for an example.

    classification_genes : list[str], optional
        Gene symbols to assign the special label.

    classification_genes_label : str, default "Unknown"
        Label for classification_genes.

    classify_genes_func : callable, optional
        Custom function for gene type classification.

    pseudobulk : str, optional
        Pseudobulk tag, propagated into metadata.

    targets : str, optional
        Targets tag, propagated into metadata.

    estimator : str, default "Binom"
        Keyword to load only matching Matrix Market slots.

    rename_sample : Callable[[str], str], optional
        Function to rename sample names before coldata is built
        (e.g. regex or string replacements).

    Returns
    -------
    GrandPy
        A GrandPy object containing sparse slot matrices.
    """

    if callable(design):
        design_arg = design
    if isinstance(design, pd.DataFrame):
        design_arg = design
    elif design is not None:
        design_arg = tuple(DESIGN_KEYS.get(d, d) for d in design)
    else:
        design_arg = None

    base = Path(folder_path).resolve()

    expected = f".pseudobulk.{targets}.{pseudobulk}" if targets and pseudobulk else None
    actual = Path(folder_path).name

    if expected and expected not in actual:
        reversed_expected = f".pseudobulk.{pseudobulk}.{targets}"
        if reversed_expected in actual:
            raise ValueError(
                f"Incompatible pseudobulk/targets combination: expected '.pseudobulk.{targets}.{pseudobulk}' in path, "
                f"but found '.pseudobulk.{pseudobulk}.{targets}'.")
        else:
            raise ValueError(
                f"The given path '{folder_path}' does not contain expected identifiers "
                f"(pseudobulk='{pseudobulk}', targets='{targets}').")

    return _read(base, sparse=True, default_slot=default_slot, design=design_arg,
                 semantics=semantics,
                 classification_genes=classification_genes,
                 classification_genes_label=classification_genes_label,
                 classify_genes_func=classify_genes_func,
                 pseudobulk=pseudobulk, targets=targets,
                 estimator=estimator,
                 rename_sample=rename_sample)


def read_grand(prefix, pseudobulk=None, targets=None, **kwargs) -> GrandPy:
    """
    Automatically detects dense vs. sparse GRAND-SLAM output and loads into GrandPy.

    This function locates and reads either a TSV-based (dense) or Matrix-Market–
    based (sparse) GRAND-SLAM result set, then returns a GrandPy object.

    Example
    -------
    This example renames "0.5h" -> "0_5h"
    gp = read_grand("https://zenodo.org/record/7612564/files/chase_notrescued.tsv.gz?download=1",
                    design=("Condition", "Time", "Replicate"),
                    rename_sample=lambda v: re.sub(r"\\.chase", "",
                    re.sub(r"0\\.5h", "0_5h",
                    re.sub(r"\\.nos4U", ".no4sU", v))))

    print(gp.coldata)

    Parameters
    ----------
    prefix : str
        Base path or file prefix for GRAND-SLAM outputs (no extension).

    pseudobulk : str, optional
        Pseudobulk identifier used in file‐naming conventions.

    targets : str, optional
        Target identifier used in file-naming conventions.

    **kwargs :
        Passed through to _read(), read_dense() or read_sparse()
        Supported keys include:

        * design : Sequence[str] | DataFrame | callable
            Design-Variables or DataFrame for sample metadata.

        * default_slot : str
            Slot to set as default (e.g. "count").

        * semantics : Mapping[str, Callable], optional
            Map of column -> function for derived metadata (e.g. time).
            See notebook 03 for an example.

        * classification_genes : Sequence[str], optional
            List of gene symbols to assign the special label.

        * classification_genes_label : str, optional
            Tag assigned to `classification_genes` (default "Unknown").

        * classify_genes_func : callable, optional
            Function for gene type classification. Usually classify_genes().

        * estimator : str, optional
            Keyword for slot-suffixes (default "Binom").

        * rename_sample : Callable[[str], str], optional
            Function that converts all sample names before building coldata
            + (e.g. regex or string replacements).

    Returns
    -------
    GrandPy
        A GrandPy object populated with expression slots, gene_info, and coldata.
    """

    try:
        if isinstance(prefix, str) and prefix.startswith(("http://", "https://")):
            print("Detected URL -> downloading to temp file")

            with tempfile.TemporaryDirectory() as tmpdir:
                local_file = Path(tmpdir) / _local_filename_from_url(prefix)
                with urllib.request.urlopen(prefix) as response, open(local_file, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

                result = read_grand(str(local_file.resolve()), pseudobulk=pseudobulk, targets=targets, **kwargs)
                print(f"Temporary file {local_file.name} was deleted after loading.")
                return result

        path = Path(prefix)

        sparse = _is_sparse_file(path)
        if sparse:
            print("Detected sparse format -> using sparse reader")
            return _read_sparse(path, pseudobulk=pseudobulk, targets=targets, **kwargs)

        else:
            file_path = _resolve_prefix_path(prefix, pseudobulk=pseudobulk, targets=targets)
            print("Detected dense format -> using dense reader")
            return _read_dense(str(file_path), **kwargs)

    except ValueError:
        raise

    except FileNotFoundError:
        raise

    except Exception:
        raise


def _find_existing_file(path: Path, base_name: str, extensions=(".gz", "", ".tsv", ".tsv.gz")) -> Path:
    """
    Search path for base_name with any of the given extensions.

    Parameters
    ----------
    path : pathlib.Path
        Directory to scan.
    base_name : str
        File stem without extension (e.g. "matrix.mtx" -> "matrix").
    extensions : tuple[str], optional
        Candidate extensions to test in order. Defaults to (".gz", "", ".tsv", ".tsv.gz").

    Returns
    -------
    pathlib.Path
        The first path that exists.
    """

    for ext in extensions:
        candidate = path / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Expected file '{base_name}' with one of {extensions} not found in {path}")


def _read(file_path, sparse, default_slot, design,
          classification_genes, classification_genes_label,
          classify_genes_func=None, pseudobulk=None, targets=None, estimator="Binom",
          rename_sample: Optional [Callable[[str], str]] = None,
          semantics=None) -> GrandPy:
    """
    Internal loader for GRAND-SLAM TSV or Matrix Market data.

    This helper reads the raw file or directory, parses gene metadata,
    infers or loads slot matrices, builds coldata, and returns a GrandPy.

    Parameters
    ----------
    file_path : str or Path
        Path to a single TSV (dense) file or a directory containing
        matrix.mtx(+gz), features(.tsv), and barcodes(.tsv).

    sparse : bool
        Whether to interpret as sparse (Matrix Market) format or not.

    default_slot : str
        Name of the slot to use as default (e.g. "count").

    design : tuple[str] or DataFrame or callable, optional
        Design variable names or DataFrame for sample metadata.

    classification_genes : list[str] or None
        Gene symbols to treat as a custom class.

    classification_genes_label : str
        Label for the classification_genes group.

    classify_genes_func : callable, optional
        Override function for gene type assignment.

    pseudobulk : str, optional
        Pseudobulk tag, propagated into metadata.

    estimator : str, default "Binom"
        Estimator keyword to filter or infer slot suffixes.

    rename_sample : callable [[str] -> str], optional
        Function to rename sample names before building coldata
        (e.g. regex or string replacements).

    Returns
    -------
    GrandPy
        A GrandPy instance with loaded slots, gene_info, coldata, and metadata.
    """

    path = Path(file_path)

    if sparse:
        base = path

        # matrix_path = find_existing_file(base, "matrix.mtx", (".gz", ""))
        features_path = _find_existing_file(base, "features", (".tsv.gz", ".tsv", ""))
        barcodes_path = _find_existing_file(base, "barcodes", (".tsv.gz", ".tsv", ""))

        features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
        features.iloc[:, 0] = features.iloc[:, 0].astype(str)
        features.iloc[:, 1] = features.iloc[:, 1].astype(str)
        barcodes = (
            pd.read_csv(barcodes_path, header=None, compression="infer")[0]
            .astype(str)
            .tolist()
        )

        if features.shape[1] == 4:
            features["Length"] = 1

        features.columns = ["Gene", "Symbol", "Mode", "Category", "Length"]
        gene_info = features[["Gene", "Symbol", "Length"]].copy()

        if classify_genes_func is None:
            if classification_genes:
                custom = {classification_genes_label: lambda g: g["Symbol"].isin(classification_genes)}
            else:
                custom = {}
            classify_genes_func = lambda gene_info: classify_genes(gene_info, custom_classes=custom, use_default=True)

        gene_info["Symbol"] = _make_unique(gene_info["Symbol"], warn=True)
        gene_info["Type"] = classify_genes_func(gene_info)
        gene_info.index = _make_unique(gene_info["Symbol"])

        coldata = build_coldata(barcodes, design, semantics=semantics)

        slots = {}
        slot_sample_names = {}

        for mtx_file in sorted(base.glob("*.mtx")) + sorted(base.glob("*.mtx.gz")):
            filename = mtx_file.name
            basename = filename.replace(".mtx", "").replace(".gz", "")

            # matrix.mtx = count
            if basename == "matrix":
                slot = "count"
            else:
                slot = basename.split(".")[-1].lower()

            # estimator wird angegeben -> nur Slots laden, die diesen im Namen enthalten
            if estimator and slot != "count":
                if estimator.lower() not in basename.lower():
                    continue

            try:
                with open(mtx_file, "rt") if mtx_file.suffix != ".gz" else gzip.open(mtx_file, "rt") as file:
                    mat = mmread(file).tocsr()
                slots[slot] = mat
                slot_sample_names[slot] = barcodes
            except Exception as e:
                warnings.warn(f"Slot {slot}: Corrupted File! – {filename} has been skipped ({e})")
                continue

        if not slots:
            raise ValueError(f"No valid slots found in {base}.")

        if default_slot is None:
            if "count" not in slots:
                raise ValueError("Default slot not specified and 'count' slot not found in data.")
            default_slot = "count"

        slots = _pad_slots(slots, sparse=True, coldata=coldata, slot_sample_names=slot_sample_names)

        # version = 3
        # runtime_path = base / "runtime"
        # if runtime_path.exists():
        #     try:
        #         with open(runtime_path) as f:
        #             for line in f:
        #                 if line.lower().startswith("version"):
        #                     version = int(line.strip().split()[-1])
        #                     break
        #     except Exception:
        #         pass

        metadata = {
            "Description": "Loaded via read_grand() (Matrix Market)",
            "default_slot": default_slot,
            "Output": "sparse",
            # "Version": version,
            "pseudobulk": pseudobulk,
            "targets": targets,
            "prefix": str(Path(file_path).resolve()).replace("\\", "/")
        }

        return GrandPy(
            prefix=base.name,
            gene_info=gene_info[["Symbol", "Gene", "Length", "Type"]],
            slots=slots,
            coldata=coldata,
            metadata=metadata
        )

    else:
        df = pd.read_csv(file_path, sep="\t", compression="infer")
        prefix = str(Path(file_path).resolve())

        slot_suffixes = _infer_suffixes_from_df(df, estimator=estimator, sparse=False)
        slots, sample_names, slot_sample_names = _parse_slots(df, slot_suffixes, sparse)

        if rename_sample is not None:
            slot_sample_names = {
                slot: [rename_sample(v) for v in names]
                for slot, names in slot_sample_names.items()
            }

        if "count" in slot_sample_names:
            sample_names = slot_sample_names["count"]
        else:
            from itertools import chain
            seen = set()
            sample_names = [
                s for s in chain.from_iterable(slot_sample_names.values())
                if not (s in seen or seen.add(s))
            ]

        if default_slot is None:
            if "count" not in slots:
                raise ValueError("Default slot not specified and 'count' slot not found in data.")
            default_slot = "count"

        if "ntr" not in slots:
            if "alpha" in slots and "beta" in slots:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ntr = slots["alpha"] / (slots["alpha"] + slots["beta"])
                if sparse:
                    ntr = _to_sparse(ntr)
                slots["ntr"] = ntr
                slot_sample_names["ntr"] = slot_sample_names.get("alpha", sample_names)
            else:
                warnings.warn("Slot 'ntr' is missing.", UserWarning)

        if classify_genes_func is None:
            if classification_genes:
                custom = {classification_genes_label: lambda g: g["Symbol"].isin(classification_genes)}
            else:
                custom = {}
            classify_genes_func = lambda gene_info: classify_genes(gene_info, custom_classes=custom, use_default=True)

        gene_info = _build_gene_info(df, classify_genes_func)
        coldata = build_coldata(sample_names, design, semantics=semantics)
        slots = _pad_slots(slots, sparse, coldata, slot_sample_names)

        # version = 2
        # dense_indicators = ("umi", "cell", "replicate")
        # columns_lower = [c.lower() for c in df.columns]
        # if any(col.startswith(dense_indicators) for col in columns_lower):
        #     version = 3

        metadata = {
            "Description": "Loaded via read_grand() (TSV)",
            "default_slot": default_slot,
            "Output": "sparse" if sparse else "dense",
            # "Version": version,
            "pseudobulk": pseudobulk,
            "targets": targets,
            "prefix": str(Path(file_path).resolve()).replace("\\", "/")
        }

        return GrandPy(
            prefix=prefix,
            gene_info=gene_info,
            slots=slots,
            coldata=coldata,
            metadata=metadata
        )


def get_table_qc(data: GrandPy, slot="count"):
    """
    Returns a QC table per sample with detected genes, totals, statistics and optional percentages per gene type.

    Parameters
    ----------
    data : GrandPy
        A fully initialised GrandPy object.

    slot : str, default "count"
        Data slot on which QC statistics are computed.

    Returns
    -------
    pandas.DataFrame
        One row per sample with basic stats (detected genes, mean, ...) and
        fractions per gene type; all columns from grand.coldata are appended.
    """
    mat = data.get_matrix(slot)

    gene_info = data.gene_info
    if "Type" not in gene_info.columns:
        raise ValueError("Gene type classification (column 'Type') not found in gene_info.")

    gene_types = gene_info["Type"]
    col_names = data.columns

    df = pd.DataFrame({
        "Name": col_names,
        "Detected": (mat > 0).sum(axis=0),
        "Avg": mat.mean(axis=0),
        "Median": np.median(mat, axis=0),
        "Min": mat.min(axis=0),
        "Max": mat.max(axis=0),
    })

    total_per_sample = mat.sum(axis=0)

    for typ in gene_types.unique():
        mask = gene_types == typ
        frac = mat[mask.values, :].sum(axis=0) / total_per_sample
        df[f"Fraction.{typ}"] = frac

    coldata = data.coldata.reset_index(drop=True)
    df = df.merge(coldata, on="Name", how="left")

    return df