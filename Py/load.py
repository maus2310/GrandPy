import tempfile
import urllib.request
import shutil
import gzip
import re
import warnings
from typing import Any, TYPE_CHECKING
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from pathlib import Path
from scipy.io import mmread

from Py.utils import _to_sparse, _make_unique
from Py.grandPy import GrandPy


# Predefined design variable names for harmonized analysis (mirrors R's Design list)
DESIGN_KEYS = {
    "has_4sU": "has.4sU",
    "conc_4sU": "concentration.4sU",
    "dur_4sU": "duration.4sU",
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
    "duration.4sU": "time",
    "Experimental.time": "time",
    "Time": "time",
    "concentration.4sU": "concentration"
}


def infer_suffixes_from_df(df, known_suffixes=None, estimator=None) -> dict:
    """
    Automatically tries to recognize slots (count, ntr, alpha, beta, ...) and their suffixes from column names.

    Parameters
    ----------
    df : pd.DataFrame
         Input data with column names such as “Mock.1h.1 alpha” or “WT.2 NTR MAP”

    known_suffixes : dict[str, list[str]]
        Optional: known suffix suggestions for known slots (e.g. taken from grandR)

    Returns
    -------
    dict[str, str]
        Slot names with recognized (single) suffix
    """

    if known_suffixes is None:
        if estimator:
            suffix_map = {
                "ntr": f" {estimator} NTR",
                "alpha": f" {estimator} alpha",
                "beta": f" {estimator} beta",
                "shape": f" {estimator} shape"
            }
        else:
            suffix_map = {}

        known_suffixes = {
            "count": [" Readcount", " Read count", "Readcount", "Read count"],
            "ntr": [" MAP", " NTR MAP", " Binom NTR MAP", " TbBinom NTR MAP", " TbBinomShape NTR MAP"],
            "alpha": [" alpha", " Binom alpha", " TbBinom alpha", " TbBinomShape alpha"],
            "beta": [" beta", " Binom beta", " TbBinom beta", " TbBinomShape beta"],
            "shape": [" shape"],
            "ll": [" ll"],
            "llr": [" llr"]
        }

    result = {}
    for slot, suffix_list in known_suffixes.items():
        for suffix in suffix_list:
            matching = [col for col in df.columns if col.endswith(suffix)]
            if matching:
                result[slot] = suffix
                break

    return result


def remove_suffixes(name, suffixes):
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


def parse_slots(df, suffixes, sparse, *, strict=True):
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

    strict : bool, default=True
        If True, raises an error if **the same sample appears in multiple slots (e.g., 'count', 'alpha')**.
        This is usually expected behavior in GRAND-SLAM, so 'strict=False' is recommended.

    Returns
    -------
    tuple
        A tuple containing:
        - dict[str, np.ndarray or sp.csr_matrix]: data matrices per slot,
        - list[str]: inferred sample names common across slots,
        - dict[str, list[str]]: slot-specific sample names
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

        sample_names_this_slot = [remove_suffixes(c, suffix) for c in cols]
        slot_sample_names[slot] = sample_names_this_slot

        if sample_names is None:
            sample_names = sample_names_this_slot

    pass

    return slots, sample_names, slot_sample_names


def build_gene_info(df, classify_func):
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

    validate_input(df, ["Gene", "Symbol", "Length"], context="gene_info")
    gene_info = df[["Gene", "Symbol", "Length"]].copy()

    gene_info["Type"] = classify_func(gene_info)
    gene_info["Symbol"] = _make_unique(gene_info["Symbol"], warn = True)
    gene_info.index = gene_info["Symbol"]
    return gene_info[["Symbol", "Gene", "Length", "Type"]]


def parse_time_string(s):
    """
    Converts Strings (e.g. '90min', '1h', '-') to float-hours.
    """
    if pd.isna(s) or s in ["-", "no4sU", "nos4U"]:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)

    s = str(s).strip().lower()
    if s.endswith("min"):
        return float(s.replace("min", "")) / 60
    elif s.endswith("h"):
        return float(s.replace("h", ""))
    elif re.fullmatch(r"\d+", s):
        return float(s) / 60
    else:
        return None


def apply_design_semantics(coldata: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a _semantics dictionary to coldata.attrs with semantic hints
    for selected design columns (e.g. time, concentration).
    """
    semantics = {}

    for key, kind in SEMANTICS.items():
        if key in coldata.columns:
            semantics[key] = kind

    coldata.attrs["_semantics"] = semantics
    return coldata


def build_coldata(names, design=None):
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

    for col in coldata.columns:
        if col.lower() in {"time", "duration.rsu"}:
            coldata[f"{col}.original"] = coldata[col]
            coldata[col] = coldata[col].map(parse_time_string)

    no4su_col = next(
        (c for c in coldata.columns if c.endswith(".original") and coldata[c].isin(["no4sU", "nos4U", "-"]).any()),
        None)
    if no4su_col:
        coldata["no4sU"] = coldata[no4su_col].isin(["no4sU", "nos4U", "-"])
    else:
        coldata["no4sU"] = False

    coldata = coldata[["Name"] + [c for c in coldata.columns if c != "Name"]]
    return apply_design_semantics(coldata)


def pad_slots(slots, sparse, coldata, slot_sample_names) -> dict:
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
    warned_once = set()

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
                # Sample fehlt im Slot - hier muss dann 'gepadded' werden
                is_no4su = False
                if "no4sU" in coldata.columns:
                    try:
                        is_no4su = bool(coldata.loc[sample, "no4sU"])
                    except KeyError:
                        pass

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


def validate_input(df, required_columns: list[str], context: str = "", warn_only: bool = False):
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
                   name_unknown: str = "Unknown") -> pd.Series:
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

    name_unknown : str, default="Unknown"
        Label for unmatched genes.

    Returns
    -------
    pd.Series
        A categorical Series assigning each gene to a type.
    """
    classes = {}

    # Benutzerdefinierte Klassen übernehmen
    if custom_classes:
        classes.update(custom_classes)

    # Standardklassen ergänzen
    if use_default:
        classes.update({
            "mito": lambda df: df["Symbol"].str.startswith("MT-", na=False),
            "ERCC": lambda df: df["Gene"].str.contains(r"ERCC-\d{5}", na=False),
            "Cellular": lambda df: df["Gene"].str.match(r"^ENS.*G\d+$", na=False)
        })

    # Unknown-Klasse als Fallback
    classes[name_unknown] = lambda df: pd.Series([True] * len(df), index=df.index)

    # Ergebnis-Vektor initialisieren
    gene_type = pd.Series(index=gene_info.index, dtype="object")

    # Klassifikation rückwärts (benutzerdefiniert überschreibt Standard)
    for name, func in reversed(list(classes.items())):
        matches = func(gene_info)
        gene_type[matches] = name

    return gene_type.astype("category")


def resolve_prefix_path(prefix, pseudobulk=None, targets=None):
    """
    Resolves the actual data file path based on GRAND-SLAM prefix and optional parameters.

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
    Path
        Resolved Path to the 'data.tsv.gz' file.
    """
    base = Path(prefix)

    # <prefix>.pseudobulk.<targets>.<pseudobulk>
    if pseudobulk and targets:
        path = base.parent / f"{base.name}.pseudobulk.{targets}.{pseudobulk}" / "data.tsv.gz"
        if path.exists():
            return path

    # <prefix>.pseudobulk.targets.<pseudobulk>
    if pseudobulk:
        path = base.parent / f"{base.name}.pseudobulk.targets.{pseudobulk}" / "data.tsv.gz"
        if path.exists():
            return path

        # <prefix>.pseudobulk.<pseudobulk>
        path = base.parent / f"{base.name}.pseudobulk.{pseudobulk}" / "data.tsv.gz"
        if path.exists():
            return path

    # <prefix>.pseudobulk.<targets>.*
    if targets:
        pattern = re.compile(f"^{re.escape(base.name)}\\.pseudobulk\\.{re.escape(targets)}\\..+$")
        for i in base.parent.iterdir():
            if i.is_dir() and pattern.match(i.name):
                path = i / "data.tsv.gz"
                if path.exists():
                    return path

    # <prefix>/data.tsv.gz
    path = base / "data.tsv.gz"
    if path.exists():
        return path

    # direct file path
    if base.is_file() and base.name.endswith((".tsv", ".tsv.gz")):
        return base

    raise FileNotFoundError(
        f"No valid 'data.tsv.gz' found.\n"
        f"Checked prefix='{prefix}' with pseudobulk='{pseudobulk}', targets='{targets}'."
    )



def is_sparse_file(path) -> bool:
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

def read_dense(file_path, default_slot="count", design=None, *, classification_genes=None, classification_genes_label="Viral", classify_genes_func=None, estimator=None):
    """
    Reads a GRAND-SLAM TSV file as dense (NumPy) matrices and returns a GrandPy object.

    Parameters
    ----------
    file_path : str
        Path to the input .tsv file.

    default_slot : str, default="count"
        The slot to use as the default slot for plotting and analysis (e.g., "count", "ntr").

    design : tuple[str], optional
        Tuple of design variables to extract from sample names (e.g., ("Condition", "Time")).

    classification_genes : list[str], optional
        List of gene symbols considered viral, to be assigned a custom type.

    classification_genes_label : str, default="Viral"
        Type label to assign to the genes listed in `classification_genes`.

    classify_genes_func : callable, optional
        Custom function to classify gene types. Overrides `classification_genes`.

    estimator : str, optional
        Is responsible for which value (e.g. MAP, Mean, TbBinom, ...) is used for NTR, alpha, beta, etc.

    Returns
    -------
    GrandPy
        A GrandPy object populated with dense matrices and metadata.
    """

    return _read(file_path, sparse=False, default_slot=default_slot, design=design,
                 classification_genes=classification_genes, classification_genes_label=classification_genes_label,
                 classify_genes_func=classify_genes_func, estimator=estimator)


def read_sparse(folder_path, default_slot="count", design=None, classification_genes=None, classification_genes_label="Viral", classify_genes_func=None, pseudobulk=None, targets=None, estimator=None):
    """
    Reads a GRAND-SLAM sparse dataset from a directory.

    Parameters
    ----------
    folder_path : str or Path
        Path to the directory containing matrix.mtx.gz etc.

    default_slot : str
        The Default-Slot is set to "count".

    design : tuple[str], optional
        E.g. ("Condition", "Time")

    classification_genes : list[str], optional
        List of gene symbols considered viral, to be assigned a custom type.

    classification_genes_label : str
        The Default is set to "Viral".

    classify_genes_func : callable, optional

    pseudobulk : callable, optional

    targets : list[str], optional

    Returns
    -------
    GrandPy
    """

    return _read(Path(folder_path), sparse=True, default_slot=default_slot, design=design,
                 classification_genes=classification_genes, classification_genes_label=classification_genes_label,
                 classify_genes_func=classify_genes_func,
                 pseudobulk=pseudobulk, targets=targets, estimator=estimator)

def read_grand(prefix, pseudobulk=None, targets=None, **kwargs):
    """
    Automatically detects whether a GRAND-SLAM dataset is in dense or sparse format
    (Matrix Market or TSV) and loads it accordingly into a GrandPy object.

    Parameters
    ----------
    prefix : str
        Path to file or folder.

    pseudobulk : str, optional
        For resolving TSV-style file paths with prefixes.

    targets : str, optional
        For resolving TSV-style file paths with prefixes.

    **kwargs :
        Passed on to _read().

    Returns
    -------
    GrandPy
    """

    try:
        if isinstance(prefix, str) and prefix.startswith(("http://", "https://")):
            print("Detected URL -> downloading to temp file")

            with tempfile.TemporaryDirectory() as tmpdir:
                local_file = Path(tmpdir) / Path(prefix).name
                with urllib.request.urlopen(prefix) as response, open(local_file, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

                return read_grand(local_file, pseudobulk=pseudobulk, targets=targets, **kwargs)

        path = Path(prefix)

        sparse = is_sparse_file(path)
        if sparse:
            print("Detected sparse format -> using sparse reader")
            return read_sparse(path, pseudobulk=pseudobulk, targets=targets, **kwargs)

        else:
            file_path = resolve_prefix_path(prefix, pseudobulk=pseudobulk, targets=targets)
            print("Detected dense format -> using dense reader")
            return read_dense(str(file_path), **kwargs)

    except ValueError as e:
        print(f"ValueError: {e}")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")


def find_existing_file(path: Path, base_name: str, extensions=(".gz", "", ".tsv", ".tsv.gz")):
    """
    Tries to find an existing file with one of the given extensions.
    Returns the full path if found, else raises FileNotFoundError.
    """
    for ext in extensions:
        candidate = path / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Expected file '{base_name}' with one of {extensions} not found in {path}")


def _read(file_path, sparse, default_slot, design,
          classification_genes, classification_genes_label,
          classify_genes_func=None, pseudobulk=None, targets=None, estimator=None):
    """
    Reads GRAND-SLAM TSV or Matrix Market file and creates a GrandPy object.

    Parameters
    ----------
    file_path : str or Path
        Path to the input file or directory.

    sparse : bool
        Whether to read in sparse mode (Matrix Market).

    default_slot : str
        The default slot to use (e.g., "count").

    design : tuple[str], optional
        Design variable names extracted from sample names.

    classification_genes : list[str], optional
        List of viral gene symbols.

    classification_genes_label : str
        Label for viral genes.

    classify_genes_func : callable, optional
        Gene classification function.

    Returns
    -------
    GrandPy
    """
    path = Path(file_path)

    if sparse:
        base = path

        matrix_path = find_existing_file(base, "matrix.mtx", extensions=(".gz", ""))
        features_path = find_existing_file(base, "features", extensions=(".tsv.gz", ".tsv", ""))
        barcodes_path = find_existing_file(base, "barcodes", extensions=(".tsv.gz", ".tsv", ""))

        with open(matrix_path, "rt") if matrix_path.suffix != ".gz" else gzip.open(matrix_path, "rt") as file:
            count_matrix = mmread(file).tocsr()

        features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
        barcodes = pd.read_csv(barcodes_path, header=None, compression="infer")[0].tolist()

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

        coldata = build_coldata(barcodes, design)

        slots = {}
        slot_sample_names = {}

        for mtx_file in sorted(base.glob("*.mtx")) + sorted(base.glob("*.mtx.gz")):
            filename = mtx_file.name
            basename = filename.replace(".mtx", "").replace(".gz", "")

            if basename == "matrix":
                slot = default_slot
            else:
                slot = basename.split(".")[-1].lower()

            try:
                with open(mtx_file, "rt") if mtx_file.suffix != ".gz" else gzip.open(mtx_file, "rt") as file:
                    mat = mmread(file).tocsr()
                slots[slot] = mat
                slot_sample_names[slot] = barcodes
            except Exception as e:
                warnings.warn(f"Slot {slot}: Datei beschädigt – {filename} wird übersprungen ({e})")
                continue

        slots = pad_slots(slots, sparse=True, coldata=coldata, slot_sample_names=slot_sample_names)

        metadata = {
            "Description": "Loaded via read_grand() (Matrix Market)",
            "default_slot": default_slot,
            "Output": "sparse",
            "Version": 3,
            "pseudobulk": pseudobulk,
            "targets": targets
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
        prefix = Path(file_path).stem

        slot_suffixes = infer_suffixes_from_df(df, estimator=estimator)
        slots, sample_names, slot_sample_names = parse_slots(df, slot_suffixes, sparse, strict=False)
        # strict=False, obwohl default=True ist, denn sonst wird ein Fehler geworfen, dass Duplicates bei dem sample names existieren - dies ist bei sars_R gerade der Fall

        if default_slot not in slots:
            raise ValueError(
                f"Missing required slot(s): ['{default_slot}']. "
                f"Ensure the input file includes columns ending with: {slot_suffixes.get(default_slot, '?')}"
            )

        if "ntr" not in slots:
            warnings.warn("Slot 'ntr' is missing.", UserWarning)

        if classify_genes_func is None:
            if classification_genes:
                custom = {classification_genes_label: lambda g: g["Symbol"].isin(classification_genes)}
            else:
                custom = {}
            classify_genes_func = lambda gene_info: classify_genes(gene_info, custom_classes=custom, use_default=True)

        gene_info = build_gene_info(df, classify_genes_func)
        coldata = build_coldata(sample_names, design)
        slots = pad_slots(slots, sparse, coldata, slot_sample_names)

        metadata = {
            "Description": "Loaded via read_grand() (TSV)",
            "default_slot": default_slot,
            "Output": "sparse" if sparse else "dense",
            "Version": 2,
            "pseudobulk": pseudobulk,
            "targets": targets
        }

        return GrandPy(
            prefix=prefix,
            gene_info=gene_info,
            slots=slots,
            coldata=coldata,
            metadata=metadata
        )


def get_table_qc(grand, slot="count"):
    """
    Returns a QC table per sample with detected genes, totals, statistics and optional percentages per gene type.

    Parameter
    ---------
    grand : GrandPy-object
        grandpy-object

    slot : str
        Slot (e.g. "count", "ntr", ...)

    Returns
    -------
    pd.DataFrame with QC-metrics + coldata-columns
    """

    if not grand._check_slot(slot):
        raise ValueError(f"Slot '{slot}' not found in grand._adata.layers.")

    mat = grand._adata.layers[slot]
    if sp.issparse(mat):
        mat = mat.toarray()

    gene_info = grand.gene_info
    if "Type" not in gene_info.columns:
        raise ValueError("Gene type classification (column 'Type') not found in gene_info.")

    gene_types = gene_info["Type"]
    col_names = grand.coldata["Name"].tolist()

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

    coldata = grand.coldata.reset_index(drop=True)  # "Name" wird zur Spalte, Index wird entfernt, da sonst pandas Probleme mach
    df = df.merge(coldata, on="Name", how="left")

    return df


# grand_obj = read_grand("https://zenodo.org/record/5834034/files/sars.tsv.gz", design=("Condition", "Time", "Replicate"))
# print(grand_obj)
#
# sars = read_grand("data/sars_R.tsv", design=("Condition", "Time", "Replicate"))
# print(sars) # funktioniert
#
# sparse_data = read_grand("test-datasets/test_sparse.targets", design=("Time", "Replicate"))
# print(sparse_data) # funktioniert
#
# grand_sparse = read_grand("test-datasets/test_sc_sparse.targets", design=("Condition", "Time", "Replicate"))
# print(sparse)
#
# sc_dense = read_grand("test-datasets/test_sc_dense.targets", design=("Time", "Replicate"))
# print(sc_dense)

# qc = get_table_qc(grand_obj, slot="count")
# print(qc.head())