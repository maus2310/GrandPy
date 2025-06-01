import pandas as pd
import numpy as np
import warnings
import scipy.sparse as sp
from Py.grandPy import GrandPy, _to_sparse, Any, ModeSlot, _make_unique
from pathlib import Path


# hier muss noch einiges gemacht werden -> absolute Rohversion, sparse-Tests stehen noch aus, ich bin noch beim design dran

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
        String without the suffix if matched; otherwise, original string.
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


def parse_slots(df, suffixes, sparse):
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
    tuple[dict[str, np.ndarray or sp.csr_matrix], list[str]]
        Dictionary of data slots and list of sample names (inferred from column names).
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
    gene_info["Symbol"] = gene_info["Symbol"]
    gene_info.index = _make_unique(gene_info["Symbol"])
    return gene_info[["Symbol", "Gene", "Length", "Type"]]


def build_coldata(sample_names, design):
    """
    Builds sample metadata (coldata) from sample names and an optional experimental design.

    Parameters
    ----------
    sample_names : list[str]
        List of sample identifiers extracted from column names.

    design : tuples[str] or None
        Tuple of design variables to extract from sample names via splitting on '.'.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sample, containing design info and a 'no4sU' flag (if applicable).
    """

    sample_index = pd.Index(sample_names, name="Name")

    split_names = [name.split(".") for name in sample_names]
    max_len = max(len(s) for s in split_names)

    if design is None:
        design = tuple(f"Design_{i+1}" for i in range(max_len))

    elif len(design) != max_len:
        design = design[:max_len] if len(design) > max_len else design + tuple(f"Design_{i+1}" for i in range(len(design), max_len))
    aligned_splits = [s + [None] * (max_len- len(s)) for s in split_names]

    if len(design) > max_len:
        design = design[:max_len]

    coldata = pd.DataFrame(aligned_splits, columns = design, index=sample_index)
    coldata["Name"] = coldata.index
    coldata = coldata[["Name"] + [c for c in coldata.columns if c != "Name"]]

    if "Time" in coldata.columns:
        coldata["no4sU"] = coldata["Time"].isin(["no4sU", "nos4U", "-"])
    else:
        coldata["no4sU"] = False

    return coldata


def pad_slots(slots, sparse, coldata, slot_sample_names) -> dict:
    """
    Pads all slot matrices to have the same columns (samples), based on coldata["Name"].
    If a sample is missing in a slot:
    - If no4sU == True: fill with 0 (sparse) or NaN (dense)
    - else: warn (as in grandR)
    Ensures slot columns align exactly with coldata["Name"] order.

    Parameters
    ----------
    slots : dict[str] -> np.ndarray or sparse matrix
        Dictionary of data matrices, e.g. count, ntr, alpha, beta

    sparse : bool
        If True, output matrices are sparse, otherwise they are dense.

    coldata : pd.DataFrame
        Sample metadata; must include "Name" and optionally "no4sU".

    slot_sample_names : dict[str, list[str]]
        Slot-specific sample names, parsed from column names (e.g. from 'Mock.1h.A alpha").

    Returns
    -------
    dict[str, np.ndarray or sparse matrix]
        Updated slots with padded sample columns
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
                warn_key = (slot_name, sample)
                # Sample fehlt im Slot - hier muss dann 'gepadded' werden
                if "no4sU" in coldata.columns and coldata.loc[sample, "no4sU"]:
                    # Falls no4sU == True -> auffüllen mit 0 oder NaN (abhängig von Matrix-Art)
                    col = np.zeros(n_genes) if sparse else np.full(n_genes, np.nan)
                else:
                    # Sample fehlt, ist aber 4sU-behandelt -> Warnung wird ausgegeben
                    # das ist anders als in grandR - da wird ein Fehler ausgeworfen
                    if warn_key not in warned_once:
                        warnings.warn(f"Sample '{sample}' missing in slot '{slot_name}' but not marked as no4sU.",
                                  stacklevel=2)
                    col = np.zeros(n_genes) if sparse else np.full(n_genes, np.nan)
            # Hinzufügen der Spalte
            new_matrix.append(col)

        # Neue Matrix wird zusammengesetzt aus Gene x Sample
        stacked = np.stack(new_matrix, axis=1)

        # Optional als Sparse-Matrix zurückgeben
        if sparse:
            stacked = _to_sparse(stacked)

        # Slot ersetzen durch gepaddete Matrix
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
        message = (f"Missing required column(s) in {context or 'input'}: {missing}")
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


def read_dense(file_path, default_slot="count", design=None, *, viral_genes=None, viral_genes_label="Viral", classify_genes_func=None):
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
    viral_genes : list[str], optional
        List of gene symbols considered viral, to be assigned a custom type.
    viral_genes_label : str, default="Viral"
        Type label to assign to the genes listed in `viral_genes`.
    classify_genes_func : callable, optional
        Custom function to classify gene types. Overrides `viral_genes`.

    Returns
    -------
    GrandPy
        A GrandPy object populated with dense matrices and metadata.
    """

    return _read(file_path, sparse=False, default_slot=default_slot, design=design,
                 viral_genes=viral_genes, viral_genes_label=viral_genes_label,
                 classify_genes_func=classify_genes_func)


def read_sparse(file_path, default_slot="count", design=None, *, viral_genes=None, viral_genes_label="Viral", classify_genes_func=None):
    """
    Reads a GRAND-SLAM TSV file and returns a GrandPy object with sparse matrices.

    Parameters
    ----------
    file_path : str
        Path to the input .tsv file.
    default_slot : str, default="count"
        The slot to use as the default slot for plotting and analysis.
    design : tuple[str], optional
        Tuple of design variables to extract from sample names.
    viral_genes : list[str], optional
        List of gene symbols considered viral.
    viral_genes_label : str, default="Viral"
        Label to assign to those viral genes in the gene type column.
    classify_genes_func : callable, optional
        Function to classify genes into categories. If None, a default classifier is used.

    Returns
    -------
    GrandPy
        A GrandPy object populated with sparse matrices and gene/sample metadata.
    """

    return _read(file_path, sparse=True, default_slot=default_slot, design=design,
                 viral_genes=viral_genes, viral_genes_label=viral_genes_label,
                 classify_genes_func=classify_genes_func)


def read_grand_auto(prefix: str, **kwargs):
    """
    Automatically detects and reads GRAND-SLAM data from a directory.

    Parameters
    ----------
    prefix : str
        Path to the directory containing GRAND-SLAM result files.

    **kwargs :
        Additional keyword arguments passed to `read_dense` or `read_sparse`.

    Returns
    -------
    GrandPy
        A GrandPy object with loaded data in dense or sparse format.

    """

    prefix_path = Path(prefix)

    if not prefix_path.exists():
        raise FileNotFoundError(f"{prefix} does not exist")

    dense_file = prefix_path / "data.tsv.gz"
    if dense_file.exists():
        return read_dense(str(dense_file), **kwargs)
    else:
        return read_sparse(prefix, **kwargs)


def _read(file_path, sparse, default_slot, design, viral_genes, viral_genes_label, classify_genes_func=None) -> GrandPy:
    """
    Reads GRAND-SLAM TSV-File and creates a GrandPy-Object.

    Parameters
        ----------
    file_path : str
        Path to GRAND-SLAM TSV-File:
    default_slot : str
        The slot to be used as the default ("count", "ntr", etc.)
    sparse : bool
        If True, sparse matrices are being used.
    design : tuple of str, optional
        Column names to extract from sample names by splitting on "." (e.g., ("Condition", "Time", "Replicate")).
        Used to construct the coldata DataFrame.
    Returns
    -------
        GrandPy-Object.
    """

    df = pd.read_csv(file_path, sep="\t")
    prefix = Path(file_path).stem

    slot_suffixes = {
        "count": (" Readcount", " Read count"), # work in progress
        "ntr": (" MAP", " NTR MAP", " Binom NTR MAP", " TbBinom NTR MAP", " TbBinomShape NTR MAP"),
        "alpha": (" alpha", " Binom alpha", " TbBinom alpha", " TbBinomShape alpha"),
        "beta": (" beta", " Binom beta", " TbBinom beta", " TbBinomShape beta")
    }

    slots, sample_names, slot_sample_names = parse_slots(df, slot_suffixes, sparse)

    # Check if the default_slot exists
    if default_slot not in slots:
        raise ValueError(
            f"Missing required slot(s): ['{default_slot}']. "
            f"Ensure the input file includes columns ending with: {slot_suffixes[default_slot]}"
        )

    # Optional: Warn if "ntr" is missing - commonly expected in GRAND-SLAM output
    if "ntr" not in slots:
        warnings.warn("Slot 'ntr' is missing.", UserWarning)


    if classify_genes_func is None:
        custom = {viral_genes_label: lambda g: g["Symbol"].isin(viral_genes) if viral_genes else None}
        classify_genes_func = lambda gene_info: classify_genes(gene_info, custom_classes=custom, use_default=True)

    gene_info = build_gene_info(df, classify_genes_func)
    coldata = build_coldata(sample_names, design)
    slots = pad_slots(slots, sparse, coldata, slot_sample_names)

    # Metadata muss noch angepasst werden, die Version fehlt
    metadata = {
        "Description": "Loaded via read_grand()",
        "default_slot": default_slot,
        "Output": "sparse" if sparse else "dense",
        "Version": 2 # Hardcoded
    }

    return GrandPy(
        prefix=prefix,
        gene_info=gene_info,
        slots=slots,
        coldata=coldata,
        metadata=metadata
    )


# sars = _read("data/sars_R.tsv", False, "count", ("Condition", "Time", "Replicate"), None, None)
# print(sars.slot_data["ntr"]) # funktioniert

# sparse = _read("test-data/test_sparse.pseudobulk.all.tsv/test_sparse.pseudobulk.all.tsv", True, "count", ("Condition", "Time", "Replicate"), None, None) # fehler

# gp = _read("test-datasets/test_dense.targets/data.tsv/data.tsv", False, "count", ("Time", "Replicate"), None, None)
# print(gp.slot_data["ntr"]) # wird nicht richtig gefüllt ... bin noch dran