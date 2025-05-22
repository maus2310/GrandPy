import pandas as pd
import numpy as np
import warnings
import scipy.sparse as sp
from Py.grandPy import GrandPy, _to_sparse, Any


# Behavior mirrors grandR::read.grand():
# - Critical annotation columns must exist (Gene, Symbol, Length)
# - 'count' slot must exist (Readcount columns)
# - 'ntr' (MAP) is optional, only a warning


# implemented some help-/subfunctions -> read_grand() becomes more readable
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

    for slot, suffix in suffixes.items():
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            continue
        mat = df[cols].to_numpy()
        mat = np.where(np.isnan(mat), 0, mat)
        if sparse:
            mat = _to_sparse(mat)
        slots[slot] = mat

        if sample_names is None:
            sample_names = [c.replace(suffix, "") for c in cols]
    return slots, sample_names


def build_gene_info(df, classify_func):
    """Extracts gene metadata and assigns a gene type to each entry.

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
    if design is not None:
        split_names = [name.split(".") for name in sample_names]
        if not all(len(s) == len(design) for s in split_names):
            warnings.warn("Design vector does not match the structure of sample names.")
            coldata = pd.DataFrame(index=sample_index)
            coldata["Condition"] = ["sample"] * len(sample_index)
        else:
            coldata = pd.DataFrame(split_names, columns=design, index=sample_index)
    else:
        coldata = pd.DataFrame(index=sample_index)
        coldata["Condition"] = ["sample"] * len(sample_index)

    coldata["Name"] = coldata.index
    coldata = coldata[["Name"] + [c for c in coldata.columns if c != "Name"]]

    if "Time" in coldata.columns:
        coldata["no4sU"] = coldata["Time"].isin(["no4sU", "nos4U", "-"])

    return coldata


def pad_slots(slots, sparse):
    """
    Ensures that all slot matrices have the same number of columns.

    Parameters
    ----------
    slots : dict[str, np.ndarray or sp.csr_matrix]
        Dictionary of slot matrices (e.g. count, ntr).

    sparse : bool
        Whether to use sparse padding matrices.

    Returns
    -------
    dict[str, np.ndarray or sp.csr_matrix]
        Dictionary with padded matrices, aligned by number of samples.

    """
    max_samples = max(mat.shape[1] for mat in slots.values())
    for key, mat in slots.items():
        if mat.shape[1] < max_samples:
            missing = max_samples - mat.shape[1]
            pad = np.zeros((mat.shape[0], missing))
            if sparse:
                pad = sp.csr_matrix(pad)
                slots[key] = sp.hstack([mat, pad])
            else:
                slots[key] = np.hstack([mat, pad])

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


def read_grand(file_path,
               default_slot="count",
               sparse=False,
               design=None, *,
               viral_genes=None,
               viral_genes_label="Viral",
               use_default_classification=True,
               classify_genes_func=None) -> GrandPy:
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

    slot_suffixes = {
        "count": " Readcount",
        "ntr": " MAP",
        "alpha": "alpha",
        "beta": "beta"
    }

    slots, sample_names = parse_slots(df, slot_suffixes, sparse)

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
        if viral_genes is not None:
            custom = {viral_genes: lambda g: g["Symbol"].isin(viral_genes)}
        else:
            custom = None

        classify_genes_func = lambda gene_info: classify_genes(
            gene_info,
            custom_classes=custom,
            use_default=True
            )
    # else:
    #     classify_genes_func = default_classify_genes

    gene_info = build_gene_info(df, classify_genes_func)
    coldata = build_coldata(sample_names, design)
    slots = pad_slots(slots, sparse)

    # Metadata muss noch angepasst werden, die Version fehlt
    metadata = {
        "Description": "Loaded via read_grand()",
        "default_slot": default_slot,
        "Output": "sparse" if sparse else "dense",
        "Version": 2 # Hardcoded
    }

    return GrandPy(
        prefix=file_path,
        gene_info=gene_info,
        slots=slots,
        coldata=coldata,
        metadata=metadata
    )

# wird bald gelöscht, wenn wir mit der Test-Einheit vorangekommen sind!
# gp_dense = read_grand("data/sars.tsv", design =("Condition", "Time", "Replicate"))
# print(gp_dense.coldata)

# print(gp_dense._adata.uns["prefix"])    # Output: "data/sars.tsv"
# print(gp_dense._adata.n_obs)            # Output: No. of samples "12"
# print(gp_dense._adata.n_vars)           # Output: No. of genes "19659"

# test_data = {
#     "Gene": ["ENSG000001", "ENSG000002", "ENSG000003", "ENSG000004"],
#     "Symbol": ["GAPDH", "ACTB", "ERCC-00001", "MT-CO1"],
#     "Length": [1000, 1200, 500, 800],
#     "Sample1 Readcount": [100, 0, 0, 0],
#     "Sample2 Readcount": [0, 200, 0, 0],
#     "Sample1 MAP": [0.9, 0.0, 0.0, 0.0],
#     "Sample2 MAP": [0.0, 0.95, 0.0, 0.0],
#     "Sample1 alpha": [10, 0, 0, 0],
#     "Sample2 alpha": [0, 20, 0, 0],
#     "Sample1 beta": [1, 1, 0, 0],
#     "Sample2 beta": [0, 2, 0, 0],
# }
#
# df = pd.DataFrame(test_data)
# df.to_csv("sars_sparse_test.tsv", sep="\t", index=False)
# gp_sparse = read_grand("sars_sparse_test.tsv", sparse=True)
# print(gp_sparse)

# Beispiel aufruf für die Funktion
# sars = read_grand(
#     "data/sars_R.tsv",
#     viral_genes=["ORF3a", "E", "M", "N"],
#     viral_genes_label="SARS-CoV-2",
#     use_default_classification=False
# )
#
# print(sars)