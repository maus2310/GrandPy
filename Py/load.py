import pandas as pd
import numpy as np
import warnings
import scipy.sparse as sp
from Py.grandPy import GrandPy, _to_sparse, Any


def default_classify_genes(df) -> any: # Hilfsfunktion, um Gene zu klassifizieren
    """
    Classify genes into types based on patterns in "Gene" or "Symbol".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing gene annotations with at least the columns "Gene" (e.g. ID) and "Sybol" (e.g. gene symbol).

    Returns
    -------
        pd.Series
        A pandas Series containing the gene type for each row in `df`,
        indexed identically to the input.
    """

    types = pd.Series("Unknown", index=df.index) # Initialisierung, unten: Einteilung
    types[df["Gene"].str.contains("ERCC-")] = "ERCC"
    types[df["Gene"].str.match(r"^ENS.*G\d+$")] = "Cellular" # Regex-Muster: Start mit ENS, irgendwo ein G und dann eine Folge von Zahlen
    types[df["Symbol"].str.startswith("MT-")] = "mito"
    return types

# Beispiel aufruf für die Funktion
# sars = read_grand(
#     "data/sars_R.tsv",
#     viral_genes=["ORF3a", "E", "M", "N"],
#     viral_genes_label="SARS-CoV-2",
#     use_default_classification=False
# )
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
               viral_genes_label="Default",
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

    slots = {}
    sample_names = None

    for slot, suffix in slot_suffixes.items():
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            continue
        mat = df[cols].to_numpy().T
        mat = np.where(np.isnan(mat), 0, mat)
        if sparse:
            mat = _to_sparse(mat)
        slots[slot] = mat

        if sample_names is None:
            sample_names = [c.replace(suffix, "") for c in cols]

    sample_index = pd.Index(sample_names, name="Name")

    if classify_genes_func is None:
        if viral_genes is not None:
            classify_genes_func = lambda gene_info: classify_genes(
                gene_info,
                custom_classes={viral_genes_label: lambda g: g["Symbol"].isin(viral_genes)},
                use_default=use_default_classification
            )
        else:
            classify_genes_func = default_classify_genes

    gene_info = df[["Gene", "Symbol", "Length"]].copy()
    gene_info["Type"] = classify_genes_func(gene_info)
    gene_info = gene_info[["Symbol", "Gene", "Length", "Type"]]

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

    if "Time" in coldata.columns:
        coldata["no4sU"] = coldata["Time"].isin(["no4sU", "nos4U", "-"])
    else:
        coldata["no4sU"] = False

    # Metadata muss noch angepasst werden, die Version fehlt
    metadata = {
    "Description": "Loaded via read_grand()",
    "default_slot": default_slot,
    "Output": "sparse" if sparse else "dense"
    }

    # correct_matrix() Ersatz, damit werden die Slots korrigiert, falls einzelne Slots weniger Samples enthalten -> selbe Zeilenanzahl durch 0-Ergänzung
    max_samples = max(mat.shape[0] for mat in slots.values())
    for key, mat in slots.items():
        if mat.shape[0] < max_samples:
            missing = max_samples - mat.shape[0]
            pad = np.zeros((missing, mat.shape[1]))
            if sparse:
                pad = sp.csr_matrix(pad)
                slots[key] = sp.vstack([mat, pad])
            else:
                slots[key] = np.vstack([mat, pad])

    return GrandPy(
        prefix=file_path,
        gene_info=gene_info,
        slots=slots,
        coldata=coldata,
        metadata=metadata
    )

# gp_dense = read_grand("data/sars.tsv", design=("Condition", "duration", "Replicate"), default_slot="count", sparse=False)
# print(gp_dense)                          # Output: GrandPy-Object

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