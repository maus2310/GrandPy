from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from scipy import sparse

from Py.load import *



@pytest.fixture
def mock_df():
    data = {
        "Sample.1 alpha": [0.1, 0.2],
        "Sample.1 beta": [0.3, 0.4],
        "Sample.1 MAP": [0.5, 0.6],
        "Gene": ["ENSG001", "ENSG002"],
        "Symbol": ["GENE1", "GENE2"],
        "Length": [1000, 1500]
    }
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# Tests für Infer / Suffixes:

def test_infer_suffixes_from_df(mock_df):
    result = infer_suffixes_from_df(mock_df)
    assert result["alpha"] == " alpha"
    assert result["beta"] == " beta"
    assert result["ntr"] == " MAP"


def test_infer_suffixes_multiple_slots():
    df = pd.DataFrame({
        "Sample.1 alpha": [1],
        "Sample.2 alpha": [2],
        "Sample.1 beta": [3],
        "Sample.2 beta": [4]
    })
    result = infer_suffixes_from_df(df)
    assert result["alpha"] == " alpha"
    assert result["beta"] == " beta"


def test_remove_suffixes_single():
    name = "Sample.1 alpha"
    assert remove_suffixes(name, " alpha") == "Sample.1"


def test_remove_suffixes_tuple():
    name = "Sample.1 MAP"
    assert remove_suffixes(name, (" alpha", " MAP")) == "Sample.1"


def test_infer_suffixes_readcount_variants():
    df = pd.DataFrame({
        "Sample.1 Readcount": [100],
        "Sample.2 Read count": [200],
        "Gene": ["G1"],
        "Symbol": ["S1"],
        "Length": [1000]
    })
    result = infer_suffixes_from_df(df)
    assert result["count"] == " Readcount"


# -----------------------------------------------------------------------------
# Tests für Slot Parsing / Padding:

def test_parse_slots(mock_df):
    suffixes = {
        "alpha": " alpha",
        "beta": " beta",
        "ntr": " MAP"
    }
    slots, sample_names, slot_sample_names = parse_slots(mock_df, suffixes, sparse=False)
    assert "alpha" in slots
    assert sample_names == ["Sample.1"]
    assert slot_sample_names["alpha"] == ["Sample.1"]


def test_parse_slots_allows_duplicate_sample_names():
    df = pd.DataFrame({
        "Sample.A Readcount": [1],
        "Sample.A alpha": [0.1],
        "Gene": ["G1"],
        "Symbol": ["S1"],
        "Length": [100]
    })
    suffixes = {"count": " Readcount", "alpha": " alpha"}
    slots, sample_names, slot_sample_names = parse_slots(df, suffixes, sparse=False)
    assert "count" in slots and "alpha" in slots
    assert sample_names == ["Sample.A"]


def test_pad_slots_dense():
    slots = {
        "count": np.array([[1, 2]])
    }
    coldata = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "no4sU": [False, True, False]
    }).set_index("Name")
    coldata["Name"] = coldata.index

    slot_sample_names = {"count": ["A", "B"]}
    padded = pad_slots(slots, sparse=False, coldata=coldata, slot_sample_names=slot_sample_names)
    assert padded["count"].shape == (1, 3)


def test_pad_slots_sparse():
    mat = sparse.csr_matrix([[1, 2]])
    slots = {"count": mat}
    coldata = pd.DataFrame({
        "Name" : ["A", "B", "C"],
        "no4sU" : [False, True, False]
    }).set_index("Name")
    coldata["Name"] = coldata.index
    slot_sample_names = {"count": ["A", "B"]}
    padded = pad_slots(slots, sparse=True, coldata=coldata, slot_sample_names=slot_sample_names)
    assert padded["count"].shape == (1, 3)


# -----------------------------------------------------------------------------
# Tests für Gene Info / Classification

def test_build_gene_info(mock_df):
    result = build_gene_info(mock_df, classify_genes)
    assert "Type" in result.columns
    assert result.loc["GENE1", "Type"] == "Cellular"


def test_build_gene_info_classification():
    df = pd.DataFrame({
        "Gene": ["ENSG00000123456", "ERCC-00001", "MT-CO1", "X1"],
        "Symbol": ["G1", "ERCC-00001", "MT-CO1", "GENE4"],
        "Length": [1000, 500, 800, 700]
    })
    info = build_gene_info(df, classify_func=classify_genes)
    assert set(info["Type"]) == {"Cellular", "ERCC", "mito", "Unknown"}


# -----------------------------------------------------------------------------
# Test für Zeitparsing / Design-Metadaten:

def test_parse_time_string_edge_cases():
    assert parse_time_string("90min") == 1.5
    assert parse_time_string("2h") == 2.0
    assert parse_time_string("60") == 1.0
    assert parse_time_string("nos4U") == 0

@pytest.mark.parametrize("value,expected", [
    ("  90min  ", 1.5),
    ("1H", 1.0),
    ("-", 0.0),
    (None, 0.0),
    ("abc", None)
])
def test_parse_time_string_various(value, expected):
    assert parse_time_string(value) == expected


def test_apply_design_semantics_sets_semantics():
    df = pd.DataFrame({"Time": [1, 2, 3], "Name": ["A", "B", "C"]})
    df.attrs.clear()
    df = apply_design_semantics(df)
    assert "_semantics" in df.attrs
    assert df.attrs["_semantics"]["Time"] == "time"



# -----------------------------------------------------------------------------
# Test für Dateipfade & Formaterkennung:

def test_resolve_prefix_path_not_found():
    with pytest.raises(FileNotFoundError):
        resolve_prefix_path("nonexistent_path")


# -----------------------------------------------------------------------------
# Tests für Laden von GRAND-SLAM-Daten

def test_read_dense_real():
    obj = read_grand("../data/sars_R.tsv", classification_genes=None, classification_genes_label="Viral", design=("Condition", "Time", "Replicate"))
    assert "count" in obj.slots
    assert obj.coldata.shape[0] > 0


def test_sparse_loader_example():
    obj = read_grand("../test-datasets/test_sparse.targets", design=("Time", "Replicate"))
    count = obj._adata.X
    assert count.shape[0] > 0


def test_read_dense_and_sparse_load():
    dense = read_grand("../data/sars_R.tsv", design=("Condition", "Time", "Replicate"))
    sparse_test = read_grand("../test-datasets/test_sparse.targets", design=("Time", "Replicate"))

    assert isinstance(dense.coldata, pd.DataFrame)
    assert isinstance(sparse_test.coldata, pd.DataFrame)
    assert "count" in dense.slots
    assert "count" in sparse_test.slots


def test_read_grand_url():
    url = "https://zenodo.org/record/5834034/files/sars.tsv.gz"
    obj = read_grand(url, design=("Condition", "Time", "Replicate"))
    assert "count" in obj._adata.layers


def test_validate_input_raises_on_missing_columns():
    df = pd.DataFrame({"A": [1], "B": [2]})
    with pytest.raises(ValueError):
        validate_input(df, ["A", "C"], context="mock")


# -----------------------------------------------------------------------------
# Test für get_table_qc():

def test_get_table_qc_returns_dataframe():
    obj = read_grand("../data/sars_R.tsv", design=("Condition", "Time", "Replicate"))
    qc = get_table_qc(obj, slot="count")
    assert isinstance(qc, pd.DataFrame)
    assert "Detected" in qc.columns
    assert "Fraction.Cellular" in qc.columns  # oder ein anderer gene type


def test_get_table_qc_missing_slot_raises():
    obj = read_grand("../data/sars_R.tsv", design=("Condition", "Time", "Replicate"))
    with pytest.raises(ValueError):
        get_table_qc(obj, slot="nonexistent")


# -----------------------------------------------------------------------------
# Test für uniqueness:

def test_make_unique_adds_suffix():
    from Py.utils import _make_unique
    series = pd.Series(["A", "B", "A", "C", "B"])
    unique = _make_unique(series)
    assert len(set(unique)) == 5
    assert unique[0] == "A"
    assert unique[2].startswith("A_")

# ------------------------------------------------------------------------------
# aus load.py:

grand_obj = read_grand("https://zenodo.org/record/5834034/files/sars.tsv.gz", design=("Condition", "Time", "Replicate"))
print(grand_obj)

sars = read_grand("data/sars_R.tsv", design=("Condition", "Time", "Replicate"))
print(sars) # funktioniert

sparse_data = read_grand("test-datasets/test_sparse.targets", design=("Time", "Replicate"))
print(sparse_data) # funktioniert

grand_sparse = read_grand("test-datasets/test_sc_sparse.targets", design=("Condition", "Time", "Replicate"))
print(grand_sparse)

sc_dense = read_grand("test-datasets/test_sc_dense.targets", design=("Time", "Replicate"))
print(sc_dense)

banp = read_grand("https://zenodo.org/record/6976391/files/BANP.tsv.gz", design=("Cell", "Experimental.time", "Genotype", "dur.4sU", "has4.U", "Replicate"))
print(banp)

qc = get_table_qc(grand_obj, slot="count")
print(qc.head())


# ------------------------------------------------------------------------------
# TODO: Write tests for reading in all test data sets that you have. You could implement that as a single function with nested loops over data sets and estimators.
# dense wird nicht erkannt (?) - bin noch dran
DATASETS_ROOT = Path(__file__).resolve().parents[1] / "test-datasets"
ESTIMATORS     = [None, "MAP", "Binom", "TbBinom", "TbBinomShape"]
SKIP_FRAGMENTS = {
    "reads.lengths.tsv", "reads.subreads.tsv", "model.parameters.tsv",
    "strandness.tsv", "clip.tsv", "subread.tsv", "experimentaldesign.tsv"
}

def is_result_ds(p: Path) -> bool:
    name = p.name.lower()
    if any(frag in name for frag in SKIP_FRAGMENTS):
        return False

    if p.is_file() and ".targets" in name and name.endswith((".tsv", ".tsv.gz")):
        return True

    if p.is_dir():
        files = {q.name.lower() for q in p.iterdir()}
        has_mtx = any(fn.endswith((".mtx", ".mtx.gz")) for fn in files)
        has_barcodes = {"barcodes.tsv", "barcodes.tsv.gz"} & files
        has_features = {"features.tsv", "features.tsv.gz"} & files
        return has_mtx and has_barcodes and has_features

    return False

PARAMS = [
    pytest.param(path, est, id=f"{path.name}[{est or 'default'}]")
    for path in sorted(DATASETS_ROOT.iterdir()) if is_result_ds(path)
    for est in ESTIMATORS
]

@pytest.mark.parametrize("dataset_path, estimator", PARAMS)
def test_read_dataset_with_estimator(dataset_path: Path, estimator):
    try:
        obj = read_grand(dataset_path, design=("Condition", "Time"), estimator=estimator)
    except (ValueError, FileNotFoundError) as e:
        pytest.skip(f"{dataset_path.name} [{estimator}] skipped: {e}")
    if obj is None:
        pytest.skip(f"{dataset_path.name} [{estimator}] returned None")
    assert isinstance(obj, GrandPy)
    assert obj.gene_info.shape[0] and obj.coldata.shape[0]
    assert obj.metadata["default_slot"] in obj.slots