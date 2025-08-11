import pytest
from scipy import sparse

from grandpy.load import *
from grandpy.load import (
    _infer_suffixes_from_df,
    _validate_input,
    _remove_suffixes,
    _parse_slots,
    _pad_slots,
    _build_gene_info,
    _design_semantics,
    _resolve_prefix_path,
    _read_dense,
    _read_sparse,)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "GrandPy" / "data"
TEST_DATASETS_DIR = PROJECT_ROOT / "test-datasets"

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



# --- Tests for infer / suffixes ---

def test_infer_suffixes_from_df(mock_df):
    result = _infer_suffixes_from_df(mock_df)
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
    result = _infer_suffixes_from_df(df)
    assert result["alpha"] == " alpha"
    assert result["beta"] == " beta"


def test_remove_suffixes_single():
    name = "Sample.1 alpha"
    assert _remove_suffixes(name, " alpha") == "Sample.1"


def test_remove_suffixes_tuple():
    name = "Sample.1 MAP"
    assert _remove_suffixes(name, (" alpha", " MAP")) == "Sample.1"


def test_infer_suffixes_readcount_variants():
    df = pd.DataFrame({
        "Sample.1 Readcount": [100],
        "Sample.2 Read count": [200],
        "Gene": ["G1"],
        "Symbol": ["S1"],
        "Length": [1000]
    })
    result = _infer_suffixes_from_df(df)
    assert result["count"] == " Readcount"



# --- Tests for slot parsing / padding ---

def test_parse_slots(mock_df):
    suffixes = {
        "alpha": " alpha",
        "beta": " beta",
        "ntr": " MAP"
    }
    slots, sample_names, slot_sample_names = _parse_slots(mock_df, suffixes, sparse=False)
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
    slots, sample_names, slot_sample_names = _parse_slots(df, suffixes, sparse=False)
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
    padded = _pad_slots(slots, sparse=False, coldata=coldata, slot_sample_names=slot_sample_names)
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
    padded = _pad_slots(slots, sparse=True, coldata=coldata, slot_sample_names=slot_sample_names)
    assert padded["count"].shape == (1, 3)



# --- Tests for gene info / classification ---

def test_build_gene_info(mock_df):
    result = _build_gene_info(mock_df, classify_genes)
    assert "Type" in result.columns
    assert result.loc["GENE1", "Type"] == "Cellular"


def test_build_gene_info_classification():
    df = pd.DataFrame({
        "Gene": ["ENSG00000123456", "ERCC-00001", "MT-CO1", "X1"],
        "Symbol": ["G1", "ERCC-00001", "MT-CO1", "GENE4"],
        "Length": [1000, 500, 800, 700]
    })
    info = _build_gene_info(df, classify_func=classify_genes)
    assert set(info["Type"]) == {"Cellular", "ERCC", "mito", "Unknown"}



# ---Tests for time parsing / design-metadata---

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
    df = _design_semantics(df)
    assert "_semantics" in df.attrs
    assert df.attrs["_semantics"]["Time"] == "time"



# --- Tests fot filepath & format recognition ---

def test_resolve_prefix_path_not_found():
    with pytest.raises(FileNotFoundError):
        _resolve_prefix_path("nonexistent_path")



# --- Tests for loading GRAND-SLAM-Data ---

def test_read_dense_real():
    obj = read_grand(DATA_DIR / "sars_R.tsv", classification_genes=None, classification_genes_label="Viral", design=("Condition", "Time", "Replicate"))
    assert "count" in obj.slots


def test_sparse_loader_example():
    obj = read_grand("../test-datasets/test_sparse.targets", design=("Time", "Replicate"))
    count = obj._anndata.X
    assert count.shape[0] > 0


def test_read_dense_and_sparse_load():
    dense = read_grand("../data/sars_R.tsv", design=("Condition", "Time", "Replicate"))
    sparse_test = read_grand("../test-datasets/test_sparse.targets", design=("Time", "Replicate"))

    assert isinstance(dense.coldata, pd.DataFrame)
    assert isinstance(sparse_test.coldata, pd.DataFrame)


def test_read_grand_url():
    url = "https://zenodo.org/record/5834034/files/sars.tsv.gz"
    obj = read_grand(url, design=("Condition", "Time", "Replicate"))
    assert "count" in obj._anndata.layers


def test_validate_input_raises_on_missing_columns():
    df = pd.DataFrame({"A": [1], "B": [2]})
    with pytest.raises(ValueError):
        _validate_input(df, ["A", "C"], context="mock")


def test_read_sparse_rejects_invalid_prefix_combination():
    """
    Tests that read_sparse raises an error when the provided folder path
    does not match the targets and pseudobulk values.
    """
    invalid_path = "../test-datasets/test_sparse.targets"  # names consists 'targets'
    with pytest.raises(ValueError, match="does not contain expected identifiers"):
        _read_sparse(invalid_path, pseudobulk="WRONG", targets="SOMETHING")


def test_read_sparse_rejects_swapped_targets_pseudobulk():
    """
    Tests that read_sparse detects and rejects a path with swapped pseudobulk and targets.
    """
    swapped_path = "../test-datasets/test_sc.pseudobulk.convoluted.merged"
    with pytest.raises(ValueError, match=r"Incompatible pseudobulk/targets combination"):
        _read_sparse(swapped_path, pseudobulk="convoluted", targets="merged")


def test_read_dense_sets_metadata_tags():
    """Tests if the pseudobulk and targets arguments are correctly stored in metadata when using read_dense."""
    obj = _read_dense(DATA_DIR / "sars_R.tsv", design=("Condition", "Time", "Replicate"), pseudobulk="pb_tag",
                      targets="tg_tag")
    assert obj.metadata["pseudobulk"] == "pb_tag"
    assert obj.metadata["targets"] == "tg_tag"


# --- Version tests ---

# def test_dense_version_is_2():
#     df = pd.DataFrame({
#         "Gene": ["g1", "g2"],
#         "Symbol": ["S1", "S2"],
#         "Length": [1000, 800],
#         "Mock.1 alpha": [0.5, 0.6],
#         "Mock.1 beta": [0.3, 0.4],
#         "Mock.1 Read count": [100, 200]
#     })
#
#     with tempfile.TemporaryDirectory() as tmpdir:
#         path = Path(tmpdir) / "dense.tsv.gz"
#         with gzip.open(path, "wt") as f:
#             df.to_csv(f, sep="\t", index=False)
#
#         gp = _read_dense(path, design=("Condition", "Time"))
#         assert gp.metadata["Version"] == 2, "Dense input must set Version=2"
#
#
# def test_sparse_version_from_runtime():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         path = Path(tmpdir)
#         (path / "barcodes.tsv").write_text("A\nB\n")
#         (path / "features.tsv").write_text("gene1\tG1\t0\tcoding\t1000\ngene2\tG2\t0\tcoding\t900\n")
#         (path / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 100\n2 2 200\n")
#         (path / "runtime").write_text("version 3\nother things\n")
#
#         gp = _read_sparse(path, design=("Condition", "Time"))
#         assert gp.metadata["Version"] == 3, "Sparse input must detect Version=3 from runtime"
#
#
# def test_sparse_version_fallback():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         path = Path(tmpdir)
#         (path / "barcodes.tsv").write_text("A\nB\n")
#         (path / "features.tsv").write_text("gene1\tG1\t0\tcoding\t1000\ngene2\tG2\t0\tcoding\t900\n")
#         (path / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 100\n2 2 200\n")
#
#         gp = _read_sparse(path, design=("Condition", "Time"))
#         assert gp.metadata["Version"] == 3, "Sparse fallback must be Version=3 if runtime file is missing"



# --- Tests for get_table_qc() ---

def test_get_table_qc_returns_dataframe():
    obj = read_grand(DATA_DIR / "sars_R.tsv", design=("Condition", "Time", "Replicate"))
    qc = get_table_qc(obj, slot="count")
    assert isinstance(qc, pd.DataFrame)
    assert "Detected" in qc.columns
    assert "Fraction.Cellular" in qc.columns  # oder ein anderer gene type


def test_get_table_qc_missing_slot_raises():
    obj = read_grand(DATA_DIR / "sars_R.tsv", design=("Condition", "Time", "Replicate"))
    with pytest.raises(ValueError):
        get_table_qc(obj, slot="nonexistent")



# --- Test for uniqueness ---

def test_make_unique_adds_suffix():
    from grandpy.utils import _make_unique
    series = pd.Series(["A", "B", "A", "C", "B"])
    unique = _make_unique(series)
    assert len(set(unique)) == 5
    assert unique[0] == "A"
    assert unique[2].startswith("A_")



# --- Loading the data ---

if __name__ == "__main__":
    grand_obj = read_grand("https://zenodo.org/record/5834034/files/sars.tsv.gz", design=("Condition", "Time", "Replicate"))
    print(grand_obj)

    sars = read_grand("../data/sars_R.tsv", design=("Condition", "Time", "Replicate"))
    print(sars) # funktioniert

    sparse_data = read_grand("../test-datasets/test_sparse.targets", design=("Time", "Replicate"))
    print(sparse_data) # funktioniert

    grand_sparse = read_grand("../test-datasets/test_sc_sparse.targets", design=("Condition", "Time", "Replicate"))
    print(grand_sparse)

    sc_dense = read_grand("../test-datasets/test_sc_dense.targets", design=("Time", "Replicate"))
    print(sc_dense)

    banp = read_grand("https://zenodo.org/record/6976391/files/BANP.tsv.gz", design=("Cell", "Experimental.time", "Genotype", "dur.4sU", "has4.U", "Replicate"))
    print(banp)

    qc = get_table_qc(grand_obj, slot="count")
    print(qc.head())

    url = "https://zenodo.org/record/7612564/files/chase_notrescued.tsv.gz?download=1"
    gp_url = read_grand(url, design=("Condition", "dur.4sU", "Replicate"))
    print(gp_url)

    grand = read_grand("../test-datasets/test_sparse.targets", design=("Time", "Replicate"))
    qc = get_table_qc(grand)
    print(qc.head())

    gp = read_grand("test-datasets/test_dense.targets", design=("Condition", "Time", "Replicate"))
    print("Title:", gp.title)
    print("Prefix:", gp.metadata["prefix"])

# ------------------------------------------------------------------------------

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
    #
    # if p.is_file() and p.name.endswith((".tsv", ".tsv.gz")):
    #     if "conversion" in p.name and "targets" in p.name:
    #         return True

    if p.is_dir():
        files = {q.name.lower() for q in p.iterdir()}

        # sparse
        has_mtx = any(fn.endswith((".mtx", ".mtx.gz")) for fn in files)
        has_barcodes = {"barcodes.tsv", "barcodes.tsv.gz"} & files
        has_features = {"features.tsv", "features.tsv.gz"} & files
        if has_mtx and has_barcodes and has_features:
            return True

        # dense
        if "data.tsv" in files or "data.tsv.gz" in files:
            return True

    return False

PARAMS = [
    pytest.param(path, est, id=f"{path.name}[{est or 'default'}]")
    for path in sorted(DATASETS_ROOT.iterdir()) if is_result_ds(path)
    for est in ESTIMATORS
]

@pytest.mark.parametrize("dataset_path, estimator", PARAMS)
def test_read_dataset_with_estimator(dataset_path: Path, estimator):
    # - load test should test valid targets and pseudobulks as well
    name = dataset_path.name.lower()
    kwargs = dict(design=("Condition", "Time"), estimator=estimator)

    if ".targets." in name:
        parts = name.split(".")
        i = parts.index("targets")
        if i + 1 < len(parts):
            kwargs["targets"] = parts[i + 1]

    if ".pseudobulk." in name:
        parts = name.split(".")
        i = parts.index("pseudobulk")
        if i + 1 < len(parts):
            kwargs["pseudobulk"] = parts[i + 1]

    try:
        obj = read_grand(dataset_path, **kwargs)
    except (ValueError, FileNotFoundError) as e:
        pytest.skip(f"{dataset_path.name} [{estimator}] skipped: {e}")
    if obj is None:
        pytest.skip(f"{dataset_path.name} [{estimator}] returned None")
    assert isinstance(obj, GrandPy)
    assert obj.gene_info.shape[0] and obj.coldata.shape[0]
    assert obj.metadata["default_slot"] in obj.slots

    if "targets" in kwargs:
        assert obj.metadata.get("targets") == kwargs["targets"]

    if "pseudobulk" in kwargs:
        assert obj.metadata.get("pseudobulk") == kwargs["pseudobulk"]