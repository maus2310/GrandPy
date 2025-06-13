import pytest
import pandas as pd
import numpy as np
from scipy import sparse
from Py.load import *
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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


def test_infer_suffixes_from_df(mock_df):
    result = infer_suffixes_from_df(mock_df)
    assert result["alpha"] == " alpha"
    assert result["beta"] == " beta"
    assert result["ntr"] == " MAP"


def test_remove_suffixes_single():
    name = "Sample.1 alpha"
    assert remove_suffixes(name, " alpha") == "Sample.1"


def test_remove_suffixes_tuple():
    name = "Sample.1 MAP"
    assert remove_suffixes(name, (" alpha", " MAP")) == "Sample.1"


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


def test_build_gene_info(mock_df):
    result = build_gene_info(mock_df, classify_genes)
    assert "Type" in result.columns
    assert result.loc["GENE1", "Type"] == "Cellular"


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


def test_read_dense_real():
    obj = read_grand("../data/sars_R.tsv", classification_genes=None, classification_genes_label="Viral", design=("Condition", "Time", "Replicate"))
    assert "count" in obj.slots
    assert obj.coldata.shape[0] > 0


def test_sparse_loader_example():
    obj = read_grand("../test-datasets/test_sparse.targets", design=("Time", "Replicate"))
    count = obj._adata.X
    assert count.shape[0] > 0
