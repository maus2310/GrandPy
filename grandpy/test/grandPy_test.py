from io import StringIO
import numpy as np
import pandas as pd
import pytest

from grandpy import read_grand, ModeSlot
from grandpy.utils import _make_unique

@pytest.fixture(scope="module")
def sars():
    return read_grand("../data/sars_R.tsv", design=("Condition", "dur.4sU", "Replicate"))


def test_default_slot_is_count(sars):
    assert sars.default_slot == "count"

def test_with_default_slot_sets_value_and_is_immutable(sars):
    updated = sars.with_default_slot("alpha")
    assert updated.default_slot == "alpha"
    assert updated.default_slot !=sars.default_slot

def test_slot_names_are_known(sars):
    known_slots = {"count", "ntr", "alpha", "beta"}
    assert known_slots.union(sars.slots) == known_slots

def test_with_dropped_slots_removes_named_slot(sars):
    dropped = sars.with_dropped_slots("ntr")
    expected_slots = ['count', 'alpha', 'beta']
    assert sorted(dropped.slots) == sorted(expected_slots)

def test_with_dropped_slots_warns_on_missing(sars):
    dropped = sars.with_dropped_slots("not_a_slot")
    assert "not_a_slot" not in dropped.slots

def test_with_condition_sets_correct_values(sars):
    updated_list = sars.with_condition(["MOGGED"] * 12)
    updated_dict = sars.with_condition({"Mock.1h.A": "Test1", "Mock.3h.A": "Test2"})

    assert updated_dict.coldata["Condition"]["Mock.1h.A"] == "Test1"
    assert updated_dict.coldata["Condition"]["Mock.3h.A"] == "Test2"
    assert all(condition == "MOGGED" for condition in updated_list.condition)

def test_with_condition_raises_on_invalid_length(sars):
    with pytest.raises(ValueError):
        sars.with_condition(["a", "b"])

def test_with_condition_raises_on_invalid_mapping(sars):
    with pytest.raises(ValueError):
        sars.with_condition({"UnknownSample": "oops"})

def test_with_renamed_columns_is_immutable(sars):
    updated = sars.with_renamed_columns({"Mock.no4sU.A": "Test1", "SARS.no4sU.A": "Test2"})
    assert sars.coldata.index[0] != "Test1"
    assert sars.coldata.index[6] != "Test2"

def test_with_renamed_columns_changes_index(sars):
    updated = sars.with_renamed_columns({"Mock.no4sU.A": "Test1", "SARS.no4sU.A": "Test2"})
    assert updated.coldata.index[0] == "Test1"
    assert updated.coldata.index[6] == "Test2"

def test_with_swapped_columns_reverses_values(sars):
    swapped = sars.with_swapped_columns("Mock.1h.A", "Mock.2h.A")

    assert all(swapped.get_matrix(columns="Mock.1h.A") == sars.get_matrix(columns="Mock.2h.A"))
    assert all(swapped.get_matrix(columns="Mock.2h.A") == sars.get_matrix(columns="Mock.1h.A"))

def test_get_table_extracts_correct_values(sars):
    expected = {
        "Mock.no4sU.A": [0.0, 0.0, 0.0],
        "Mock.1h.A": [201.79815, 118.64680, 531.08850],
        "Mock.2h.A": [434.818800, 81.973500, 1197.084926]
    }
    result = sars.get_table(ModeSlot("new", "count"), [0, 1, 2], [0, 1, 2])

    for col in expected:
        for row in range(3):
            assert round(result[col][row], 3) == round(expected[col][row], 3)

def test_merge_coldata_merges_rows_correctly(sars):
    egp = sars[0:10]
    zgp = sars[0:10]

    data = """Name,Condition,duration.4sU,Replicate,no4sU
    Mock.no4sU.A,Mock,0,A,True
    Mock.1h.A,Mock,1,A,False
    Mock.2h.A,Mock,2,A,False
    Mock.2h.B,Mock,2,B,False
    Mock.3h.A,Mock,3,A,False
    Mock.4h.A,Mock,4,A,False
    SARS.no4sU.A,SARS,0,A,True
    SARS.1h.A,SARS,1,A,False
    SARS.2h.A,SARS,2,A,False
    SARS.2h.B,SARS,2,B,False
    SARS.3h.A,SARS,3,A,False
    SARS.4h.A,SARS,4,A,False
    Mock.no4sU.A,Mock,0,A,True
    Mock.1h.A,Mock,1,A,False
    Mock.2h.A,Mock,2,A,False
    Mock.2h.B,Mock,2,B,False
    Mock.3h.A,Mock,3,A,False
    Mock.4h.A,Mock,4,A,False
    SARS.no4sU.A,SARS,0,A,True
    SARS.1h.A,SARS,1,A,False
    SARS.2h.A,SARS,2,A,False
    SARS.2h.B,SARS,2,B,False
    SARS.3h.A,SARS,3,A,False
    SARS.4h.A,SARS,4,A,False
    """
    reference = pd.read_csv(StringIO(data))
    reference.index = _make_unique(pd.Series(reference["Name"]))
    merged = egp.merge(zgp, axis=0).coldata

    # reference.index has unexpected whitespaces which makes the comparison impossible
    reference.index = reference.index.str.strip()
    merged.index = merged.index.str.strip()

    for col in ["Condition", "duration.4sU", "Replicate", "no4sU"]:
        assert all(merged[col] == reference[col])

def test_with_gene_info_list_is_immutable(sars):
    sars = sars[0:10]
    updated = sars.with_gene_info(name="Gene", value=list(range(10)))
    assert not np.array_equal(updated.gene_info["Gene"], sars.gene_info.get("Gene", []))

def test_with_gene_info_dict_sets_values(sars):
    updated = sars.with_gene_info(name="Gene", value={"UHMK1": "Control", "ATF3": "Treatment"})
    assert updated.gene_info["Gene"][0] == "Control"
    assert updated.gene_info["Gene"][1] == "Treatment"

def test_with_gene_info_series_sets_values(sars):
    sars = sars[0:10]
    series = pd.Series(range(10), index=sars.gene_info.index[:10])
    updated = sars.with_gene_info(name="Gene", value=series)
    assert all(updated.gene_info["Gene"].values == series.values)

def test_with_gene_info_fails_on_length_mismatch(sars):
    sars = sars[0:5]
    values = list(range(10))  # too long
    with pytest.raises(ValueError):
        sars.with_gene_info(name="X", value=values)

def test_with_coldata_is_immutable(sars):
    updated = sars.with_coldata(name="new_condition", value=list(range(12)))
    assert "new_condition" not in sars.coldata.columns
    assert "new_condition" in updated.coldata.columns

def test_with_coldata_fails_on_length_mismatch(sars):
    sars = sars[0:5]
    values = list(range(10))  # too long
    with pytest.raises(ValueError):
        sars.with_coldata(name="X", value=values)

def test_get_index_from_various_inputs(sars):
    assert sars.get_index("UHMK1", regex=False) == [0]
    assert sars.get_index(["UHMK1", "ATF3"], regex=False) == [0, 1]
    assert sars.get_index(0, regex=False) == [0]

def test_get_genes_with_various_inputs(sars):
    assert len(sars.get_genes()) == sars.gene_info.shape[0]
    assert sars.get_genes(0) == ["UHMK1"]
    assert sars.get_genes(0, get_gene_symbols=False) == ["ENSG00000152332"]
    assert sars.get_genes("UHMK1") == ["UHMK1"]
    assert sars.get_genes("UHMK1", get_gene_symbols=False) == ["ENSG00000152332"]
    assert sars.get_genes([0, 1]) == ["UHMK1", "ATF3"]
    assert sars.get_genes(["UHMK1", "ATF3"]) == ["UHMK1", "ATF3"]
    assert sars.get_genes([0, "UHMK1"]) == ["UHMK1"]

    regex_genes = sars.get_genes(r"^U.*1$", regex=True)
    regex_names = sars.get_genes(r"^U.*1$", regex=True, get_gene_symbols=False)
    assert regex_genes == ["UHMK1", "UPP1", "UBA1"]
    assert regex_names == ["ENSG00000152332", "ENSG00000183696", "ENSG00000130985"]

def test_replace_overrides_selected_components(sars):
    sars_small = sars[0:3, 0:3]

    new_prefix = "NEW_PREFIX"
    new_gene_info = sars_small.gene_info
    new_gene_info["extra_column"] = [1, 2, 3]

    new_coldata = sars_small.coldata
    new_coldata["new_column"] = [True, False, True]

    new_slots = {"count": np.ones((3, 3))}
    new_metadata = {"default_slot": "count"}
    new_analyses = {"analysis1": "done"}

    replaced = sars_small.replace(
        prefix=new_prefix,
        gene_info=new_gene_info,
        coldata=new_coldata,
        slots=new_slots,
        metadata=new_metadata,
        analyses=new_analyses,
    )

    # --- Check if fields were correctly replaced ---
    assert replaced.title == new_prefix
    assert "extra_column" in replaced.gene_info.columns
    assert "new_column" in replaced.coldata.columns
    assert np.array_equal(replaced.get_matrix(), np.ones((3, 3)))
    assert replaced.default_slot == "count"
    assert replaced.analyses == list(new_analyses.keys())

    # --- Check if unchanged parts remain the same ---
    assert replaced.get_genes() == sars_small.get_genes()
    assert list(replaced.coldata.index) == list(sars_small.coldata.index)
    assert list(replaced.gene_info.index) == list(sars_small.gene_info.index)

def test_replace_with_invalid_slot_shape_raises(sars):
    sars_small = sars[0:3]
    invalid_slots = {"count": np.ones((5, 5))}  # wrong Shape
    with pytest.raises(ValueError):
        sars_small.replace(slots=invalid_slots)