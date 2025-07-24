from io import StringIO
import numpy as np
import pandas as pd
import pytest

from grandpy import read_grand, ModeSlot
from grandpy.utils import _make_unique


@pytest.fixture(scope="module")
def sars():
    return read_grand("../data/sars_R.tsv", design=("Condition", "dur.4sU", "Replicate"))


def test_with_default_slot(sars):
    assert sars.default_slot == "count"

    updated = sars.with_default_slot("alpha")
    assert updated.default_slot == "alpha"

def test_slots_and_with_dropped_slots(sars):
    known_slots = {"count", "ntr", "alpha", "beta"}
    assert known_slots.union(sars.slots) == known_slots

    dropped = sars.with_dropped_slots("ntr")
    expected_slots = ['count', 'alpha', 'beta']
    assert sorted(dropped.slots) == sorted(expected_slots)

def test_with_condition(sars):
    updated_list = sars.with_condition(["MOGGED"] * 12)
    updated_dict = sars.with_condition({"Mock.1h.A": "Test1", "Mock.3h.A": "Test2"})

    assert updated_dict.coldata["Condition"]["Mock.1h.A"] == "Test1"
    assert updated_dict.coldata["Condition"]["Mock.3h.A"] == "Test2"
    assert all(condition == "MOGGED" for condition in updated_list.condition)

    with pytest.raises(ValueError):
        sars.with_condition(["a", "b"])
    with pytest.raises(ValueError):
        sars.with_condition({"UnknownSample": "oops"})

def test_with_renamed_columns(sars):
    updated = sars.with_renamed_columns({"Mock.no4sU.A": "Test1", "SARS.no4sU.A": "Test2"})
    assert sars.coldata.index[0] != "Test1"
    assert updated.coldata.index[0] == "Test1"
    assert updated.coldata.index[6] == "Test2"

def test_with_swapped_columns(sars):
    swapped = sars.with_swapped_columns("Mock.1h.A", "Mock.2h.A")
    assert all(swapped.get_matrix(columns="Mock.1h.A") == sars.get_matrix(columns="Mock.2h.A"))
    assert all(swapped.get_matrix(columns="Mock.2h.A") == sars.get_matrix(columns="Mock.1h.A"))

def test_get_table(sars):
    expected = {
        "Mock.no4sU.A": [0.0, 0.0, 0.0],
        "Mock.1h.A": [201.79815, 118.64680, 531.08850],
        "Mock.2h.A": [434.818800, 81.973500, 1197.084926]
    }
    result = sars.get_table(ModeSlot("new", "count"), [0, 1, 2], [0, 1, 2])
    for col in expected:
        for row in range(3):
            assert round(result[col][row], 3) == round(expected[col][row], 3)

def test_merge_coldata(sars):
    gp1 = sars[0:10]
    gp2 = sars[0:10]
    data = """Name,Condition,Replicate,no4sU
Mock.no4sU.A,Mock,A,True
Mock.1h.A,Mock,A,False
Mock.2h.A,Mock,A,False
Mock.2h.B,Mock,B,False
Mock.3h.A,Mock,A,False
Mock.4h.A,Mock,A,False
SARS.no4sU.A,SARS,A,True
SARS.1h.A,SARS,A,False
SARS.2h.A,SARS,A,False
SARS.2h.B,SARS,B,False
SARS.3h.A,SARS,A,False
SARS.4h.A,SARS,A,False
""" * 2

    # Drop is due to the *2 of the 'data' leading to the column names appearing as a row in the resulting DataFrame.
    reference = pd.read_csv(StringIO(data)).drop(12, axis=0)
    merged = gp1.merge(gp2, axis=0).coldata[["Name","Condition","Replicate","no4sU"]]

    reference.index = _make_unique(reference["Name"].str.strip())
    merged.index = merged.index.str.strip()

    assert (merged.astype(str) == reference.astype(str)).all().all()

def test_merge_gene_info(sars):
    gp1 = sars[0:10]
    gp2 = sars[10:20]
    data = """Symbol,Gene,Length,Type
UHMK1,ENSG00000152332,8478,Cellular
ATF3,ENSG00000162772,2103,Cellular
PABPC4,ENSG00000090621,3592,Cellular
ROR1,ENSG00000185483,5832,Cellular
ZC3H11A,ENSG00000058673,11825,Cellular
ZBED6,ENSG00000257315,12481,Cellular
PRDX6,ENSG00000117592,1751,Cellular
PRRC2C,ENSG00000117523,10366,Cellular
ATP1B1,ENSG00000143153,2608,Cellular
NEK7,ENSG00000151414,4149,Cellular
RPS27,ENSG00000177954,625,Cellular
DHX9,ENSG00000135829,4240,Cellular
KHDRBS1,ENSG00000121774,2757,Cellular
PTPRF,ENSG00000142949,7727,Cellular
KIAA1522,ENSG00000162522,5438,Cellular
CD46,ENSG00000117335,7826,Cellular
GALNT2,ENSG00000143641,4454,Cellular
RPS8,ENSG00000142937,1115,Cellular
TINAGL1,ENSG00000142910,4759,Cellular
CAPN2,ENSG00000162909,4130,Cellular"""

    reference = pd.read_csv(StringIO(data))
    merged = gp1.merge(gp2, axis=1).gene_info

    reference.index = reference["Symbol"].str.strip()
    merged.index = merged.index.str.strip()

    assert (merged.astype(str) == reference.astype(str)).all().all()

def test_with_gene_info(sars):
    small = sars[0:10]
    updated_list = small.with_gene_info(name="Gene", value=list(range(10)))
    assert not np.array_equal(updated_list.gene_info["Gene"], small.gene_info.get("Gene", []))

    updated_dict = small.with_gene_info(name="Gene", value={"UHMK1": "Control", "ATF3": "Treatment"})
    assert updated_dict.gene_info["Gene"][0] == "Control"
    assert updated_dict.gene_info["Gene"][1] == "Treatment"

    series = pd.Series(range(10), index=small.gene_info.index[:10])
    updated_series = small.with_gene_info(name="Gene", value=series)
    assert all(updated_series.gene_info["Gene"].values == series.values)

    with pytest.raises(ValueError):
        sars[0:5].with_gene_info(name="X", value=list(range(10)))  # too long

def test_with_coldata(sars):
    updated = sars.with_coldata(name="new_condition", value=list(range(12)))
    assert "new_condition" not in sars.coldata.columns
    assert "new_condition" in updated.coldata.columns

    with pytest.raises(ValueError):
        sars[0:5].with_coldata(name="X", value=list(range(10)))  # length mismatch

def test_get_index_and_get_genes(sars):
    assert sars.get_index("UHMK1", regex=False) == [0]
    assert sars.get_index(["UHMK1", "ATF3"], regex=False) == [0, 1]
    assert sars.get_index(0, regex=False) == [0]

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

def test_replace(sars):
    sars_small = sars[0:3, 0:3]
    new_prefix = "NEW_PREFIX"
    new_gene_info = sars_small.gene_info.copy()
    new_gene_info["extra_column"] = [1, 2, 3]
    new_coldata = sars_small.coldata.copy()
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

    assert replaced.title == new_prefix
    assert "extra_column" in replaced.gene_info.columns
    assert "new_column" in replaced.coldata.columns
    assert np.array_equal(replaced.get_matrix(), np.ones((3, 3)))
    assert replaced.default_slot == "count"
    assert replaced.analyses == list(new_analyses.keys())

    assert replaced.get_genes() == sars_small.get_genes()
    assert list(replaced.coldata.index) == list(sars_small.coldata.index)
    assert list(replaced.gene_info.index) == list(sars_small.gene_info.index)

    with pytest.raises(ValueError):
        sars[0:3].replace(slots={"count": np.ones((5, 5))})  # invalid shape

