from io import StringIO

import numpy as np
import pandas as pd
import pytest

from grandpy import read_grand, ModeSlot
from grandpy.utils import _make_unique


# TODO: Funktions not yet tested: get_significant_genes(), get_analysis_table(), (get_references()), to_anndata, from_anndata


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
    updated_str = sars.with_condition("Control")
    updated_list = sars.with_condition(["Control"] * 12)
    updated_dict = sars.with_condition({"Mock.1h.A": "Test1", "Mock.3h.A": "Test2"})
    updated_coldata_columns = sars.with_condition(["Condition", "Replicate"])

    coldata_columns_exp = ["Mock A", "Mock A", "Mock A", "Mock B", "Mock A", "Mock A", "SARS A", "SARS A", "SARS A", "SARS B", "SARS A", "SARS A"]

    assert all(condition == "Control" for condition in updated_str.condition)
    assert all(condition == "Control" for condition in updated_list.condition)
    assert updated_dict.coldata["Condition"]["Mock.1h.A"] == "Test1"
    assert updated_dict.coldata["Condition"]["Mock.3h.A"] == "Test2"
    assert all(updated_coldata_columns.coldata["Condition"].values == coldata_columns_exp)

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

def test_get_matrix(sars):
    # --- Basic Test: Should return a ndarray with rows and columns ---
    result = sars.get_matrix(mode_slot="count", genes=["UHMK1", "ATF3"], columns=[0, 1])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert np.allclose(result[0, 0], 650.0)
    assert np.allclose(result[1, 1], 188.0)

    # --- Test mode slot ---
    result = sars.get_matrix(mode_slot="old_count", genes=["UHMK1", "ATF3"], columns=[0, 1], ntr_nan=False)
    assert result.shape == (2, 2)
    assert not np.isnan(result[0, 0])
    assert np.allclose(result[0, 1], 4312.0, atol=1)
    assert np.allclose(result[1, 1], 69.0, atol=1)

    # --- Test ntr_nan=True with mode slot ---
    result = sars.get_matrix(mode_slot="new_count", genes=["UHMK1", "ATF3"], columns=[0, 1], ntr_nan=True)
    assert np.isnan(result[0, 0])
    assert not np.isnan(result[1, 1])
    assert np.allclose(result[0, 1], 201.0, atol=1)
    assert np.allclose(result[1, 1], 118.0, atol=1)

    # --- Test with invalid mode_slot ---
    with pytest.raises(ValueError):
        sars.get_matrix(mode_slot="invalid_slot", genes=["UHMK1", "ATF3"], columns=[0, 1])

    # --- Test without genes ---
    result = sars.get_matrix(mode_slot="new_count", genes=[], columns=[0, 1])
    assert result.shape[0] == 0

def test_get_table(sars):
    # --- Basic Test: Should return a DataFrame with rows and columns ---
    result = sars.get_table()
    assert result.shape[0] == 1045
    assert result.shape[1] == 12

    # --- Test with Specific Genes ---
    genes = ["UHMK1", "ATF3"]
    result = sars.get_table(genes=genes)
    assert all(gene in result.index for gene in genes)
    assert result.shape[0] == len(genes)

    # --- Test with Specific Columns (Samples) ---
    columns = [0, 1]
    result = sars.get_table(columns=columns)
    assert result.shape[1] == len(columns)

    # --- Test with gene_info included ---
    result = sars.get_table(with_gene_info=True)
    assert "Symbol" in result.columns
    assert "Length" in result.columns

    # --- Test with different `name_genes_by` (Ensembl IDs instead of Symbols) ---
    result = sars.get_table(with_gene_info=True, name_genes_by="Gene")
    assert result.index.name == "Gene"

    # --- Test with summarize DataFrame ---
    summarize = sars.get_summary_matrix()
    result = sars.get_table(summarize=summarize)
    assert result.shape == (1045,2)

    # --- Test with Prefix for Column Names ---
    result = sars.get_table(prefix="Test_")
    assert all(col.startswith("Test_") for col in result.columns)

    # --- Test with ntr_nan=True (Check if no4sU has NaN values) ---
    result = sars.get_table(mode_slot="new_count", ntr_nan=True)
    assert all(np.isnan(result.loc[:,["Mock.no4sU.A", "SARS.no4sU.A"]]))

    # --- Test with reorder_columns=True (Columns should match original order) ---
    result = sars.get_table(columns=list(range(4,8)) + list(range(0,4)) + list(range(8,12)), reorder_columns=True)
    assert list(result.columns) == list(sars.coldata.index)

    # --- Test with invalid genes (should return empty DataFrame) ---
    result = sars.get_table(genes=["NonExistentGene"])
    assert result.shape[0] == 0

def test_get_data(sars):
    # --- Basic Test: Should return a DataFrame with rows and columns ---
    result = sars.get_data(with_coldata=False)
    assert result.shape[0] == 12
    assert result.shape[1] == 1045

    # --- Test with Specific Genes ---
    genes = ["UHMK1", "ATF3"]
    result = sars.get_data(genes=genes)
    assert all(gene in result.columns for gene in genes)
    assert result.shape[1] == len(genes) + len(sars.coldata.columns)

    # --- Test with Specific Columns (Samples) ---
    columns = [0, 1]
    result = sars.get_data(columns=columns)
    assert result.shape[0] == len(columns)

    # --- Test with coldata included ---
    result = sars.get_data(with_coldata=True)
    assert isinstance(result, pd.DataFrame)
    assert "Condition" in result.columns

    # --- Test with different `name_genes_by` (Ensembl IDs instead of Symbols) ---
    result = sars.get_data(with_coldata=False, name_genes_by="Gene")
    assert result.columns[0] == "ENSG00000152332"

    # --- Test with by_rows=True ---
    result = sars.get_data(by_rows=True)
    assert result.shape[0] > result.shape[1]

    # --- Test with invalid genes (should return empty DataFrame) ---
    result = sars.get_data(genes=["NonExistentGene"], with_coldata=False)
    assert result.shape[1] == 0

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

def test_with_updated_symbols(sars):
    small_sars = sars[0:3]

    # Test 1: Test on already correct symbols
    t1_sars = small_sars
    try:
        updated_sars = t1_sars.with_updated_symbols(species="human")

        assert "Symbol" in updated_sars.gene_info.columns
        assert updated_sars.genes == small_sars.genes
    except Exception as e:
        pytest.fail(f"Test 1 failed due to exception: {e}")


    # Test 2: Update wrong symbols
    wrong_symbols = ["s1", "s2", "s3"]
    t2_sars = small_sars.with_gene_info(name="Symbol", value=wrong_symbols)
    try:
        updated_sars = t2_sars.with_updated_symbols(species="human")

        assert "Symbol" in updated_sars.gene_info.columns
        assert updated_sars.genes == small_sars.genes
    except Exception as e:
        pytest.fail(f"Test 2 failed due to exception: {e}")


    # Test 3: No symbol column
    t3_sars = small_sars._dev_replace(gene_info=small_sars.gene_info.drop(columns=["Symbol"]))
    try:
        updated_sars = t3_sars.with_updated_symbols(species="human")

        assert "Symbol" in updated_sars.gene_info.columns
        assert updated_sars.genes == small_sars.genes
    except Exception as e:
        pytest.fail(f"Test 3 failed due to exception: {e}")


    # Test 4: Ensemble ID has no matching symbol
    wrong_symbols = [None, "S", "s3"]
    t4_sars = sars[["ORF1ab", "S", "UHMK1"]].with_gene_info(name="Symbol", value=wrong_symbols)
    try:
        updated_sars = t4_sars.with_updated_symbols(species="human")

        assert "Symbol" in updated_sars.gene_info.columns
        assert updated_sars.genes == [None, "S", "UHMK1"]
    except Exception as e:
        pytest.fail(f"Test 4 failed due to exception: {e}")


