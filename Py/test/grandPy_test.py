from itertools import count
from operator import contains

import numpy as np
import pandas as pd
import pytest

from Py.load import *
from Py.slot_tool import ModeSlot

read_input = {
    "prefix": "../data/sars_R.tsv",
    "design": ("Condition", "dur.4sU", "Replicate")
}

def test_default_slots_test():

    gp = read_grand(**read_input)
    default_slots_test = gp.default_slot
    assert default_slots_test in "count"

def test_with_default_slots():

    gp = read_grand(**read_input)
    with_default_slots_test_alpha = gp.with_default_slot("alpha")
    assert with_default_slots_test_alpha.default_slot == "alpha"

def test_with_default_slots_immutability():

    gp = read_grand(**read_input)
    with_default_slots_test_immutability = gp.with_default_slot("alpha")
    assert with_default_slots_test_immutability.default_slot != gp.default_slot

def test_slot_names():

    gp = read_grand(**read_input)
    control_list = ['count', 'ntr', 'alpha', 'beta']
    slots_test = gp.slots
    for slots in slots_test:
        assert slots in control_list

def test_with_dropped_slots():

    gp = read_grand(**read_input)
    control_list_ntr = ['alpha', 'beta', 'count']
    with_dropped_slots = gp.with_dropped_slots("ntr")
    assert with_dropped_slots.slots == control_list_ntr

    # Dieser Test klappt aber nur wenn man count beibehält (wie gedacht), kann default slots löschen & neue setzen

    # test_with_dropped_default_slots = gp.with_default_slot("ntr").with_dropped_slots("ntr")
    # control_list_count = ['count', 'alpha', 'beta']
    # assert test_with_dropped_default_slots.slot_names == control_list_count

# ich schaue nochmal wie man das am besten testet, habe atm keine neue Matrix zur Hand, mit der ich abgleichen könnte
# def test_with_slots():
#
#     gp = read_grand("../data/sars_R.tsv")

def test_with_condition():
    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_condition = gp.with_condition(["MOGGED"]*12)
    with_condition_dict = gp.with_condition({"Mock.1h.A": "Test1", "Mock.3h.A": "Test2"})
    assert with_condition_dict.coldata["Condition"]["Mock.1h.A"] == "Test1" and with_condition_dict.coldata["Condition"]["Mock.3h.A"] == "Test2"
    for i in range(0,12):
        assert with_condition.condition[i] == "MOGGED"

def test_with_renamed_columns_immutability():
    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_renamed_columns_immmutibility = gp.with_renamed_columns({"Mock.no4sU.A": "Test1", "SARS.no4sU.A": "Test2"})
    assert gp.coldata.index[0] != "Test1" and gp.coldata.index[6] != "Test2"

def test_with_renamed_columns_dict():
    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_renamed_columns_dict = gp.with_renamed_columns({"Mock.no4sU.A": "Test1", "SARS.no4sU.A": "Test2"})
    assert with_renamed_columns_dict.coldata.index[0] == "Test1" and with_renamed_columns_dict.coldata.index[6] == "Test2"

def test_with_swapped_columns():
    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_swapped_columns = gp.with_swapped_columns("Mock.1h.A", "Mock.2h.A")
    assert np.all(np.isin(with_swapped_columns.get_matrix(columns="Mock.1h.A"), gp.get_matrix(columns="Mock.2h.A")))
    assert np.all(np.isin(with_swapped_columns.get_matrix(columns="Mock.2h.A"), gp.get_matrix(columns="Mock.1h.A")))

def test_get_table():
    test_table = {
        "Mock.no4sU.A" : [0.0, 0.0, 0.0],
        "Mock.1h.A": [201.79815, 118.64680, 531.08850],
        "Mock.2h.A": [434.818800, 81.973500, 1197.084926]
    }

    gp = read_grand(**read_input)
    test_get_table = gp.get_table(ModeSlot("new", "count"), [0,1,2], [0,1,2])
    columns = ["Mock.no4sU.A","Mock.1h.A", "Mock.2h.A"]
    rows = [0,1,2]
    for i in columns:
        for j in rows:
            assert round(test_get_table[i][j], 3) == round(test_table[i][j], 3)

def test_merge_coldata():
    from io import StringIO

    gp =read_grand(**read_input)
    egp = gp[0:10]
    zgp = gp[10:20]

    data = """
    Name,Condition,duration.4sU,Replicate,no4sU
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
    Mock.no4sU.A_1,Mock,0,A,True
    Mock.1h.A_1,Mock,1,A,False
    Mock.2h.A_1,Mock,2,A,False
    Mock.2h.B_1,Mock,2,B,False
    Mock.3h.A_1,Mock,3,A,False
    Mock.4h.A_1,Mock,4,A,False
    SARS.no4sU.A_1,SARS,0,A,True
    SARS.1h.A_1,SARS,1,A,False
    SARS.2h.A_1,SARS,2,A,False
    SARS.2h.B_1,SARS,2,B,False
    SARS.3h.A_1,SARS,3,A,False
    SARS.4h.A_1,SARS,4,A,False
    """

    test_dataframe = pd.read_csv(StringIO(data), index_col=0)
    test_concat_coldata_df = egp.merge(zgp, axis= 0).coldata

    cols = ["Condition", "duration.4sU", "Replicate", "no4sU"]
    rows = [0,1,2,3]
    print (test_concat_coldata_df.columns)
    for i in cols:
        for j in rows:
            assert test_concat_coldata_df[i][j] == test_dataframe[i][j]

def test_with_gene_info_immutability():

    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_gene_info_immutability = gp.with_gene_info("Gene", [1,2,3,4,5,6,7,8,9,10])
    for el in with_gene_info_immutability.gene_info["Gene"].values: assert el not in gp.gene_info["Gene"].values

def test_with_gene_info_dict():
    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_gene_info_dict = gp.with_gene_info("Gene", {"UHMK1": "Control", "ATF3": "Treatment"})
    assert (with_gene_info_dict.gene_info["Gene"][0] == "Control"
            and with_gene_info_dict.gene_info["Gene"][1] == "Treatment")

def test_with_gene_info_series():
    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_gene_info_series = gp.with_gene_info("Gene", pd.Series([1,2,3,4,5,6,7,8,9,10], index = gp.gene_info.index[0:10]))
    for i in range(0,10):
        assert with_gene_info_series.gene_info["Gene"][i] == i+1

def test_with_coldata():

    gp = read_grand(**read_input)
    gp = gp[0:10]
    with_coldata_immutability = gp.with_coldata("new_condition", [1,2,3,4,5,6,7,8,9,10,11,12])
    assert "new_condition" not in gp.coldata.columns

def test_get_index():

    gp = read_grand(**read_input)
    gp = gp[0:10]

    one_gene_regex_false_test = gp.get_index("UHMK1", regex = False)
    assert one_gene_regex_false_test == [0]

    two_genes_regex_false_test = gp.get_index(["UHMK1", "ATF3"], regex = False)
    assert two_genes_regex_false_test == [0, 1]

    one_index_regex_false_test = gp.get_index(0, regex = False)
    assert one_index_regex_false_test == [0]

def test_get_genes():

    gp = read_grand(**read_input)
    all_genes_test = gp.get_genes()
    all_genes_test_names = gp.get_genes(get_gene_symbols=False)
    assert gp.gene_info.shape[0] == len(all_genes_test) and gp.gene_info.shape[0] == len(all_genes_test_names)
    int_genes_test = gp.get_genes(0)
    int_genes_test_names = gp.get_genes(0, get_gene_symbols=False)
    assert int_genes_test == ["UHMK1"] and int_genes_test_names == ["ENSG00000152332"]
    str_genes_test = gp.get_genes("UHMK1")
    str_genes_test_names = gp.get_genes("UHMK1", get_gene_symbols=False)
    assert str_genes_test == ["UHMK1"] and str_genes_test_names == ["ENSG00000152332"]
    list_int_genes_test = gp.get_genes([0, 1])
    assert list_int_genes_test == ["UHMK1", "ATF3"]
    list_str_genes_test = gp.get_genes(["UHMK1", "ATF3"])
    assert list_str_genes_test == ["UHMK1", "ATF3"]
    mixed_genes_test = gp.get_genes([0, "UHMK1"])
    assert mixed_genes_test == ["UHMK1"]
    regex_genes_test = gp.get_genes(r"^U.*1$", regex = True)
    regex_genes_test_names = gp.get_genes(r"^U.*1$", regex = True, get_gene_symbols=False)
    assert regex_genes_test == ["UHMK1", "UPP1", "UBA1"] and regex_genes_test_names == ["ENSG00000152332", "ENSG00000183696", "ENSG00000130985"]

def test_get_genes_immutability():
    gp = read_grand(**read_input)
    get_genes_immutability_test = gp.get_genes()
    get_genes_immutability_test[0] = "UHMK2"
    assert gp.get_genes()[0] != "UHMK2"