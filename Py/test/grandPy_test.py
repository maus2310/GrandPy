from itertools import count

import pandas as pd
import pytest
from Py.load import *

def test_default_slots():

    gp = read_grand("../data/sars.tsv")
    default_slots_test = gp.default_slot
    assert default_slots_test in "count"

def test_with_default_slots():

    gp = read_grand("../data/sars.tsv")
    with_default_slots_test_alpha = gp.with_default_slot("alpha")
    assert with_default_slots_test_alpha.default_slot == "alpha"
    with_default_slots_test_beta = gp.with_default_slot("beta")
    assert with_default_slots_test_beta.default_slot in "beta"
    with_default_slots_test_ntr = gp.with_default_slot("ntr")
    assert with_default_slots_test_ntr.default_slot in "ntr"
    with_default_slots_test_count = gp.with_default_slot("count" )
    assert with_default_slots_test_count.default_slot in "count"

def with_default_slots_test_immutability():

    gp = read_grand("../data/sars.tsv")
    with_default_slots_test_immutability = gp.with_default_slot("alpha")
    assert with_default_slots_test_immutability.default_slot != gp.default_slot

def slots_test():

    gp = read_grand("../data/sars.tsv")
    control_list = list['count', 'ntr', 'alpha', 'beta']
    slots_test = gp.slot_data
    for slots in slots_test:
        assert slots in control_list

def with_slots_test():

    gp = read_grand("../data/sars.tsv")




def test_get_genes():
    gp = read_grand("../data/sars_R.tsv")
    all_genes_test = gp.get_genes()
    all_genes_test_names = gp.get_genes(use_gene_symbols=False)
    assert gp.gene_info.shape[0] == len(all_genes_test) and gp.gene_info.shape[0] == len(all_genes_test_names)
    int_genes_test = gp.get_genes(0)
    int_genes_test_names = gp.get_genes(0, use_gene_symbols=False)
    assert int_genes_test == ["UHMK1"] and int_genes_test_names == ["ENSG00000152332"]
    str_genes_test = gp.get_genes("UHMK1")
    str_genes_test_names = gp.get_genes("UHMK1", use_gene_symbols=False)
    assert str_genes_test == ["UHMK1"] and str_genes_test_names == ["ENSG00000152332"]
    list_int_genes_test = gp.get_genes([0, 1])
    assert list_int_genes_test == ["UHMK1", "ATF3"]
    list_str_genes_test = gp.get_genes(["UHMK1", "ATF3"])
    assert list_str_genes_test == ["UHMK1", "ATF3"]
    mixed_genes_test = gp.get_genes([0, "UHMK1"])
    assert mixed_genes_test == ["UHMK1"]
    regex_genes_test = gp.get_genes(r"^U.*1$", regex = True)
    regex_genes_test_names = gp.get_genes(r"^U.*1$", regex = True, use_gene_symbols=False)
    assert regex_genes_test == ["UHMK1", "UPP1", "UBA1"] and regex_genes_test_names == ["ENSG00000152332", "ENSG00000183696", "ENSG00000130985"]

def test_get_genes_immutability():
    gp = read_grand("../data/sars_R.tsv")
    get_genes_immutability_test = gp.get_genes()
    get_genes_immutability_test[0] = "UHMK2"
    assert gp.get_genes()[0] != "UHMK2"



