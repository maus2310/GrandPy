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
    ueberpruefliste = list['count', 'ntr', 'alpha', 'beta']
    slots_test = gp.slots
    for slots in slots_test:
        assert slots in ueberpruefliste


    # test_data = {
    #     "Gene": ["ENSG000001", "ENSG000002"],
    #     "Symbol": ["GAPDH", "ACTB"],
    #     "Length": [1000, 1200],
    #     "Sample1 MAP": [0.9, 0.1],
    #     "Sample2 MAP": [0.8, 0.2]
    # }
    # test_df = pd.DataFrame(test_data)
    # test_file = tmp_path / "test_de.tsv"
    # test_df.to_csv(test_file, sep="\t", index=False)

