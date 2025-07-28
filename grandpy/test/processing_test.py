import pytest

import grandpy as gp

def test_filter_genes():
    sars = gp.read_grand("../data/sars.tsv", design=("Condition", "dur.4sU", "Replicate"))

    # --- Default behaviour ---
    filtered_default = sars.filter_genes()
    assert len(filtered_default.genes) == 9162
    assert all(g in sars.genes for g in filtered_default.genes)

    # --- Filtering with min_expression + min_condition ---
    filtered_expr_cond = sars.filter_genes(min_expression=1000, min_condition=2)
    assert len(filtered_expr_cond.genes) == 5611

    # --- Filtering with keep ---
    filtered_with_keep = sars.filter_genes(min_expression=1000, min_condition=2, keep=["ATF3"])
    assert "ATF3" in filtered_with_keep.genes
    assert len(filtered_with_keep.genes) == 5612

    # --- Subsetting with use ---
    viral_genes = sars.get_classified_genes("Unknown")
    filtered_subset = sars.filter_genes(use=viral_genes)
    assert set(filtered_subset.genes) == set(viral_genes)

    # --- return_genes ---
    returned_indices = sars.filter_genes(min_expression=1000, min_condition=2, return_genes=True)
    assert isinstance(returned_indices, list)
    assert len(returned_indices) == 5611

    # --- Error on invalid slot ---
    with pytest.raises(ValueError):
        sars.filter_genes(mode_slot="not_a_slot")