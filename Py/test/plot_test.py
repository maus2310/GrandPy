from itertools import count
from operator import contains

import pytest
from Py.load import *
from Py.plot import *

sars = read_grand("../data/sars_R.tsv",
                          design=("Condition", "Time", "Replicate"),
                          classification_genes=['UHMK1', 'ATF3', 'PABPC4', 'ROR1', 'ZC3H11A', 'ZBED6', 'PRDX6', 'PRRC2C'], classification_genes_label="Moin")
def test_plot_test_run(sars=sars, gene = "SRSF6", pathforsave = None ): #"C:/Users/Andre/Desktop/plots"
    try:
        sars = sars.normalize()
        sars = sars.compute_ntr_ci()
        lol = sars.get_classified_genes("Moin")
        plot_scatter(sars, x="SARS.1h.A", highlight = lol, label=[], log=True, y_label_offset=0.01, remove_outlier=True, path_for_save=pathforsave)
        plot_pca(sars, path_for_save=pathforsave)
        plot_gene_old_vs_new(sars, gene, show_ci=True, path_for_save=pathforsave)
        plot_gene_total_vs_ntr(sars, gene, slot="total_count" ,show_ci=True, path_for_save=pathforsave)
        plot_gene_groups_points(sars, gene, show_ci=True, path_for_save=pathforsave, dodge=True)
        plot_gene_groups_bars(sars, gene, xlabels="Condition + '.' + Replicate", show_ci=True, path_for_save=pathforsave)
        plot_gene_snapshot_timecourse(sars, gene, show_ci=True, mode_slot="ntr", path_for_save=pathforsave, dodge=True)
        plot_gene_progressive_timecourse(sars, gene, path_for_save=pathforsave)
        plot_expression_test(sars, "SARS.1h.A", "SARS.no4sU.A", path_for_save=pathforsave)
        plot_type_distribution(sars, relative=True, path_for_save=pathforsave)
    except Exception as e:
        pytest.fail(f"test_plot_test_run failed: {e}")