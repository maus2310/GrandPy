import pytest
import tempfile
from pathlib import Path
from grandpy import *

TOLERANCE_KB = 10

EXPECTED_FILE_SIZES = {
    "Expression_Test.svg": 204,
    "Heatmap_norm.svg": 63,
    "PCA_norm.svg": 40,
    "SARS.1h.A_Mock.1h.A_norm.svg": 212,
    "SRSF6_Groups_Bars.svg": 36,
    "SRSF6_Groups_Points.svg": 40,
    "SRSF6_Old_vs_New.svg": 52,
    "SRSF6_Progressive_Timecourse.svg": 48,
    "SRSF6_Snapshot_Timecourse.svg": 41,
    "SRSF6_Total_vs_Ntr.svg": 47,
    "Type_Distribution.svg": 49
}
# TODO Read/Imports checken!!!!!!!!!!!!!!!!! #MA und Vulcano fehlt
sars = read_grand("../data/sars_R.tsv",
                          design=("Condition", "duration.4sU", "Replicate"),
                          classification_genes=['UHMK1', 'ATF3', 'PABPC4', 'ROR1', 'ZC3H11A', 'ZBED6', 'PRDX6', 'PRRC2C'], classification_genes_label="Moin")

def test_plot_outputs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = Path(tmpdirname)
        plot_run(sars=sars, gene="SRSF6", pathforsave=path)
        out_of_range = []
        missing = []

        print(f"\n--- Check plot-size with tolerance of +/- {TOLERANCE_KB} KB ---")
        for filename, expected_kb in EXPECTED_FILE_SIZES.items():
            filepath = path / filename
            if not filepath.is_file():
                missing.append(filename)
                print(f"{filename}: MISSING")
                continue

            actual_kb = filepath.stat().st_size / 1024
            diff = actual_kb - expected_kb
            print(f"{filename}: {actual_kb:.1f} KB (expected {expected_kb} ± {TOLERANCE_KB}, Δ={diff:+.1f})")

            if abs(diff) > TOLERANCE_KB:
                out_of_range.append(
                    f"(Failed Plot: '{filename}', actual size: {actual_kb:.2f} KB, expected size: {expected_kb} KB)"
                )

        if missing:
            pytest.fail("Missing plot files: " + ", ".join(missing))
        if out_of_range:
            pytest.fail("Files with size out of range:\n" + "\n".join(out_of_range))

def plot_run(sars, gene, pathforsave):
    sars = sars.normalize()
    sars = sars.compute_ntr_ci()
    highlightgenes = sars.get_classified_genes("Moin")

    plot_scatter(sars, x="SARS.1h.A", highlight=highlightgenes, label=highlightgenes, log=True, y_label_offset=0.01, remove_outlier=True, diagonal=True, path_for_save=pathforsave)
    plot_pca(sars, path_for_save=pathforsave)
    plot_heatmap(sars, transform="vst", cluster_genes=False, path_for_save=pathforsave, title="Heatmap", genes=highlightgenes)
    plot_gene_old_vs_new(sars, gene, show_ci=True, path_for_save=pathforsave)
    plot_gene_total_vs_ntr(sars, gene, slot="total_count", show_ci=True, path_for_save=pathforsave)
    plot_gene_groups_points(sars, gene, show_ci=True, path_for_save=pathforsave, dodge=True)
    plot_gene_groups_bars(sars, gene, xlabels="Condition + '.' + Replicate", show_ci=True, path_for_save=pathforsave)
    plot_gene_snapshot_timecourse(sars, gene, show_ci=True, mode_slot="ntr", path_for_save=pathforsave, dodge=True)
    plot_gene_progressive_timecourse(sars, gene, path_for_save=pathforsave)
    plot_expression_test(sars, "SARS.1h.A", "SARS.no4sU.A", path_for_save=pathforsave)
    plot_type_distribution(sars, relative=True, path_for_save=pathforsave)