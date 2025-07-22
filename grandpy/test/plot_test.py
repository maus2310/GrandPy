import pytest
import tempfile
from pathlib import Path
from grandpy import *

TOLERANCE_KB = 10
FAILED_PLOTS = []
OUT_OF_RANGE = []

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
    "Type_Distribution.svg": 49,
    "scatter.svg": 60
}

@pytest.fixture(scope="module")
def sars_dataset():
    sars = read_grand("../data/sars_R.tsv",
                      design=("Condition", "duration.4sU", "Replicate"),
                      classification_genes=['UHMK1', 'ATF3', 'PABPC4', 'ROR1', 'ZC3H11A', 'ZBED6', 'PRDX6', 'PRRC2C'],
                      classification_genes_label="Moin")
    sars = sars.normalize().compute_ntr_ci()
    return sars

@pytest.fixture(scope="module")
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

def check_file_size(path, filename):
    expected_kb = EXPECTED_FILE_SIZES[filename]
    filepath = path / filename
    if not filepath.is_file():
        pytest.fail(f"{filename} missing at: {filepath}")

    actual_kb = filepath.stat().st_size / 1024
    diff = actual_kb - expected_kb

    if abs(diff) > TOLERANCE_KB:
        pytest.fail(
            f"{filename}: {actual_kb:.1f} KB (expected {expected_kb} ± {TOLERANCE_KB}, Δ={diff:+.1f})"
        )


def test_plot_scatter(sars_dataset, temp_output_dir):
    try:
        highlight = sars_dataset.get_classified_genes("Moin")
        plot_scatter(sars_dataset, x="SARS.1h.A", highlight=highlight, label=highlight, log=True,
                     y_label_offset=0.01, remove_outlier=True, diagonal=True, path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "SARS.1h.A_Mock.1h.A_norm.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"SARS.1h.A_Mock.1h.A_norm.svg: {e}")

def test_plot_pca(sars_dataset, temp_output_dir):
    try:
        plot_pca(sars_dataset, path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "PCA_norm.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"PCA_norm.svg: {e}")

def test_plot_heatmap(sars_dataset, temp_output_dir):
    try:
        highlight = sars_dataset.get_classified_genes("Moin")
        plot_heatmap(sars_dataset, transform="vst", cluster_genes=False, title="Heatmap",
                     genes=highlight, path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "Heatmap_norm.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"Heatmap_norm.svg: {e}")

def test_plot_gene_old_vs_new(sars_dataset, temp_output_dir):
    try:
        plot_gene_old_vs_new(sars_dataset, "SRSF6", show_ci=True, path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "SRSF6_Old_vs_New.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"SRSF6_Old_vs_New.svg: {e}")

def test_plot_gene_total_vs_ntr(sars_dataset, temp_output_dir):
    try:
        plot_gene_total_vs_ntr(sars_dataset, "SRSF6", slot="total_count", show_ci=True, path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "SRSF6_Total_vs_Ntr.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"SRSF6_Total_vs_Ntr.svg: {e}")

def test_plot_gene_groups_points(sars_dataset, temp_output_dir):
    try:
        plot_gene_groups_points(sars_dataset, "SRSF6", show_ci=True, path_for_save=temp_output_dir, dodge=True, show_plot=False)
        check_file_size(temp_output_dir, "SRSF6_Groups_Points.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"SRSF6_Groups_Points.svg: {e}")

def test_plot_gene_groups_bars(sars_dataset, temp_output_dir):
    try:
        plot_gene_groups_bars(sars_dataset, "SRSF6", xlabels="Condition + '.' + Replicate", show_ci=True,
                              path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "SRSF6_Groups_Bars.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"SRSF6_Groups_Bars.svg: {e}")

def test_plot_snapshot(sars_dataset, temp_output_dir):
    try:
        plot_gene_snapshot_timecourse(sars_dataset, "SRSF6", show_ci=True, mode_slot="ntr", path_for_save=temp_output_dir, dodge=True, show_plot=False)
        check_file_size(temp_output_dir, "SRSF6_Snapshot_Timecourse.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"SRSF6_Snapshot_Timecourse.svg: {e}")

def test_plot_progressive(sars_dataset, temp_output_dir):
    try:
        plot_gene_progressive_timecourse(sars_dataset, "SRSF6", path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "SRSF6_Progressive_Timecourse.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"SRSF6_Progressive_Timecourse.svg: {e}")

def test_plot_expression_test(sars_dataset, temp_output_dir):
    try:
        plot_expression_test(sars_dataset, "SARS.1h.A", "SARS.no4sU.A", path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "Expression_Test.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"Expression_Test.svg: {e}")

def test_plot_type_distribution(sars_dataset, temp_output_dir):
    try:
        plot_type_distribution(sars_dataset, relative=True, path_for_save=temp_output_dir, show_plot=False)
        check_file_size(temp_output_dir, "Type_Distribution.svg")
    except Exception as e:
        FAILED_PLOTS.append(f"Type_Distribution.svg: {e}")

import warnings
def test_summary():
    red = "\033[91m"
    yellow = "\033[93m"
    green = "\033[92m"
    reset = "\033[0m"

    if FAILED_PLOTS:
        print(f"\n\n\n{red} Failed Plots:{reset}")
        for fail in FAILED_PLOTS:
            print(f"   - {red}{fail}{reset}")

    if OUT_OF_RANGE:
        print(f"\n\n\n{yellow} Plots with unexpected size:{reset}")
        for warn in OUT_OF_RANGE:
            print(f"   - {yellow}{warn}{reset}")

    if FAILED_PLOTS or OUT_OF_RANGE:
        warnings.warn("\nSome plots failed or had wrong sizes. See above.")
    else:
        print(f"\n\n\n{green} All plots passed successfully.{reset}")
