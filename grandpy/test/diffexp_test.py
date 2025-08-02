import numpy as np
import pandas as pd
import pytest
from mpmath import polygamma

import grandpy as gp
from grandpy.lfc import psi_lfc, norm_lfc, empirical_bayes_prior, center_median

np.random.seed(123)
A = np.random.normal(loc=200, scale=1, size=1000)
B = np.random.normal(loc=100, scale=1, size=1000)

def test_empirical_bayes_prior_returns_floats():
    """
    Testet, dass empirical_bayes_prior ein Tuple aus zwei floats zurückgibt.

    Beispielaufruf (R-Äquivalent):
        a, b = empirical_bayes_prior(rnorm(1000,200), rnorm(1000,100))
    """
    a, b = empirical_bayes_prior(A, B)
    assert isinstance(a, float), "Parameter a sollte float sein"
    assert isinstance(b, float), "Parameter b sollte float sein"


def test_empirical_bayes_prior_enforces_min_sd():
    """
    Testet, dass min_sd korrekt durchgesetzt wird: die resultierende Prior-SD >= min_sd.
    """
    min_sd = 0.5
    a, b = empirical_bayes_prior(A, B, min_sd=min_sd)

    sd = np.sqrt((polygamma(1, a) + polygamma(1, b)) / (np.log(2)**2))
    sd = float(sd)  # Mpf zu float konvertieren für Vergleich

    tol = 1e-6
    assert sd >= min_sd - tol, f"Erwartet SD >= {min_sd}, aber SD = {sd:.6f}"


def test_center_median_median_zero():
    """Nach Anwendung von center_median ist der Median 0."""
    x = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    y = center_median(x)
    assert np.isclose(np.median(y), 0.0)


def test_center_median_values():
    """center_median verschiebt alle Werte korrekt um den Median."""
    x = np.array([10.0, 20.0, 30.0])
    median = np.median(x)
    expected = x - median
    y = center_median(x)
    assert np.allclose(y, expected)

def test_psi_lfc_with_ci_output():
    """Psi_LFC mit credible intervals liefert Tuple (lfc_array, ci_matrix) ohne Fehler und konsistentes CI-Layout."""
    A6 = np.random.normal(loc=50, scale=5, size=200)
    B6 = np.random.normal(loc=50, scale=5, size=200)
    lfc, ci = psi_lfc(A6, B6, cre=True)
    assert isinstance(lfc, np.ndarray)
    assert isinstance(ci, np.ndarray)
    assert lfc.shape == A6.shape
    assert ci.shape == (A6.shape[0], 2)
    lowers = ci[:, 0]
    uppers = ci[:, 1]
    assert np.all(lowers <= uppers), "First Column must be the lower CI."
    assert np.all(lowers <= lfc), "Some LFC-values are below your CI."
    assert np.all(lfc <= uppers), "Some LFC-vaLues are above your CI."

def test_norm_lfc_output_shape_and_type():
    """Prüft, dass norm_lfc ein numpy-Array gleicher Länge zurückgibt."""
    A2 = np.random.normal(loc=5, scale=2, size=50)
    B2 = np.random.normal(loc=3, scale=2, size=50)
    out = norm_lfc(A2, B2)
    assert isinstance(out, np.ndarray)
    assert out.shape == A2.shape

def test_norm_lfc_median_zero():
    """Median des Outputs sollte 0 sein (median-centering)."""
    A3 = np.array([1.0, 2.0, 4.0])
    B3 = np.array([1.0, 1.0, 1.0])
    out = norm_lfc(A3, B3)
    assert np.isclose(np.median(out), 0.0)

def test_norm_lfc_known_values():
    """Überprüfung gegen manuell berechnete Werte für kleinen Input."""
    A4 = np.array([1.0, 2.0, 3.0])
    B4 = np.array([1.0, 1.0, 1.0])
    # raw lfc: log2(A+1)-log2(B+1)
    raw = np.log2(A4 + 1) - np.log2(B4 + 1)
    centered = raw - np.median(raw)
    out = norm_lfc(A4, B4, pseudo=(1.0, 1.0))
    assert np.allclose(out, centered)

def test_psi_lfc_no_ci_output():
    """Psi_LFC ohne credible intervals liefert numpy-Array der richtigen Länge und Median nahe 0."""
    A5 = np.random.normal(loc=50, scale=5, size=100)
    B5 = np.random.normal(loc=50, scale=5, size=100)
    lfc = psi_lfc(A5, B5, cre=False)
    assert isinstance(lfc, np.ndarray)
    assert lfc.shape == A5.shape
    assert np.isclose(np.median(lfc), 0.0, atol=1e-6)

def test_psi_lfc_zero_input():
    """Bei identischen Bedingungen ist LFC-Vektor praktisch Null."""
    A7 = np.full(10, 100.0)
    B7 = np.full(10, 100.0)
    lfc = psi_lfc(A7, B7, cre=False)
    assert np.allclose(lfc, np.zeros(10), atol=1e-6)
    assert np.allclose(lfc, 0.0, atol=1e-6)
    assert np.allclose(lfc, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # EmpiricalBayesPrior(rnorm(1000,200),rnorm(1000,100))
    a, b = empirical_bayes_prior(A, B)
    print(f"Estimated priors: a = {a:.3f}, b = {b:.3f}")

    # CenterMedian(rnorm(1000,200))
    A_centered = center_median(A)
    print(f"Median before: {np.median(A):.6f}")
    print(f"Median after : {np.median(A_centered):.6f}")

    # NormLFC(rnorm(1000, 200), rnorm(1000, 100))
    lfc_values = norm_lfc(A, B)
    print(lfc_values)

    # PsiLFC(rnorm(1000, 200), rnorm(1000, 100))
    lfc_centered = psi_lfc(A, B, cre=False, verbose=True)
    print("Shrunk, median-centered LFC:")
    print(lfc_centered[:10])  # ersten 10 Werte
    # with CI
    lfc_ci, qlfc = psi_lfc(A, B, cre=True, verbose=False)
    print("LFC with 95% credible intervals (first 10 genes):")
    for i in range(10):
        print(f"Gene{i + 1}: LFC={lfc_ci[i]:.3f}, CI=({qlfc[i, 0]:.3f}, {qlfc[i, 1]:.3f})")


    # ------------------------------------------------------------------------------------------

    # sars <- ReadGRAND(system.file("extdata", "sars.tsv.gz", package = "grandR"), design=c(Design$Condition,Design$dur.4sU,Design$Replicate))
    # sars <- subset(sars, Coldata(sars,Design$dur.4sU)==2)
    # sars<-LFC(sars,mode="total",contrasts=GetContrasts(sars,contrast=c("Condition","Mock")))
    # sars<-LFC(sars,mode="new",normalization="total", contrasts=GetContrasts(sars,contrast=c("Condition","Mock")))
    # head(GetAnalysisTable(sars))

    # Ausgabe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)

    sars = gp.read_grand("data/sars_R.tsv", design=("Condition", "dur.4sU", "Replicate"))

    # subset:
    mask = sars.coldata["duration.4sU"] == 2
    sars = sars[:, mask]

    contrasts = sars.get_contrasts(contrast=["Condition", "Mock"])

    sars = sars.compute_lfc(mode_slot="count", contrasts=contrasts)
    sars = sars.compute_lfc(mode_slot="new_count", normalization="total", contrasts=contrasts)

    result = sars.get_analysis_table(with_gene_info=True)
    print(result)

    # ---------------------------------------------------------------------------------------------

    # sars <- ReadGRAND(system.file("extdata", "sars.tsv.gz", package = "grandR"), design=c(Design$Condition,Design$dur.4sU,Design$Replicate))
    # sars <- subset(sars,Coldata(sars,Design$dur.4sU)==2)
    # sars<-PairwiseDESeq2(sars,mode="total", contrasts=GetContrasts(sars,contrast=c("Condition","Mock")))
    # sars<-PairwiseDESeq2(sars,mode="new",normalization="total", contrasts=GetContrasts(sars,contrast=c("Condition","Mock")))
    # head(GetAnalysisTable(sars,column="Q"))

    sars = gp.read_grand("data/sars_R.tsv", design=("Condition", "dur.4sU", "Replicate"))

    mask = sars.coldata["duration.4sU"] == 2
    sars = sars[:, mask]
    # sars = sars._dev_replace(anndata=sars._anndata[:, mask])

    contrasts = sars.get_contrasts(contrast=("Condition", "Mock"))

    sars = sars.pairwise_deseq2(mode_slot="count", contrasts=contrasts)
    sars = sars.pairwise_deseq2(mode_slot="new_count", normalization="total", contrasts=contrasts)

    df = sars.get_analysis_table(columns="Q", with_gene_info=True)
    print(df)