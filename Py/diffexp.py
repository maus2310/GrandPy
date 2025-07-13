import pandas as pd
import numpy as np
from typing import Union, TYPE_CHECKING, Optional, Literal

try:
    from pydeseq2.ds import DeseqDataSet, DeseqStats
except ImportError:
    DeseqDataSet = DeseqStats = None

from Py.utils import _ensure_list

if TYPE_CHECKING:
    from Py.grandPy import GrandPy


def get_summary_matrix(
        data: "GrandPy",
        no4sU: bool = False,
        columns: Union[None, str, list[str]] = None,
        average: bool = True
) -> pd.DataFrame:
    coldata = data.coldata
    sample_names = coldata.index.tolist()

    if "Condition" not in coldata.columns:
        raise ValueError("Object does not have 'Condition' information!")

    if columns is None:
        columns = sample_names
    else:
        columns = _ensure_list(columns)
        columns = [c for c in columns if c in sample_names]

    # Exclude 4sU-marked samples if requested
    if not no4sU:
        no4su_samples = coldata.index[coldata["no4sU"]]
        columns = list(set(columns) - set(no4su_samples))

    # Mapping: sample_name → condition
    condition_series = coldata.loc[columns, "Condition"]

    # Create indicator matrix
    unique_conditions = condition_series.unique()
    matrix = pd.DataFrame(
        {
            cond: (condition_series == cond).astype(float)
            for cond in unique_conditions
        },
        index=condition_series.index
    )

    # Ensure all samples included, fill 0 if not matched
    all_samples = pd.Index(sample_names)
    matrix = matrix.reindex(index=all_samples).fillna(0)

    # Drop columns with all zeros
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]

    if average:
        matrix = matrix.div(matrix.sum(axis=0), axis=1).fillna(0)

    return matrix

# TODO: Compute M bisher nicht umgesetzt & Werte bei method="deseq2" fixen & Doc-String ofc
# für Pairwise(): compute_lfc(grand_obj, method="simple", contrasts=[...]) - Rückgabe ist dict
# mit allen gewünschten Vergleichen, LFC-Werte = float
# für PairwiseDeSeq2(): compute_lfc(grand_obj, method="deseq2", contrasts=[...], n_cpus=...)
# hier ist zusätzlich die Q-Spalte für adjusted p-Werte enthalten
def compute_lfc(data,
                contrasts: Optional[list[tuple[str, str]]] = None,
                method: Literal["deseq2", "simple"] = "deseq2",
                slot: Optional[str] = None,
                condition_col: str = "Condition",
                design: Optional[Union[str, list[str]]] = None,
                n_cpus: int = 1) -> dict[str, pd.DataFrame]:
    """
    - contrasts ist hier eine Liste von Tupeln (base, compare)
    - method wählt zwischen deseq2 (pydeseq2 wird für jeden einzelnen contrast genutzt) &
                            simple (Mittelwert-LFC ohne p-Werte)
    - output ist ein Dictionary, in dem jeder key z.B. "SARS vs. Mock" ist und der Wert
      das zugehörige DataFrame mit Spalten["LFC", "Q" (optional), ...]
    """

    # Table und meta vorbereiten
    slot = slot or data.default_slot
    table = data.get_table(mode_slots=slot)  # Genes x Samples
    meta = data.coldata.copy()

    # DESeq2 initialisieren, wenn angegeben bei method
    if method == "deseq2":
        if DeseqDataSet is None:
            raise ImportError("pydeseq2 not found. Install pydeseq2 package.")
        # Design-Faktoren bestimmen
        if design is not None:
            design_factors = design
        elif hasattr(data, 'design') and data.design is not None:
            design_factors = data.design
        else:
            design_factors = condition_col
        dds = DeseqDataSet(
            counts=table.T.round().astype(int),
            metadata=meta,
            design_factors=design_factors)
        dds.deseq2()

    # Kontraste bilden, wenn nicht angegeben
    if contrasts is None:
        levels = meta[condition_col].unique().tolist()
        contrasts = [(base, comp) for base in levels for comp in levels if comp != base]

    results: dict[str, pd.DataFrame] = {}
    for base, compare in contrasts:
        key = f"{compare}_vs_{base}"
        col_lfc = f"total.{base} vs {compare}.LFC"

        if method == "deseq2":
            stats = DeseqStats(
                dds,
                contrast=[condition_col, compare, base],
                n_cpus=n_cpus)

            stats.summary()
            all_coefs = list(stats.LFC.columns)
            test_coefs = [c for c in all_coefs if "Intercept" not in c]
            coef_to_shrink = test_coefs[0] if test_coefs else all_coefs[0]
            stats.lfc_shrink(coeff=coef_to_shrink)
            df_raw = stats.results_df

            if df_raw is None:
                raise RuntimeError("pydeseq2 returned no results.")
            df = (
                df_raw
                .rename(columns={"log2FoldChange": col_lfc, "padj": "Q"})
                .loc[:, [col_lfc, "Q"]])
            df.index.name = "Symbol"
            results[key] = df
        else:
            if "no4sU" in meta.columns:
                keep = ~meta["no4sU"]
                tbl = table.loc[:, keep]
                cond = meta.loc[keep, condition_col]
            else:
                tbl = table
                cond = meta[condition_col]

            sums = tbl.groupby(cond, axis=1).sum()
            missing = [x for x in (base, compare) if x not in sums.columns]
            if missing:
                raise KeyError(f"Condition(s) {missing} not found; available: {sums.columns.tolist()}")

            lfc = np.log2((sums[compare] + 1) / (sums[base] + 1))
            df = lfc.to_frame(name=col_lfc)
            df.index.name = "Symbol"
            results[key] = df

    return results