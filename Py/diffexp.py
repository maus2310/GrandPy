import pandas as pd
import numpy as np
from typing import Union, TYPE_CHECKING, Optional, Literal, Sequence
from itertools import combinations

try:
    from pydeseq2.ds import DeseqDataSet, DeseqStats
except ImportError:
    DeseqDataSet = DeseqStats = None

from Py.utils import _ensure_list

if TYPE_CHECKING:
    from Py.grandPy import GrandPy


def _get_summary_matrix(
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

def _get_contrasts(
        data: "GrandPy",
        contrast: list = ["Condition"],
        columns: Union[Sequence[str], str] = None,
        group: Union[Sequence[str], str] = None,
        name_format: str = None,
        no4sU: bool = False,
        ) -> Union["GrandPy", pd.DataFrame]:

    our_coldata = data.coldata

    if len(contrast) not in [1, 2, 3]:
        raise ValueError("Contrast must be of length 1, 2, or 3.")

    if contrast[0] not in our_coldata.columns:
        raise ValueError(f"Column {contrast[0]} not found in coldata.")
    col = contrast[0]

    # Untermenge der Daten, falls columns gesetzt ist
    use_mask = np.ones(len(our_coldata), dtype=bool)
    if columns is not None:
        use_mask = our_coldata.index.isin(columns) if isinstance(columns, list) else columns

    if name_format is None:
        name_format = "$A vs $B" if group is None else "$A vs $B.$GRP"

    if not no4sU and "no4sU" in our_coldata.columns:
        use_mask &= ~our_coldata["no4sU"].fillna(False)

    def make_name(a, b, grp=""):
        return (name_format
                .replace("$A", str(a))
                .replace("$B", str(b))
                .replace("$COL", col)
                .replace("$GRP", grp))

    def make_vector(a, b, use):
        re = np.zeros(len(our_coldata))
        re[(our_coldata[col] == a) & use] = 1
        re[(our_coldata[col] == b) & use] = -1
        return re

    def contrast_df_for_level_pairs(level_pairs, group_name=""):
        df = {}
        for a, b in level_pairs:
            vec = make_vector(a, b, use_mask)
            name = make_name(a, b, group_name)
            df[name] = vec
        return pd.DataFrame(df, index=data.coldata.index)

    # Hauptlogik für Kontrastgenerierung
    match len(contrast):
        case 1:
            levels = our_coldata.loc[use_mask, col].dropna().unique().tolist()
            level_pairs = list(combinations(levels, 2))
            contrast_df = contrast_df_for_level_pairs(level_pairs)
        case 2:
            levels = our_coldata.loc[use_mask, col].dropna().unique().tolist()
            other_levels = [l for l in levels if l != contrast[1]]
            contrast_df = contrast_df_for_level_pairs([(l, contrast[1]) for l in other_levels])
        case 3:
            contrast_df = contrast_df_for_level_pairs([(contrast[1], contrast[2])])
        case _: raise ValueError("Illegal contrasts (either a name from design (all pairwise comparisons), a name and a reference level (all comparisons vs. the reference), or a name and two levels (exactly this comparison))")


    # Falls Gruppierung gewünscht → innerhalb jeder Gruppe Kontraste berechnen
    if group is not None:
        all_dfs = []
        for grp_val in data.coldata[group].dropna().unique():
            group_mask = (data.coldata[group] == grp_val)
            use_mask_group = use_mask & group_mask
            if len(our_coldata.loc[use_mask_group]) < 2:
                continue
            levels = our_coldata.loc[use_mask_group, col].dropna().unique().tolist()
            if len(levels) < 2:
                continue
            level_pairs = list(combinations(levels, 2)) if len(contrast) == 1 else \
                [(l, contrast[1]) for l in levels if l != contrast[1]] if len(contrast) == 2 else \
                    [(contrast[1], contrast[2])]
            df = contrast_df_for_level_pairs(level_pairs, group_name=str(grp_val))
            all_dfs.append(df)
        contrast_df = pd.concat(all_dfs, axis=1)

    # Entferne Spalten mit nur 0 (irrelevant)
    contrast_df = contrast_df.loc[:, ~(contrast_df == 0).all(axis=0)]

    # Entferne Kontraste mit nur +1 oder nur -1 (nicht sinnvoll)
    remove_mask = ((contrast_df >= 0).all(axis=0)) | ((contrast_df <= 0).all(axis=0))
    if remove_mask.any():
        removed_cols = contrast_df.columns[remove_mask].tolist()
        print(f"Removed uninformative contrasts: {', '.join(removed_cols)}")
        contrast_df = contrast_df.loc[:, ~remove_mask]

    return contrast_df