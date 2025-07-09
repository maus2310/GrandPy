from scipy.stats import beta
import numpy as np
from math import log
import pandas as pd
from typing import TYPE_CHECKING, Union, Sequence, Optional
import warnings

if TYPE_CHECKING:
    from Py.grandPy import GrandPy
    from Py.slot_tool import ModeSlot

def _comp_hl(p, time=1):
    """Computes half-life from NTR-value p and time t"""
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = log(2) / (-1.0 / time * np.log(1 - p))
        hl = np.where(np.isfinite(hl), hl, np.nan)
    return hl

def _comp_tpm(
    count_matrix: np.ndarray,
    lengths: np.ndarray,
    subset: Union[Sequence[int], int, None] = None
) -> np.ndarray:
    """
    Computes the TPM (transcripts per million) from the given slot and gene lengths.

    Parameters
    ----------
    count_matrix : np.ndarray
        Count matrix

    lengths : np.ndarray
        Lengths of the genes

    subset : int, list[int], np.ndarray, optional
        Optionaler Index oder Liste von Indizes von Genen zur Berechnung der Skalenfaktoren

    Returns
    -------
    np.ndarray
        TPM-normalisierte Matrix (shape: Gene × Samples)
    """

    # Verhindere Division durch Null
    lengths = lengths.copy()
    zero_len = lengths == 0
    lengths[zero_len] = 1

    # RPK berechnen (Reads Per Kilobase)
    reads_per_kilo = count_matrix / (np.atleast_2d(lengths / 1000).transpose())

    # Subset in ein array umwandeln, falls notwendig
    if subset is not None:
        subset = np.atleast_1d(subset)
        scale = np.nansum(reads_per_kilo[subset, :], axis=0) / 1e6
    else:
        scale = np.nansum(reads_per_kilo, axis=0) / 1e6

    # TPM berechnen
    tpm = reads_per_kilo / scale

    # NaNs für Gene mit Länge 0 setzen
    tpm[zero_len, :] = np.nan

    return tpm


def _comp_fpkm(
        count_matrix: np.ndarray,
        lengths: np.ndarray,
        subset: Union[Sequence[int], np.ndarray, None] = None
) -> np.ndarray:
    """
    Berechnet FPKM aus einer Count-Matrix und Transkriptlängen.
    Entspricht exakt der R-Funktion comp.fpkm.

    Parameters
    ----------
    count_matrix : np.ndarray
        Count-Matrix (Gene × Samples)
    lengths : np.ndarray
        Längen der Transkripte (1D, für alle Gene in cmat)
    subset : Optional[Sequence[int]]
        Welche Gene für die Skalierung (RPM) verwendet werden sollen.

    Returns
    -------
    np.ndarray
        FPKM-Matrix (Gene × Samples)
    """
    count_matrix = np.asarray(count_matrix)
    lengths = np.asarray(lengths)

    # Subset verwenden, falls angegeben
    if subset is not None:
        subset = np.atleast_1d(subset)
        scale = np.nansum(count_matrix[subset, :], axis=0) / 1e6
    else:
        scale = np.nansum(count_matrix, axis=0) / 1e6

    scale[scale == 0] = np.nan  # Verhindert Division durch 0

    # RPM: jede Spalte durch zugehörigen Skalierungsfaktor teilen
    rpm = count_matrix / scale  # Broadcasting funktioniert hier direkt

    # Behandlung von Länge = 0
    zerolen = lengths == 0
    lengths = lengths.copy()
    lengths[zerolen] = 1  # temporär auf 1 setzen um Division durch 0 zu vermeiden

    # FPKM berechnen
    fpkm = rpm / (lengths[:, np.newaxis] / 1000)
    fpkm[zerolen, :] = np.nan  # ursprüngliche Länge 0 → NA setzen

    return fpkm

def _comp_rpm(
    cmat: np.ndarray,
    subset: Union[Sequence[int], np.ndarray, None] = None,
    factor: float = 1e6
) -> np.ndarray:
    """
    Berechnet RPM (Reads per Million) aus einer Count-Matrix.

    Parameters
    ----------
    cmat : np.ndarray
        Zählmatrix (Gene × Samples)
    subset : Liste von Indizes, optional
        Nur diese Gene werden zur Skalierung verwendet.
    factor : float
        Skalenfaktor (Standard: 1e6)

    Returns
    -------
    np.ndarray
        RPM-normalisierte Matrix
    """
    cmat = np.asarray(cmat)

    if subset is not None:
        subset = np.atleast_1d(subset)
        scale = np.nansum(cmat[subset, :], axis=0) / factor
    else:
        scale = np.nansum(cmat, axis=0) / factor

    scale[scale == 0] = np.nan  # Verhindere Division durch 0

    rpm = cmat / scale  # Broadcasting über Spalten

    return rpm

def compute_ntr_posterior_quantile(data: "GrandPy", quantile: float, name: str) -> "GrandPy":
    """
    Compute a posterior quantile of the NTR beta distribution and store as a new slot.

    Parameters:
    - data: GrandPy object
    - quantile: float in [0,1], quantile to compute
    - name: str, name of the new slot
    """
    alpha = data.get_matrix(mode_slot="alpha")
    beta_ = data.get_matrix(mode_slot="beta")

    q = beta.ppf(quantile, alpha, beta_)
    return data.with_slot(name, q)

def compute_ntr_posterior_lower(data: "GrandPy", ci_size: float = 0.95, name: str = "lower") -> "GrandPy":
    """
    Compute lower bound of the NTR credible interval.
    """
    quantile = (1 - ci_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def compute_ntr_posterior_upper(data: "GrandPy", ci_size: float = 0.95, name: str = "upper") -> "GrandPy":
    """
    Compute upper bound of the NTR credible interval.
    """
    quantile = 1 - (1 - ci_size) / 2
    return compute_ntr_posterior_quantile(data, quantile, name)

def _compute_ntr_ci(data: "GrandPy", ci_size: float = 0.95, name_lower: str = "lower", name_upper: str = "upper") -> "GrandPy":
    """
    Compute both lower and upper bounds of the credible interval.
    """
    data = compute_ntr_posterior_lower(data, ci_size, name_lower)
    data = compute_ntr_posterior_upper(data, ci_size, name_upper)
    return data

# Beispielaufruf: sars = sars.compute_steady_state_half_lives(compute_ci=True)
#                 print(sars.get_analysis_table())
def _compute_steady_state_half_lives(
    data: "GrandPy",
    time = None,
    name="HL",
    columns=None,
    max_hl=48.0,
    ci_size=0.95,
    compute_ci=False,
    as_analysis=False
) -> "GrandPy":
    if time is None:
        time = data.coldata["Time"]

    if isinstance(time, str):
        time = data.coldata[time]

    ntrs = data.get_table(mode_slots="ntr", name_genes_by="Symbol")

    if np.isscalar(time):
        time = pd.Series([time] * ntrs.shape[1], index=ntrs.columns)
    else:
        time = pd.Series(time, index=ntrs.columns)

    if columns is None:
        selected_columns = list(ntrs.columns)
    elif isinstance(columns, str):
        selected_columns = list(data.coldata.query(columns).index)
    elif isinstance(columns, list):
        selected_columns = columns
    else:
        raise ValueError("Unsupported column specification for `columns`.")

    time = time[selected_columns]
    if len(selected_columns) != ntrs.shape[1]:
        as_analysis = True

    if compute_ci:
        as_analysis = True

        existing_slots = data.slots
        needs_lower = "lower" not in existing_slots
        needs_upper = "upper" not in existing_slots

        if needs_lower or needs_upper:
            data = data.compute_ntr_ci(ci_size=ci_size)

        lower = data.get_table(mode_slots="lower", name_genes_by="Symbol")
        upper = data.get_table(mode_slots="upper", name_genes_by="Symbol")

        frames = []
        for col in selected_columns:
            col_time = time[col]
            lower_hl = np.minimum(_comp_hl(upper[col].values, col_time), max_hl)
            map_hl   = np.minimum(_comp_hl(ntrs[col].values, col_time), max_hl)
            upper_hl = np.minimum(_comp_hl(lower[col].values, col_time), max_hl)

            df = pd.DataFrame({
                f"{name}.Half-life.lower.{col}": lower_hl,
                f"{name}.Half-life.MAP.{col}": map_hl,
                f"{name}.Half-life.upper.{col}": upper_hl
            }, index=ntrs.index)
            frames.append(df)

        hls = pd.concat(frames, axis=1)

    else:
        hls = pd.DataFrame({
            col: np.minimum(_comp_hl(ntrs[col].values, time[col]), max_hl)
            for col in selected_columns
        }, index=ntrs.index)
    if as_analysis:
        return data.with_analysis(name, hls)
    else:
        return data.with_slot(name, hls)


def _filter_genes(
    data,
    mode_slot: Union[str, "ModeSlot"] = None,
    min_expression: int = 100,
    min_columns: int = None,
    min_condition: int = None,
    keep: Union[str, int, Sequence[Union[int, str]]] = None,
    use: Union[str, int, Sequence[Union[int, str, bool]]] = None,
    return_genes: bool = False
) -> Union["GrandPy", list[str]]:
    if use is not None and keep is not None:
        raise ValueError("Do not specify both use and keep!")

    if mode_slot is None:
        mode_slot = data.default_slot

    if not data._check_slot(mode_slot):
        raise ValueError(f"Slot '{mode_slot}' unknown!")

    if use is None:
        aggregation_matrix = None
        if min_condition is not None:
            aggregation_matrix = data.get_summary_matrix(no4sU=True, average=False)

        matrix = data.get_table(mode_slots=mode_slot, summarize=aggregation_matrix)

        if min_columns is None:
            min_columns = min_condition if min_condition is not None else matrix.shape[1] / 2

        use_mask = (matrix >= min_expression).sum(axis=1) >= min_columns

        if keep is not None:
            keep_idx = data.get_index(keep)
            use_mask |= np.isin(np.arange(matrix.shape[0]), keep_idx)

        use = use_mask

    gene_idx = data.get_index(use)

    if return_genes:
        return data.get_genes(gene_idx)

    # apply adjusts slots according to changes to gene_info
    return data._apply(
        function_gene_info=lambda t: t.iloc[gene_idx, :]
    )

def _normalize(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "norm",
    slot: str = "count",
    set_to_default: bool = True,
    size_factors: np.ndarray = None,
    return_size_factors: bool = False
) -> Union["GrandPy", np.ndarray]:
    """
    DESeq2-ähnliche Normalisierung einer Slot-Matrix durch Size Factors.

    Parameters
    ----------
    data : GrandPy
        Das GrandPy-Objekt.
    genes : list[str] oder bool-Maske, optional
        Gene zur Berechnung der Size Factors. Default: alle.
    name : str
        Name des neuen Slots.
    slot : str
        Slot zur Normalisierung, z.B. "count".
    set_to_default : bool
        Ob der neue Slot als default gesetzt wird.
    size_factors : np.ndarray, optional
        Falls gegeben, verwende diese Size Factors direkt.
    return_size_factors : bool
        Wenn True, gib die Size Factors zurück.

    Returns
    -------
    GrandPy oder np.ndarray
        Normalisiertes GrandPy-Objekt oder Size Factors.
    """

    matrix_for_size = data.get_matrix(slot, genes=genes)

    if size_factors is None:
        if matrix_for_size.ndim == 1:
            matrix_for_size = matrix_for_size[np.newaxis, :]

        jitter = 1e-8
        safe_matrix = np.where(matrix_for_size == 0, jitter, matrix_for_size)
        log_mat = np.log(safe_matrix)
        log_geomeans = np.mean(log_mat, axis=1)

        # Größe des Arrays
        n_cols = safe_matrix.shape[1]
        size_factors = np.zeros(n_cols)

        for i in range(n_cols):
            counts = safe_matrix[:, i]
            valid = np.isfinite(log_geomeans) & (counts > 0)
            diffs = np.log(counts[valid]) - log_geomeans[valid]
            size_factors[i] = np.exp(np.median(diffs))

    if return_size_factors:
        return size_factors

    matrix_for_normalization = data.get_matrix(slot)

    normalized_matrix = matrix_for_normalization / size_factors

    return data.with_slot(name, normalized_matrix, set_to_default=set_to_default)

def _normalize_fpkm(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "fpkm",
    slot: str = "count",
    set_to_default: bool = True,
    total_len: np.ndarray = None
) -> "GrandPy":
    """
    FPKM-Normalisierung eines GrandPy-Objekts, analog zu DESeq2::NormalizeFPKM in R.

    Parameters
    ----------
    data : GrandPy
        Das GrandPy-Objekt mit Count-Daten.
    genes : Sequenz von Genen oder bool-Maske
        Gene, die für die Skalierung verwendet werden sollen (nicht zur Ausgabe!).
    name : str
        Name des neuen Slots (z. B. "fpkm").
    slot : str
        Welcher Slot verwendet werden soll (z. B. "count").
    set_to_default : bool
        Ob der neue Slot als Default gesetzt wird.
    total_len : np.ndarray
        Transkriptlängen aller Gene (Standard: data.gene_info["Length"]).

    Returns
    -------
    GrandPy
        Das aktualisierte Objekt mit dem neuen Slot.
    """

    # Hole Transkriptlängen
    if total_len is None:
        total_len = np.asarray(data.gene_info["Length"])

    # Hole die vollständige Matrix (alle Gene × Samples)
    mat = data.get_matrix(slot)

    # Konvertiere genes (z. B. Namen, bools oder Indizes) zu numerischem Index
    gene_indices = data.get_index(genes=genes)  # kann auch None sein

    # Berechne FPKM-Werte
    fpkm = _comp_fpkm(count_matrix=mat, lengths=total_len, subset=gene_indices)

    # Speichere als neuen Slot
    return data.with_slot(name, fpkm, set_to_default=set_to_default)

def _normalize_tpm(
        data: "GrandPy",
        genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
        name: str = "tpm",
        slot: str = "count",
        set_to_default: bool = True,
        total_len: np.ndarray = None
) -> "GrandPy":
    """
    TPM-normalization of the GrandPy-dataset.

    Parameters
    ----------
    data: GrandPy
        a GrandPy-object
    genes: sequence of genes or indices
        genes or indices used for the tpm-normalization
    name: str
        name of the slot that the tpm-normalization will create
    slot: str
        the slot with the data to tpm-normalize
    set_to_default: bool
        whether to set the new slot as the default or not
    total_len: np.ndarray, optional
        array with the transcript length of the genes

    Returns
    -------
    GrandPy
        a new GrandPy-object with the appended data
    """
    # Hole Genlängen
    if total_len is None:
        total_len = np.asarray(data.gene_info["Length"])

    # Hole vollständige Zählmatrix
    count_matrix = data.get_matrix(slot)  # shape: (n_genes, n_samples)

    # Berechne, welche Gene für das Scaling benutzt werden sollen
    if genes is not None:
        subset_indices = data.get_index(genes=genes)
    else:
        subset_indices = None

    # Berechne TPM normalisiert auf subset
    tpm_matrix = _comp_tpm(count_matrix, total_len, subset=subset_indices)

    # Optional: falls nur bestimmte Gene als Ergebnis gewünscht sind
    # z. B. Rückgabe nur für subset, nicht das ganze TPM
    # → das hier NICHT tun, wenn gesamte Matrix gespeichert werden soll
    # tpm_matrix = tpm_matrix[subset_indices, :]  # optional

    return data.with_slot(name, tpm_matrix, set_to_default=set_to_default)

def _normalize_rpm(
    data: "GrandPy",
    genes: Union[str, int, Sequence[Union[str, int, bool]]] = None,
    name: str = "rpm",
    slot: str = "count",
    set_to_default: bool = True,
    factor: float = 1e6
) -> "GrandPy":
    """
    RPM-Normalisierung eines GrandPy-Objekts.

    Parameters
    ----------
    data : GrandPy
        Das GrandPy-Objekt mit Count-Daten.
    genes : Liste von Genen oder Indizes
        Gene, die zur Skalierung verwendet werden (nicht Ausgabe!).
    name : str
        Name des neuen Slots (z. B. "rpm").
    slot : str
        Welcher Slot verwendet werden soll (z. B. "count").
    set_to_default : bool
        Ob der neue Slot als Standard gesetzt wird.
    factor : float
        Skalierungsfaktor (Standard: 1e6)

    Returns
    -------
    GrandPy
        Das aktualisierte Objekt mit dem neuen RPM-Slot.
    """

    # Zählmatrix laden (alle Gene)
    mat = data.get_matrix(slot)

    # Index der Gene, die zur Skalierung verwendet werden sollen
    gene_indices = data.get_index(genes=genes) if genes is not None else None

    # Berechne RPM
    rpm = _comp_rpm(cmat=mat, subset=gene_indices, factor=factor)

    # Füge neuen Slot ein
    return data.with_slot(name, rpm, set_to_default=set_to_default)

# def _normalize_baseline(data: "GrandPy",
#                         reference = None,
#                         name: str = "baseline",
#                         slot: str = None,
#                         set_to_default: bool = False,
#                         LFC_func = None) -> "GrandPy":
#
#     if reference is None: reference = data.get_references(reference="condition", )
#     matrix_for_baseline = data.get_matrix(mode_slot=slot)



def _compute_absolute(
        data: "GrandPy",
        dilution: float = 4e4,
        volume: float = 10.0,
        slot: str = "tpm",
        name: str = "absolute"
) -> "GrandPy":
    """
    Schätzt absolute Molekülzahlen aus TPM-Daten mithilfe von ERCC-Spike-ins.
    Annäherung an monocle::relative2abs auf Basis von Spike-Summen.
    """

    # Hole TPM-Matrix (angenommen: numpy-Array mit shape (n_genes, n_cells))
    mat = data.get_matrix(slot)
    gene_types = data.gene_info["Type"]
    is_ercc = (gene_types == "ERCC")

    if not np.any(is_ercc):
        raise ValueError("Keine ERCC-Gene im Datensatz gefunden.")

    ercc_mat = mat[is_ercc, :]  # TPMs nur der ERCCs
    ercc_tpm_sum = np.nansum(ercc_mat, axis=0)  # Summe je Zelle

    # Ersetze Nullen durch np.nan, um Division durch 0 zu vermeiden
    ercc_tpm_sum[ercc_tpm_sum == 0] = np.nan

    # Berechne Skalierungsfaktor je Zelle
    scaling_factor = (dilution * volume) / ercc_tpm_sum  # shape: (n_cells,)

    # Multipliziere jede Spalte mit passendem Skalierungsfaktor
    absolute = mat * scaling_factor[np.newaxis, :]  # Broadcasting erzwingen

    # Setze absolute = 0, wo TPM = 0 war
    absolute[mat == 0] = 0

    # Ergebnis als neuen Slot speichern
    return data.with_slot(name, absolute)

def _compute_expression_percentage(
    data,
    name: str,
    genes: Union[str, Sequence[str]] = None,
    slot: str = None,
    genes_total: Union[str, Sequence[str]] = None,
    slot_total: str = None,
    float_to_percent: bool = True,
):
    """
    Compute the percentage of expression for a set of genes per column and
    store it in the coldata (sample metadata).

    Parameters
    ----------
    data : GrandPy
        The GrandPy object containing the expression data.

    name : str
        Name of the new column in coldata where the percentage will be stored.

    genes : list, optional
        List of genes for which to compute expression fraction.
        Defaults to all genes.

    slot : str, optional
        Data slot to use for numerator values. Defaults to data.default_slot.

    genes_total : list, optional
        List of genes to use for total expression. Defaults to all genes.

    slot_total : str, optional
        Data slot to use for total expression. Defaults to slot.

    percent_to_float : bool, default=True
        If True, percentages are scaled to [0, 100].

    Returns
    -------
    GrandPy
        The modified GrandPy object with a new column in coldata.
    """

    numerator = data.get_matrix(mode_slot=slot, genes=genes).sum(axis=0)
    denominator = data.get_matrix(mode_slot=slot_total, genes=genes_total).sum(axis=0)

    percentage = numerator / denominator
    if float_to_percent:
        percentage *= 100

    data = data.with_coldata(column=name, value=percentage)
    return data

def _filter_genes(
    data: "GrandPy",
    mode_slot: Union[str, "ModeSlot"] = None,
    *,
    min_expression: Number = 100,
    min_columns: int = None,
    min_condition: int = None,
    keep: Union[str, int, Sequence[Union[int, str]]] = None,
    use: Union[str, int, Sequence[Union[int, str, bool]]] = None,
    return_genes: bool = False
) -> Union["GrandPy", list[int]]:
    """
    Filter genes based on expression/value thresholds.

    Parameters
    ----------
    mode_slot : str or ModeSlot, optional
        Which data slot to use.

    min_expression : Number, default 100
        Minimum value threshold to consider a gene expressed.

    min_columns : int, optional
        Minimum number of samples the gene must meet `min_expression` in.
        Defaults to half the number of columns in the matrix.
        Will be ignored if `min_condition` is provided

    min_condition : int, optional
        Overrides `min_columns` if set.

    keep : str or int or Sequence[str or int], optional
        Genes to force-keep, regardless of threshold filtering.

    use : str or int or Sequence[bool or int or str], optional
        Only these genes will be kept if provided (boolean mask, indices, or names).
        Filtering will not be applied to them. (Basically just subsetting)

    return_genes : bool, default False
        If True, return the list of selected gene indices instead of a filtered GrandPy object.

    Returns
    -------
    list[str] or GrandPy
    """
    return filter_genes(data, mode_slot, min_expression=min_expression, min_columns=min_columns, min_condition=min_condition, use=use, keep=keep, return_genes=return_genes)
