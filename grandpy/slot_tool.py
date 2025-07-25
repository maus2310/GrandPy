import warnings
from collections.abc import Sequence
from typing import Literal, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


class ModeSlot:
    """
    Represents a mode slot.

    Modes can either be 'new', 'old' or 'total'.

    For new the data slot value is multiplied by ntr.
    For old the data slot value is multiplied by 1-ntr.

    Examples
    --------
    Create a ModeSlot object containing counts.

    >>> t_count = ModeSlot("total", "count")
    >>> t_count
    "total_count"

    Now with new counts, meaning the counts are multiplied by ntr.

    >>> n_count = ModeSlot("new", "count")
    >>> n_count
    "new_count"

    These objects can be used in all methods requiring a `mode_slot`.

    >>> gp.plot_scatter(data = ..., mode_slot = n_count)

    Parameters
    ----------
    mode: str
        A mode string. Can either be 'new', 'old' or 'total'.

    slot: str
        An available slot.

    """
    def __init__(self, mode: Literal["new", "n", "old", "o", "total", "t"], slot: str):
        self._set_mode(mode)
        self.slot = slot

    def __str__(self):
        return f"{self.mode}_{self.slot}"

    def _set_mode(self, mode):
        if mode == "n" or mode == "new":
            self.mode = "new"
        elif mode == "o" or mode == "old":
            self.mode = "old"
        elif mode == "t" or mode == "total" or mode == "" or mode is None:
            self.mode = "total"
        else:
            raise ValueError(f"Invalid mode: {mode}. Can either be 'new', 'old' or 'total'.")

def _parse_as_mode_slot(mode_slot: Union[str, ModeSlot]) -> ModeSlot:
    """
    Helper function to parse a mode_slot string.
    """
    if isinstance(mode_slot, ModeSlot):
        return mode_slot

    mode_slot_candidate = mode_slot.split("_", 1)

    if len(mode_slot_candidate) == 1:
        return ModeSlot("total", mode_slot)

    if len(mode_slot_candidate) != 2:
        raise ValueError(
            f"Invalid mode_slot: '{mode_slot}'. Expected format: '<mode>_<slot>' or ModeSlot('<mode>', '<slot>').")

    mode, slot = mode_slot_candidate

    return ModeSlot(mode, slot)


class SlotTool:
    def __init__(self, adata: ad.AnnData, is_sparse: bool):
        self._adata = adata
        self._is_sparse = is_sparse

    def slots(self) -> list[str]:
        return list(self._adata.layers.keys())

    def slot_data(self) -> dict[str, Union[np.ndarray, sp.csr_matrix]]:
        def safe_copy(obj):
            return obj.copy() if hasattr(obj, "copy") else obj

        return {k: safe_copy(v) for k, v in self._adata.layers.items()}

    def get_slot(self, slot: str) -> Union[np.ndarray, sp.csr_matrix]:
        if slot not in self._adata.layers:
            raise KeyError(f"Slot '{slot}' not found.")
        return self._adata.layers[slot].copy()

    def with_slot(self, name: str, new_slot: Union[np.ndarray, pd.DataFrame, sp.csr_matrix, Sequence], *, set_to_default=False) -> tuple[dict, dict]:
        new_slots = self.slot_data()
        # rows and columns are not modified so there is no need to copy
        rows = self._adata.obs.index
        columns = self._adata.var.index

        if name in new_slots.keys():
            warnings.warn(f"Slot '{name}' already exists. It will be overwritten.")

        def validate_and_convert_new_data(matrix: Union[pd.DataFrame, sp.csr_matrix, np.ndarray, Sequence]
                                           ) -> Union[np.ndarray, sp.csr_matrix]:
            # If DataFrame → to NumPy
            if isinstance(matrix, pd.DataFrame):
                from grandpy.utils import _make_unique

                matrix.index = _make_unique(pd.Series(matrix.index))
                matrix = matrix.reindex(index=rows, columns=columns)

                matrix = matrix.values

            # If sparse, but not csr → to csr
            if self._is_sparse and not isinstance(matrix, sp.csr_matrix):
                matrix = sp.csr_matrix(matrix)

            # If dense, but not ndarray → to ndarray
            if not self._is_sparse and not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix)

            return matrix

        new_slot = validate_and_convert_new_data(new_slot)

        new_slots[name] = new_slot

        new_metadata = self._adata.uns.get('metadata', {}).copy()
        if set_to_default:
            new_metadata['default_slot'] = name

        return new_slots, new_metadata

    def with_dropped_slots(self,slots_to_remove: Sequence[str]) -> tuple[dict, dict]:
        current_slots = self.slot_data()

        remaining = [s for s in current_slots if s not in slots_to_remove]

        if not remaining:
            raise ValueError("Cannot drop all slots - at least one must remain.")

        new_slots = {k: current_slots[k] for k in remaining}

        new_metadata = self._adata.uns.get('metadata', {}).copy()
        if self._adata.uns["metadata"]["default_slot"] in slots_to_remove:
            new_metadata['default_slot'] = remaining[0]

        return new_slots, new_metadata

    def check_slot(self, slot: Union[str, ModeSlot], *, allow_ntr: bool = True) -> bool:
        if isinstance(slot, ModeSlot):
            slot = slot.slot
        if not allow_ntr and slot == "ntr":
            return False
        return slot in self.slots()

    def resolve_mode_slot(self, mode_slot: Union[str, ModeSlot], *, allow_ntr = True, ntr_nan = False) -> Union[np.ndarray, sp.csr_matrix]:
        def one_minus_csr_matrix(matrix: sp.csr_matrix) -> sp.csr_matrix:
            """
            Helper funktion to compute one minus a sparse matrix.
            """
            ones = sp.csr_matrix(np.ones(matrix.shape), dtype=matrix.dtype)

            return ones - matrix

        # if mode_slot is a string, it gets parsed into a ModeSlot Object
        mode_slot = _parse_as_mode_slot(mode_slot)

        if not self.check_slot(mode_slot.slot, allow_ntr=allow_ntr):
            raise ValueError(f"Slot '{mode_slot.slot}' not found in data slots.")

        slot_total = self._adata.layers[mode_slot.slot].copy()
        ntr = self._adata.layers.get("ntr")

        if ntr_nan:
            boolean_mask = self._adata.var["no4sU"].values
            ntr[:, boolean_mask] = np.nan

        resulting_mode_slot = slot_total

        # The resulting data is computed, depending on the mode
        if mode_slot.mode != "total":
            if self._is_sparse:
                resulting_mode_slot = slot_total.multiply(ntr) if mode_slot.mode == "new" else slot_total.multiply(one_minus_csr_matrix(ntr))
            else:
                resulting_mode_slot = slot_total * ntr if mode_slot.mode == "new" else slot_total * (1 - ntr)

        return resulting_mode_slot

    def with_ntr_slot(self, as_ntr: str, save_ntr_as: str = None) -> dict[str, Union[np.ndarray, sp.csr_matrix]]:
        new_slots = self.slot_data()

        if save_ntr_as is not None:
            new_slots[save_ntr_as] = new_slots["ntr"]

        new_slots["ntr"] = new_slots[as_ntr]

        return new_slots