import warnings
from typing import Literal, Union, Sequence
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp




class ModeSlot:
    """
    Used to store a mode slot.

    Modes can either be 'new', 'old' or 'total'.

    For new the data slot value is multiplied by ntr.
    For old the data slot value is multiplied by 1-ntr.

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


class SlotManager:
    def __init__(self, adata: ad.AnnData, is_sparse: bool):
        self._adata = adata
        self._is_sparse = is_sparse

    def slots(self):
        return list(self._adata.layers.keys())

    def slot_data(self):
        def safe_copy(obj):
            return obj.copy() if hasattr(obj, "copy") else obj

        return {k: safe_copy(v) for k, v in self._adata.layers.items()}

    def get_slot(self, slot: str) -> Union[np.ndarray, sp.csr_matrix]:
        if slot not in self._adata.layers:
            raise KeyError(f"Slot '{slot}' not found.")
        return self._adata.layers[slot].copy()

    def with_slot(self, name: str, new_slot: Union[np.ndarray, pd.DataFrame, sp.csr_matrix, list], *,
                  set_to_default=False) -> ad.AnnData:
        new_adata = self._adata.copy()

        if name in new_adata.layers.keys():
            raise ValueError(f"Slot '{name}' already exists. Please choose a different name.")

        def validate_and_convert_new_data(matrix: Union[pd.DataFrame, sp.csr_matrix, np.ndarray]
                                           ) -> Union[np.ndarray, sp.csr_matrix]:
            # If DataFrame → to NumPy
            if isinstance(matrix, pd.DataFrame):
                from Py.utils import _make_unique

                matrix.index = _make_unique(pd.Series(matrix.index))
                matrix = matrix.reindex(index=self._adata.obs.index, columns=self._adata.var.index)

                # Row and column names of the new matrix must be equal to the existing ones.
                for i in range(self._adata.n_obs):
                    if matrix.index[i] != self._adata.obs.index[i]:
                        warnings.warn(f"Row name mismatch for slot '{name}' at index {i}")
                for i in range(self._adata.n_vars):
                    if matrix.columns[i] != self._adata.var["Name"].iloc[i]:
                        warnings.warn(f"Column name mismatch for slot '{name}' at index {i}")

                matrix = matrix.values

            # If sparse, but not csr → to csr
            if self._is_sparse and not isinstance(matrix, sp.csr_matrix):
                matrix = sp.csr_matrix(matrix)

            # If dense, but not ndarray → to ndarray
            if not self._is_sparse and not isinstance(matrix, np.ndarray):
                try:
                    matrix = np.array(matrix)
                except:
                    raise TypeError("Matrix must be ndarray, DataFrame, or scipy sparse matrix")

            return matrix

        new_slot = validate_and_convert_new_data(new_slot)

        new_slots = self.slot_data()
        new_slots[name] = new_slot
        new_adata.layers = new_slots

        if set_to_default:
            new_metadata = self._adata.uns.get('metadata', {}).copy()
            new_metadata['default_slot'] = name
            new_adata.uns['metadata'] = new_metadata

        return new_adata

    def with_dropped_slots(self, slots_to_remove: Sequence[str]) -> ad.AnnData:
        new_adata = self._adata.copy()

        current_slots = self.slots()
        remaining = [s for s in current_slots if s not in slots_to_remove]

        if not remaining:
            raise ValueError("Cannot drop all slots - at least one must remain.")

        new_slots = self.slot_data()
        new_slots = {k: new_slots[k] for k in remaining}
        new_adata.layers = new_slots

        if new_adata.uns["metadata"]["default_slot"] in slots_to_remove:
            new_metadata = new_adata.uns.get('metadata', {}).copy()
            new_metadata['default_slot'] = remaining[0]
            new_adata.uns['metadata'] = new_metadata

        return new_adata

    def check_slot(self, slot: str, *, allow_ntr: bool = True) -> bool:
        if not allow_ntr and slot == "ntr":
            return False
        return slot in self.slots()

    def resolve_mode_slot(self, mode_slot: Union[str, ModeSlot], *, allow_ntr = True) -> Union[np.ndarray, sp.csr_matrix]:
        def parse_mode_slot(mode_slot_unparsed: str) -> ModeSlot:
            """
            Helper function to parse a mode_slot string.
            """
            mode_slot_candidate = mode_slot_unparsed.split("_", 1)

            if len(mode_slot_candidate) == 1:
                return ModeSlot("total", mode_slot_unparsed)

            if len(mode_slot_candidate) != 2:
                raise ValueError(
                    f"Invalid mode_slot: '{mode_slot_unparsed}'. Expected format: '<mode>_<slot>' or ModeSlot('<mode>', '<slot>').")

            mode, slot = mode_slot_candidate

            return ModeSlot(mode, slot)

        def one_minus_csr_matrix(matrix: sp.csr_matrix) -> sp.csr_matrix:
            """
            Helper funktion to compute one minus a sparse matrix.
            """
            ones = sp.csr_matrix(np.ones(matrix.shape), dtype=matrix.dtype)

            return ones - matrix

        # if mode_slot is a string, it gets parsed into a ModeSlot Object
        if isinstance(mode_slot, str):
            if self.check_slot(mode_slot, allow_ntr = allow_ntr):
                return self._adata.layers[mode_slot]
            mode_slot = parse_mode_slot(mode_slot)

        if not self.check_slot(mode_slot.slot, allow_ntr=allow_ntr):
            raise ValueError(f"Slot '{mode_slot.slot}' not found in data slots.")

        slot_total = self._adata.layers[mode_slot.slot].copy()
        ntr = self._adata.layers["ntr"]

        resulting_mode_slot = slot_total

        # The resulting data is computed, depending on the mode
        if mode_slot.mode != "total":
            if self._is_sparse:
                resulting_mode_slot = slot_total.multiply(ntr) if mode_slot.mode == "new" else slot_total.multiply(one_minus_csr_matrix(ntr))
            else:
                resulting_mode_slot = slot_total * ntr if mode_slot.mode == "new" else slot_total * (1 - ntr)

        return resulting_mode_slot