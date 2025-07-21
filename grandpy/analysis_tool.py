import re
import warnings
import anndata as ad
import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Sequence

from .utils import _ensure_list, _make_unique, _reindex_by_index_name


class AnalysisTool:
    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    def analyses(self):
        analyses = self._adata.uns["analyses"]
        return list(analyses.keys())

    def get_analyses(
            self,
            pattern: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            regex: bool = True,
            description: bool = False
    ) -> Union[list[str], dict[str, list[str]]]:
        available_analyses = self.analyses()

        if pattern is None:
            if description:
                return {name: df.columns.tolist() for name, df in self._adata.uns["analyses"].items()}
            else:
                return available_analyses

        pattern = _ensure_list(pattern)

        def check_analyses(pattern, available_analyses, regex) -> list[bool]:
            """
            Helper function to check if the given names or regex pattern match the available analyses.
            """
            if regex:
                return [any((re.search(pat, analysis)) for pat in pattern) for analysis in available_analyses]
            else:
                if all(isinstance(pat, (int, np.integer)) for pat in pattern):
                    mask = np.zeros(len(available_analyses), dtype=bool)
                    mask[pattern] = True
                    return mask.tolist()

                elif(all(isinstance(pat, str) for pat in pattern)):
                    return [analysis in pattern for analysis in available_analyses]

                elif(all(isinstance(pat, (bool, np.bool)) for pat in pattern)):
                    return pattern

                else:
                    raise TypeError(f"pattern must be either int, str, bool, or a Sequence of those, not {pattern}")

        checks = check_analyses(pattern, available_analyses, regex)

        result = np.array(available_analyses)[checks].tolist()

        if description:
            all_descriptions = {name: df.columns.tolist() for name, df in self._adata.uns["analyses"].items()}
            result = {k: v for (k, v), m in zip(all_descriptions.items(), checks) if m}

        return result

    def with_analysis(self, name: str, table: pd.DataFrame, by: str = None) -> dict[str, pd.DataFrame]:
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"'table' has to be a pd.DataFrame, not {type(table)}")

        new_analyses = self._adata.uns["analyses"].copy()

        new_analyses = {} if new_analyses is None else new_analyses

        if new_analyses.get(name, None) is not None:
            warnings.warn(f"An analysis named {name} already exists! It will be overwritten.")

        if by is not None:
            table = table.set_index(by, drop=False, verify_integrity=False)

        table.index = _make_unique(pd.Series(table.index), warn=False)
        table = _reindex_by_index_name(table, self._adata.obs)

        new_analyses[name] = table

        return new_analyses

    def drop_analyses(self, pattern: str = None) -> dict[str, pd.DataFrame]:
        new_analyses = self._adata.uns["analyses"]

        if pattern is None:
            new_analyses = {}
        else:
            new_analyses = {key: value for key, value in new_analyses.items() if not re.search(pattern, key)}

        return new_analyses