import re
import warnings
from collections.abc import Sequence
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from .utils import _ensure_list, _make_unique, _reindex_by_index_name


class AnalysisTool:
    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    def analyses(self):
        """
        For detailed documentation see GrandPy.analyses.
        """
        analyses = self._adata.uns["analyses"]
        return list(analyses.keys())

    def get_analyses(
            self,
            pattern: Union[str, int, Sequence[Union[str, int, bool]]] = None,
            regex: bool = True,
            description: bool = False
    ) -> Union[list[str], dict[str, list[str]]]:
        """
        For detailed documentation see GrandPy.get_analyses.
        """
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

    def with_analysis(self, name: str, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        For detailed documentation see GrandPy.with_analysis.
        """
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"'table' has to be a pd.DataFrame, not {type(table)}")

        if not table.index.name == "Symbol":
            raise ValueError(f"Index of 'table' has to be named 'Symbol'. It is currently named: {table.index.name}.")

        new_analyses = self._adata.uns["analyses"].copy()

        new_analyses = {} if new_analyses is None else new_analyses

        if new_analyses.get(name, None) is not None:
            warnings.warn(f"An analysis named {name} already exists! It will be overwritten.")

        table.index = _make_unique(pd.Series(table.index), warn=False)
        table = _reindex_by_index_name(table, self._adata.obs)

        new_analyses[name] = table

        return new_analyses

    def drop_analyses(self, pattern: Union[str, Sequence[str]] = None) -> dict[str, pd.DataFrame]:
        """
        For detailed documentation see GrandPy.with_dropped_analyses.
        """
        new_analyses = self._adata.uns["analyses"]

        if pattern is None:
            new_analyses = {}
        else:
            pattern = _ensure_list(pattern)

            new_analyses = {
                key: value
                for key, value in new_analyses.items()
                if not any(re.search(pat, key) for pat in pattern)
            }

        return new_analyses