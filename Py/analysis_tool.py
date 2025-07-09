import re
import warnings
import anndata as ad
import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Sequence

from Py.utils import _ensure_list, _make_unique, _reindex_by_index_name


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
                return [any(re.search(pat, analysis) for analysis in available_analyses) for pat in pattern]
            else:
                if all(isinstance(pat, (bool, int)) for pat in pattern):
                    return [True] * len(available_analyses)
                else:
                    return [analysis in available_analyses for analysis in pattern]

        checks = check_analyses(pattern, available_analyses, regex)

        if not all(checks):
            missing = [analysis for analysis, check in zip(pattern, checks) if not check]
            raise ValueError(f"No analysis found for pattern: {', '.join(map(str, missing))}")

        if all(isinstance(pat, (bool, int)) for pat in pattern):
            checks = pattern

        result = np.array(available_analyses)[checks].tolist()

        result = list(dict.fromkeys(result))

        if description:
            return {name: df.columns.tolist() if name in result else {} for name, df in self._adata.uns["analyses"].items()}

        return result

    def with_analysis(self, name: str, table: pd.DataFrame, by: str = None) -> dict[str, pd.DataFrame]:
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