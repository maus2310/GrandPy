import re
import warnings
from typing import Union, Sequence
import anndata as ad
import numpy as np
import pandas as pd
from Py.utils import _ensure_list, _make_unique


class AnalysisTool:
    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    def analyses(self) -> list[str]:
        return list(self._adata.uns["analyses"].keys())

    def get_analyses(self, pattern: Union[str, int, Sequence[Union[str, int, bool]]] = None, regex: bool = True) -> list[str]:
        available_analyses = self.analyses()

        if pattern is None:
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
        print(checks)
        if not all(checks):
            missing = [analysis for analysis, check in zip(pattern, checks) if not check]
            raise ValueError(f"No analysis found for pattern: {', '.join(map(str, missing))}")

        if all(isinstance(pat, (bool, int)) for pat in pattern):
            checks = pattern

        result = np.array(available_analyses)[checks].tolist()

        result = list(dict.fromkeys(result))

        return result

    def with_analysis(self, name: str, table: pd.DataFrame, by: str = None) -> dict[str, pd.DataFrame]:
        new_analyses = self._adata.uns["analyses"].copy()

        new_analyses = {} if new_analyses is None else new_analyses

        if new_analyses.get(name, None) is not None:
            warnings.warn(f"An analyses named {name} already exists! It will be overwritten.")

        if by is not None:
            table = table.set_index(by)
            table.index.name = None

        if re.search("^ENS", table.index[0]) is not None:
            table = table.reindex(self._adata.obs["Gene"])
        else:
            table.index = _make_unique(pd.Series(table.index), warn=False)
            table = table.reindex(self._adata.obs.index)

        new_analyses[name] = table

        return new_analyses

    def drop_analyses(self, pattern: str = None) -> dict[str, pd.DataFrame]:
        new_analyses = self._adata.uns["analyses"]

        if pattern is None:
            new_analyses = {}
        else:
            new_analyses = {key: value for key, value in new_analyses.items() if not re.search(pattern, key)}

        return new_analyses