import numpy as np
import pandas as pd
from typing import Union, TYPE_CHECKING

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