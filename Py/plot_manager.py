import re
import warnings
from typing import Literal, Callable, Mapping, Any, Union
import anndata as ad

from Py.grandPy import Plot


class PlotManager:
    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    def plots(self) -> dict[str, dict[str, Plot]]:
        data = self._adata.uns
        result = {}

        if data is not None and data.get("plots") is not None:
            if data.get("plots", {}).get("gene") is not None:
                result["gene"] = list(data["plots"]["gene"].keys())

            if data.get("plots", {}).get("global") is not None:
                result["global"] = list(data["plots"]["global"].keys())

        return result

    def add_plot(self, name: str, plot_type: Literal["gene", "global"], function: Union[Plot, Callable]) -> dict[str, dict[str, Plot]]:
        def function_to_plot(fun: Callable) -> Plot:
            from inspect import signature

            sig = signature(fun)
            params = dict(sig.parameters)

            return Plot(fun, params, plot_type=plot_type)

        new_plots = self._adata.uns["plots"].copy()

        if new_plots is None:
            new_plots = {}
        if new_plots.get(plot_type) is None:
            new_plots[plot_type] = {}
        if name in new_plots[plot_type].keys():
            warnings.warn(f"A {plot_type} plot with the name '{name}' already exists. It will be overwritten.")

        if isinstance(function, Plot):
            pass

        elif callable(function):
            function = function_to_plot(function)

        else:
            raise TypeError("Expected Plot or function")

        new_plots[plot_type][name] = function

        return new_plots

    def drop_plot(self, pattern: str) -> dict[str, dict[str, Plot]]:
        new_plots = self._adata.uns["plots"].copy()

        if pattern is None:
            new_plots = None

        else:
            for key in ("gene", "global", "floating"):
                if new_plots.get(key) is not None:
                    plots_dict = new_plots[key]

                    new_plots[key] = {
                        name: value
                        for name, value in plots_dict.items()
                        if not re.search(pattern, name)
                    }

        return new_plots