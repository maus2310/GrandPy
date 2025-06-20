import re
import warnings
from typing import Literal, Callable, Mapping, Any, Union
import anndata as ad


class Plot:
    """
    Used to store a plot function.

    Parameters
    ----------
    function: Callable
        A plot function

    parameters: Mapping[str, Any]
        Parameter names mapped to their values.

    plot_type: Literal["global", "gene"], optional
        The type of plot. Either 'global' or 'gene'.
    """
    def __init__(self, function: Callable, parameters: Mapping[str, Any] = None, plot_type: Literal["global", "gene"] = "global"):
        self.function = function
        self.parameters = parameters if parameters is not None else {}
        self.plot_type = plot_type

    def __repr__(self):
        return f"Plot(type={self.plot_type!r}, function={self.function}, parameters={self.parameters})"

    def __call__(self, data, gene: str = None):
        if self.plot_type == "gene":
            if gene is None:
                raise ValueError("Gene must be provided for a gene plot.")
            return self.function(data, gene, **self.parameters)
        elif self.plot_type == "global":
            return self.function(data, **self.parameters)
        else:
            raise ValueError(f"Invalid plot type: {self.plot_type}")


class PlotTool:
    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    def plots(self) -> dict[str, dict[str, Plot]]:
        plots = self._adata.uns["plots"]

        result = {}

        if plots is not None:
            result = {}
            if plots.get("gene") is not None:
                result["gene"] = list(plots["gene"].keys())

            if plots.get("global") is not None:
                result["global"] = list(plots["global"].keys())

        return result

    def add_plot(self, name: str, function: Union[Plot, Callable]) -> dict[str, dict[str, Plot]]:
        def function_to_plot(fun: Callable) -> Plot:
            from inspect import signature

            sig = signature(fun)
            params = dict(sig.parameters)

            p_type = "gene" if "gene" in params else "global"

            clean_params = {k: v.default for k, v in params.items()
                            if k not in ("data", "gene") and v.default is not v.empty}

            return Plot(fun, clean_params, plot_type=p_type)

        new_plots = self._adata.uns["plots"].copy()

        if not isinstance(function, Plot):
            function = function_to_plot(function)

        plot_type = function.plot_type

        if new_plots is None:
            new_plots = {}
        if new_plots.get(plot_type) is None:
            new_plots[plot_type] = {}
        if name in new_plots[plot_type].keys():
            warnings.warn(f"A {plot_type} plot with the name '{name}' already exists. It will be overwritten.")


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