<img src="logo/grandPy_hex_logo.png" alt="grandPy logo" align="right" width="190"/>

# GrandPy
Nucleotide conversion sequencing experiments have been developed to add a temporal dimension to RNA-seq and single-cell
RNA seq. Such experiments require specialized tools for primary processing such as GRAND-SLAM, and specialized tools
for downstream analyses. GrandPy provides a comprehensive toolbox for quality control, kinetic modeling, differential
gene expression analysis and visualization of such data. It mimics the core functionality of the original
[grandR](https://grandr.erhard-lab.de/) package, by which it is inspired.

## Installation
GrandPy is <span style="color:red">available</span> from [[PyPi]](...). Install GrandPy using the following commands on the Python console:

<pre> pip install grandpy</pre>

You can also install the development version from GitLab:

<pre> pip install git+https://git.uni-regensburg.de/se24/g03/grandpy.git </pre>

## System Requirements
GrandPy has mostly been tested on Windows but should also run on Linux and macOS.
The package runs on standard laptops (multicore CPUs are recommended; memory requirements
depend on the size of your datasets).

Installing it via `pip` will make sure that the following (standard) packages are available:

<pre>numpy, pandas, scipy, anndata, tqdm, matplotlib, seaborn, pydeseq2</pre>

Additional packages are optional and important for particular functions:

<pre>scikit-learn, mygene, numdifftools</pre>

## Cheatsheet

[<img src="cheatsheet/cheatsheet_preview_Version_1.png" alt="grandPy Cheatsheet" width="600"/>](cheatsheet/grandPy_Cheat_Sheet_Version_1.pdf)

## How to get started
First, have a look at the [getting started](./notebooks/notebook_00_getting_started.ipynb) notebook.

Next, explore [differential expression](./notebooks/notebook_01_differential_expression.ipynb) or
[kinetic modeling](./notebooks/notebook_02_kinetic_modeling.ipynb), which provide an overview of the two primary
settings for nucleotide conversion experiments.

There are also additional notebooks:

- [Loading data and working with GrandPy objects](./notebooks/notebook_03_loading_data_and_working_with_grandpy_objects.ipynb): Learn more about programming with GrandPy
- [Working with data matrices and analysis results](./notebooks/notebook_04_working_with_data_matrices_and_analysis_results.ipynb): Learn more about how to retrieve data from GrandPy objects
- [Plotting](./notebooks/notebook_05_plotting.ipynb): Learn how to create and store plots with GrandPy
- [Pulse-chase](./notebooks/notebook_06_fitting_pulse-chase_data.ipynb): Learn how to fit pulse-chase data with GrandPy


## Acknowledgements
GrandPy is heavily inspired by the [grandR](https://grandr.erhard-lab.de/) R package
by Prof. Dr. Florian Erhard, and we gratefully acknowledge his work
as well as the team behind it, especially Julian-Andres Selke and Rahaf Issa,
whose contributions made `GrandPy` possible.