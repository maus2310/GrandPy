# grandPy

<span style="color:palevioletred"> # Logo - Work in progress</span>


Nucleotide conversion sequencing experiments have been developed to add a temporal dimension to RNA-seq and single-cell RNA seq. Such experiments require specialized tools for primary processing such as GRAND-SLAM, and specialized tools for downstream analyses. grandPy provides a comprehensive toolbox for quality control, kinetic modeling, differential gene expression analysis and visualization of such data. It mimics the core functionality of the original package `grandR` [[1]](https://grandr.erhard-lab.de/).

## Installation
grandPy is <span style="color:red">available</span> from [[PyPi]](...). Install grandPy using the following commands on the Python console:

<pre> pip install grandpy</pre>

You can also install the development version from GitLab:

<pre> pip install git + https://git.uni-regensburg.de/se24/g03/grandpy.git </pre>

## System Requirements
grandPy should be compatible with Windows operating systems <span style="color:red">(?)</span>, we recommend using grandPy on a Windows machine, where it has been tested. grandPy runs on standard laptops (multi-core CPUs are recommended and memory requirements depend on the size of your data sets).

Installing it via `pip` will make sure that the following (standard) packages are available:

<pre><span style="color:red">List of used packages</span> </pre>

Additional packages are optional and important for particular functions:

<pre>pydeseq2</pre>

With all dependencies available, installation of grandPy typically takes <span style="color:red">less than a minute</span>.

## Cheatsheet
<span style="color:palevioletred"># Cheatsheet - Work in progress </span>

## How to get started
First have a look at the [Getting started](../grandpy/grandpy/vignettes/vignette_00_getting_started.ipynb) vignette.

Then, go through the [Differential expression](../grandpy/grandpy/vignettes/vignette_01_differential_expression_(snapshot_data).ipynb) or the [Kinetic modeling](../grandpy/grandpy/vignettes/vignette_02_kinetic_modeling_(progressive_labeling_time_courses).ipynb) vignette, which provide a comprehensive walk-through of the two main settings of nucleotide conversion experiments.

There are also additional vignettes:

- [Loading data and working with grandPy objects](../grandpy/grandpy/vignettes/vignette_03_loading_data_and_working_with_grandPy_objects.ipynb): Learn more about programming with grandPy
- [Working with data matrices and analysis results](../grandpy/grandpy/vignettes/vignette_04_working_with_data_matrices_and_analysis_results.ipynb): Learn more about how to retrieve data from grandPy objects
- [...]
- [Pulse-chase](../grandpy/grandpy/vignettes/vignette_06_fitting_pulse-chase_data.ipynb): Learn how to fit pulse-chase data with grandPy