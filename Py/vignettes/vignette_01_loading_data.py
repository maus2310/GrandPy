"""
Loading data and working with grandPy objects

Throughout this vignette, we will be using the GRAND-SLAM processed SLAM-seq
data set from Finkel et al. 2021 https://www.nature.com/articles/s41586-021-03610-3).
The data set contains time series (progressive labeling) samples from a human
epithelial cell line (Calu3 cells); half of the samples were infected with SARS-CoV-2
for different periods of time.

The output of GRAND-SLAM is a tsv file where rows are genes and columns are read counts
and other statistics (e.g., the new-to-total RNA ratio) for all samples.
The data set is available on zenodo (“https://zenodo.org/record/5834034/files/sars.tsv.gz”).
We start by reading this file into Python:
"""

import pandas as pd
import numpy as np
from Py.load import read_grand

# ------------------------------------------------------------------------------------------
# 1. specify file path
file_path = "../data/sars.tsv"

# ------------------------------------------------------------------------------------------
# 2. read data
gp = read_grand(file_path)

# ------------------------------------------------------------------------------------------
# 3. print out the grandPy Object
print(gp) # outputs the number of genes, samples, slots, metadata etc.

# ------------------------------------------------------------------------------------------
# Examples (work in progress)
# 4. check whether certain slots are available
try:
    gp.check_mode_slot("ntr", "raw")
    print("Slot 'ntr' contains the mode 'raw")
except Exception as e:
    print(f"{e}")

# ------------------------------------------------------------------------------------------
# 5. print out Gen-Symbols, sample-names,
print("the first 5 genes: ", gp.adata.var["Symbol"].head().tolist())
print("Samples: ", gp.adata.obs["Name"].tolist())
print("Gen-types: ", gp.adata.var["Type"].value_counts())

# plots in progress