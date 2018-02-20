# STSN
Project modeling the propagation of light through some unknown material.

Please feel free to add or modify the code accordingly and make suggestions with push requests or via e-mail.

I.
1. travelNorm.py contains the Tensorflow Python script for the "unmasked version" of the STSN
2. travelMask.py contains the Tensorflow ... ... ... ... ...   "masked version" where a certain region is omitted during the loss function calculation.
3. plotCostsAndWeights.m contains the MATLAB script for plotting the costs and weights of each test
4. (OUTDATED) scatter_n4_T40.py contains the original (non-Tensorflow) NumPy and Autograd implementation of the STSN (no masking applied).

II.
The naming of the data files under data/... is:
scatter_wL_TnL_all_io.csv

where:
- wL: number of non-one weights
- nL: number of scatter/prop layers (time-depth)
- io: either in or out to identify input/output pairs

III.
The results/ sub-directory contains:
- CSV files with training loss and network weights at the respective epoch
- Figures for results of the CSV files
