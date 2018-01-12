# lightPropagation
Project modeling the propagation of light through some material.

Please feel free to add or modify the code accordingly and make suggestions with pull requests or via e-mail.


The naming of the data files under data/... is:
scatter_wL_TnL_all_io.csv

where:
- wL: number of non-one weights
- nL: number of scatter/prop layers (time-depth)
- io: either in or out to identify input/output pairs

For the cases:
1. 2-weights cases (wL = 2), the weight values should be 0.5 at indices 52 and 60 respectively
2. 4-weights cases (wL = 4), the wight values should be 0.5, 0.9, 0.7, 0.3 at indices 52, 57, 59, amd 66 respectively
3. Have to clarify with Andrew about the 1-weight case
