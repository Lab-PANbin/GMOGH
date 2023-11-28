# GMOGH
GMOGH: Toward Convergence: A Gradient-Based Multiobjective Method With Greedy Hash for Hyperspectral Unmixing
======================================================
GMOGH is a gradient-based method for hyperspectral unmixing problem. The proposed method contains two main steps. First, the continuous solutions are searched using a multiobjective gradient-based approach. Then, the optimal discrete solutions (endmembers) are updated using a greedy hash method. The proposed method is evaluated on 2 simulated and 3 real remote sensing data for a range of SNR values (i.e., from 20 to 40 dB). The results show considerable improvements compared to sparse and multiobjective methods.

If you use this code please cite the following paper

[1]R. Li, B. Pan, X. Xu, T. Li and Z. Shi, "Toward Convergence: A Gradient-Based Multiobjective Method With Greedy Hash for Hyperspectral Unmixing," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-14, 2023, Art no. 5509114, doi: 10.1109/TGRS.2023.3267080.

Urban_F210.mat is provided as an example dataset, data_example.mat is provided as spectral library, end6_groundTruth.mat is provided as groundtruth of Urban.

You can run run.py to try our method.

