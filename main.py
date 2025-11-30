from signals import random_signal, periodic_signal, chaotic_signal
from tau_estimates import estimate_tau_autocorrelation

import numpy as np
import matplotlib.pyplot as plt
import nolds


n = 10000
display_plot = False

random_sig = random_signal(n)
periodic_sig = periodic_signal(n, display_plot=display_plot)
lorenz_sig = chaotic_signal("data/lorenz_classic.csv", display_plot=display_plot)
rossler_sig = chaotic_signal("data/rossler_classic.csv", display_plot=display_plot)
rossler09_sig = chaotic_signal("data/rossler_fractional_0_9.csv", display_plot=display_plot)
rossler088_sig = chaotic_signal("data/rossler_fractional_0_88.csv", display_plot=display_plot)

# rvals = nolds.logarithmic_r(1, np.e, 1.1)
# corr_dim_args = dict(emb_dim=5, lag=10, fit="poly", rvals=rvals)
# cdx = nolds.corr_dim(lorenz_sig[:, 0], **corr_dim_args)
# cdy = nolds.corr_dim(lorenz_sig[:, 1], **corr_dim_args)
# cdz = nolds.corr_dim(lorenz_sig[:, 2], **corr_dim_args)

# print("Expected correlation dimension:  2.05")
# print("corr_dim(x)                   : ", cdx)
# print("corr_dim(y)                   : ", cdy)
# print("corr_dim(z)                   : ", cdz)

signals = [random_sig, periodic_sig, lorenz_sig, rossler_sig, rossler09_sig, rossler088_sig]

for signal in signals:
    estimate_tau_autocorrelation(signal)
