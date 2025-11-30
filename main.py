from signals import random_signal, periodic_signal, chaotic_signal
from tau_estimates import estimate_tau_autocorrelation

import numpy as np
import matplotlib.pyplot as plt
#Spoko libki moga sie przydac
#import nolds
#import hundun


n = 10000
display_plot = False

random_sig = random_signal(n)
periodic_sig = periodic_signal(n, display_plot=display_plot)
lorenz_sig = chaotic_signal("data/lorenz_classic.csv", display_plot=display_plot)
rossler_sig = chaotic_signal("data/rossler_classic.csv", display_plot=display_plot)
rossler09_sig = chaotic_signal("data/rossler_fractional_0_9.csv", display_plot=display_plot)
rossler088_sig = chaotic_signal("data/rossler_fractional_0_88.csv", display_plot=display_plot)

# TODO
# Tau estimation - autocorrelation + mutual information
# embedded dimension - po tau, GP do testa
# rekonstrukcja na podstawie ed i tau
# hurst
# lapunov
# analiza fraktalna
# bifurkacja

signals = [random_sig, periodic_sig, lorenz_sig, rossler_sig, rossler09_sig, rossler088_sig]

for signal in signals:
    estimate_tau_autocorrelation(signal)
