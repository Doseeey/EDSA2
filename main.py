from signals import random_signal, periodic_signal, chaotic_signal
from tau_estimates import estimate_tau_autocorrelation, estimate_tau_mutual
from estimate_ed import estimate_ed
from phase_space_reconstruction import phase_space_reconstruction


n = 10000
display_plot = False

random_sig = random_signal(n)
periodic_sig = periodic_signal(n, display_plot=display_plot)
lorenz_sig = chaotic_signal("data/lorenz_classic.csv", display_plot=display_plot)
rossler_sig = chaotic_signal("data/rossler_classic.csv", display_plot=display_plot)
rossler09_sig = chaotic_signal("data/rossler_fractional_0_9.csv", display_plot=display_plot)
rossler088_sig = chaotic_signal("data/rossler_fractional_0_88.csv", display_plot=display_plot)

# TODO
# hurst
# lapunov
# analiza fraktalna
# bifurkacja

signals = [random_sig, periodic_sig, lorenz_sig, rossler_sig, rossler09_sig, rossler088_sig]

# for signal in signals[2:]:
#     tau_acr = estimate_tau_autocorrelation(signal)
#     tau_mutual = estimate_tau_mutual(signal)

signal = lorenz_sig[:, 0]

#Lag estimation
tau_acr = estimate_tau_autocorrelation(signal)
print(f"Tau ACR: {tau_acr}")

tau_mutual = estimate_tau_mutual(signal)
print(f"Tau ACR: {tau_mutual}")


#Embedding dimension estimation (very long)
#estimate_ed(signal=signal[:20000], m_min=1, m_max=8, tau=tau_acr)

#Phase space reconstruction
ed = 3

phase_space_reconstruction(signal, ed, tau_acr)
phase_space_reconstruction(signal, ed, tau_mutual)
