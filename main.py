from signals import random_signal, periodic_signal, chaotic_signal
from tau_estimates import estimate_tau_autocorrelation, estimate_tau_mutual
from estimate_ed import estimate_correlation_dimension_range, estimate_ed
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

# tau_acr = estimate_tau_autocorrelation(signal)

# print(f"Tau ACR: {tau_acr}")

# tau_mutual = estimate_tau_mutual(signal)

# print(f"Tau ACR: {tau_mutual}")


estimated_D2 = estimate_ed(
        signal=signal[:20000],
        m_min=1,
        m_max=8,
        tau=55, 
    )

# phase_space_reconstruction(signal, 3, tau_acr)
# phase_space_reconstruction(signal, 3, tau_mutual)

#print(ed_acr)

# import nolds

# d2, debug_data = nolds.corr_dim(signal, emb_dim=1, lag=55, debug_data=True)
# print(d2)
# print(debug_data)