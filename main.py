from analysis import analyze_signal
from bifurcation import generate_bifurcation
from signals import random_signal, periodic_signal, chaotic_signal
# from tau_estimates import estimate_tau_autocorrelation, estimate_tau_mutual
# from estimate_ed import estimate_ed
# from phase_space_reconstruction import phase_space_reconstruction


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
# analiza fraktalna - entropia
# bifurkacja

# tau i ed na sprawku wypisane, wszystkie sa dla signal[:, 0] (pierwszego wektora)

signals = [
    (random_sig, {
        "name": "random",
        "tau": 4,
        "dim": 2
    }),
    (periodic_sig, {
        "name": "periodic",
        "tau": 6,
        "dim": 2
    }),
    (lorenz_sig, {
        "name": "lorenz",
        "tau": 33,
        "dim": 3
    }),
    (rossler_sig, {
        "name": "rossler",
        "tau": 194,
        "dim": 3
    }),
    (rossler09_sig, {
        "name": "rossler_fractional_0_9",
        "tau": 329,
        "dim": 3
    }),
    (rossler088_sig, {
        "name": "rossler_fractional_0_88",
        "tau": 249,
        "dim": 3
    })
]

for index, item in enumerate(signals):
    signal, params = item
    # tau_acr = estimate_tau_autocorrelation(signal[:, 0])
    # tau_mutual = estimate_tau_mutual(signal[:, 0])
    # estimate_ed(signal[:4000], m_min = 1, m_max = 8, tau = tau_acr)
    signal_to_pass = signal if index < 2 else signal[:, 0]
    signal_to_pass = signal_to_pass[:10000].flatten() if index == 2 else signal_to_pass
    analyze_signal(signal_to_pass, params["name"], params["tau"], params["dim"])

# generate_bifurcation()
# Example

# signal = rossler088_sig[:, 0]

#Lag estimation
# tau_acr = estimate_tau_autocorrelation(signal, display_plot=False)
# tau_mi = estimate_tau_mutual(signal, display_plot=False)

#Embedding dimension estimation (dlugo trwa)
# estimate_ed(signal[:4000], m_min = 1, m_max = 8, tau = tau_acr)
#estimate_ed(signal[:5000], m_min = 1, m_max = 8, tau = tau_mi)

ed = 3

#Phase space reconstruction
#phase_space_reconstruction(signal, ed, tau_acr)
#phase_space_reconstruction(signal, ed, tau_mi)