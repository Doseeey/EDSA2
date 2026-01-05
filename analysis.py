import os
import numpy as np
import nolds


def hurst(data, name):
    try:
        hurst = nolds.hurst_rs(data, fit="poly", debug_plot=True, plot_file=os.path.join("graphs", f"hurst_{name}.png"))
    except Exception as e:
        print(e)
        hurst = np.nan
    return hurst


def lyapunov(data, tau_mi, ed, name):
    try:
        lle = nolds.lyap_r(
            data, emb_dim=ed, lag=tau_mi, min_tsep=tau_mi, debug_plot=True, plot_file=os.path.join("graphs", f"lyap_{name}.png")
        )
    except Exception as e:
        print(e)
        lle = np.nan
    return lle


def entropy(data, ed, name):
    try:
        tolerance = 0.2 * np.std(data)
        sampen = nolds.sampen(
            data, emb_dim=ed, tolerance=tolerance, debug_plot=True, plot_file=os.path.join("graphs", f"sampen_emb{ed}_{name}.png")
        )
    except Exception as e:
        print(e)
        sampen = np.nan
    return sampen


def analyze_signal(signal, name, tau_mi, ed):
    if len(signal) > 12000:
        signal = signal[:12000]

    hurst_result = hurst(signal, name)
    lyapunov_result = lyapunov(signal, tau_mi, ed, name)
    entropy_result = entropy(signal, ed, name)

    res = {
        "Hurst": np.round(hurst_result, 4),
        "LLE": np.round(lyapunov_result, 4),
        "SampEn": np.round(entropy_result, 4),
        "Signal": name,
    }
    print(res)
