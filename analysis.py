import numpy as np
import nolds


def hurst(data):
    try:
        hurst = nolds.hurst_rs(data, fit="poly")
    except Exception as e:
        print(e)
        hurst = np.nan
    return hurst

def lyapunov(data, tau_mi, ed):
    try:
        lle = nolds.lyap_r(data, emb_dim=ed, lag=tau_mi, min_tsep=tau_mi)
    except Exception as e:
        print(e)
        lle = np.nan
    return lle

def entropy(data, ed):
    try:
        tolerance = 0.2 * np.std(data)
        sampen = nolds.sampen(data, emb_dim=ed, tolerance=tolerance)
    except Exception as e:
        print(e)
        sampen = np.nan
    return sampen

def analyze_signal(signal, name, tau_mi, ed):
    if len(signal) > 12000:
        signal = signal[:12000]

    hurst_result = hurst(signal)
    lyapunov_result = lyapunov(signal, tau_mi, ed)
    entropy_result = entropy(signal, ed)
    
    res = {
        "Hurst": np.round(hurst_result, 4),
        "LLE": np.round(lyapunov_result, 4),
        "SampEn": np.round(entropy_result, 4),
        "Signal": name
    }
    print(res)
