import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def estimate_tau_autocorrelation(data: np.ndarray, alpha_level: float = 0.001) -> int:
    # Prepare data
    if data.ndim > 1:
        # Założenie: Sygnały są sprzężone i wystarczy analiza jednego.
        time_series = data[:, 0]
    elif data.ndim == 1:
        time_series = data
        
    time_series = time_series - np.mean(time_series)
    n_lags = 1000
    acf_values, confint = acf(time_series, nlags=n_lags, alpha=alpha_level)
    
    lower_bound = confint[:, 0] - acf_values
    
    # tau = pierwszy indeks ponizej interwalu ufnosci
    significant_lags = np.where(acf_values[1:] > np.abs(lower_bound[1:]))[0]
    
    if len(significant_lags) < len(acf_values) - 1:
        # Pierwsze opóźnienie, które NIE jest znaczące
        # Indeks w tablicy acf_values to (significant_lags[-1] + 2)
        # Indeks w tablicy (indeks w Pythonie) jest przesunięty
        tau = significant_lags[-1] + 2 if len(significant_lags) > 0 else 1
    else:
        tau = 1
        print("Brak znaczącego spadku ACF. Wybrano domyślne tau=1.")
    
    plt.figure(figsize=(10, 5))
    plt.plot(acf_values, linestyle='-', color='b', label='ACF')
    plt.fill_between(range(len(acf_values)), confint[:, 0], confint[:, 1], color='gray', alpha=0.3, label=f'Interwał ufności $\\alpha={alpha_level}$')
    
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(x=tau, color='red', linestyle='--', label=f'Estymowane $\\tau = {tau}$')
    
    plt.title('Estymacja Opóźnienia Czasowego $\\tau$ (Metoda Autokorelacji)')
    plt.xlabel('Opóźnienie (Lag)')
    plt.ylabel('Autokorelacja')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    return tau