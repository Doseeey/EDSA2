import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from utils import calculate_average_mutual_information, find_first_local_minimum, find_zero_or_minimum

def estimate_tau_autocorrelation(data: np.ndarray) -> int:
    # Prepare data
    if data.ndim > 1:
        # Założenie: Sygnały są sprzężone i wystarczy analiza jednego.
        time_series = data[:, 0]
    elif data.ndim == 1:
        time_series = data
        
    time_series = time_series - np.mean(time_series)
    n_lags = 1000
    acf_values = acf(time_series, nlags=n_lags)

    tau = find_zero_or_minimum(acf_values)
    
    plt.figure(figsize=(10, 5))
    plt.plot(acf_values, linestyle='-', color='b', label='ACF')    
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(x=tau, color='red', linestyle='--', label=f'Estymowane $\\tau = {tau}$')
    
    plt.title('Estymacja Opóźnienia Czasowego $\\tau$ (Metoda Autokorelacji)')
    plt.xlabel('Opóźnienie (Lag)')
    plt.ylabel('Autokorelacja')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    return tau

def estimate_tau_mutual(data: np.ndarray) -> int:
    if data.ndim > 1:
        # Założenie: Sygnały są sprzężone i wystarczy analiza jednego.
        time_series = data[:, 0]
    elif data.ndim == 1:
        time_series = data

    n_lags = 1000
    ami_values = calculate_average_mutual_information(time_series, n_lags)
    tau, _ = find_first_local_minimum(ami_values)

    plt.figure(figsize=(10, 5))
    plt.plot(ami_values, linestyle='-', color='b', label='ACF')    
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(x=tau, color='red', linestyle='--', label=f'Estymowane $\\tau = {tau}$')
    
    plt.title('Estymacja Opóźnienia Czasowego $\\tau$ (Metoda Autokorelacji)')
    plt.xlabel('Opóźnienie (Lag)')
    plt.ylabel('Autokorelacja')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    return tau