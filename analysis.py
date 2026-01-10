import os
from matplotlib import pyplot as plt
import numpy as np
import nolds
from scipy.spatial.distance import pdist
from scipy.stats import linregress


def embed_data(data, m, tau):
    N = len(data) - (m - 1) * tau
    if N <= 0:
        return np.array([])
    
    indices = np.arange(N)[:, None] + np.arange(m)[None, :] * tau
    return data[indices]


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


def entropy(data, name, max_dim=8):
    """
    Generuje rodzinę krzywych całki korelacyjnej ln(C) vs ln(eps).
    Zwraca słownik z danymi do obliczenia K2.
    """
    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Normalizacja danych (ważna dla stałego zakresu epsilon)
    data_norm = (data - np.mean(data)) / np.std(data)
    
    # Zakres epsilon (logarytmiczny)
    r_vals = np.logspace(-2.0, 1.0, 50)
    
    curves_data = {} # Słownik: wymiar -> (log_r, log_c)
    
    # Używamy tau podanego w argumencie 'ed' lub stałego mniejszego, 
    # jeśli ed (tau_mi) jest bardzo duże, co psuje wykresy C(r).
    # Tutaj zakładam, że 'ed' w Twoim wywołaniu to wymiar, a 'tau' brakuje?
    # W analyze_signal przekazujesz: entropy(signal, ed, name...) 
    # Zakładam, że w Twoim kodzie 'ed' to wymiar zanurzenia (np. 2 lub 3).
    # Potrzebujemy tau. Użyjemy tau=10 jako bezpiecznej wartości dla wykresów C(r),
    # lub przekaż tau_mi jeśli chcesz (ale tau_mi=194 zniszczy wykres entropy).
    tau_used = 10 
    
    # Optymalizacja: Próbkujemy punkty, bo pdist dla 12000 pkt to za dużo
    num_samples = 2000

    for m in range(1, max_dim + 1):
        try:
            # 1. Zanurzanie
            vectors = embed_data(data_norm, m, tau_used)
            
            if len(vectors) == 0: continue

            # Subsampling dla wydajności
            if len(vectors) > num_samples:
                idx = np.random.choice(len(vectors), num_samples, replace=False)
                vectors = vectors[idx]
            
            # 2. Liczenie odległości (pdist)
            dists = pdist(vectors, metric='euclidean')
            
            # 3. Liczenie C(r)
            c_r_vals = []
            for r in r_vals:
                count = np.sum(dists < r)
                # Normalizacja przez liczbę par
                val = count / len(dists)
                c_r_vals.append(val)
            
            c_r_vals = np.array(c_r_vals)

            # Filtrowanie zer do logarytmu
            mask = c_r_vals > 1e-5
            if np.sum(mask) > 2:
                log_r = np.log(r_vals[mask])
                log_c = np.log(c_r_vals[mask])
                
                # Zapisujemy
                curves_data[m] = (log_r, log_c)
                
                # Rysujemy
                plt.plot(log_r, log_c, label=f'$d_E={m}$', color=colors[m-1], linewidth=1.5)

        except Exception as e:
            print(f"Error entropy m={m}: {e}")

    plt.title(rf"Wykres zależności $\ln(C(\epsilon))$ = f($\ln(\epsilon)$) - układ {name}", fontsize=16)
    plt.xlabel(r"$\ln(\epsilon)$", fontsize=14)
    plt.ylabel(r"$\ln(C(\epsilon))$", fontsize=14)
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if not os.path.exists("graphs"): os.makedirs("graphs")
    plt.savefig(os.path.join("graphs", f"entropy_lines_{name}.png"), dpi=150)
    plt.close()
    
    return curves_data


def k2_vs_dim(curves_data, name, max_dim=8):
    """
    Oblicza K2 na podstawie nachylenia D2 i odległości krzywych (Wzór 14 ze slajdów).
    """
    k2_results = []
    dims_x = []
    
    # 1. Estymacja D2 (Wymiar korelacyjny) z krzywej dla najwyższego wymiaru
    # Szukamy nachylenia w najbardziej liniowym fragmencie
    if max_dim in curves_data:
        x_last, y_last = curves_data[max_dim]
        # Bierzemy środek zakresu
        mid = len(x_last) // 2
        start, end = max(0, mid - 5), min(len(x_last), mid + 5)
        if len(x_last) > 5:
            res = linregress(x_last[start:end], y_last[start:end])
            D2_est = res.slope
            # Wybieramy target_ln_eps (punkt odniesienia)
            target_ln_eps = x_last[mid]
        else:
            D2_est = 0
            target_ln_eps = -1.0
    else:
        D2_est = 0
        target_ln_eps = 0

    # 2. Obliczanie K2 dla każdego wymiaru
    for m in range(1, max_dim + 1):
        if m in curves_data:
            x_vals, y_vals = curves_data[m]
            
            # Znajdujemy wartość ln(C) dla target_ln_eps
            idx = np.argmin(np.abs(x_vals - target_ln_eps))
            ln_C = y_vals[idx]
            
            # WZÓR ZE SLAJDU: K2 = |ln(C) - D2 * ln(eps)| / m
            # (pomijamy tau w mianowniku dla uproszczenia skali wykresu)
            if ln_C > -20: # Zabezpieczenie przed szumem
                val_k2 = abs(ln_C - D2_est * target_ln_eps) / m
                k2_results.append(val_k2)
                dims_x.append(m)

    plt.figure(figsize=(8, 6))
    plt.plot(dims_x, k2_results, 'ko-', markersize=6, linewidth=1.5, label='K2')
    
    plt.title(rf"Wykres zależnosci $K_2$ od f($d_E$) - układ {name}", fontsize=16)
    plt.xlabel(r"$d_E$", fontsize=14)
    plt.ylabel(r"$K_2$", fontsize=14)
    plt.xticks(dims_x)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if len(k2_results) > 0:
        plt.ylim(0, max(k2_results) * 1.3)
    
    filename = os.path.join("graphs", f"k2_vs_dim_{name}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return k2_results

def analyze_signal(signal, name, tau_mi, ed):
    if len(signal) > 12000:
        signal = signal[:12000]

    # hurst_result = hurst(signal, name)
    # lyapunov_result = lyapunov(signal, tau_mi, ed, name)

    max_dim = 8
    
    curves_data = entropy(signal, name, max_dim=max_dim)
    k2_vs_dim_result = k2_vs_dim(curves_data, name, max_dim=max_dim)

    res = {
        # "Hurst": np.round(hurst_result, 4),
        # "LLE": np.round(lyapunov_result, 4),
        "entropy_lines": np.round(k2_vs_dim_result, 4) if len(k2_vs_dim_result) > 0 else 0,
        "k2_vs_dim": k2_vs_dim_result,
        "Signal": name,
    }
    print(res)
