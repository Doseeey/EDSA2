import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def get_gl_coefficients(q, n_steps):
    c = np.zeros(n_steps)
    c[0] = 1.0
    for j in range(1, n_steps):
        c[j] = (1.0 - (1.0 + q) / j) * c[j - 1]
    return c

def run_fractional_rossler(q, n_steps=2000, h=0.05, keep_last=500):
    a = 0.5
    b = 0.2
    c_param = 10.0
    
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    
    x[0], y[0], z[0] = 1.0, 1.0, 1.0  # Y0 = [1, 1, 1]

    coeffs = get_gl_coefficients(q, n_steps)
    
    h_q = h ** q
    
    for k in range(1, n_steps):
        current_coeffs = coeffs[1:k+1]
        
        history_x = x[k-1::-1]
        history_y = y[k-1::-1]
        history_z = z[k-1::-1]
        
        mem_x = np.dot(current_coeffs, history_x)
        mem_y = np.dot(current_coeffs, history_y)
        mem_z = np.dot(current_coeffs, history_z)
        
        fx = -y[k-1] - z[k-1]
        fy = x[k-1] + a * y[k-1]
        fz = b + z[k-1] * (x[k-1] - c_param)
                
        x[k] = (fx * h_q) - mem_x
        y[k] = (fy * h_q) - mem_y
        z[k] = (fz * h_q) - mem_z

        if np.abs(x[k]) > 1000:
            return x[:k]

    return x[-keep_last:]

def generate_bifurcation():    
    q_values = np.linspace(0.7, 1.1, 60) 
    
    bif_q = []
    bif_x = []

    steps = 2000 
    h_step = 0.01
    analyze_last = 500

    for q in q_values:
        try:
            x_series = run_fractional_rossler(q, n_steps=steps, h=h_step, keep_last=analyze_last)
            
            peaks_idx = argrelextrema(x_series, np.greater)[0]
            peaks = x_series[peaks_idx]
            
            if len(peaks) > 0:
                bif_q.extend([q] * len(peaks))
                bif_x.extend(peaks)
                
        except Exception as e:
            print(f"error q={q}: {e}")

    # Rysowanie
    plt.figure(figsize=(10, 6))
    plt.scatter(bif_q, bif_x, s=1.5, c='black', alpha=0.6)
    plt.title("Diagram bifurkacyjny systemu Rösslera względem rzędu pochodnej q")
    plt.xlabel("Rząd pochodnej q")
    plt.ylabel("Lokalne maksima x(t)")
    plt.xlim(0.7, 1.1)
    plt.grid(True, alpha=0.3)
    
    filename = "bifurkacja_q.png"
    plt.savefig(filename, dpi=300)
    print(f"saved as {filename}")
