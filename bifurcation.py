import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def get_gl_coefficients(q, n_steps):
    c = np.zeros(n_steps)
    c[0] = 1.0
    for j in range(1, n_steps):
        c[j] = (1.0 - (1.0 + q) / j) * c[j - 1]
    return c

def run_fast_fractional_rossler(q, n_steps, h, keep_last, memory_length=500):
    a, b, c_param = 0.5, 0.2, 10.0
    
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    
    x[0], y[0], z[0] = 1.0, 1.0, 1.0

    coeffs = get_gl_coefficients(q, memory_length + 1)
    h_q = h ** q

    for k in range(1, n_steps):
        start_mem = max(0, k - memory_length)
        len_mem = k - start_mem
        curr_coeffs = coeffs[1:len_mem+1]
        
        mem_x = np.dot(curr_coeffs, x[start_mem:k][::-1])
        mem_y = np.dot(curr_coeffs, y[start_mem:k][::-1])
        mem_z = np.dot(curr_coeffs, z[start_mem:k][::-1])
        
        xk, yk, zk = x[k-1], y[k-1], z[k-1]
        
        fx = -yk - zk
        fy = xk + a * yk
        fz = b + zk * (xk - c_param)
        
        x[k] = (fx * h_q) - mem_x
        y[k] = (fy * h_q) - mem_y
        z[k] = (fz * h_q) - mem_z
        
        if abs(x[k]) > 200: 
            return np.array([]) 

    return x[-keep_last:]

def generate_bifurcation(show_plot=False):
    steps = 5000       
    analyze_last = 2000
    h_step = 0.02
    
    q_values = np.linspace(0.7, 0.92, 1000) 
    
    bif_q = []
    bif_x = []

    for q in q_values:
        x_series = run_fast_fractional_rossler(q, steps, h_step, analyze_last)
        
        if len(x_series) == 0:
            continue
        peaks_idx = argrelextrema(x_series, np.greater)[0]
        peaks = x_series[peaks_idx]
        
        if len(peaks) == 0:
            continue
        bif_q.extend([q] * len(peaks))
        bif_x.extend(peaks)

    plt.figure(figsize=(12, 7))
    plt.scatter(bif_q, bif_x, s=1.5, c='black', alpha=0.6)
    
    plt.title("Diagram bifurkacyjny systemu Rösslera", fontsize=14)
    plt.xlabel("Rząd pochodnej q", fontsize=12)
    plt.ylabel("Lokalne maksima x(t)", fontsize=12)
    plt.xlim(0.7, 0.92)
    
    plt.ylim(-25, 25) 
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = "bifurkacja_q.png"
    filepath = os.path.join("graphs", filename)
    plt.savefig(filepath, dpi=300)
    if show_plot:
        plt.show()

if __name__ == "__main__":
    generate_bifurcation()