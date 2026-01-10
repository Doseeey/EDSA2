import os
import numpy as np
import matplotlib.pyplot as plt

def generate_bifurcation(showFig=False):
    a = 0.5
    b = 2.0
    c = 10.0

    h = 0.002
    steps = 60000
    transient = 40000    

    q_values = np.linspace(0.7, 1.05, 100)


    def get_gl_coeffs(q, n):
        coeffs = np.empty(n)
        coeffs[0] = 1.0
        for j in range(1, n):
            coeffs[j] = (1.0 - (q + 1.0) / j) * coeffs[j-1]
        return coeffs

    def simulate_rossler(q, steps, h, a, b, c):
        x = np.zeros(steps)
        y = np.zeros(steps)
        z = np.zeros(steps)

        x[0], y[0], z[0] = 0.5, 0.5, 0.5

        coeffs = get_gl_coeffs(q, steps)
        h_q = h ** q

        limit = 1e5

        for n in range(0, steps - 1):
            if abs(x[n]) > limit or abs(z[n]) > limit:
                return np.zeros(steps)

            xn, yn, zn = x[n], y[n], z[n]

            mem_x = 0.0
            mem_y = 0.0
            mem_z = 0.0

            for j in range(1, n + 1):
                w = coeffs[j]
                mem_x += w * x[n - j + 1]
                mem_y += w * y[n - j + 1]
                mem_z += w * z[n - j + 1]

            fx = -(yn + zn)
            fy = xn + a * yn
            fz = b + zn * (xn - c)

            x[n+1] = h_q * fx - mem_x
            y[n+1] = h_q * fy - mem_y
            z[n+1] = h_q * fz - mem_z

        return x


    bifurcation_q = []
    bifurcation_x = []

    for q in q_values:
        x_series = simulate_rossler(q, steps, h, a, b, c)

        if x_series[-1] != 0:
            x_steady = x_series[transient:]

            for i in range(1, len(x_steady) - 1):
                val = x_steady[i]
                if x_steady[i-1] < val and val > x_steady[i+1] and val > 0.1 and val < 30:
                    bifurcation_q.append(q)
                    bifurcation_x.append(val)

    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(bifurcation_q, bifurcation_x, s=0.5, c='blue', alpha=0.6)

    plt.title(f'Diagram bifurkacyjny systemu Rosslera')
    plt.xlabel('Rząd pochodnej (q)')
    plt.ylabel('Maksima X')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.7, 1.05)
    plt.ylim(0, 20)
    filename = "bifurkacja_q.png"
    filepath = os.path.join("graphs", filename)
    plt.savefig(filepath)
    if showFig:
        plt.show()