import numpy as np
import matplotlib.pyplot as plt


def random_signal(n: int, display_plot: bool = False):  
    data = np.random.rand(n)

    # if display_plot:
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111)
    #     ax.set_title(f'Szereg Czasowy z sygnału losowego')
    #     ax.plot(data, color='black', label=f'Szereg Czasowy z sygnału losowego', linewidth=0.8)
    #     plt.show()

    return data

def periodic_signal(n: int, display_plot: bool = False):
    arr = np.arange(n)
    data = np.sin(arr * 0.1) + np.cos(arr * 0.1)

    # if display_plot:
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111)
    #     ax.set_title(f'Szereg Czasowy z sygnału periodycznego')
    #     ax.plot(data, color='black', label=f'Szereg Czasowy z sygnału periodycznego', linewidth=0.8)
    #     plt.show()

    return data

def chaotic_signal(filename: str, display_plot: bool = False):
    data = np.genfromtxt(filename, delimiter=',')

    # if display_plot:
    #     x = data[:, 0]
    #     y = data[:, 1]
    #     z = data[:, 2]

    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_title(f'Szereg Czasowy z pliku {filename}')
    #     ax.plot(x, y, z, color='black', label=f'Szereg Czasowy z pliku {filename}', linewidth=0.8)
    #     plt.show()

    return data


n = 10000
random_signal(n, display_plot=True)
periodic_signal(n, display_plot=True)
chaotic_signal("data/lorenz_classic.csv", display_plot=True)
chaotic_signal("data/rossler_classic.csv", display_plot=True)
chaotic_signal("data/rossler_fractional_0_9.csv", display_plot=True)
chaotic_signal("data/rossler_fractional_0_88.csv", display_plot=True)

