from utils import create_delay_vectors
import matplotlib.pyplot as plt

def phase_space_reconstruction(signal, m, tau, plot=True):
    embedded_vectors = create_delay_vectors(signal, m, tau)

    if m >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d') 
        
        X, Y, Z = embedded_vectors[:, 0], embedded_vectors[:, 1], embedded_vectors[:, 2]
        
        ax.plot(X, Y, Z, lw=0.5, alpha=0.8)
        ax.set_title(f'Phase Space Reconstruction ($e_D$={m}, $\\tau$={tau})')
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t + $\\tau$)')
        ax.set_zlabel('x(t + 2$\\tau$)')
        plt.show()
        
    elif m == 2:
        plt.figure(figsize=(8, 6))
        X, Y = embedded_vectors[:, 0], embedded_vectors[:, 1]
        plt.plot(X, Y, lw=0.5, alpha=0.8)
        plt.title(f'Phase Space Reconstruction ($e_D$={m}, $\\tau$={tau})')
        plt.xlabel('x(t)')
        plt.ylabel('x(t + $\\tau$)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()