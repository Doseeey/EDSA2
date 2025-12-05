from utils import create_delay_vectors
import matplotlib.pyplot as plt

def phase_space_reconstruction(signal, m, tau, plot=True):
    print(f"\n--- Phase Space Reconstruction (m={m}, tau={tau}) ---")
    embedded_vectors = create_delay_vectors(signal, m, tau)

    if m >= 3:
        # 3D Plot for m >= 3
        fig = plt.figure(figsize=(10, 8))
        # The '3d' projection is needed for 3D plots
        ax = fig.add_subplot(111, projection='3d') 
        
        # Since we only have three physical axes to plot, we take the first three dimensions
        X, Y, Z = embedded_vectors[:, 0], embedded_vectors[:, 1], embedded_vectors[:, 2]
        
        ax.plot(X, Y, Z, lw=0.5, alpha=0.8)
        ax.set_title(f'3D Phase Space Reconstruction (m={m}, $\\tau$={tau})')
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t + $\\tau$)')
        ax.set_zlabel('x(t + 2$\\tau$)')
        plt.show()
        
    elif m == 2:
        # 2D Plot for m = 2
        plt.figure(figsize=(8, 6))
        X, Y = embedded_vectors[:, 0], embedded_vectors[:, 1]
        plt.plot(X, Y, lw=0.5, alpha=0.8)
        plt.title(f'2D Phase Space Reconstruction (m={m}, $\\tau$={tau})')
        plt.xlabel('x(t)')
        plt.ylabel('x(t + $\\tau$)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()