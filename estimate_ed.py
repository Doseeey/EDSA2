import numpy as np
import matplotlib.pyplot as plt
import nolds

def estimate_ed(signal, m_min=1, m_max=10, tau=10):
    print(f"--- Embedding Dimension Analysis (tau={tau}) ---")

    embedding_dims = range(m_min, m_max + 1)
    D2_estimates = []
    log_r = []
    log_C = []
    
    for emb_dim in embedding_dims:
        print(f"Estimating dimension {emb_dim}")
        D2_final, debug_data = nolds.corr_dim(
            signal,
            emb_dim=emb_dim,
            lag=tau,
            fit="poly",
            debug_data=True
        )

        D2_estimates.append(D2_final)
        log_r.append(debug_data[0])
        log_C.append(debug_data[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.set_title(f'$\\log C_2(r)$ vs $\\log r$ ($\\tau$={tau})')
    ax1.set_xlabel('$\\log r$')
    ax1.set_ylabel('$\\log C_2(r)$')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    for m in range(m_max):
        ax1.plot(log_r[m], log_C[m], label=f'm={m}', marker='.', markersize=4, alpha=0.7)
        

    ax1.legend(loc='lower right')
    print("--- D2 Estimates per Embedding Dimension (m) ---")
    for D2 in D2_estimates:
        print(f"D2 estimate = {D2:.4f}")
        
    ax2 = axes[1]
    ax2.set_title('Correlation Dimension')
    ax2.set_xlabel('$e_D$')
    ax2.set_ylabel('$D_2$')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    m_values = np.array(range(m_min, m_max+1))
    D2_values = np.array(list(D2_estimates))
    
    ax2.plot(m_values, D2_values, 'o-', color='blue', label='D2 Estimate')

    plt.tight_layout()
    plt.show()

    return D2_final