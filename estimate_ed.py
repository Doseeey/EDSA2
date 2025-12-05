import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import nolds

def create_delay_vectors(signal, m, tau):
    """
    Creates delay embedded vectors from a time series (Takens' Theorem).

    Parameters:
    - signal (np.array): 1D time series (e.g., [x(t), x(t+1), ...]).
    - m (int): Embedding dimension.
    - tau (int): Time delay (lag).

    Returns:
    - np.array: Array of embedded vectors (N_vectors x m).
    """
    N = len(signal)
    # The number of vectors N_m that can be created
    N_vectors = N - (m - 1) * tau
    
    if N_vectors <= 0:
        raise ValueError("Signal is too short for the given m and tau.")
    
    # Create the matrix of embedded vectors
    embedded_vectors = np.array([
        signal[i : i + m * tau : tau]
        for i in range(N_vectors)
    ])
    return embedded_vectors

def correlation_sum(embedded_vectors, r_values):
    """
    Calculates the correlation sum C(r) for a set of radii r.

    C(r) = (2 / N(N-1)) * Sum(Theta(r - ||Xi - Xj||)) where Theta is the Heaviside
    step function, counting the fraction of pairs closer than r.

    Parameters:
    - embedded_vectors (np.array): Phase space vectors.
    - r_values (np.array): Array of radius thresholds.

    Returns:
    - np.array: The correlation sum C(r) for each radius in r_values.
    """
    N = embedded_vectors.shape[0]
    
    # Calculate all pairwise Euclidean distances
    # pdist returns a condensed distance matrix (a 1D array of distances)
    distances = pdist(embedded_vectors, metric='euclidean')
    
    # Total number of unique pairs
    N_total_pairs = len(distances)
    if N_total_pairs == 0:
        return np.zeros_like(r_values)

    C_r = np.zeros_like(r_values, dtype=float)
    
    # Calculate C(r) for each radius
    for i, r in enumerate(r_values):
        # Count pairs whose distance is less than or equal to r
        N_pairs_less_than_r = np.sum(distances <= r)
        C_r[i] = N_pairs_less_than_r / N_total_pairs

    return C_r

def grassberger_procaccia(signal, m, tau, log_r, r_range_indices):
    """
    Performs delay embedding, calculates C(r), and estimates D2 for a single m.

    Parameters:
    - signal (np.array): 1D time series.
    - m (int): Embedding dimension.
    - tau (int): Time delay (lag).
    - log_r (np.array): Logarithm of radii (r).
    - r_range_indices (tuple): (start_idx, end_idx) for the scaling region fit.

    Returns:
    - tuple: (D2_estimate, log_C) where log_C is log(C(r)).
    """
    try:
        # 1. Delay Embedding
        embedded = create_delay_vectors(signal, m, tau)
    except ValueError as e:
        print(f"Error for m={m}: {e}")
        return 0.0, None

    # Radii are calculated outside and passed as log_r (log(r))
    radii = np.exp(log_r)

    # 2. Calculate Correlation Sums
    C_r = correlation_sum(embedded, radii)

    # Filter out C(r) == 0 to avoid log(0)
    # Note: We keep the full range for plotting, but use 1e-10 as a minimum for log
    log_C = np.log(np.maximum(C_r, 1e-10))
    
    # 3. Estimate D2 (Slope) in the Scaling Region
    start_idx, end_idx = r_range_indices
    
    # Ensure the scaling region is valid (C(r) > 0 and enough points)
    scaling_log_r = log_r[start_idx:end_idx]
    scaling_log_C = log_C[start_idx:end_idx]
    
    # Only fit if we have enough points (at least 2 for a line)
    if len(scaling_log_r) < 2 or np.all(scaling_log_C == scaling_log_C[0]):
        D2_estimate = 0.0
    else:
        # Linear least squares fit (log(C(r)) = D2 * log(r) + intercept)
        slope, intercept = np.polyfit(scaling_log_r, scaling_log_C, 1)
        D2_estimate = slope
        
    return D2_estimate, log_C


# --- 4. Main Estimation and Plotting Function ---

def estimate_correlation_dimension_range(
    signal,
    m_min=1,
    m_max=10,
    tau=1,
    r_min_perc=5,
    r_max_perc=70,
    r_fit_start_perc=20,
    r_fit_end_perc=50,
    num_r=50
):
    """
    Estimates the correlation dimension (D2) over a range of embedding dimensions (m),
    plots the log-log curves and the D2 convergence plot.

    Parameters:
    - signal (np.array): 1D time series.
    - m_min (int): Minimum embedding dimension.
    - m_max (int): Maximum embedding dimension.
    - tau (int): Time delay (lag).
    - r_min_perc (int): Lower bound of radii range as a percentage of max distance.
    - r_max_perc (int): Upper bound of radii range as a percentage of max distance.
    - r_fit_start_perc (int): Start of the scaling region for D2 fitting (percentage).
    - r_fit_end_perc (int): End of the scaling region for D2 fitting (percentage).
    - num_r (int): Number of logarithmically spaced radii.

    Returns:
    - float: The converged (estimated) correlation dimension.
    """
    print(f"--- Grassberger-Procaccia Analysis (tau={tau}) ---")

    # --- Setup Radii (r) ---
    # Use the first embedded vector set (m=m_min) to determine a sensible range for r
    temp_embedded = create_delay_vectors(signal, m_min, tau)
    # The max distance in the phase space
    max_dist = np.max(pdist(temp_embedded, metric='euclidean'))
    
    # Calculate log-spaced radii within the specified percentile range
    r_min = max_dist * r_min_perc / 100
    r_max = max_dist * r_max_perc / 100
    
    if r_max <= r_min:
        print("Error: Max radius must be greater than min radius. Check r_max_perc.")
        return 0.0

    r_values = np.logspace(np.log10(r_min), np.log10(r_max), num=num_r)
    log_r = np.log(r_values)

    # Determine the indices for the fitting region
    r_fit_start_val = max_dist * r_fit_start_perc / 100
    r_fit_end_val = max_dist * r_fit_end_perc / 100
    
    start_idx = np.searchsorted(r_values, r_fit_start_val)
    end_idx = np.searchsorted(r_values, r_fit_end_val)
    
    r_range_indices = (start_idx, end_idx)
    
    if end_idx <= start_idx + 1:
        print(f"Warning: Scaling region too narrow for fit. Using index range: {r_range_indices}")
        print("Adjust r_fit_start_perc and r_fit_end_perc.")


    # --- Run GP for all embedding dimensions (m) ---
    embedding_dims = range(m_min, m_max + 1)
    D2_estimates = []
    log_C_data = [] # Stores log_C for plotting

    for m in embedding_dims:
        D2, log_C = grassberger_procaccia(signal, m, tau, log_r, r_range_indices)
        if log_C is not None:
            D2_estimates.append(D2)
            log_C_data.append(log_C)
            print(f"  m={m}: D2 estimate = {D2:.4f}")

    # --- Plot 1: Log-Log Plot (log C(r) vs log r) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.set_title(f'Log-Log Plot: $\\log C_m(r)$ vs $\\log r$ ($\\tau$={tau})')
    ax1.set_xlabel('$\\log r$ (Radius)')
    ax1.set_ylabel('$\\log C_m(r)$ (Correlation Sum)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Add the fitting region as a shaded area
    fit_range = log_r[start_idx:end_idx]
    if len(fit_range) > 0:
        ax1.axvspan(fit_range[0], fit_range[-1], color='gray', alpha=0.1, label='Scaling Region')

    for m_idx, m in enumerate(embedding_dims):
        if m_idx < len(log_C_data):
            ax1.plot(log_r, log_C_data[m_idx], label=f'm={m}', marker='.', markersize=4, alpha=0.7)
            
            # Optionally plot the fitted line segment (slope D2)
            fitted_line = D2_estimates[m_idx] * log_r + (log_C_data[m_idx] - D2_estimates[m_idx] * log_r)[start_idx]
            ax1.plot(fit_range, fitted_line[start_idx:end_idx], color='black', linestyle='--')

    ax1.legend(loc='lower right')

    # --- Plot 2: Convergence Plot (D2 vs m) ---
    ax2 = axes[1]
    ax2.set_title('Correlation Dimension Convergence')
    ax2.set_xlabel('Embedding Dimension ($m$)')
    ax2.set_ylabel('Estimated Dimension ($D_2$)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    m_estimates = np.array(list(embedding_dims)[:len(D2_estimates)])
    ax2.plot(m_estimates, D2_estimates, 'o-', color='blue', label='D2 Estimate')

    # --- Estimate Converged Dimension (D_c) ---
    # D_c is estimated as the stable plateau value. We check if the last few values
    # have stabilized within a tolerance (e.g., 5% change).
    converged_D2 = 0.0
    if len(D2_estimates) > 2:
        # Check if the last 3 points are close
        last_three = np.array(D2_estimates[-3:])
        if np.ptp(last_three) / np.mean(last_three) < 0.05: # Peak-to-peak change < 5%
            converged_D2 = np.mean(last_three)
            ax2.axhline(converged_D2, color='red', linestyle='-', linewidth=2, label=f'Converged $D_2 \\approx {converged_D2:.4f}$')
            ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    return converged_D2

def estimate_ed(
    signal,
    m_min=1,
    m_max=10,
    tau=10
):
    """
    Estimates the correlation dimension (D2) over a range of embedding dimensions (m)
    using the nolds library, and plots the results.

    Parameters:
    - signal (np.array): 1D time series.
    - m_min (int): Minimum embedding dimension.
    - m_max (int): Maximum embedding dimension.
    - tau (int): Time delay (lag).

    Returns:
    - float: The final estimated correlation dimension (D2) from nolds.
    """
    if nolds is None:
        return 0.0

    print(f"--- Nolds Grassberger-Procaccia Analysis (tau={tau}) ---")

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

        # D2_estimates is a dictionary mapping m to the D2 value found for that m
        D2_estimates.append(D2_final)
        # log_r is the common array of log(r) values used for all m
        log_r.append(debug_data[0])
        # log_C_data is a dictionary mapping m to the array of log(C(r)) values
        log_C.append(debug_data[1])
    
    # 1. Plot the Log-Log Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.set_title(f'Nolds Log-Log Plot: $\\log C_m(r)$ vs $\\log r$ ($\\tau$={tau})')
    ax1.set_xlabel('$\\log r$ (Radius)')
    ax1.set_ylabel('$\\log C_m(r)$ (Correlation Sum)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    for m in range(m_max):
        ax1.plot(log_r[m], log_C[m], label=f'm={m}', marker='.', markersize=4, alpha=0.7)
        

    ax1.legend(loc='lower right')
    print("--- D2 Estimates per Embedding Dimension (m) ---")
    for D2 in D2_estimates:
        print(f"D2 estimate = {D2:.4f}")
        
    # 2. Plot the Convergence Plot (D2 vs m)
    ax2 = axes[1]
    ax2.set_title('Correlation Dimension Convergence (Nolds)')
    ax2.set_xlabel('Embedding Dimension ($m$)')
    ax2.set_ylabel('Estimated Dimension ($D_2$)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    m_values = np.array(range(m_min, m_max+1))
    D2_values = np.array(list(D2_estimates))
    
    # Plot D2 vs m
    ax2.plot(m_values, D2_values, 'o-', color='blue', label='D2 Estimate')

    # Mark the final converged dimension (D2_final is the plateau value)
    if D2_final is not None:
        ax2.axhline(D2_final, color='red', linestyle='-', linewidth=2, label=f'Converged $D_2 \\approx {D2_final:.4f}$')
        ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    return D2_final