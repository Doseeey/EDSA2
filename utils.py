import numpy as np
from scipy.stats import entropy

def find_zero_or_minimum(acf_values: np.array) -> int:
    threshold = 0.5 * acf_values[0]
    for i in range(len(acf_values) - 1):
        #tau jezeli przekroczy 0
        if acf_values[i] > 0 and acf_values[i+1] <= 0:
            return i+1
        #tau jezeli minimum lokalne
        if i + 2 < len(acf_values):
            if acf_values[i] > acf_values[i+1] and acf_values[i+1] <= acf_values[i+2]:
                return i+1
        #tau jezeli acf[i] <= 0.5*acf[0]
        if acf_values[i] < threshold:
            return i
    return 1 

def calculate_average_mutual_information(time_series, max_tau, num_bins=32):
    N = len(time_series)
    ami_values = []
    hist_range = (time_series.min(), time_series.max())
    P_x, bin_edges = np.histogram(time_series, bins=num_bins, range=hist_range, density=True)
    P_x[P_x == 0] = np.finfo(float).eps

    # calc ami for all taus
    for tau in range(1, max_tau + 1):

        n_overlap = N - tau

        X_t = time_series[:n_overlap]
        X_t_plus_tau = time_series[tau:N]
        
        P_joint, _, _ = np.histogram2d(
            X_t, 
            X_t_plus_tau, 
            bins=num_bins, 
            range=[hist_range, hist_range], 
            density=True
        )
        # Ensure no zero probabilities
        P_joint[P_joint == 0] = np.finfo(float).eps
        
        H_marginal = entropy(P_x, base=np.e)
        H_joint = entropy(P_joint.flatten(), base=np.e)
        
        I_tau = 2 * H_marginal - H_joint
        ami_values.append(I_tau)

    return np.array(ami_values)

def find_first_local_minimum(ami_values):
    for i in range(1, len(ami_values) - 1):
        if ami_values[i] < ami_values[i-1] and ami_values[i] <= ami_values[i+1]:
            return i + 1, ami_values[i]
    return None, None

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