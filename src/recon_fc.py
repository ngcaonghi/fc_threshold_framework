import numpy as np
from utils import prep_parcellation
import warnings


def weak_recoverability(
    fc, 
    parcellation, 
    tau,
    binarize=False
):
    """
    Computes the weak recoverability of a functional connectivity (FC) matrix.

    Parameters:
    fc (numpy.ndarray): Functional connectivity matrix (n x n).
    parcellation (numpy.ndarray): Parcellation vector of length n, indicating the community assignment of each node.
    tau (float): Threshold.
    binarize (bool, optional): If True, binarize the FC matrix based on the threshold tau. Defaults to False.

    Returns:
    float: The Sandon and Abbe signal-to-noise ratio (SNR).

    Raises:
    Warning: If the largest eigenvalue of the community profile matrix is close to zero, which indicates potential numerical instability.
    """
    parcellation = prep_parcellation(parcellation, fc.shape[-1])
    n = len(parcellation)
    k = len(np.unique(parcellation))
    Omega = np.zeros((k))
    for i in range(k):
        Omega[i] = np.sum(parcellation == i)
    P = np.diag(Omega) / n
    if binarize:
        A = np.ones_like(fc) * (np.abs(fc) > tau)
    else:
        A = np.abs(fc) * (np.abs(fc) > tau)
    Q = np.zeros((k,k))
    for i in range(n): # unordered nodes {0, 1,2,3,4..,n-1} in matrix A_tau
        for j in range(n):
            Q[parcellation[i],parcellation[j]] = \
                Q[parcellation[i],parcellation[j]] + A[i,j]
    # Eig decompostion of PQ
    H = np.matmul(P,Q)
    w,_ = np.linalg.eig(H)
    w = np.sort(w)
    # Sandon and Abbe SNR
    if w[k-1] <= 0.001:
        warnings.warn(
            'Largest eignvalue of community profile matrix P @ Q is close to zero.', 
            InfWarning
        )
        return np.inf
    return w[k-2]**2 / w[k-1]


def vetting(
    fcs,
    taus,
    parcellation   
):
    """
    Determine the weak recoverability interval.

    Parameters:
    fcs (numpy.ndarray): A 3D array of shape (n_subjects, n_nodes, n_nodes) containing individual functional connectivity matrices.
    taus (numpy.ndarray): An array of thresholds.
    parcellation (numpy.ndarray): Parcellation vector indicating community assignment of each node.

    Returns:
    numpy.ndarray: Array of SNRs that are greater than 1.
    
    Raises:
    ValueError: If fcs is not a 3D array or if its last two dimensions are not equal.
    """
    if np.ndim(fcs) != 3:
        raise ValueError('Make sure fcs is an array containing all individual fcs, i.e., np.ndim(fcs) == 3.')
    if fcs.shape[-1] != fcs.shape[-2]:
        raise ValueError('Make sure fcs has shape (n_subjects, n_nodes, n_nodes).')
    fc_avg = fcs.mean(axis=0)
    snrs = np.zeros_like(taus)
    for t, tau in enumerate(taus):
        snrs[t] = weak_recoverability(fc_avg, parcellation, tau, binarize=True)
    wr_interval = snrs[1 < snrs]
    return wr_interval


def recon_fc(
    fcs,
    taus,
    parcellation
):
    """
    Identify weakly recoverable subjects and their optimal thresholds.

    Parameters:
    fcs (numpy.ndarray): A 3D array of shape (n_subjects, n_nodes, n_nodes) containing individual functional connectivity matrices.
    taus (numpy.ndarray): An array of thresholds.
    parcellation (numpy.ndarray): Parcellation vector indicating community assignment of each node.

    Returns:
    tuple: 
        - numpy.ndarray: Optimal threshold (tau) for each subject.
        - numpy.ndarray: Boolean array indicating whether each subject's FC matrix is weakly recoverable.
    """
    parcellation = prep_parcellation(parcellation, fcs.shape[-1])
    wr_interval = vetting(fcs, taus, parcellation)
    recoverable = np.zeros([fcs.shape[0]], dtype=bool)
    tau_opt = np.zeros((fcs.shape[0]))
    for subject, fc in enumerate(fcs):
        snrs = np.zeros((taus.shape[0]))
        for t, tau in enumerate(taus):
            snrs[t] = weak_recoverability(fc, parcellation, tau)
        topt = taus[np.argmax(snrs)]
        if topt in wr_interval:
            recoverable[subject] = True
            tau_opt[subject] = topt
    return tau_opt, recoverable


def weak_recoverability_null_model(
    fc, 
    parcellation, 
    tau,
    n_permutations=100,
    seed=42,
    binarize=False
):
    """
    Computes the distribution of weak recoverability (SNR) under a null model by permuting the parcellation.

    Parameters:
    fc (numpy.ndarray): Functional connectivity matrix (n x n).
    parcellation (numpy.ndarray): Parcellation vector of length n, indicating the community assignment of each node.
    tau (float): Threshold value.
    n_permutations (int, optional): Number of permutations to generate the null model. Defaults to 100.
    seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 42.
    binarize (bool, optional): If True, binarize the FC matrix based on the threshold tau. Defaults to False.

    Returns:
    numpy.ndarray: An array of SNR values obtained from the null model (one for each permutation).
    """
    rng = np.random.default_rng(seed)
    snr_distribution = np.zeros((n_permutations))
    parcellation = prep_parcellation(parcellation, fc.shape[-1])
    for n in range(n_permutations):
        parcellation_permed = rng.permuted(parcellation)
        snr_distribution[n] = weak_recoverability(
            fc, parcellation_permed, tau, binarize
        )
    return snr_distribution


class InfWarning(UserWarning):
    pass