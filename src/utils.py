import numpy as np
import graph_tool as gt
from sklearn.metrics.cluster import adjusted_mutual_info_score


def to_graph_tool(adj):
    """
    Converts an adjacency matrix to a graph-tool Graph object.

    Parameters:
    adj (numpy.ndarray): Adjacency matrix (2D numpy array) representing the graph.

    Returns:
    gt.Graph: A graph-tool Graph object with edge weights from the adjacency matrix.
    """
    g = gt.Graph(directed=False)
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights
    nnz = np.nonzero(np.triu(adj,1))
    nedges = len(nnz[0])
    g.add_edge_list(
        np.hstack(
            [np.transpose(nnz),
             np.reshape(adj[nnz], (nedges,1))]
        ), 
        eprops=[edge_weights]
    )
    return g


def prep_parcellation(parcellation, n):
    """
    Prepares the parcellation vector by ensuring it is 1D and adjusting the values if necessary.

    Parameters:
    parcellation (numpy.ndarray): Parcellation vector indicating community assignment of each node.
    n (int): Number of nodes in the functional connectivity matrix.

    Returns:
    numpy.ndarray: Processed parcellation vector.

    Raises:
    ValueError: If parcellation is not a 1D array or if its length does not match the number of nodes.
    """
    if np.ndim(parcellation) != 1:
        raise ValueError('Make sure parcellation is a 1D array.')
    if np.all(np.unique(parcellation)==np.arange(1, 9)):
        parcellation = parcellation - 1
        parcellation = np.delete(parcellation, np.where(parcellation == 7))
    if len(parcellation) != n:
        raise ValueError('Make sure the parcellation size matches the number of fc vertices.')
    return parcellation


def sbm_ami(
    fc,
    parcellation,
    tau,
    n_samples
):
    """
    Computes the adjusted mutual information (AMI) score between the parcellation and the inferred partition using a stochastic block model (SBM).

    Parameters:
    fc (numpy.ndarray): Functional connectivity matrix (n x n).
    parcellation (numpy.ndarray): Parcellation vector indicating community assignment of each node.
    tau (float): Threshold for binarizing the FC matrix or for element-wise comparison.
    n_samples (int): Number of samples to draw for SBM inference.

    Returns:
    float: Average AMI score over the specified number of samples.
    """
    parcellation = prep_parcellation(parcellation, fc.shape[-1])
    A_masked = np.abs(fc) * (np.abs(fc) > tau)
    adj_matx = np.around(A_masked * 100)
    graph = to_graph_tool(adj_matx)
    n = len(parcellation)
    mi_snr = 0
    for _ in range(n_samples):
        infer_partition = np.zeros(n)
        state = gt.inference.minimize_blockmodel_dl(
            graph, state_args=dict(
                recs=[graph.ep.weight], 
                rec_types=["discrete-poisson"], 
                deg_corr=False)
            )
        b = state.get_blocks()
        for i in range(n):
            infer_partition[i] = b[i]
        infer_partition = infer_partition.astype(np.uint8)
        mi_snr += adjusted_mutual_info_score(
            parcellation, infer_partition, average_method='max'
        )
    mi_snr /= n_samples
    return mi_snr

