import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def compute_svd_embeddings(graph: nx.DiGraph, k: int = 6) -> dict[str, np.ndarray]:
    """
    Compute SVD on the adjacency matrix and return U and V matrices.
    U represents source node features, V represents destination node features.
    """
    nodes = sorted(list(graph.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create sparse adjacency matrix
    rows, cols = [], []
    for u, v in graph.edges():
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])
    
    adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
    
    # Compute SVD
    # svds returns U, S, Vt
    u_mat, s_mat, vt_mat = svds(adj_matrix.astype(float), k=k)
    
    # Store results in dictionaries for fast lookup
    u_dict = {node: u_mat[i] for node, i in node_to_idx.items()}
    v_dict = {node: vt_mat.T[i] for node, i in node_to_idx.items()}
    
    return {
        "U": u_dict,
        "V": v_dict
    }


def get_svd_features(src: int, dst: int, svd_embeddings: dict[str, dict]) -> dict[str, float]:
    """
    Compute dot product and other metrics from SVD embeddings.
    """
    u_src = svd_embeddings["U"].get(src, np.zeros(6))
    v_dst = svd_embeddings["V"].get(dst, np.zeros(6))
    
    # Dot product
    dot_prod = np.dot(u_src, v_dst)
    
    # Concatenated (if needed by model indirectly, but usually better to let model handle)
    # Here we just return dot product as it's the most common "link" feature from SVD
    return {
        "svd_dot_product": float(dot_prod)
    }
