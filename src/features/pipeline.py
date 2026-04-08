import pandas as pd 
import networkx as nx 

from src.features.basic import (
    follows_back,
    compute_shortest_path_length,
)
from src.features.similarity import (
    get_graph_neighbors,
    jaccard_for_followees,
    jaccard_for_followers,
    cosine_for_followees,
    cosine_for_followers,
    adar_for_followees,
    adar_for_followers,
    intersection_of_followers,
    intersection_of_followees,
    preferential_attachment_followees,
    preferential_attachment_followers,
    resource_allocation_for_followees,
    resource_allocation_for_followers,
)
from src.features.embeddings import get_svd_features
import numpy as np
REQUIRED_COLUMNS = {"source_node","destination_node","label"}

def validate_pairs_df (df:pd.DataFrame) -> None :
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing :
        raise ValueError(f"Missing columns: {missing}")
    if df.isnull().any().any():
        raise ValueError("DataFrame contains NaN values")

def build_features_for_pairs(pairs_df: pd.DataFrame, graph: nx.DiGraph, precomputed: dict) -> pd.DataFrame:
    validate_pairs_df(pairs_df)
    
    node_features_df = None
    
    all_nodes = set()
    for key in ["in_degree", "out_degree", "pagerank", "katz_centrality", "hits_hubs", "hits_auth", "weight_in", "weight_out"]:
        if key in precomputed:
            all_nodes.update(precomputed[key].keys())
    
    nodes_list = sorted(list(all_nodes))
    node_df = pd.DataFrame({"node": nodes_list})
    
    node_df["in_degree"] = node_df["node"].map(precomputed.get("in_degree", {})).fillna(0)
    node_df["out_degree"] = node_df["node"].map(precomputed.get("out_degree", {})).fillna(0)
    node_df["pagerank"] = node_df["node"].map(precomputed.get("pagerank", {})).fillna(0.0)
    node_df["katz"] = node_df["node"].map(precomputed.get("katz_centrality", {})).fillna(0.0)
    node_df["hubs"] = node_df["node"].map(precomputed.get("hits_hubs", {})).fillna(0.0)
    node_df["auth"] = node_df["node"].map(precomputed.get("hits_auth", {})).fillna(0.0)
    node_df["weight_in"] = node_df["node"].map(precomputed.get("weight_in", {})).fillna(0.0)
    node_df["weight_out"] = node_df["node"].map(precomputed.get("weight_out", {})).fillna(0.0)

    src_node_df = node_df.rename(columns={
        "node": "source_node",
        "in_degree": "src_in_degree",
        "out_degree": "src_out_degree",
        "pagerank": "src_pagerank",
        "katz": "src_katz",
        "hubs": "src_hubs",
        "auth": "src_auth",
        "weight_in": "src_weight_in",
        "weight_out": "weight_out" 
    })
    
    src_node_df["pagerank_src"] = src_node_df["src_pagerank"]
    src_node_df["katz_src"] = src_node_df["src_katz"]
    src_node_df["hubs_src"] = src_node_df["src_hubs"]
    src_node_df["authorities_src"] = src_node_df["src_auth"]

    dst_node_df = node_df.rename(columns={
        "node": "destination_node",
        "in_degree": "dist_in_degree",
        "out_degree": "dist_out_degree",
        "pagerank": "dist_pagerank",
        "katz": "dist_katz",
        "hubs": "dist_hubs",
        "auth": "dist_auth",
        "weight_in": "weight_in", 
        "weight_out": "dist_weight_out"
    })
    
    dst_node_df["pagerank_dst"] = dst_node_df["dist_pagerank"]
    dst_node_df["katz_dst"] = dst_node_df["dist_katz"]
    dst_node_df["hubs_dst"] = dst_node_df["dist_hubs"]
    dst_node_df["authorities_dst"] = dst_node_df["dist_auth"]

    df = pairs_df.copy()
    df = df.merge(src_node_df, on="source_node", how="left")
    df = df.merge(dst_node_df, on="destination_node", how="left")
    
    df = df.fillna(0)

    df["weight_in_plus_weight_out"] = df["weight_in"] + df["weight_out"]
    df["weight_in_mul_weight_out"] = df["weight_in"] * df["weight_out"]
    df["weight_in_2x_plus_weight_out"] = 2.0 * df["weight_in"] + df["weight_out"]
    df["weight_in_plus_weight_out_2x"] = df["weight_in"] + 2.0 * df["weight_out"]

    successors, predecessors = get_graph_neighbors(graph)
    
    svd_embeddings = {
        "U": precomputed.get("svd_u", {}),
        "V": precomputed.get("svd_v", {}),
    }

    # Optimized feature generation
    sources = df.source_node.values
    dists = df.destination_node.values

    # Skip shortest_path if it's too slow (keeping it for now but optimizing loop)
    # To keep old code as requested:
    """
    results = []
    for row in df.itertuples(index=False):
        src, dist = row.source_node, row.destination_node
        
        graph_feats = {
            "follows_back": follows_back(graph, src, dist),
            "jaccard_for_followees": jaccard_for_followees(src, dist, successors),
            "jaccard_for_followers": jaccard_for_followers(src, dist, predecessors),
            "cosine_for_followees": cosine_for_followees(src, dist, successors),
            "cosine_for_followers": cosine_for_followers(src, dist, predecessors),
            "adar_for_followees": adar_for_followees(src, dist, successors, predecessors),
            "adar_for_followers": adar_for_followers(src, dist, predecessors, successors),
            "shortest_path_length": compute_shortest_path_length(graph, src, dist),
            "number_of_followers_s": len(predecessors.get(src, set())),
            "number_of_followees_s": len(successors.get(src, set())),
            "number_of_followers_d": len(predecessors.get(dist, set())),
            "number_of_followees_d": len(successors.get(dist, set())),
            "intersection_of_followers": intersection_of_followers(src, dist, predecessors),
            "intersection_of_followees": intersection_of_followees(src, dist, successors),
            "preferential_attachment_followees": preferential_attachment_followees(src, dist, successors),
            "preferential_attachment_followers": preferential_attachment_followers(src, dist, predecessors),
            "ra_for_followees": resource_allocation_for_followees(src, dist, successors, predecessors),
            "ra_for_followers": resource_allocation_for_followers(src, dist, predecessors, successors),
            **get_svd_features(src, dist, svd_embeddings),
        }
        results.append(graph_feats)
    """

    results = [
        {
            "follows_back": 1 if graph.has_edge(d, s) else 0,
            "jaccard_for_followees": jaccard_for_followees(s, d, successors),
            "jaccard_for_followers": jaccard_for_followers(s, d, predecessors),
            "cosine_for_followees": cosine_for_followees(s, d, successors),
            "cosine_for_followers": cosine_for_followers(s, d, predecessors),
            "adar_for_followees": adar_for_followees(s, d, successors, predecessors),
            "adar_for_followers": adar_for_followers(s, d, predecessors, successors),
            "shortest_path_length": compute_shortest_path_length(graph, s, d),
            "number_of_followers_s": len(predecessors.get(s, set())),
            "number_of_followees_s": len(successors.get(s, set())),
            "number_of_followers_d": len(predecessors.get(d, set())),
            "number_of_followees_d": len(successors.get(d, set())),
            "intersection_of_followers": intersection_of_followers(s, d, predecessors),
            "intersection_of_followees": intersection_of_followees(s, d, successors),
            "preferential_attachment_followees": len(successors.get(s, set())) * len(successors.get(d, set())),
            "preferential_attachment_followers": len(predecessors.get(s, set())) * len(predecessors.get(d, set())),
            "ra_for_followees": resource_allocation_for_followees(s, d, successors, predecessors),
            "ra_for_followers": resource_allocation_for_followers(s, d, predecessors, successors),
            **get_svd_features(s, d, svd_embeddings),
        }
        for s, d in zip(sources, dists)
    ]
    
    graph_features_df = pd.DataFrame(results)
    final_df = pd.concat([df, graph_features_df], axis=1)
    
    return final_df
