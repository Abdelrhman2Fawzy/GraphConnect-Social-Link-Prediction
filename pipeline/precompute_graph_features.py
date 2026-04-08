from pathlib import Path 
import pickle 
import json 
import math
from pathlib import Path 
import pickle 
import json 
import math
import networkx as nx 
import numpy as np
from src.features.embeddings import compute_svd_embeddings

def load_graph(path : str | Path) -> nx.Graph:
    path = Path(path)
    with open(path,'rb') as f : 
        graph = pickle.load(f)
    return graph 


def save_pickle(obj , path : str | Path ) -> None :
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok= True)
    with open(path,'wb') as f :
        pickle.dump(obj,f , protocol=pickle.HIGHEST_PROTOCOL)


def save_json(obj , path : str | Path ) -> None :
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok= True)
    with open(path,'w') as f :
        json.dump(obj,f , indent=4)
def compute_weight_in_map(graph: nx.DiGraph) -> dict[int, float]:
    return {node: 1.0 / math.sqrt(1.0 + deg) for node, deg in graph.in_degree()}


def compute_weight_out_map(graph: nx.DiGraph) -> dict[int, float]:
    return {node: 1.0 / math.sqrt(1.0 + deg) for node, deg in graph.out_degree()}

def main() -> None:
    graph_path = Path("data/processed/graph.pkl")
    artifacts_dir = Path("artifacts")

    in_degree_path = artifacts_dir / "in_degree.pkl"
    out_degree_path = artifacts_dir / "out_degree.pkl"
    pagerank_path = artifacts_dir / "pagerank.pkl"
    metadata_path = artifacts_dir / "precompute_metadata.json"
    katz_path = artifacts_dir / "katz_centrality.pkl"
    hits_hubs_path = artifacts_dir / "hits_hubs.pkl"
    hits_auth_path = artifacts_dir / "hits_auth.pkl"
    weight_in_path = artifacts_dir / "weight_in.pkl"
    weight_out_path = artifacts_dir / "weight_out.pkl"
    svd_u_path = artifacts_dir / "svd_u.pkl"
    svd_v_path = artifacts_dir / "svd_v.pkl"

    print(f"Loading graph from: {graph_path}")
    graph = load_graph(graph_path)

    print("Computing in-degree map...")
    in_degree_map = dict(graph.in_degree())

    print("Computing out-degree map...")
    out_degree_map = dict(graph.out_degree())

    print("Computing PageRank...")
    pagerank_map = nx.pagerank(graph)

    print("Computing Katz Centrality...")
    katz_map = nx.katz_centrality(graph, alpha=0.005, beta=1.0)

    print("Computing HITS score...")
    hits_hubs, hits_auth = nx.hits(graph)

    print("Computing weight maps...")
    weight_in_map = compute_weight_in_map(graph)
    weight_out_map = compute_weight_out_map(graph)

    print("Computing SVD embeddings...")
    svd_results = compute_svd_embeddings(graph, k=6)

    print(f"Saving in-degree map to: {in_degree_path}")
    save_pickle(in_degree_map, in_degree_path)

    print(f"Saving out-degree map to: {out_degree_path}")
    save_pickle(out_degree_map, out_degree_path)

    print(f"Saving PageRank map to: {pagerank_path}")
    save_pickle(pagerank_map, pagerank_path)

    print(f"Saving Katz map to: {katz_path}")
    save_pickle(katz_map, katz_path)

    print(f"Saving HITS hubs map to: {hits_hubs_path}")
    save_pickle(hits_hubs, hits_hubs_path)

    print(f"Saving HITS authorities map to: {hits_auth_path}")
    save_pickle(hits_auth, hits_auth_path)
    print(f"Saving weight_in map to: {weight_in_path}")
    save_pickle(weight_in_map, weight_in_path)

    print(f"Saving weight_out map to: {weight_out_path}")
    save_pickle(weight_out_map, weight_out_path)

    print(f"Saving SVD U map to: {svd_u_path}")
    save_pickle(svd_results["U"], svd_u_path)

    print(f"Saving SVD V map to: {svd_v_path}")
    save_pickle(svd_results["V"], svd_v_path)

    metadata = {
        "graph_path": str(graph_path),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "saved_artifacts": {
            "in_degree": str(in_degree_path),
            "out_degree": str(out_degree_path),
            "pagerank": str(pagerank_path),
            "katz_centrality": str(katz_path),
            "hits_hubs": str(hits_hubs_path),
            "hits_auth": str(hits_auth_path),
            "weight_in": str(weight_in_path),
            "weight_out": str(weight_out_path),

        },
    }

    print(f"Saving metadata to: {metadata_path}")
    save_json(metadata, metadata_path)

    print("Done precomputing graph features.")


if __name__ == "__main__":
    main()