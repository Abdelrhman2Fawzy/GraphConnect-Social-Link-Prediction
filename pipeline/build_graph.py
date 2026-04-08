from pathlib import Path 

from src.data.build_graph import(
    build_graph,
    compute_graph_stats,
    save_graph,
    save_graph_stats,
)
from src.data.load_edges import load_edges
def main() -> None: 
    print("Loading edges...")
    edges_df = load_edges("data/processed/edges.csv")
    print("Building graph...")
    graph = build_graph(edges_df)
    print("Computing graph stats...")
    stats = compute_graph_stats(graph)
    print("Saving graph...")
    save_graph(graph, "data/processed/graph.pkl")
    print("Saving graph stats...")
    save_graph_stats(stats, "data/processed/graph_stats.json")
    print("Done!")

if __name__ == "__main__":
    main()