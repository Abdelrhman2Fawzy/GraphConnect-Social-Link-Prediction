from pathlib import Path
import pickle

import pandas as pd
import networkx as nx

from src.features.load_precomputed import load_precomputed_features
from src.features.pipeline import build_features_for_pairs


def load_graph(path: str | Path) -> nx.DiGraph:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)


def process_in_batches(pairs_df: pd.DataFrame, graph: nx.DiGraph, precomputed: dict, output_path: Path, batch_size: int = 10000) -> None:
    num_batches = (len(pairs_df) + batch_size - 1) // batch_size
    print(f"Total rows: {len(pairs_df)}, batch size: {batch_size}, total batches: {num_batches}")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(pairs_df))
        batch_df = pairs_df.iloc[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches} (rows {start_idx} to {end_idx})...")
        features_batch = build_features_for_pairs(
            pairs_df=batch_df,
            graph=graph,
            precomputed=precomputed,
        )
        
        if i == 0:
            features_batch.to_csv(output_path, index=False)
        else:
            features_batch.to_csv(output_path, mode='a', header=False, index=False)
        
        del features_batch

def main() -> None:
    train_pairs_path = Path("data/processed/train_pairs.csv")
    test_pairs_path = Path("data/processed/test_pairs.csv")
    graph_path = Path("data/processed/graph.pkl")

    train_features_path = Path("data/processed/train_features.csv")
    test_features_path = Path("data/processed/test_features.csv")

    print(f"Loading train pairs from: {train_pairs_path}")
    train_pairs = pd.read_csv(train_pairs_path)

    print(f"Loading test pairs from: {test_pairs_path}")
    test_pairs = pd.read_csv(test_pairs_path)

    print(f"Loading graph from: {graph_path}")
    graph = load_graph(graph_path)

    print("Loading precomputed artifacts...")
    precomputed = load_precomputed_features("artifacts")

    print("Building train feature table in batches...")
    process_in_batches(train_pairs, graph, precomputed, train_features_path)

    print("Building test feature table in batches...")
    process_in_batches(test_pairs, graph, precomputed, test_features_path)

    print("Done.")


if __name__ == "__main__":
    main()