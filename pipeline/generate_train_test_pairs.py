from pathlib import Path
import json
import pickle

import networkx as nx
import pandas as pd

from src.data.split_edges import split_positive_edges
from src.data.sample_negatives import sample_negative_edges

def load_graph(path : Path) -> nx.DiGraph : 
    with open(path,"rb") as f :
        return pickle.load(f)
def save_csv(df : pd.DataFrame , path : Path) -> None : 
    df.to_csv(path,index=False)

def save_metadata(metadata : dict ,path : Path ) -> None : 
    with open(path , "w",encoding="utf-8") as f : 
        json.dump(metadata,f,ensure_ascii=False,indent=2)


def main() -> None : 
    edges_path = Path("data/interim/train_edges.csv")
    full_graph_path = Path("data/processed/graph.pkl")

    train_pos_path = Path("data/interim/train_pos_edges.csv")
    test_pos_path = Path("data/interim/test_pos_edges.csv")
    train_neg_path = Path("data/interim/train_neg_edges.csv")
    test_neg_path = Path("data/interim/test_neg_edges.csv")
    metadata_path = Path("data/interim/pair_generation_metadata.json")
    random_state = 42
    test_size = 0.2
    
    print(f"Loading positive edges from: {edges_path}")
    df_edges = pd.read_csv(edges_path)

    # ---------- split positives ----------
    print("Splitting positive edges into train/test...")
    train_pos, test_pos = split_positive_edges(
        df_pos=df_edges,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Train positives: {len(train_pos)}")
    print(f"Test positives: {len(test_pos)}")

    print(f"Loading full graph from: {full_graph_path}")
    graph = load_graph(full_graph_path)

    # ---------- sample negatives ----------
    print("Sampling negative edges for train...")
    train_neg = sample_negative_edges(
        graph=graph,
        n_samples=len(train_pos),
        random_state=random_state,
    )

    print("Sampling negative edges for test...")
    test_neg = sample_negative_edges(
        graph=graph,
        n_samples=len(test_pos),
        random_state=random_state,
    )

    print(f"Train negatives: {len(train_neg)}")
    print(f"Test negatives: {len(test_neg)}")

    # ---------- save everything ----------
    print(f"Saving train positives to: {train_pos_path}")
    save_csv(train_pos, train_pos_path)

    print(f"Saving test positives to: {test_pos_path}")
    save_csv(test_pos, test_pos_path)

    print(f"Saving train negatives to: {train_neg_path}")
    save_csv(pd.DataFrame(train_neg, columns=["source_node", "destination_node"]), train_neg_path)

    print(f"Saving test negatives to: {test_neg_path}")
    save_csv(pd.DataFrame(test_neg, columns=["source_node", "destination_node"]), test_neg_path)

    # ---------- metadata ----------
    metadata = {
        "train_pos_count": len(train_pos),
        "test_pos_count": len(test_pos),
        "train_neg_count": len(train_neg),
        "test_neg_count": len(test_neg),
        "random_state": random_state,
        "test_size": test_size,
        "train_edges_path": str(edges_path),
        "full_graph_path": str(full_graph_path),
        "train_pos_path": str(train_pos_path),
        "test_pos_path": str(test_pos_path),
        "train_neg_path": str(train_neg_path),
        "test_neg_path": str(test_neg_path),
    }

    print(f"Saving metadata to: {metadata_path}")
    save_metadata(metadata, metadata_path)

    print("Done!")

if __name__ == "__main__":
    main()
