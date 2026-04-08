from src.data.load_edges import load_edges, validate_edges

def prepare_edges():
    edges_df = load_edges("data/raw/train.csv")
    validate_edges(edges_df)
  #  edges_df.to_csv("data/processed/edges.csv", index=False)
    edges_df.to_csv("data/interim/train_edges.csv", index=False)


if __name__ == "__main__":
    prepare_edges()