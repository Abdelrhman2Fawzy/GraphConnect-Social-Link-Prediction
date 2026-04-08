from pathlib import Path
import pandas as pd

from src.data.build_pairs import combine_pairs_with_labels, save_pairs


def main() -> None:
    train_pos_path = Path("data/interim/train_pos_edges.csv")
    train_neg_path = Path("data/interim/train_neg_edges.csv")
    test_pos_path = Path("data/interim/test_pos_edges.csv")
    test_neg_path = Path("data/interim/test_neg_edges.csv")

    train_pairs_path = Path("data/processed/train_pairs.csv")
    test_pairs_path = Path("data/processed/test_pairs.csv")

    print(f"Loading {train_pos_path}")
    train_pos = pd.read_csv(train_pos_path)

    print(f"Loading {train_neg_path}")
    train_neg = pd.read_csv(train_neg_path)

    print(f"Loading {test_pos_path}")
    test_pos = pd.read_csv(test_pos_path)

    print(f"Loading {test_neg_path}")
    test_neg = pd.read_csv(test_neg_path)

    print("Building labeled train pairs...")
    train_pairs = combine_pairs_with_labels(
        pos_df=train_pos,
        neg_df=train_neg,
        shuffle=True,
        random_state=42,
    )

    print("Building labeled test pairs...")
    test_pairs = combine_pairs_with_labels(
        pos_df=test_pos,
        neg_df=test_neg,
        shuffle=True,
        random_state=42,
    )

    print(f"Saving train pairs to {train_pairs_path}")
    save_pairs(train_pairs, train_pairs_path)

    print(f"Saving test pairs to {test_pairs_path}")
    save_pairs(test_pairs, test_pairs_path)

    print("Done.")
    print(f"train_pairs shape: {train_pairs.shape}")
    print(f"test_pairs shape: {test_pairs.shape}")
    print("\nTrain label counts:")
    print(train_pairs["label"].value_counts())
    print("\nTest label counts:")
    print(test_pairs["label"].value_counts())


if __name__ == "__main__":
    main()