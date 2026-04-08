import pandas as pd
from pathlib import Path

input_path = "data/processed/test_features.csv"
output_dir = Path("data/processed/test_parquet_chunks2")
output_dir.mkdir(exist_ok=True)

for i, chunk in enumerate(pd.read_csv(input_path, chunksize=500_000)):
    chunk["label"] = chunk["label"].astype("int8")

    for col in chunk.columns:
        if col != "label":
            chunk[col] = chunk[col].astype("float32")

    chunk.to_parquet(
        output_dir / f"part_{i}.parquet",
        engine="pyarrow",
        compression="snappy"
    )

    print(f"Saved chunk {i}")
