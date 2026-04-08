import subprocess
import sys
import os


def run_step(script_path: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Running: {script_path}")
    print(f"{'=' * 70}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run([sys.executable, script_path], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_path}")


def main() -> None:
    run_step("pipeline/prepare_edges.py")
    run_step("pipeline/build_graph.py")
    run_step("pipeline/precompute_graph_features.py")
    run_step("pipeline/generate_train_test_pairs.py")
    run_step("pipeline/build_labeled_pairs.py")
    run_step("pipeline/build_feature_tables.py")
    run_step("pipeline/convert.py")
    run_step("pipeline/train_model.py")
    run_step("pipeline/evaluate_model.py")  
    print("\nOffline data pipeline completed successfully.")


if __name__ == "__main__":
    main()