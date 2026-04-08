import pickle 
from pathlib import Path 


def load_pickle(path : str | Path) -> dict : 
    path = Path (path)
    with open(path, 'rb') as f : 
        return pickle.load(f)
def load_precomputed_features(artifacts_dir : str | Path = "artifacts") -> dict : 
    artifacts_dir= Path(artifacts_dir)
    return {
        "in_degree" : load_pickle(artifacts_dir / "in_degree.pkl"),
        "out_degree" : load_pickle(artifacts_dir / "out_degree.pkl"),
        "pagerank" : load_pickle(artifacts_dir / "pagerank.pkl"),
        "katz_centrality" : load_pickle(artifacts_dir / "katz_centrality.pkl"),
        "hits_hubs" : load_pickle(artifacts_dir / "hits_hubs.pkl"),
        "hits_auth" : load_pickle(artifacts_dir / "hits_auth.pkl"),
        "weight_in" : load_pickle(artifacts_dir / "weight_in.pkl"),
        "weight_out" : load_pickle(artifacts_dir / "weight_out.pkl"),
        "svd_u" : load_pickle(artifacts_dir / "svd_u.pkl"),
        "svd_v" : load_pickle(artifacts_dir / "svd_v.pkl"),
    }
