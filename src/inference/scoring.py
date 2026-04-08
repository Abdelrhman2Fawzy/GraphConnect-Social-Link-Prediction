import networkx as nx 
import pandas as pd 
import joblib 
from src.features.pipeline import build_features_for_pairs

EXPLANATION_COLUMNS = [
    "intersection_of_followers",
    "intersection_of_followees",
    "jaccard_for_followers",
    "jaccard_for_followees",
    "cosine_for_followers",
    "cosine_for_followees",
    "follows_back",
    "shortest_path_length",
    "page_rank_source",
    "page_rank_destination",
    "preferential_attachment"
]
def build_candidate_pairs(user_id: int, candidates: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_node": [user_id] * len(candidates),
            "destination_node": candidates,
            "label": [0] * len(candidates),  
        }
    )

def score_candidates(model , feature_columns : list[str] , graph  : nx.DiGraph , precomputed : dict , user_id : int ,candidates : list[int]) -> pd.DataFrame : 
    if not candidates:
        return pd.DataFrame(["source_node","destination_node","score"])
    pairs_df = build_candidate_pairs(user_id,candidates)
    features_df = build_features_for_pairs(pairs_df=pairs_df , graph=graph , precomputed=precomputed)
    X = features_df[feature_columns]
    raw_scores = model.predict_proba(X)[:,1]

    keep_cols=["source_node", "destination_node"]+[c for c in EXPLANATION_COLUMNS if c in features_df.columns]
    result = features_df[keep_cols].copy()
    result["raw_score"] = raw_scores

    return result.sort_values("raw_score", ascending=False).reset_index(drop=True)
    