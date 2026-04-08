from pathlib import Path
import json
import pickle

import joblib
import networkx as nx

from src.features.load_precomputed import load_precomputed_features
from src.inference.candidate_generation import generate_candidates
from src.inference.scoring import score_candidates
from src.inference.presentation import score_to_confidence, format_reasons


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


class LinkRecommender:
    def __init__(
        self,
        graph_path: str | Path,
        model_path: str | Path,
        feature_columns_path: str | Path,
        artifacts_dir: str | Path = "artifacts",
        model_version: str = "rf_v1",
    ) -> None:
        self.graph = self._load_graph(graph_path)
        self.model = joblib.load(model_path)
        self.precomputed = load_precomputed_features(artifacts_dir)
        self.model_version = model_version

        with open(feature_columns_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

    def _load_graph(self, path: str | Path) -> nx.DiGraph:
        with open(path, "rb") as f:
            return pickle.load(f)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        max_candidates: int = 1000,
    ) -> dict:
        candidates = generate_candidates(
            graph=self.graph,
            user_id=user_id,
            max_candidate=max_candidates,
        )

        generated_candidates_count = len(candidates)

        if not candidates:
            return {
                "user_id": user_id,
                "generated_candidates_count": 0,
                "returned_count": 0,
                "model_version": self.model_version,
                "recommendations": [],
            }

        scored_df = score_candidates(
            model=self.model,
            feature_columns=self.feature_columns,
            graph=self.graph,
            precomputed=self.precomputed,
            user_id=user_id,
            candidates=candidates,
        )

        if scored_df.empty:
            return {
                "user_id": user_id,
                "generated_candidates_count": generated_candidates_count,
                "returned_count": 0,
                "model_version": self.model_version,
                "recommendations": [],
            }

        top_df = scored_df.head(top_k)

        recommendations = []
        for rank, row in enumerate(top_df.to_dict(orient="records"), start=1):
            raw_score = float(row["raw_score"])

            recommendations.append(
                {
                    "rank": rank,
                    "candidate_id": int(row["destination_node"]),
                    "raw_score": raw_score,
                    "confidence": score_to_confidence(raw_score),
                    "reasons": format_reasons(row, EXPLANATION_COLUMNS),
                }
            )

        return {
            "user_id": user_id,
            "generated_candidates_count": generated_candidates_count,
            "returned_count": len(recommendations),
            "model_version": self.model_version,
            "recommendations": recommendations,
        }