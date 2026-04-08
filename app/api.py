from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.recommender import LinkRecommender


app = FastAPI(title="GraphConnect API")


recommender = LinkRecommender(
    graph_path="data/processed/graph.pkl",
    model_path="models/model.joblib",
    feature_columns_path="models/feature_columns.json",
    artifacts_dir="artifacts",
    model_version="rf_v1",
)


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10
    max_candidates: int = 1000


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return recommender.recommend(
        user_id=req.user_id,
        top_k=req.top_k,
        max_candidates=req.max_candidates,
    )