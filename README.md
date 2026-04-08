# 🔗 GraphConnect — Social Link Prediction Engine

A production-style **link prediction system** for directed social graphs, built on the Facebook recruiting challenge dataset. Given a social network, GraphConnect predicts which missing links are likely to appear and serves explainable "People You May Know" recommendations via a FastAPI endpoint.

---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| **F1 Score** | 0.8659 |
| **ROC-AUC** | 0.9741 |
| **Accuracy** | 0.8755 |
| **Graph Nodes** | 1,862,220 |
| **Graph Edges** | 9,437,519 |
| **Engineered Features** | 42 |
| **Test Samples** | 3,775,008 |

---

## 🏗️ Project Structure

```
facebook_link_prediction/
├── README.md
├── requirements.txt
├── data/
│   └── processed/              # Processed edges, graph, stats
│       ├── edges.csv
│       ├── graph.pkl
│       ├── graph_stats.json
│       ├── train_pairs.csv
│       ├── test_pairs.csv
│       ├── train_features.csv
│       └── test_features.csv
├── pipeline/                   # Orchestration scripts (run in order)
│   ├── run_offline_data_pipeline.py
│   ├── prepare_edges.py
│   ├── build_graph.py
│   ├── precompute_graph_features.py
│   ├── generate_train_test_pairs.py
│   ├── build_labeled_pairs.py
│   ├── build_feature_tables.py
│   ├── convert.py
│   ├── train_model.py
│   └── evaluate_model.py
├── src/                        # Core library modules
│   ├── data/                   # Edge loading, graph building, sampling
│   │   ├── load_edges.py
│   │   ├── build_graph.py
│   │   ├── split_edges.py
│   │   └── sample_negatives.py
│   ├── features/               # Feature engineering (7 families)
│   │   ├── basic.py            # Degree, reciprocity, shortest path
│   │   ├── similarity.py       # Jaccard, Cosine, Adamic-Adar, RA
│   │   ├── weights.py          # Weighted degree features
│   │   ├── ranking.py          # PageRank, Katz, HITS
│   │   ├── embeddings.py       # SVD dot-product features
│   │   ├── pipeline.py         # Combines all feature families
│   │   └── load_precomputed.py
│   ├── inference/              # Online recommendation pipeline
│   │   ├── candidate_generation.py
│   │   ├── scoring.py
│   │   ├── recommender.py      # Main LinkRecommender class
│   │   └── presentation.py     # Confidence buckets & explanations
│   └── visualization/
├── artifacts/                  # Precomputed graph features & metrics
│   ├── metrics/
│   │   ├── test_metrics.json
│   │   └── train_metrics.json
│   ├── pagerank.pkl
│   ├── katz_centrality.pkl
│   ├── hits_hubs.pkl
│   ├── hits_auth.pkl
│   ├── in_degree.pkl
│   ├── out_degree.pkl
│   ├── weight_in.pkl
│   └── weight_out.pkl
├── models/                     # Trained model artifacts
│   ├── model.joblib
│   ├── feature_columns.json
│   └── feature_importance.csv
├── app/                        # FastAPI application
│   └── api.py
└── tests/
    └── test_features.py
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone <repo-url>
cd facebook_link_prediction
pip install -r requirements.txt
```

### Dependencies

```
joblib, networkx, numpy, pandas, pyarrow, scikit-learn , FastAPI , uvicorn
```

---

## 🚀 Usage

### 1. Run the Full Offline Pipeline

A single command runs all stages end-to-end:

```bash
python pipeline/run_offline_data_pipeline.py
```

This executes the following stages in order:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `prepare_edges.py` | Clean and validate the raw edge list |
| 2 | `build_graph.py` | Build a `networkx.DiGraph` and save stats |
| 3 | `precompute_graph_features.py` | Compute PageRank, Katz, HITS, weights |
| 4 | `generate_train_test_pairs.py` | 80/20 edge split + negative sampling |
| 5 | `build_labeled_pairs.py` | Combine positives and negatives |
| 6 | `build_feature_tables.py` | Engineer all 42 features per pair |
| 7 | `convert.py` | Convert CSVs to Parquet for efficiency |
| 8 | `train_model.py` | Train the Random Forest classifier |
| 9 | `evaluate_model.py` | Evaluate on held-out test set |

### 2. Serve Recommendations via API

```bash
uvicorn app.api:app --reload
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/recommend` | Get top-K recommendations for a user |

**Example request:**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "top_k": 10}'
```

**Example response:**

```json
{
  "user_id": 42,
  "generated_candidates_count": 512,
  "returned_count": 10,
  "model_version": "rf_v1",
  "recommendations": [
    {
      "rank": 1,
      "candidate_id": 987,
      "raw_score": 0.942,
      "confidence": "high",
      "reasons": ["follows_you_back", "high_mutual_followers"]
    }
  ]
}
```

---

## 🧠 Feature Engineering

42 features are organized into **7 families**:

| Family | Count | Importance | Examples |
|--------|-------|------------|----------|
| **Interactions** | 4 | 36.5% | `weight_in * weight_out`, `weight_in + weight_out` |
| **Similarity** | 8 | 18.9% | Jaccard, Cosine, Adamic-Adar, Resource Allocation |
| **Reciprocity** | 1 | 18.4% | `follows_back` |
| **Degree** | 8 | 18.2% | In/out degree, follower/followee counts |
| **Weight** | 4 | 7.0% | Inverse-sqrt dampened degree |
| **Ranking** | 16 | 1.0% | PageRank, Katz, HITS hubs/authorities |
| **Shortest Path** | 1 | 0.0% | BFS shortest path length |

---

## 🔍 Model

**Algorithm:** Random Forest (scikit-learn)

```
RandomForestClassifier(
    max_depth=14,
    n_estimators=121,
    min_samples_leaf=28,
    min_samples_split=111,
    random_state=25,
    n_jobs=-1
)
```

**Confusion Matrix (Test Set — 3.7M samples):**

```
                    Predicted Neg   Predicted Pos
Actual Negative      1,788,074        99,430
Actual Positive        370,526      1,516,978
```

---

## 📂 Key Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Trained model | `models/model.joblib` | Serialized Random Forest |
| Feature columns | `models/feature_columns.json` | Ordered list of 42 features |
| Feature importance | `models/feature_importance.csv` | Gini importance per feature |
| Graph | `data/processed/graph.pkl` | Full `nx.DiGraph` |
| Graph stats | `data/processed/graph_stats.json` | Node/edge counts, avg degree |
| Test metrics | `artifacts/metrics/test_metrics.json` | F1, AUC, confusion matrix |

---

