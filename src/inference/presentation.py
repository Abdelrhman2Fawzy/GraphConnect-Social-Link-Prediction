REASON_NAME_MAP = {
    "intersection_of_followers": "mutual_followers",
    "intersection_of_followees": "mutual_followees",
    "jaccard_for_followers": "follower_similarity",
    "jaccard_for_followees": "followee_similarity",
    "cosine_for_followers": "follower_cosine_similarity",
    "cosine_for_followees": "followee_cosine_similarity",
    "follows_back": "follows_you_back",
    "shortest_path_length": "graph_distance",
    "page_rank_source": "source_importance",
    "page_rank_destination": "candidate_importance",
}


def score_to_confidence(score: float) -> str:
    """
    Convert raw model score to user-friendly confidence bucket.
    """
    if score >= 0.95:
        return "High"
    if score >= 0.80:
        return "Medium"
    return "Low"


def rename_reason_key(key: str) -> str:
    return REASON_NAME_MAP.get(key, key)


def format_reasons(row: dict, explanation_columns: list[str]) -> dict:
    reasons = {}

    for key in explanation_columns:
        if key not in row:
            continue

        value = row[key]

        if key == "follows_back":
            reasons[rename_reason_key(key)] = bool(value)
        else:
            reasons[rename_reason_key(key)] = value

    return reasons