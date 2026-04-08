
def ranking_features(precomputed: dict, src: int, dst: int) -> dict:
    pagerank = precomputed["pagerank"]
    katz = precomputed["katz_centrality"]
    hubs = precomputed["hits_hubs"]
    authorities = precomputed["hits_auth"]

    return {
        "pagerank_src": pagerank.get(src, 0.0),
        "pagerank_dst": pagerank.get(dst, 0.0),
        "katz_src": katz.get(src, 0.0),
        "katz_dst": katz.get(dst, 0.0),
        "hubs_src": hubs.get(src, 0.0),
        "hubs_dst": hubs.get(dst, 0.0),
        "authorities_src": authorities.get(src, 0.0),
        "authorities_dst": authorities.get(dst, 0.0),
    }