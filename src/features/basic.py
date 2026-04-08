import networkx as nx 

def follows_back ( graph : nx.DiGraph , u : int, v : int ) -> int : 
    return 1 if graph.has_edge(v,u) else 0 


def basic_node_features (precomputed_features : dict , src : int , dist : int ) -> dict : 
    in_degree = precomputed_features["in_degree"]
    out_degree= precomputed_features["out_degree"]
    pagerank = precomputed_features["pagerank"]
    katz = precomputed_features.get("katz_centrality", {})
    hubs = precomputed_features.get("hits_hubs", {})
    auth = precomputed_features.get("hits_auth", {})

    return {
        "src_in_degree" : in_degree.get(src,0),
        "src_out_degree" : out_degree.get(src,0),
        "src_pagerank" : pagerank.get(src,0),
        "src_katz": katz.get(src, 0),
        "src_hubs": hubs.get(src, 0),
        "src_auth": auth.get(src, 0),
        "dist_in_degree" : in_degree.get(dist,0),
        "dist_out_degree" : out_degree.get(dist,0),
        "dist_pagerank" : pagerank.get(dist,0),
        "dist_katz": katz.get(dist, 0),
        "dist_hubs": hubs.get(dist, 0),
        "dist_auth": auth.get(dist, 0),
    }
def compute_shortest_path_length(graph : nx.DiGraph , u : int , v : int, cutoff: int = 6 ) -> int : 
    p=-1
    try:
        if graph.has_edge(u,v):
            graph.remove_edge(u,v)
            try:
                p= nx.shortest_path_length(graph,source=u,target=v, cutoff=cutoff)
            except nx.NetworkXNoPath:
                p = -1
            graph.add_edge(u,v)
        else:
            try:
                p= nx.shortest_path_length(graph,source=u,target=v, cutoff=cutoff)
            except nx.NetworkXNoPath:
                p = -1
        return p
    except:
        return -1
def katz_centrality (graph : nx.DiGraph , src : int = None , dist : int = None ) -> float : 
    
    katz = nx.katz_centrality(graph, alpha=0.005, beta=1.0)
    if src is not None and dist is not None:
        return katz.get(src, 0) * katz.get(dist, 0)
    return katz

def Hits_score(graph : nx.DiGraph , src : int = None , dist : int = None ) -> float : 
    hubs, auth = nx.hits(graph, max_iter=100, tol=1e-08, nstart=None, normalized=True)
    if src is not None and dist is not None:
        return hubs.get(src, 0) * auth.get(dist, 0)
    return hubs, auth

def number_of_followers(graph: nx.DiGraph, node: int) -> int:
    """
    Return the number of followers (in-degree) of a node.
    """
    return len(list(graph.predecessors(node)))

def number_of_followees(graph: nx.DiGraph, node: int) -> int:
    """
    Return the number of followees (out-degree) of a node.
    """
    return len(list(graph.successors(node)))

