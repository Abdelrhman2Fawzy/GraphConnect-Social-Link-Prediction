import networkx as nx 

def generate_candidates(graph : nx.DiGraph , user_id : int , max_candidate  = 1000) -> list[int]:
    if user_id not in graph:
        return []
    direct_followers = set(graph.successors(user_id))
    direct_followees = set(graph.predecessors(user_id))
    candidates = set()
    # followers of followees
    for nbr in direct_followees:
        if nbr in graph:
            candidates.update(graph.successors(nbr))
    # followees of followers
    for nbr in direct_followers:
        if nbr in graph:
            candidates.update(graph.predecessors(nbr))
    for nbr in direct_followers:
        if nbr in graph:
            candidates.update(graph.successors(nbr))
    for nbr in direct_followees:
        if nbr in graph:
            candidates.update(graph.predecessors(nbr))

    candidates.discard(user_id)
    candidates -= direct_followees

    candidates = list(candidates)
    if len(candidates) > max_candidate:
        candidates = candidates[:max_candidate]
    return candidates


    

    
