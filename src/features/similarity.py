import networkx as nx 
import numpy as np

def get_graph_neighbors(graph: nx.DiGraph):
    """
    Pre-materialize successors and predecessors as sets for fast lookups.
    """
    successors = {node: set(graph.successors(node)) for node in graph.nodes()}
    predecessors = {node: set(graph.predecessors(node)) for node in graph.nodes()}
    return successors, predecessors


def jaccard_for_followees(a: int, b: int, successors: dict) -> float:
    set_a = successors.get(a, set())
    set_b = successors.get(b, set())
    if not set_a or not set_b:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)


def jaccard_for_followers(a: int, b: int, predecessors: dict) -> float:
    set_a = predecessors.get(a, set())
    set_b = predecessors.get(b, set())
    if not set_a or not set_b:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)





def cosine_for_followees(a: int, b: int, successors: dict) -> float:
    set_a = successors.get(a, set())
    set_b = successors.get(b, set())
    if not set_a or not set_b:
        return 0.0
    # Old: return len(set_a.intersection(set_b)) / (len(set_a) * len(set_b))
    return len(set_a.intersection(set_b)) / np.sqrt(len(set_a) * len(set_b))


def cosine_for_followers(a: int, b: int, predecessors: dict) -> float:
    set_a = predecessors.get(a, set())
    set_b = predecessors.get(b, set())
    if not set_a or not set_b:
        return 0.0
    # Old: return len(set_a.intersection(set_b)) / (len(set_a) * len(set_b))
    return len(set_a.intersection(set_b)) / np.sqrt(len(set_a) * len(set_b))


def adar_for_followees(a: int, b: int, successors: dict, predecessors: dict) -> float:
    set_a = successors.get(a, set())
    set_b = successors.get(b, set())
    common = set_a.intersection(set_b)
    if not common:
        return 0.0
    
    adar = 0.0
    for node in common:
        preds_len = len(predecessors.get(node, []))
        if preds_len > 1:
            adar += 1.0 / np.log10(preds_len)
    return adar


def adar_for_followers(a: int, b: int, predecessors: dict, successors: dict) -> float:
    set_a = predecessors.get(a, set())
    set_b = predecessors.get(b, set())
    common = set_a.intersection(set_b)
    if not common:
        return 0.0
    
    adar = 0.0
    for node in common:
        succs_len = len(successors.get(node, []))
        if succs_len > 1:
            adar += 1.0 / np.log10(succs_len)
    return adar


def intersection_of_followers(a: int, b: int, predecessors: dict) -> int:
    return len(predecessors.get(a, set()).intersection(predecessors.get(b, set())))


def intersection_of_followees(a: int, b: int, successors: dict) -> int:
    return len(successors.get(a, set()).intersection(successors.get(b, set())))


def preferential_attachment_followees(a: int, b: int, successors: dict) -> int:
    return len(successors.get(a, set())) * len(successors.get(b, set()))


def preferential_attachment_followers(a: int, b: int, predecessors: dict) -> int:
    return len(predecessors.get(a, set())) * len(predecessors.get(b, set()))


def resource_allocation_for_followees(a: int, b: int, successors: dict, predecessors: dict) -> float:
    set_a = successors.get(a, set())
    set_b = successors.get(b, set())
    common = set_a.intersection(set_b)
    if not common:
        return 0.0
    
    rai = 0.0
    for node in common:
        preds_len = len(predecessors.get(node, []))
        if preds_len > 0:
            rai += 1.0 / preds_len
    return rai


def resource_allocation_for_followers(a: int, b: int, predecessors: dict, successors: dict) -> float:
    set_a = predecessors.get(a, set())
    set_b = predecessors.get(b, set())
    common = set_a.intersection(set_b)
    if not common:
        return 0.0
    
    rai = 0.0
    for node in common:
        succs_len = len(successors.get(node, []))
        if succs_len > 0:
            rai += 1.0 / succs_len
    return rai