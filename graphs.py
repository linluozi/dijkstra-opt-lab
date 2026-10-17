"""Graph generators shared by the benchmark and the test-suite.

The sparse generator matters more than it looks. A naive Dijkstra stops early
when every remaining node still sits at infinity (``min_dist == inf``), so on a
disconnected graph the O(V^2) term never materialises and the naive arm looks
suspiciously fast. Building a random cycle first guarantees every node is
reachable from every other, which is the only way the comparison is honest.
"""

from __future__ import annotations

import random
from typing import Dict

Graph = Dict[int, Dict[int, float]]


def sparse_graph(n: int, avg_degree: int = 4, seed: int = 7) -> Graph:
    """Random cycle (full reachability) plus extra random edges. E ~= avg_degree * V."""
    if n < 2:
        raise ValueError("sparse_graph needs n >= 2")
    if avg_degree < 1:
        raise ValueError("avg_degree must be >= 1 (the cycle alone)")

    rng = random.Random(seed)
    graph: Graph = {i: {} for i in range(n)}

    order = list(range(n))
    rng.shuffle(order)
    for k in range(n):
        u, v = order[k], order[(k + 1) % n]
        graph[u][v] = rng.uniform(1.0, 10.0)

    for _ in range(n * (avg_degree - 1)):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v:
            graph[u][v] = rng.uniform(1.0, 10.0)

    return graph


def dense_graph(n: int, edge_prob: float, seed: int = 11) -> Graph:
    """Erdos-Renyi style directed graph. E ~= edge_prob * V^2."""
    rng = random.Random(seed)
    graph: Graph = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < edge_prob:
                graph[i][j] = rng.uniform(1.0, 10.0)
    return graph
