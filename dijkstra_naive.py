"""Naive Dijkstra — the reference implementation, kept deliberately slow.

This is the baseline arm of the comparison. It keeps the one defect that the
benchmark is about: **a linear scan over the remaining nodes on every
extraction**, so a single-source run costs O(V^2) before edges are even
considered.

What it does *not* keep are the API-level traps from the original package. A
control group has to be correct, otherwise the speed ratio measures nothing —
if the slow arm can silently return a wrong answer, "it is faster" is not a
fair statement.

Deliberate O(V^2) behaviour (do not "fix" this):

    while unseen:
        for node in unseen:          # <- O(V) per extraction, V times
            if dist[node] < best: ...

Everything else follows the textbook description (CLRS ch. 24.3).
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, List, Optional, Tuple

from _graph import INF, Graph, collect_nodes, require_target, validate_weights


def dijkstra_naive(
    graph: Graph,
    start: Hashable,
    end: Optional[Hashable] = None,
    *,
    check_weights: bool = True,
) -> Tuple[List[Hashable], float]:
    """Shortest path from ``start`` to ``end`` using a linear-scan frontier.

    Parameters
    ----------
    graph
        Adjacency dict, ``graph[u][v] = weight``. Weights must be non-negative.
    start
        Source node. May be absent from ``graph`` (returns unreachable).
    end
        Target node. Required — see :func:`_graph.require_target`.
    check_weights
        Validate every edge weight before searching. Costs one O(V + E) pass.
        Set ``False`` only for graphs you built yourself and know to be clean.

    Returns
    -------
    (path, distance)
        ``path`` is the node sequence from ``start`` to ``end``, or ``[]`` when
        ``end`` is unreachable. ``distance`` is ``inf`` in that case.
    """
    require_target(end)
    if check_weights:
        validate_weights(graph)

    if start not in graph:
        return [], INF

    nodes = collect_nodes(graph)
    dist: Dict[Hashable, float] = {node: INF for node in nodes}
    dist[start] = 0.0
    predecessor: Dict[Hashable, Hashable] = {}
    unseen = set(nodes)

    while unseen:
        # The defect under test: linear scan instead of a heap.
        current = None
        best = INF
        for node in unseen:
            if dist[node] < best:
                best = dist[node]
                current = node

        if current is None:
            break  # everything left is unreachable

        unseen.remove(current)

        if current == end:
            break

        for neighbour, weight in graph.get(current, {}).items():
            candidate = dist[current] + weight
            if candidate < dist[neighbour]:
                dist[neighbour] = candidate
                predecessor[neighbour] = current

    if end not in dist or math.isinf(dist[end]):
        return [], INF

    path: List[Hashable] = []
    node = end
    while node != start:
        path.append(node)
        node = predecessor[node]
    path.append(start)
    path.reverse()
    return path, dist[end]


def dijkstra_naive_all(
    graph: Graph,
    start: Hashable,
    include_unreachable: bool = False,
    *,
    check_weights: bool = True,
) -> Dict[Hashable, float]:
    """Shortest distances from ``start`` to every reachable node.

    Parameters
    ----------
    include_unreachable
        When ``True``, nodes that exist in the graph but cannot be reached from
        ``start`` are present in the result mapped to ``inf``. Default ``False``,
        matching ``dijkstra_optimized_all``.
    check_weights
        Validate every edge weight before searching. Costs one O(V + E) pass.
    """
    if check_weights:
        validate_weights(graph)

    if start not in graph:
        return {}

    nodes = collect_nodes(graph)
    dist: Dict[Hashable, float] = {node: INF for node in nodes}
    dist[start] = 0.0
    unseen = set(nodes)

    while unseen:
        current = None
        best = INF
        for node in unseen:
            if dist[node] < best:
                best = dist[node]
                current = node

        if current is None:
            break

        unseen.remove(current)

        for neighbour, weight in graph.get(current, {}).items():
            candidate = dist[current] + weight
            if candidate < dist[neighbour]:
                dist[neighbour] = candidate

    if include_unreachable:
        return dist
    return {node: d for node, d in dist.items() if not math.isinf(d)}


if __name__ == "__main__":
    sample = {
        1: {2: 10, 3: 20},
        2: {4: 40},
        3: {4: 5},
        4: {},
    }
    path, distance = dijkstra_naive(sample, 1, 4)
    print("naive:", path, distance)
    assert path == [1, 3, 4]
    assert distance == 25
