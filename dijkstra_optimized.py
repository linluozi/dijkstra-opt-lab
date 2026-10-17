"""Heap-based Dijkstra — the optimised arm of the comparison.

Three things make this faster than the naive version:

1. **Binary heap frontier.** ``heapq`` pops the minimum in O(log V) instead of
   scanning all remaining nodes in O(V), taking the run from O(V^2 + E) to
   O((V + E) log V).
2. **Lazy deletion.** Python's ``heapq`` has no efficient decrease-key, so an
   improved distance simply pushes a second entry and the stale one is dropped
   on pop because the node is already settled. In practice this beats manually
   maintaining a position index.
3. **Localised hot loop.** ``heappush``/``heappop`` are bound to locals and the
   queue entries are three-tuples, which keeps the inner loop cheap.

What this version does *not* do is win on dense graphs. When E = Theta(V^2) the
heap version costs Theta(V^2 log V) while the linear-scan version costs
Theta(V^2), so the heap is asymptotically *worse* there. The measured 1-3x on
dense graphs is a constant factor, not a complexity win; see README.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Dict, Hashable, List, Optional, Tuple

from _graph import INF, Graph, require_target, validate_weights


def dijkstra_optimized(
    graph: Graph,
    start: Hashable,
    end: Optional[Hashable] = None,
    *,
    check_weights: bool = True,
) -> Tuple[List[Hashable], float]:
    """Shortest path from ``start`` to ``end`` using a binary heap.

    Parameters
    ----------
    graph
        Adjacency dict, ``graph[u][v] = weight``. Weights must be non-negative.
    start
        Source node. May be absent from ``graph`` (returns unreachable).
    end
        Target node. Required — the search stops as soon as it is settled.
    check_weights
        Validate every edge weight before searching. Costs one O(V + E) pass,
        which is roughly 8-15% of the search itself on sparse graphs. Set
        ``False`` only for graphs you built yourself and know to be clean.

    Returns
    -------
    (path, distance)
        ``path`` is the node sequence from ``start`` to ``end``, or ``[]`` when
        ``end`` is unreachable. ``distance`` is ``inf`` in that case.

    Notes
    -----
    Queue entries are ``(distance, sequence, node)``. The middle field is a
    monotonically increasing counter, which guarantees the tuples never fall
    through to comparing ``node``. With a plain ``(distance, node)`` pair, two
    equal distances force Python to compare the nodes themselves, and a graph
    mixing ``int`` and ``str`` keys raises ``TypeError: '<' not supported``.
    """
    require_target(end)
    if check_weights:
        validate_weights(graph)

    if start not in graph:
        return [], INF

    heappush = heapq.heappush
    heappop = heapq.heappop
    counter = itertools.count()

    dist: Dict[Hashable, float] = {start: 0.0}
    predecessor: Dict[Hashable, Hashable] = {}
    settled: set = set()
    queue: List[Tuple[float, int, Hashable]] = [(0.0, next(counter), start)]

    while queue:
        d, _, node = heappop(queue)

        if node in settled:
            continue  # stale entry left behind by a later improvement
        settled.add(node)

        if node == end:
            break  # settled distances are final, so this is safe to stop at

        for neighbour, weight in graph.get(node, {}).items():
            candidate = d + weight
            if candidate < dist.get(neighbour, INF):
                dist[neighbour] = candidate
                predecessor[neighbour] = node
                heappush(queue, (candidate, next(counter), neighbour))

    if end not in dist:
        return [], INF

    path: List[Hashable] = []
    node = end
    while node != start:
        path.append(node)
        node = predecessor[node]
    path.append(start)
    path.reverse()
    return path, dist[end]


def dijkstra_optimized_all(
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
        matching ``dijkstra_naive_all``. Producing the full key set costs one
        extra O(V + E) pass, so it is opt-in rather than the default.
    check_weights
        Validate every edge weight before searching. Costs one O(V + E) pass.
    """
    if check_weights:
        validate_weights(graph)

    if start not in graph:
        return {}

    heappush = heapq.heappush
    heappop = heapq.heappop
    counter = itertools.count()

    dist: Dict[Hashable, float] = {start: 0.0}
    settled: set = set()
    queue: List[Tuple[float, int, Hashable]] = [(0.0, next(counter), start)]

    while queue:
        d, _, node = heappop(queue)
        if node in settled:
            continue
        settled.add(node)

        for neighbour, weight in graph.get(node, {}).items():
            candidate = d + weight
            if candidate < dist.get(neighbour, INF):
                dist[neighbour] = candidate
                heappush(queue, (candidate, next(counter), neighbour))

    if include_unreachable:
        for node, neighbours in graph.items():
            dist.setdefault(node, INF)
            for neighbour in neighbours:
                dist.setdefault(neighbour, INF)
        return dist

    return {node: d for node, d in dist.items() if not math.isinf(d)}


if __name__ == "__main__":
    sample = {
        1: {2: 10, 3: 20},
        2: {4: 40},
        3: {4: 5},
        4: {},
    }
    path, distance = dijkstra_optimized(sample, 1, 4)
    print("optimized:", path, distance)
    assert path == [1, 3, 4]
    assert distance == 25
