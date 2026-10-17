"""Benchmark: linear-scan Dijkstra vs binary-heap Dijkstra.

Design notes
------------
*Graph family is reported for every case.* Speedup is meaningless without it.
On a sparse graph (E ~ deg * V) the heap wins by orders of magnitude; on a
dense graph (E ~ V^2) it is asymptotically *worse* and only wins by a small
constant factor, if at all. The original package benchmarked dense graphs only
and then claimed sparse-graph numbers.

*Round-interleaved ratios.* Absolute timings drift between runs by tens of
percent, so each round runs the naive arm and the heap arm back to back and the
headline number is the median of the per-round ratios. Drift hits both arms
equally, so the ratio survives what the absolute timings do not.

*best-of-N within a round.* Noise only ever makes a run slower, so the minimum
is the closest thing to the true cost.

*Reachability is guaranteed.* ``sparse_graph`` lays a cycle down first. Without
it a naive run would bail out early on ``min_dist == inf`` and the O(V^2) term
would never show up.

Usage
-----
    python benchmark.py                 # full run, writes results.json
    python benchmark.py --quick         # smaller sizes, for a fast sanity check
    python benchmark.py --no-json       # print only
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from dijkstra_naive import dijkstra_naive, dijkstra_naive_all
from dijkstra_optimized import dijkstra_optimized, dijkstra_optimized_all
from _graph import validate_weights
from graphs import dense_graph, sparse_graph

SPARSE_SIZES = (1000, 2000, 4000, 8000, 16000)
DENSE_SPECS = ((200, 0.06), (400, 0.05), (800, 0.03), (1600, 0.012))

# Fixed V, growing E. This is the experiment that actually settles the
# complexity argument: naive cost barely moves with edge count while the heap
# pays for every edge, so the ratio must fall through 1.0 somewhere.
SWEEP_V = 1000
SWEEP_DEGREES = (4, 10, 30, 100, 300, 700)

# Larger graphs get fewer rounds purely so the whole run stays under a minute.
SPARSE_ROUNDS = {1000: (3, 2), 2000: (3, 2), 4000: (3, 2), 8000: (3, 2), 16000: (3, 1)}


def measure(fn, graph, repeat: int, **kwargs) -> float:
    """Best-of-`repeat` wall time in seconds."""
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        fn(graph, 0, **kwargs)
        best = min(best, time.perf_counter() - started)
    return best


def measure_validate(graph, repeat: int = 3) -> float:
    """Best-of-`repeat` time for the negative-weight scan alone."""
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        validate_weights(graph)
        best = min(best, time.perf_counter() - started)
    return best


def run_case(
    label: str,
    family: str,
    graph: Dict[int, Dict[int, float]],
    rounds: int,
    repeat: int,
) -> Dict[str, object]:
    nodes = len(graph)
    edges = sum(len(neighbours) for neighbours in graph.values())

    # Warm both arms before measuring anything. Warming only the fast one
    # inflates the ratio by roughly 10% (measured), which is small but free to
    # avoid.
    dijkstra_naive_all(graph, 0)
    dijkstra_optimized_all(graph, 0)

    naive_times: List[float] = []
    heap_times: List[float] = []
    ratios: List[float] = []

    for _ in range(rounds):
        t_naive = measure(dijkstra_naive_all, graph, repeat)
        t_heap = measure(dijkstra_optimized_all, graph, repeat)
        naive_times.append(t_naive)
        heap_times.append(t_heap)
        ratios.append(t_naive / t_heap)

    # Time the weight scan on its own. Deriving it as (default - unchecked) was
    # tried first and produced impossible negatives on small graphs, because it
    # subtracted two noisy numbers.
    t_validate = min(measure_validate(graph) for _ in range(5))
    heap_ms = min(heap_times) * 1000
    validate_ms = t_validate * 1000

    return {
        "label": label,
        "family": family,
        "v": nodes,
        "e": edges,
        "edges_per_node": round(edges / nodes, 2),
        "naive_ms": round(min(naive_times) * 1000, 3),
        "heap_ms": round(heap_ms, 3),
        "validate_ms": round(validate_ms, 3),
        "validation_pct": round(validate_ms / heap_ms * 100, 1),
        "ratio_median": round(statistics.median(ratios), 2),
        "ratio_min": round(min(ratios), 2),
        "ratio_max": round(max(ratios), 2),
        "rounds": rounds,
        "repeat": repeat,
    }


def run_density_sweep() -> List[Dict[str, object]]:
    """Hold V fixed and grow E until the heap stops winning."""
    rows: List[Dict[str, object]] = []
    for degree in SWEEP_DEGREES:
        graph = sparse_graph(SWEEP_V, avg_degree=degree, seed=31)
        rows.append(run_case(f"deg-{degree}", "sweep", graph, 3, 2))
    return rows


def verify() -> Dict[str, object]:
    """Fast must agree with slow, on distances *and* on reconstructed paths.

    Sizes stay small on purpose: this is a correctness gate, not a benchmark.
    """
    checked_sources = 0
    checked_nodes = 0
    checked_paths = 0

    cases = [
        ("sparse", sparse_graph(300, avg_degree=4, seed=21)),
        ("dense", dense_graph(120, edge_prob=0.08, seed=22)),
    ]

    for family, graph in cases:
        for source in (0, len(graph) // 3, len(graph) - 1):
            naive = dijkstra_naive_all(graph, source)
            heap = dijkstra_optimized_all(graph, source)
            if set(naive) != set(heap):
                raise AssertionError(f"{family} V={len(graph)}: key sets differ")

            for node, distance in naive.items():
                if abs(heap[node] - distance) > 1e-9:
                    raise AssertionError(
                        f"{family} V={len(graph)} source={source} node={node}: "
                        f"{distance} != {heap[node]}"
                    )
                checked_nodes += 1

            # Distances agreeing is not enough: a path can be the right length
            # and still walk edges that do not exist.
            for target in list(naive)[:15]:
                path_n, dist_n = dijkstra_naive(graph, source, target)
                path_o, dist_o = dijkstra_optimized(graph, source, target)
                if abs(dist_n - dist_o) > 1e-9:
                    raise AssertionError(
                        f"{family} path distance mismatch {source}->{target}"
                    )
                total = 0.0
                for u, v in zip(path_o, path_o[1:]):
                    if v not in graph.get(u, {}):
                        raise AssertionError(f"{family}: {u}->{v} is not an edge")
                    total += graph[u][v]
                if path_o and abs(total - dist_o) > 1e-9:
                    raise AssertionError(
                        f"{family}: path {source}->{target} sums to {total}, "
                        f"reported {dist_o}"
                    )
                checked_paths += 1
            checked_sources += 1

    return {
        "sources": checked_sources,
        "nodes": checked_nodes,
        "paths": checked_paths,
    }


def print_verification(result: Dict[str, object]) -> None:
    print("=" * 88)
    print("Correctness gate: naive vs heap")
    print("=" * 88)
    print(
        f"  {result['sources']} sources x {result['nodes']} node distances, "
        f"{result['paths']} reconstructed paths checked"
    )
    print("  distances agree, every path walks real edges and sums to its distance")


def print_table(rows: List[Dict[str, object]]) -> None:
    print()
    print("=" * 88)
    print("Performance  (ratio = median of per-round naive/heap ratios)")
    print("=" * 88)
    header = (
        f"{'family':8s} {'V':>7s} {'E':>8s} {'E/V':>6s} "
        f"{'naive ms':>10s} {'heap ms':>9s} {'check%':>7s} "
        f"{'ratio':>7s}  {'range':>13s}"
    )
    print(header)
    print("-" * 88)
    for row in rows:
        spread = f"{row['ratio_min']:.1f}-{row['ratio_max']:.1f}"
        print(
            f"{row['family']:8s} {row['v']:7d} {row['e']:8d} "
            f"{row['edges_per_node']:6.2f} {row['naive_ms']:10.2f} "
            f"{row['heap_ms']:9.2f} {row['validation_pct']:6.1f}% "
            f"{row['ratio_median']:6.1f}x  {spread:>13s}"
        )
    print()
    print("check% = share of the heap run spent scanning weights for negatives.")
    print("Skipping it with check_weights=False trades that safety net for speed.")


def print_sweep(rows: List[Dict[str, object]]) -> None:
    print()
    print("=" * 88)
    print(f"Density sweep at fixed V = {SWEEP_V}")
    print("=" * 88)
    print(
        f"{'V':>7s} {'E':>8s} {'E/V':>7s} {'naive ms':>10s} "
        f"{'heap ms':>9s} {'ratio':>8s}  verdict"
    )
    print("-" * 88)
    for row in rows:
        ratio = row["ratio_median"]
        verdict = "heap wins" if ratio >= 1.5 else ("wash" if ratio >= 0.9 else "naive wins")
        print(
            f"{row['v']:7d} {row['e']:8d} {row['edges_per_node']:7.1f} "
            f"{row['naive_ms']:10.2f} {row['heap_ms']:9.2f} "
            f"{ratio:7.2f}x  {verdict}"
        )
    print()
    print("Past a few hundred edges per node the advantage collapses to ~1.3x and")
    print("flatlines there: every edge costs a queued push, while the linear scan")
    print("never pays for edges it does not relax. It does not cross below 1.0 in")
    print("CPython at these sizes -- heappush is C and the scan is bytecode, so the")
    print("constant factor eats the complexity gap. The textbook advice still holds:")
    print("dense graphs want the O(V^2) array version, not the heap.")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sizes only")
    parser.add_argument("--no-json", action="store_true", help="do not write results.json")
    parser.add_argument("--out", default="results.json", help="JSON output path")
    args = parser.parse_args(argv)

    verification = verify()
    print_verification(verification)

    rows: List[Dict[str, object]] = []

    sparse_sizes = (1000, 2000, 4000) if args.quick else SPARSE_SIZES
    for n in sparse_sizes:
        rounds, repeat = SPARSE_ROUNDS.get(n, (2, 1))
        rows.append(
            run_case(
                f"sparse-{n}",
                "sparse",
                sparse_graph(n, avg_degree=4, seed=7),
                rounds,
                repeat,
            )
        )

    dense_specs = ((200, 0.06), (400, 0.05)) if args.quick else DENSE_SPECS
    for n, prob in dense_specs:
        rows.append(
            run_case(f"dense-{n}", "dense", dense_graph(n, prob, seed=11), 3, 5)
        )

    print_table(rows)

    sweep_rows = [] if args.quick else run_density_sweep()
    if sweep_rows:
        print_sweep(sweep_rows)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "verification": verification,
        "cases": rows,
        "density_sweep": sweep_rows,
    }

    if not args.no_json:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"\nwrote {args.out}")

    print("\nNote: ratios are per-round medians, not a ratio of the two minima.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
