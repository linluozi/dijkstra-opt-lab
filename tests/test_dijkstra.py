"""Regression and equivalence tests for the two Dijkstra implementations.

Each test class maps to one of the defects found in the original package, so the
suite doubles as a checklist: if someone reintroduces the bug, the matching test
goes red and the docstring says why it matters.
"""

from __future__ import annotations

import math
import unittest

from dijkstra_naive import dijkstra_naive, dijkstra_naive_all
from dijkstra_optimized import dijkstra_optimized, dijkstra_optimized_all
from graphs import dense_graph, sparse_graph

BOTH_PATH = (("naive", dijkstra_naive), ("optimized", dijkstra_optimized))
BOTH_ALL = (("naive_all", dijkstra_naive_all), ("optimized_all", dijkstra_optimized_all))


class PathAssertions(unittest.TestCase):
    def assert_path_valid(self, graph, path, distance):
        """The path must walk real edges and its weights must sum to `distance`."""
        if not path:
            self.assertTrue(math.isinf(distance), "empty path must come with inf")
            return
        self.assertTrue(math.isfinite(distance))
        total = 0.0
        for u, v in zip(path, path[1:]):
            self.assertIn(v, graph.get(u, {}), f"{u} -> {v} is not an edge")
            total += graph[u][v]
        self.assertAlmostEqual(total, distance, delta=1e-9)


class TestTextbookCases(PathAssertions):
    def test_simple_diamond(self):
        graph = {1: {2: 10, 3: 20}, 2: {4: 40}, 3: {4: 5}, 4: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                path, distance = fn(graph, 1, 4)
                self.assertEqual(path, [1, 3, 4])
                self.assertEqual(distance, 25)
                self.assert_path_valid(graph, path, distance)

    def test_single_node(self):
        graph = {1: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, 1, 1), ([1], 0.0))

    def test_zero_weight_edges_are_allowed(self):
        graph = {1: {2: 0.0, 3: 0.0}, 2: {4: 1.0}, 3: {4: 2.0}, 4: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                path, distance = fn(graph, 1, 4)
                self.assertEqual(distance, 1.0)
                self.assert_path_valid(graph, path, distance)

    def test_self_loop_is_ignored(self):
        graph = {1: {1: 0.0, 2: 3.0}, 2: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, 1, 2), ([1, 2], 3.0))
                self.assertEqual(fn(graph, 1, 1), ([1], 0.0))


class TestNegativeWeights(PathAssertions):
    """Defect #1 in the original package: `_all` skipped the validation that
    `dijkstra_optimized` performed, so a negative edge produced a silently
    wrong answer from one function and an exception from the other."""

    NEGATIVE = {1: {2: 1.0, 3: 2.0}, 2: {4: 1.0}, 3: {2: -5.0}, 4: {}}

    def test_all_four_entry_points_reject_negative_weights(self):
        calls = [
            ("naive", lambda: dijkstra_naive(self.NEGATIVE, 1, 4)),
            ("optimized", lambda: dijkstra_optimized(self.NEGATIVE, 1, 4)),
            ("naive_all", lambda: dijkstra_naive_all(self.NEGATIVE, 1)),
            ("optimized_all", lambda: dijkstra_optimized_all(self.NEGATIVE, 1)),
        ]
        for name, call in calls:
            with self.subTest(impl=name):
                with self.assertRaises(ValueError):
                    call()

    def test_negative_edge_far_from_source_still_rejected(self):
        graph = {1: {2: 1.0}, 2: {}, 9: {10: -3.0}, 10: {}}
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                with self.assertRaises(ValueError):
                    fn(graph, 1)


class TestTargetRequirement(PathAssertions):
    """Defect #3: `end=None` used to return `([], 0.0)`, an invented distance
    that looks like a real answer to any caller that unpacks the tuple."""

    def test_missing_target_raises(self):
        graph = {1: {2: 1.0}, 2: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                with self.assertRaises(ValueError):
                    fn(graph, 1)
                with self.assertRaises(ValueError):
                    fn(graph, 1, None)

    def test_all_variants_are_the_single_source_api(self):
        graph = {1: {2: 1.0}, 2: {}}
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, 1), {1: 0.0, 2: 1.0})


class TestMixedNodeTypes(PathAssertions):
    """Defect #2: the original heap stored `(distance, node)`. Equal distances
    forced Python to compare the nodes, and a graph mixing int and str keys blew
    up with `TypeError: '<' not supported between instances of 'int' and 'str'`."""

    def test_int_and_str_keys_with_equal_distances(self):
        graph = {"a": {"b": 1.0, 2: 1.0}, "b": {}, 2: {}}
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, "a"), {"a": 0.0, "b": 1.0, 2: 1.0})

    def test_tuple_keys_with_equal_distances(self):
        graph = {(0, 0): {(0, 1): 2.0, (1, 0): 2.0}, (0, 1): {}, (1, 0): {}}
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                self.assertEqual(
                    fn(graph, (0, 0)), {(0, 0): 0.0, (0, 1): 2.0, (1, 0): 2.0}
                )

    def test_many_ties_do_not_crash(self):
        graph = {i: {(i + 1): 1.0} for i in range(20)}
        graph[20] = {}
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                self.assertEqual(len(fn(graph, 0)), 21)


class TestUnreachable(PathAssertions):
    """Defect #4: the two `_all` functions disagreed on whether unreachable
    nodes appear in the result, so swapping implementations turned an `inf`
    lookup into a `KeyError`."""

    GRAPH = {1: {2: 1.0}, 2: {}, 3: {4: 1.0}, 4: {}}

    def test_default_omits_unreachable(self):
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                self.assertEqual(fn(self.GRAPH, 1), {1: 0.0, 2: 1.0})

    def test_opt_in_includes_unreachable(self):
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                result = fn(self.GRAPH, 1, include_unreachable=True)
                self.assertEqual(result[1], 0.0)
                self.assertEqual(result[2], 1.0)
                self.assertTrue(math.isinf(result[3]))
                self.assertTrue(math.isinf(result[4]))

    def test_both_implementations_agree_on_key_sets(self):
        for include in (False, True):
            with self.subTest(include_unreachable=include):
                n = dijkstra_naive_all(self.GRAPH, 1, include_unreachable=include)
                o = dijkstra_optimized_all(self.GRAPH, 1, include_unreachable=include)
                self.assertEqual(set(n), set(o))

    def test_unreachable_target_returns_empty_path(self):
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                path, distance = fn(self.GRAPH, 1, 4)
                self.assertEqual(path, [])
                self.assertTrue(math.isinf(distance))


class TestBoundaries(PathAssertions):
    def test_empty_graph(self):
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                self.assertEqual(fn({}, "x", "x"), ([], math.inf))
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                self.assertEqual(fn({}, "x"), {})

    def test_start_not_in_graph(self):
        graph = {1: {2: 1.0}, 2: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, 99, 2), ([], math.inf))
        for name, fn in BOTH_ALL:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, 99), {})

    def test_target_not_in_graph(self):
        graph = {1: {2: 1.0}, 2: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, 1, 99), ([], math.inf))

    def test_target_only_reachable_as_key_not_value(self):
        graph = {1: {2: 1.0}, 2: {}}
        for name, fn in BOTH_PATH:
            with self.subTest(impl=name):
                self.assertEqual(fn(graph, 1, 2), ([1, 2], 1.0))


class TestEquivalence(PathAssertions):
    """The whole point: fast must never disagree with slow."""

    def test_random_sparse_graphs(self):
        for seed in (1, 2, 3):
            graph = sparse_graph(120, avg_degree=4, seed=seed)
            for source in (0, 37, 119):
                with self.subTest(seed=seed, source=source):
                    n = dijkstra_naive_all(graph, source)
                    o = dijkstra_optimized_all(graph, source)
                    self.assertEqual(set(n), set(o))
                    for node, distance in n.items():
                        self.assertAlmostEqual(o[node], distance, delta=1e-9)

    def test_random_dense_graphs(self):
        graph = dense_graph(60, edge_prob=0.2, seed=5)
        for source in (0, 30, 59):
            with self.subTest(source=source):
                n = dijkstra_naive_all(graph, source)
                o = dijkstra_optimized_all(graph, source)
                self.assertEqual(set(n), set(o))
                for node, distance in n.items():
                    self.assertAlmostEqual(o[node], distance, delta=1e-9)

    def test_early_exit_matches_full_run(self):
        """Stopping at the target must not change the distance to it."""
        graph = sparse_graph(150, avg_degree=5, seed=9)
        for target in (0, 1, 40, 149):
            with self.subTest(target=target):
                _, early = dijkstra_optimized(graph, 0, target)
                full = dijkstra_optimized_all(graph, 0)
                self.assertAlmostEqual(early, full[target], delta=1e-9)

    def test_every_returned_path_is_walkable(self):
        graph = sparse_graph(80, avg_degree=4, seed=13)
        full = dijkstra_optimized_all(graph, 0)
        for target in list(full)[:20]:
            with self.subTest(target=target):
                for name, fn in BOTH_PATH:
                    path, distance = fn(graph, 0, target)
                    self.assertEqual(path[0], 0)
                    self.assertEqual(path[-1], target)
                    self.assert_path_valid(graph, path, distance)


class TestLazyDeletion(PathAssertions):
    def test_node_improved_many_times(self):
        """Force repeated improvements so stale heap entries actually pile up."""
        graph = {0: {1: 100.0}}
        for i in range(30):
            graph[0][i + 2] = 100.0 - i * 3
            graph[i + 2] = {1: float(60 - i)}
        graph[1] = {}
        n = dijkstra_naive_all(graph, 0)
        o = dijkstra_optimized_all(graph, 0)
        self.assertEqual(set(n), set(o))
        for node, distance in n.items():
            self.assertAlmostEqual(o[node], distance, delta=1e-9)

    def test_deep_chain_does_not_recurse(self):
        depth = 20000
        graph = {i: {i + 1: 1.0} for i in range(depth)}
        graph[depth] = {}
        path, distance = dijkstra_optimized(graph, 0, depth)
        self.assertEqual(len(path), depth + 1)
        self.assertEqual(distance, float(depth))


if __name__ == "__main__":
    unittest.main(verbosity=2)
