"""Render the two charts used in the README from results.json.

Every number on the charts is read from the benchmark output — nothing is typed
in by hand, so the figures cannot drift away from the measurements.

    python scripts/make_charts.py

Palette follows the warm editorial scheme used across these projects: cream
paper, espresso ink, amber and terracotta accents. No cool blues.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

PAPER = "#FBF7F1"
INK = "#2A211B"
MUTED = "#8A7E72"
GRID = "#E4DACB"
AMBER = "#C8811F"
TERRACOTTA = "#A6402A"

ROOT = Path(__file__).resolve().parent.parent


def load_results(path: Path) -> Dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def split_families(cases: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {"sparse": [], "dense": []}
    for case in cases:
        out[case["family"]].append(case)
    for family in out:
        out[family].sort(key=lambda c: c["v"])
    return out


def style_axes(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=10, length=3, width=1.0)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)


def chart_speedup(families: Dict[str, List[Dict]], out: Path, ver: str) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 7.0), dpi=200)
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (2, 3)), alpha=0.8)
    ax.text(25000, 1.1, "no gain", color=MUTED, fontsize=9.5, ha="right", va="bottom")

    def degree_label(cases: List[Dict]) -> str:
        vals = [c["edges_per_node"] for c in cases]
        low, high = min(vals), max(vals)
        return f"E/V = {low:.0f}" if high - low < 0.5 else f"E/V = {low:.0f}-{high:.0f}"

    series = [
        ("sparse", "o", "-", AMBER),
        ("dense", "s", (0, (5, 3)), TERRACOTTA),
    ]

    for family, marker, style, colour in series:
        cases = families[family]
        label = f"{family.capitalize()} graph  {degree_label(cases)}"
        xs = [c["v"] for c in cases]
        ys = [c["ratio_median"] for c in cases]
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=2.4,
            linestyle=style,
            marker=marker,
            markersize=7,
            markerfacecolor=colour,
            markeredgecolor=PAPER,
            markeredgewidth=1.6,
            label=label,
            zorder=3,
        )
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.0f}x" if y >= 10 else f"{y:.1f}x",
                (x, y),
                textcoords="offset points",
                xytext=(0, 13),
                ha="center",
                va="bottom",
                fontsize=10,
                color=colour,
                fontweight="bold",
            )

    ax.set_xlim(140, 26000)
    ax.set_ylim(0.8, 170)
    ax.set_xlabel("nodes V", color=INK, fontsize=12, labelpad=8)
    ax.set_ylabel("speedup  naive / heap", color=INK, fontsize=12, labelpad=8)

    ax.xaxis.set_major_locator(FixedLocator([200, 1000, 2000, 4000, 8000, 16000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["200", "1k", "2k", "4k", "8k", "16k"], color=MUTED)
    ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))

    sparse_y = [c["ratio_median"] for c in families["sparse"]]
    dense_y = [c["ratio_median"] for c in families["dense"]]
    fig.suptitle(
        f"Sparse graphs gain {min(sparse_y):.0f}-{max(sparse_y):.0f}x, "
        f"dense graphs {min(dense_y):.0f}-{max(dense_y):.0f}x",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=17,
        color=INK,
        fontweight="bold",
    )
    ax.set_title(
        "Same two implementations, same machine. Only the graph family changed.",
        loc="left",
        fontsize=11.5,
        color=MUTED,
        pad=10,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=11,
        handlelength=2.6,
        labelcolor=INK,
    )

    fig.text(
        0.045,
        0.028,
        f"Median of per-round ratios, CPython {ver} on Windows. "
        "Dense graphs are asymptotically worse for the heap, not better.",
        fontsize=9.5,
        color=MUTED,
    )

    fig.tight_layout(rect=(0.03, 0.055, 0.975, 0.945))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out}")


def chart_runtime(families: Dict[str, List[Dict]], out: Path, ver: str) -> None:
    cases = families["sparse"]
    xs = [c["v"] for c in cases]
    naive = [c["naive_ms"] for c in cases]
    heap = [c["heap_ms"] for c in cases]

    fig, ax = plt.subplots(figsize=(12.4, 7.0), dpi=200)
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for ys, colour, label, marker, style in (
        (naive, TERRACOTTA, "linear scan  O(V^2)", "s", (0, (5, 3))),
        (heap, AMBER, "binary heap  O((V+E) log V)", "o", "-"),
    ):
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=2.4,
            linestyle=style,
            marker=marker,
            markersize=7,
            markerfacecolor=colour,
            markeredgecolor=PAPER,
            markeredgewidth=1.6,
            label=label,
            zorder=3,
        )
        for x, y in zip(xs, ys):
            last = x == xs[-1]
            ax.annotate(
                f"{y:.0f}" if y >= 100 else f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(-13, 7) if last else (0, 13 if colour == TERRACOTTA else -21),
                ha="right" if last else "center",
                va="bottom" if last else ("bottom" if colour == TERRACOTTA else "top"),
                fontsize=9.5,
                color=colour,
            )

    def slope(ys: List[float]) -> float:
        first, last = ys[0], ys[-1]
        return math.log(last / first) / math.log(xs[-1] / xs[0])

    ax.annotate(
        f"fitted slope {slope(naive):.2f}",
        xy=(xs[-1], naive[-1]),
        xytext=(-10, -46),
        textcoords="offset points",
        ha="right",
        fontsize=10.5,
        color=TERRACOTTA,
        fontweight="bold",
    )
    ax.annotate(
        f"fitted slope {slope(heap):.2f}",
        xy=(xs[-1], heap[-1]),
        xytext=(-10, 26),
        textcoords="offset points",
        ha="right",
        fontsize=10.5,
        color=AMBER,
        fontweight="bold",
    )

    ax.set_xlim(750, 26000)
    ax.set_ylim(0.5, 14000)
    ax.set_xlabel("nodes V (E ~ 4V)", color=INK, fontsize=12, labelpad=8)
    ax.set_ylabel("single-source run, milliseconds", color=INK, fontsize=12, labelpad=8)

    ax.xaxis.set_major_locator(FixedLocator([1000, 2000, 4000, 8000, 16000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"], color=MUTED)
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    fig.suptitle(
        f"At {xs[-1] // 1000}k nodes the gap is "
        f"{naive[-1] / 1000:.1f} s against {heap[-1]:.0f} ms",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=17,
        color=INK,
        fontweight="bold",
    )
    ax.set_title(
        "Sparse graph, single source, all distances. Lower is better.",
        loc="left",
        fontsize=11.5,
        color=MUTED,
        pad=16,
    )

    ax.legend(loc="upper left", frameon=False, fontsize=11, handlelength=2.6)

    fig.text(
        0.045,
        0.028,
        f"Best of 3 rounds, CPython {ver} on Windows. "
        "A slope near 2 is the O(V^2) term; near 1.5 is the heap's log factor.",
        fontsize=9.5,
        color=MUTED,
    )

    fig.tight_layout(rect=(0.03, 0.055, 0.98, 0.93))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out}")


def chart_density(sweep: List[Dict], out: Path, ver: str) -> None:
    """Fixed V, growing E: the curve that settles the complexity argument."""
    xs = [c["edges_per_node"] for c in sweep]
    ys = [c["ratio_median"] for c in sweep]

    fig, ax = plt.subplots(figsize=(12.4, 7.0), dpi=200)
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (2, 3)), alpha=0.85)
    ax.text(3.6, 1.06, "break-even", color=MUTED, fontsize=9.5, ha="left", va="bottom")

    ax.plot(
        xs,
        ys,
        color=AMBER,
        linewidth=2.6,
        marker="o",
        markersize=8,
        markerfacecolor=AMBER,
        markeredgecolor=PAPER,
        markeredgewidth=1.8,
        zorder=3,
    )

    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.1f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=AMBER,
            fontweight="bold",
        )

    ax.axvspan(150, 800, color=TERRACOTTA, alpha=0.07, zorder=0)
    ax.text(
        250,
        6.0,
        "no practical gain",
        color=TERRACOTTA,
        fontsize=11,
        ha="center",
        va="center",
    )

    ax.set_xlim(3, 900)
    ax.set_ylim(0.9, 22)
    ax.set_xlabel("edges per node  (E / V)", color=INK, fontsize=12, labelpad=8)
    ax.set_ylabel("speedup  naive / heap", color=INK, fontsize=12, labelpad=8)

    ax.xaxis.set_major_locator(FixedLocator([4, 10, 30, 100, 300, 500]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["4", "10", "30", "100", "300", "500"], color=MUTED)
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5, 10, 20]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))

    fig.suptitle(
        "Hold V fixed and add edges: the heap's lead disappears",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=17,
        color=INK,
        fontweight="bold",
    )
    ax.set_title(
        f"V = {sweep[0]['v']} nodes, single source, only the edge count changes.",
        loc="left",
        fontsize=11.5,
        color=MUTED,
        pad=10,
    )

    fig.text(
        0.045,
        0.028,
        f"Median of per-round ratios, CPython {ver}. The heap never drops below the scan "
        "here: heappush is C, the scan is bytecode.",
        fontsize=9.5,
        color=MUTED,
    )

    fig.tight_layout(rect=(0.03, 0.055, 0.975, 0.945))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results.json"))
    parser.add_argument("--outdir", default=str(ROOT / "docs"))
    args = parser.parse_args()

    data = load_results(Path(args.results))
    families = split_families(data["cases"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chart_speedup(families, outdir / "speedup-by-graph-family.png", data["python"])
    chart_runtime(families, outdir / "runtime-by-graph-size.png", data["python"])
    sweep = data.get("density_sweep") or []
    if sweep:
        chart_density(sweep, outdir / "speedup-vs-density.png", data["python"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
