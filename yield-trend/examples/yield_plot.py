#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yield & Trend Analysis — lightweight open-source demo (trends + spec violations)
================================================================================
Runs the yield / parameter-trend core on bundled sample data and renders a
per-metric cross-batch trend chart with USL/LSL spec limits as PNG.
No proprietary report-template adapter included — this is the open demo layer.

Usage:
    python3 yield-trend/examples/yield_plot.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(_HERE)          # yield-trend/
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)

from process_yield_trend import Params, _load_table, analyze   # noqa: E402


def plot_trends(res, p, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = p.metric_cols
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    # index of each batch metric by metric name
    for i, metric in enumerate(metrics):
        ax = axes[i]
        xs = []
        ys = []
        labels = []
        for b in res.batch_metrics:
            if b.metric != metric:
                continue
            xs.append(len(xs))
            ys.append(b.mean)
            labels.append(b.batch)
        ax.plot(xs, ys, marker="o", color="#1f77b4", label=metric)
        # spec limits
        if p.usl and metric in p.usl:
            ax.axhline(p.usl[metric], color="red", ls="--", lw=1.2, label=f"USL={p.usl[metric]}")
        if p.lsl and metric in p.lsl:
            ax.axhline(p.lsl[metric], color="orange", ls="--", lw=1.2, label=f"LSL={p.lsl[metric]}")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{metric} cross-batch trend")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    demo_csv = os.path.join(_HERE, "demo_yield.csv")
    if not os.path.exists(demo_csv):
        print(f"[open] sample data missing: {demo_csv}")
        return 2

    p = Params(
        batch_col="batch",
        metric_cols=["yield", "temperature"],
        usage="yield",
        usl={"yield": 99, "temperature": 3.6},
        lsl={"yield": 90, "temperature": 2.6},
        lower_bound=0.0,
        upper_bound=100.0,
    )
    res = analyze(_load_table(demo_csv), p)

    print("=" * 60)
    print("Analysis summary (digest)")
    print("=" * 60)
    for line in res.digest:
        print("  " + line)

    if res.trends:
        print("\nTrend direction:")
        for t in res.trends:
            print(f"  {t.metric}: {t.direction}  (slope={getattr(t,'slope',None)})")

    if res.spec_violations:
        print("\nSpec violations:")
        for v in res.spec_violations:
            direction = "above USL" if v.kind == "above_usl" else "below LSL"
            print(f"  [{v.batch}] {v.metric} {direction} limit={v.limit}  ({v.n_outside} samples, {v.pct}%)")

    out = os.path.join(_HERE, "preview_yield_demo.png")
    plot_trends(res, p, out)
    print(f"\nSaved PNG : {out}")


if __name__ == "__main__":
    sys.exit(main())