#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPC Analysis — lightweight open-source demo (control charts + outliers + capability)
=====================================================================================
Runs the SPC core on bundled sample data and renders X-bar / R control charts as PNG.
No proprietary report-template adapter included — this is the open demo layer.
The full version (with PPT/Word report + custom template support) is released on the
commerce channel, see the repo top-level README for the link.

Usage:
    python3 examples/spc_plot.py            # from repo root
    python3 spc_plot.py                     # from spc-analysis/, or
    python3 spc-analysis/examples/spc_plot.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(_HERE)          # spc-analysis/
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)

from process_spc import Params, _load_table, analyze   # noqa: E402


def plot_control_charts(res, out_path):
    """Render one PNG per parameter: top = X-bar, bottom = R chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # group charts by param, keep xbar & r
    by_param = {}
    for c in res.control_charts:
        by_param.setdefault(c.param, []).append(c)

    n_params = len(by_param)
    if n_params == 0:
        print("  (no control charts computed — need sample col / subgroup)")
        return None

    fig, axes = plt.subplots(n_params, 2, figsize=(13, 3.4 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, 2)

    for i, (param, charts) in enumerate(by_param.items()):
        for c in charts:
            ax = axes[i, 0] if c.kind == "xbar" else axes[i, 1]
            xs = list(range(len(c.values)))
            ax.plot(xs, c.values, marker="o", color="#1f77b4", label=f"{param} {c.kind}")
            # control limits
            if c.center is not None:
                ax.axhline(c.center, color="green", ls="-", lw=1)
            if c.ucl is not None:
                ax.axhline(c.ucl, color="red", ls="--", lw=1)
            if c.lcl is not None:
                ax.axhline(c.lcl, color="red", ls="--", lw=1)
            # out-of-control points (beyond control limits) in red
            for j, v in enumerate(c.values):
                if (c.ucl is not None and v > c.ucl) or (c.lcl is not None and v < c.lcl):
                    ax.plot(xs[j], v, marker="o", color="red", ms=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(c.batches, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{param}  {c.kind.upper()}  (UCL={c.ucl:.3g}, CL={c.center:.3g}, LCL={c.lcl:.3g})" if c.ucl else
                         f"{param}  {c.kind.upper()}  (CL={c.center:.3g})")
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    demo_csv = os.path.join(_HERE, "demo_spc.csv")
    if not os.path.exists(demo_csv):
        print(f"[open] sample data missing: {demo_csv}")
        return 2

    p = Params(
        batch_col="batch",
        sample_col="sample",
        param_cols=["paramA", "paramB"],
        lower_bound=5.0,
        upper_bound=99.0,
        spec_usl={"paramA": 72.0, "paramB": 96.0},
        spec_lsl={"paramA": 38.0, "paramB": 82.0},
    )
    res = analyze(_load_table(demo_csv), p)

    print("=" * 60)
    print("Analysis summary")
    print("=" * 60)
    print(f"Batches : {len(res.batches)}")
    print(f"Outliers: {len(res.outliers)}")
    for o in res.outliers:
        sid = o.sample or "-"
        print(f"  [{o.reason}] {o.batch}/{sid}  {o.param} = {o.value}")
    if res.capability:
        print("Capability (Cp/Cpk/Ppk):")
        for param, cp in res.capability.items():
            pp = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cp.items() if k != "n_batches"}
            print(f"  {param}: {pp}")

    out = os.path.join(_HERE, "preview_spc_demo.png")
    plot_control_charts(res, out)
    if res.control_charts:
        print(f"\nControl charts: {len(res.control_charts)}")
        print(f"Saved PNG : {out}")


if __name__ == "__main__":
    sys.exit(main())