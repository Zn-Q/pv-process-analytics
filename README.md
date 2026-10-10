# PV Process Analytics — Open Toolkit

[![License: MIT](https://img.shields.io/github/license/Zn-Q/pv-process-analytics)](LICENSE)
[![Language: Python](https://img.shields.io/github/languages/top/Zn-Q/pv-process-analytics)]()
[![Stars](https://img.shields.io/github/stars/Zn-Q/pv-process-analytics?style=social)](https://github.com/Zn-Q/pv-process-analytics)

Open-source demo layer for **photovoltaic / semiconductor process data analytics**.

Lightweight, dependency-light Python toolkits covering three common engineering needs
in PV cell & wafer manufacturing:

| Module | What it does | Stack |
|---|---|---|
| `spc-analysis/` | Batch-level **SPC**: stability (mean / range / CV), cross-batch trend, **outlier detection** (batch/sample/param, with attribution hints), **X-bar & R control charts**, **capability (Cp/Cpk/Ppk)** | numpy |
| `yield-trend/` | Batch-level **yield / parameter trend**: per-metric cross-batch direction, **USL/LSL spec-limit violation alerts**, before/after & segmented comparison | stdlib + openpyxl |
| `lifetime-simulation/` | **QSSPC carrier-lifetime** solver for passivated silicon wafers: injection-dependent **τ_eff(Δn)** forward simulation + **least-squares extraction** of Seff / Nt / Et / σ | numpy, scipy, matplotlib |

> Everything here is **generic methodology with zero vendor/line-specific data** —
> no real cell-technology keywords, no thresholds, no column mappings are hard-coded.
> All parameters are user-supplied at run time.

## Quick start

```bash
# SPC control charts + outliers + capability
python3 spc-analysis/examples/spc_plot.py        # → spc-analysis/examples/preview_spc_demo.png

# Yield trend + spec-limit violations
python3 yield-trend/examples/yield_plot.py       # → yield-trend/examples/preview_yield_demo.png

# Carrier-lifetime forward simulation + parameter extraction
python3 lifetime-simulation/demo_run.py          # → curves + fit + report under demo_output/
```

Preview outputs:

- SPC control charts (X-bar / R) with red out-of-control points
- Yield cross-batch trend with USL/LSL spec lines
- QSSPC τ_eff(Δn) curve + measured-vs-fit extraction

## Repository layout

```
pv-process-analytics/
├── spc-analysis/          # SPC engine + sample data + lightweight control-chart plotting
├── yield-trend/           # yield/parameter-trend engine + lightweight plotting
├── lifetime-simulation/   # QSSPC lifetime solver + demo
├── LICENSE
└── README.md
```

## Requirements
Plain Python 3.8+; `numpy` (all), `scipy`+`matplotlib` (lifetime simulation).
No external API keys, fully local — your wafer data never leaves your machine.

## Commercial edition
This repository ships the **open demo layer** (analysis cores, sample data and
lightweight plotting). The full commercial editions — including **proprietary
report-template adapters** (render results into your own PPT/Word templates) and
polished business-report layouts, packaged as ready-to-use Agent **Skills** — are
distributed through the **ByteDance skill-evaluation platform (虾评 👉
xiaping.coze.com)**, where you can try and install them directly. Search there
for:

- **制程SPC数据分析工具** (SPC engine, full report templates)
- **制程良率与参数趋势分析器** (yield-trend engine, spec-limit alerts)
- **载流子寿命分析工具** (QSSPC solver, Word-report + fit extraction)

For bespoke template-driven reports or enterprise integration, open an issue or
reach out through the linked platform pages.

## License
MIT — see [LICENSE](LICENSE). Note the license covers only the open demo layer;
commercial/complete editions are excluded as described above.