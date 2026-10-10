# lifetime-simulation — QSSPC Carrier-Lifetime Solver

Pure-Python solver for **photovoltaic passivated-silicon carrier lifetime**
(n-type Cz-Si focus). Reproduces the classic QSSPC "low-injection plateau + high-
injection Auger roll-off" τ_eff(Δn) shape and supports **parameter extraction**
from measured multi-point data.

## Features
- **Forward simulation**: compute the injection-dependent τ_eff(Δn) curve for a
  given surface-recombination / bulk-defect model (Seff / Nt / Et / σ)
- **Measured-data extraction**: least-squares fit of measured (Δn, τ_eff) points
  (e.g. from Sinton WCT-120) to recover Seff / Nt / Et / σ
- **Report output**: fit chart (measured scatter vs fit) + Word report

## Quick start
```bash
python3 demo_run.py                  # full demo → demo_output/
python3 demo_run.py --out my_out     # custom output dir
```
Runs all three capabilities on bundled synthetic data (with 2% noise), which the
extraction should recover to the true values.

## Solver CLI
```bash
# 1) forward simulation — τ_eff(Δn) curve for given surface/defect params
python3 qsspc_lifetime.py --Seff 1.0 --Nt 1e10 --plot --out sim_out
#   other knobs: --Et 0.0 --Nd 4.5e15 --W 165e-4 --sigma-n 1e-15 --model kc2002

# 2) measured-data extraction — fit (Δn, τ_eff) CSV → Seff/Nt/Et (+ Word report)
python3 qsspc_lifetime.py --data measured.csv --fit Seff,Nt,Et \
       --report my_report --out fit_out
```
`--plot` saves the forward curve PNG; `--fit` takes a comma-separated subset of
`Seff,Nt,Et,sigma_n,sigma_p` to fit. See `demo_run.py` for wiring examples.

## Requirements
`numpy`, `scipy`, `matplotlib` (charts), `python-docx` (Word report — optional if
you only need curves). No external API keys — fully local.

## Full edition
The commercial edition adds polished template-driven reports. See the main README
for the store link when the listing is live.