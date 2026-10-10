# spc-analysis — Batch Process SPC in plain Python

Turns a batch-level process data ledger into **stability + trend + control-chart +
capability** analysis. Generic methodology — no line-specific thresholds or column
mappings are hard-coded; everything is supplied at run time.

## Features
- Batch **stability**: mean / range / coefficient of variation (CV)
- **Cross-batch trend** per parameter
- **Outlier detection** with three-level granularity (batch / sample / parameter)
  and attribution hints (single-point outlier vs whole-batch shift)
- **X-bar & R control charts** (batch = subgroup), out-of-control points flagged
- **Capability indices** Cp / Cpk / Ppu / Cpl when spec limits are provided

## Quick start
```bash
python3 examples/spc_plot.py
```
This loads the bundled sample `examples/demo_spc.csv` (8 batches × 8 samples,
two parameters `paramA` / `paramB`), runs the full pipeline and writes
`examples/preview_spc_demo.png` (X-bar / R charts per parameter):

- LOT-003 S6 `paramA=73.0` → single-point outlier (above USL)
- LOT-006 → whole-batch shift (X-bar beyond control limits)

## Use your own data
```bash
python3 -c "from process_spc import Params,_load_table,analyze; \
res=analyze(_load_table('your.csv'), Params(batch_col='batch',sample_col='sample', \
param_cols=['p1','p2'], spec_usl={'p1':72}, spec_lsl={'p1':38})); \
print(res.to_dict()['outliers']); print(res.capability)"
```
`.csv` / `.xlsx` both supported (openpyxl for xlsx).

## Data format
One row per sample, with a batch identifier column + at least one numeric
parameter column. Optional sample column enables single-sample attribution.

## Full edition
The complete commercial edition adds template-driven **PPT/Word reports**
(render results into your own templates) and polished business report layouts.
Get it from the store link once the listing is live; the open version here keeps
the same analysis core so you can evaluate the methodology first.