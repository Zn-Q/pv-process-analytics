# yield-trend — Yield & Parameter Trend Analysis

Converts a batch-level yield / performance ledger into **yield summary +
cross-batch trend direction + USL/LSL spec-limit violation alerts + before/after &
segmented comparison**.

## Features
- **Yield-metric summary** per batch (mean, std, CV, min/max, outside-spec counts)
- **Cross-batch trend direction** (up / down / flat) with slope
- **USL/LSL spec-limit violation alerts** per batch, with outside-spec sample count & %
- **Before/after & segmented comparison** via a user-supplied segment column

## Quick start
```bash
python3 examples/yield_plot.py
```
Loads `examples/demo_yield.csv` (10 batches × 5 samples, `yield` declining +
`temperature` rising, with intentionally embedded spec violations) and writes
`examples/preview_yield_demo.png`:

- B007 / B010 `yield` below LSL=90
- B008 `yield` above USL=99
- B009 / B010 `temperature` above USL=3.6

## Use your own data
```bash
python3 -c "from process_yield_trend import Params,_load_table,analyze; \
p=Params(batch_col='batch',metric_cols=['yield'],usage='yield', \
usl={'yield':99},lsl={'yield':90}); \
res=analyze(_load_table('your.csv'),p); print(res.digest)"
```

## Data format
One row per sample with a batch identifier + at least one numeric metric column.
`.csv` / `.xlsx` supported.

## Full edition
The commercial edition adds template-driven **PPT/Word reports** and layered
business report layouts. Once the store listing is live, use the linked channel;
the open core here lets you validate the methodology first.