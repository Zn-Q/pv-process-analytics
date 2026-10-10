#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
制程良率与参数趋势分析器（Process Yield & Trend Analyst）分析内核

把「批次级良率/效能台账」自动转成「良率口径汇总 + 跨批趋势方向判断 +
规格限(USL/LSL)越限预警 + 前后批对比 + 用户自定义分段对比」的分析结果。

【数据脱敏红线】
本内核是通用方法论，不绑定任何特定行业、产品、产线、工艺或参数数值。
- 不内置任何行业列名、工艺环节、真实阈值或真实参数名
- 所有批次标识列、良率列、参数列、规格限、分段列均由使用者传入
- 不输出任何"工艺定性结论"，只给可核查的分析方向

对比 SPC（process_spc.py）：
- SPC 侧重"批量内稳定性 + 离群识别"（统计控制）
- 本工具侧重"良率/效能口径 + 跨批趋势方向 + 规格限越限 + 批次间对比"（管理取向）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import csv
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 支持 xlsx 需 openpyxl；缺失时降级仅支持 CSV/tsv
try:
    import openpyxl  # noqa: F401
    _HAS_XLSX = True
except Exception:
    _HAS_XLSX = False


# ============ 通用 & 去行业常量 ============

# 越界判定的通用阈值（纯数值合理性，非行业含义；可被用户配置覆盖）
_DEFAULT_LOWER_BOUND = 0.0
_DEFAULT_UPPER_BOUND = 100.0


@dataclass
class Params:
    """分析参数（全部由使用者传入，无行业默认）"""
    batch_col: str                 # 批次标识列
    metric_cols: List[str]         # 良率/效能参数列（至少一个）
    usage: str = "yield"           # yield=良率口径(0-100或0-1) / param=参数趋势(任意范围)
    usl: Optional[Dict[str, float]] = None   # 每参数上限（可选）
    lsl: Optional[Dict[str, float]] = None   # 每参数下限（可选）
    segment_col: Optional[str] = None        # 用户自定义分段列（可选）
    lower_bound: float = _DEFAULT_LOWER_BOUND
    upper_bound: float = _DEFAULT_UPPER_BOUND


@dataclass
class BatchMetric:
    batch: str
    metric: str
    n: int
    mean: Optional[float]
    std: Optional[float]
    cv: Optional[float]          # 变异系数（%）仅 n>=2 且 mean!=0
    min_v: Optional[float]
    max_v: Optional[float]
    median: Optional[float]
    outside_usl: int = 0         # 高于 USL 的样本数
    outside_lsl: int = 0         # 低于 LSL 的样本数
    n_used: int = 0              # 参与统计的样本数（剔除越界/缺失后）


@dataclass
class TrendResult:
    metric: str
    batches: List[str]           # 按顺序排列的批次
    means: List[float]           # 各批均值
    slope: Optional[float]       # 线性回归斜率（每批变化）
    direction: str               # up / down / flat
    mean_level: Optional[float]  # 整体均值
    delta_first_last: Optional[float]  # 首尾批均值差


@dataclass
class SpecViolation:
    """规格限越限预警（按批）"""
    batch: str
    metric: str
    kind: str                    # above_usl / below_lsl
    limit: float
    n_outside: int
    pct: float                   # 该批越限样本占比（%）


@dataclass
class SegmentStat:
    segment: str
    metric: str
    n_batch: int
    mean: Optional[float]
    std: Optional[float]
    delta_vs_overall: Optional[float]  # 对比整体均值的差


@dataclass
class AnalysisResult:
    batches: List[str]
    segment_col: Optional[str] = None
    segments: List[SegmentStat] = field(default_factory=list)
    batch_metrics: List[BatchMetric] = field(default_factory=list)
    trends: List[TrendResult] = field(default_factory=list)
    spec_violations: List[SpecViolation] = field(default_factory=list)
    digest: List[str] = field(default_factory=list)   # 结论摘要


# ============ 数据清洗 ============

def _clean_rows(table: List[Dict[str, Any]], p: Params) -> List[Dict[str, Any]]:
    """剔除辅助文件行、按通用越界阈值过滤、字段缺失兜底"""
    cleaned = []
    for row in table:
        if not row or not any(str(k).strip() for k in row.keys()):
            continue
        # 过滤空行 / 全空
        if all(str(v).strip() == "" for v in row.values()):
            continue
        cleaned.append(row)
    return cleaned


def _load_table(path: str) -> List[Dict[str, Any]]:
    """读取 CSV / tsv / xlsx → List[Dict]（首行表头）"""
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for r in reader:
                rows.append({k: (v.strip() if isinstance(v, str) else v)
                             for k, v in r.items()})
    elif ext in (".xlsx", ".xlsm"):
        if not _HAS_XLSX:
            raise RuntimeError("读取 xlsx 需要 openpyxl，请先安装或改用 CSV")
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        header = None
        for row in ws.iter_rows(values_only=True):
            if row is None:
                continue
            if header is None:
                header = [str(c).strip() if c is not None else "" for c in row]
                continue
            rows.append(dict(zip(header, [c for c in row])))
        wb.close()
    else:
        raise ValueError(f"不支持的文件类型: {ext}（支持 csv/tsv/xlsx）")
    return rows


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in ("na", "nan", "null", "-", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ============ 批次/参数统计 ============

def _batch_metrics(cleaned: List[Dict[str, Any]], p: Params) -> List[BatchMetric]:
    """按批次 × 参数分组的均值/中位/标准差/CV/越限统计"""
    out: List[BatchMetric] = []
    # 批次顺序（按出现顺序去重）
    batches: "List[str]" = []
    for row in cleaned:
        b = str(row.get(p.batch_col, "")).strip()
        if b and b not in batches:
            batches.append(b)

    for b in batches:
        for mcol in p.metric_cols:
            vals: List[float] = []
            for row in cleaned:
                if str(row.get(p.batch_col, "")).strip() != b:
                    continue
                v = _to_float(row.get(mcol))
                if v is None:
                    continue
                # 通用越界过滤（用户可配置边界）
                if not (p.lower_bound <= v <= p.upper_bound):
                    continue
                vals.append(v)
            bm = BatchMetric(batch=b, metric=mcol, n=len(vals), n_used=len(vals),
                             mean=None, std=None, cv=None, min_v=None,
                             max_v=None, median=None)
            if vals:
                vals_sorted = sorted(vals)
                bm.mean = sum(vals) / len(vals)
                bm.median = _median(vals_sorted)
                bm.min_v = vals_sorted[0]
                bm.max_v = vals_sorted[-1]
                if len(vals) >= 2:
                    var = sum((x - bm.mean) ** 2 for x in vals) / (len(vals) - 1)
                    bm.std = var ** 0.5
                    if bm.mean:
                        bm.cv = bm.std / bm.mean * 100.0
                # 规格限越限统计
                usl = p.usl.get(mcol) if p.usl else None
                lsl = p.lsl.get(mcol) if p.lsl else None
                if usl is not None:
                    bm.outside_usl = sum(1 for x in vals if x > usl)
                if lsl is not None:
                    bm.outside_lsl = sum(1 for x in vals if x < lsl)
            out.append(bm)
    return out


def _median(vals_sorted: List[float]) -> Optional[float]:
    n = len(vals_sorted)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return vals_sorted[mid]
    return (vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0


# ============ 跨批趋势方向（线性回归）============

def _trend_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """最小二乘斜率 per 批次；批次<3 无法判方向"""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def _direction_from_slope(slope: Optional[float], ys: List[float]) -> str:
    """判定趋势方向：flat/up/down，阈值为相对整体波动的尺度归一"""
    if slope is None:
        return "flat"
    ys_sorted = sorted(ys)
    if len(ys_sorted) >= 3:
        scale = (ys_sorted[-1] - ys_sorted[0]) if ys_sorted[-1] != ys_sorted[0] else abs(ys_sorted[-1])
    else:
        scale = abs(ys_sorted[-1]) if ys_sorted else 1.0
    if scale == 0:
        scale = 1.0
    # 斜率绝对值相对每批波动 2% 以内视为平稳
    thresh = 0.02 * scale
    if abs(slope) < thresh:
        return "flat"
    return "up" if slope > 0 else "down"


def _trends(batch_metrics: List[BatchMetric]) -> List[TrendResult]:
    """对每个指标，按批次出现顺序排均值做趋势判断"""
    # 按参数分组
    by_metric: "Dict[str, OrderedDict[str, float]]" = OrderedDict()
    for bm in batch_metrics:
        if bm.mean is None:
            continue
        by_metric.setdefault(bm.metric, OrderedDict())
        by_metric[bm.metric][bm.batch] = bm.mean
    out: List[TrendResult] = []
    for mcol, order in by_metric.items():
        batches = list(order.keys())
        means = [order[b] for b in batches]
        xs = list(range(len(batches)))
        slope = _trend_slope(xs, means)
        direction = _direction_from_slope(slope, means)
        out.append(TrendResult(
            metric=mcol, batches=batches, means=means, slope=slope,
            direction=direction,
            mean_level=sum(means) / len(means) if means else None,
            delta_first_last=(means[-1] - means[0]) if len(means) >= 2 else None,
        ))
    return out


# ============ 规格限越限预警 ============

def _spec_violations(batch_metrics: List[BatchMetric],
                     usl: Optional[Dict[str, float]],
                     lsl: Optional[Dict[str, float]]) -> List[SpecViolation]:
    out: List[SpecViolation] = []
    for bm in batch_metrics:
        total = bm.n_used
        if total == 0:
            continue
        if usl and bm.metric in usl and bm.outside_usl > 0:
            out.append(SpecViolation(
                batch=bm.batch, metric=bm.metric, kind="above_usl",
                limit=usl[bm.metric], n_outside=bm.outside_usl,
                pct=bm.outside_usl / total * 100.0,
            ))
        if lsl and bm.metric in lsl and bm.outside_lsl > 0:
            out.append(SpecViolation(
                batch=bm.batch, metric=bm.metric, kind="below_lsl",
                limit=lsl[bm.metric], n_outside=bm.outside_lsl,
                pct=bm.outside_lsl / total * 100.0,
            ))
    return out


# ============ 分段对比 ============

def _segments(cleaned: List[Dict[str, Any]], batch_metrics: List[BatchMetric],
              p: Params) -> List[SegmentStat]:
    if not p.segment_col:
        return []
    # 每个批次所属分段（取该批首条记录的分段值）
    batch_seg: "Dict[str,str]" = {}
    for row in cleaned:
        b = str(row.get(p.batch_col, "")).strip()
        s = str(row.get(p.segment_col, "")).strip()
        if b and s and b not in batch_seg:
            batch_seg[b] = s
    seg_order: "List[str]" = []
    for b in batch_seg:
        if batch_seg[b] not in seg_order:
            seg_order.append(batch_seg[b])

    out: List[SegmentStat] = []
    # 对每个指标 × 分段，聚合该段内各批均值的平均
    for mcol in p.metric_cols:
        # 各批均值
        batch_mean: "Dict[str,float]" = {}
        for bm in batch_metrics:
            if bm.metric == mcol and bm.mean is not None:
                batch_mean[bm.batch] = bm.mean
        overall = [batch_mean[b] for b in batch_mean if b in batch_seg]
        overall_mean = sum(overall) / len(overall) if overall else None
        for seg in seg_order:
            seg_batches = [b for b in batch_seg if batch_seg[b] == seg and b in batch_mean]
            svals = [batch_mean[b] for b in seg_batches]
            if not svals:
                continue
            seg_mean = sum(svals) / len(svals)
            diff = (seg_mean - overall_mean) if overall_mean is not None else None
            n = len(svals)
            if n >= 2:
                var = sum((x - seg_mean) ** 2 for x in svals) / (n - 1)
                sd = var ** 0.5
            else:
                sd = None
            out.append(SegmentStat(segment=seg, metric=mcol, n_batch=n,
                                   mean=seg_mean, std=sd, delta_vs_overall=diff))
    return out


# ============ 摘要 ============

def _digest(trends: List[TrendResult], spec_violations: List[SpecViolation],
            segments: List[SegmentStat], usage: str) -> List[str]:
    lines: List[str] = []
    lines.append(f"共分析 {len(trends)} 个指标，趋势方向与整体水平如下：")
    for t in trends:
        d = {"up": "上升", "down": "下降", "flat": "平稳"}[t.direction]
        level = f"{t.mean_level:.4g}" if t.mean_level is not None else "N/A"
        dl = (f"（首尾差 {t.delta_first_last:+.4g}）"
              if t.delta_first_last is not None else "")
        lines.append(f"· 参数「{t.metric}」：整体水平 {level}，趋势{d}{dl}")
    if spec_violations:
        lines.append("规格限越限预警：")
        for v in spec_violations:
            kind = "高于上限USL" if v.kind == "above_usl" else "低于下限LSL"
            lines.append(f"· 批次「{v.batch}」参数「{v.metric}」{kind}={v.limit:.4g}，"
                         f"{v.n_outside} 个样本越限（{v.pct:.1f}%）")
    else:
        lines.append("未发现规格限越限（若未提供USL/LSL，则此项不参与判定）。")
    if segments:
        lines.append("分段对比：")
        for s in segments:
            diff = f"{s.delta_vs_overall:+.4g}" if s.delta_vs_overall is not None else "N/A"
            lines.append(f"· 分段「{s.segment}」参数「{s.metric}」均值 {s.mean:.4g} "
                         f"（{s.n_batch}批，对比整体 {diff}）")
    if usage == "yield":
        lines.append("口径提示：良率类指标通常越接近上限/UCL 越好，请结合传入的规格方向解读。")
    else:
        lines.append("口径提示：参数趋势方向需结合该参数的目标方向（越高越好或越低越好）解读。")
    return lines


# ============ 入口 ============

def analyze(table: List[Dict[str, Any]], p: Params) -> AnalysisResult:
    cleaned = _clean_rows(table, p)
    batch_metrics = _batch_metrics(cleaned, p)
    trends = _trends(batch_metrics)
    spec_violations = _spec_violations(batch_metrics, p.usl, p.lsl)
    segments = _segments(cleaned, batch_metrics, p)
    batches: "List[str]" = []
    for bm in batch_metrics:
        if bm.batch not in batches:
            batches.append(bm.batch)
    digest = _digest(trends, spec_violations, segments, p.usage)
    return AnalysisResult(
        batches=batches, segment_col=p.segment_col,
        segments=segments, batch_metrics=batch_metrics,
        trends=trends, spec_violations=spec_violations, digest=digest,
    )


def _to_json(res: AnalysisResult) -> Dict[str, Any]:
    def _b(bm):
        return {"batch": bm.batch, "metric": bm.metric, "n": bm.n, "n_used": bm.n_used,
                "mean": bm.mean, "median": bm.median, "std": bm.std, "cv": bm.cv,
                "min": bm.min_v, "max": bm.max_v,
                "outside_usl": bm.outside_usl, "outside_lsl": bm.outside_lsl}
    return {
        "batches": res.batches,
        "digest": res.digest,
        "trends": [{"metric": t.metric, "direction": t.direction,
                    "slope": t.slope, "mean_level": t.mean_level,
                    "delta_first_last": t.delta_first_last,
                    "batches": t.batches, "means": t.means} for t in res.trends],
        "spec_violations": [{"batch": v.batch, "metric": v.metric, "kind": v.kind,
                             "limit": v.limit, "n_outside": v.n_outside,
                             "pct": v.pct} for v in res.spec_violations],
        "segments": [{"segment": s.segment, "metric": s.metric, "n_batch": s.n_batch,
                      "mean": s.mean, "std": s.std,
                      "delta_vs_overall": s.delta_vs_overall} for s in res.segments],
        "batch_metrics": [_b(bm) for bm in res.batch_metrics],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="制程良率与参数趋势分析器（通用方法论）")
    ap.add_argument("--data", required=True, help="数据文件 (csv/tsv/xlsx)，首行为表头")
    ap.add_argument("--batch", required=True, help="批次标识列名")
    ap.add_argument("--metrics", required=True, help="良率/参数列名，逗号分隔")
    ap.add_argument("--usage", default="yield", choices=["yield", "param"],
                    help="yield=良率口径 / param=参数趋势")
    ap.add_argument("--usl", default="", help="规格上限，如 'colA:99,colB:1.5'")
    ap.add_argument("--lsl", default="", help="规格下限，如 'colA:90,colB:1.0'")
    ap.add_argument("--segment", default="", help="用户自定义分段列名（可选）")
    ap.add_argument("--lower-bound", type=float, default=_DEFAULT_LOWER_BOUND)
    ap.add_argument("--upper-bound", type=float, default=_DEFAULT_UPPER_BOUND)
    ap.add_argument("--out-json", default="", help="输出JSON路径（可选）")
    args = ap.parse_args(argv)

    metric_cols = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_cols:
        print("错误: --metrics 至少需要一个指标列")
        return 2
    usl = {}
    for kv in args.usl.split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                usl[k] = float(v)
    lsl = {}
    for kv in args.lsl.split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                lsl[k] = float(v)

    p = Params(batch_col=args.batch, metric_cols=metric_cols, usage=args.usage,
               usl=usl or None, lsl=lsl or None,
               segment_col=args.segment or None,
               lower_bound=args.lower_bound, upper_bound=args.upper_bound)
    table = _load_table(args.data)
    res = analyze(table, p)

    print("=" * 60)
    print("分析摘要")
    print("=" * 60)
    for line in res.digest:
        print(line)
    print()
    print("=" * 60)
    print("批次指标明细（批次/指标/n/均值/中位/标准差/CV/越限）")
    print("=" * 60)
    for bm in res.batch_metrics:
        print(f"  {bm.batch:>8} | {bm.metric:<14} | n={bm.n_used} | "
              f"mean={bm.mean if bm.mean is None else round(bm.mean,4)} | "
              f"med={bm.median if bm.median is None else round(bm.median,4)} | "
              f"std={bm.std if bm.std is None else round(bm.std,4)} | "
              f"cv={bm.cv if bm.cv is None else round(bm.cv,2)}% | "
              f"U={bm.outside_usl}/L={bm.outside_lsl}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(_to_json(res), f, ensure_ascii=False, indent=2)
        print(f"\nJSON结果已写入: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())