#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process SPC Data Analyst - 分析内核（通用方法论，无任何行业/工艺硬编码）
计算：批量内稳定性、跨批次趋势、参数相关性、离群预警（带归因线索）、用户自定义分段分析。

v1.1 改动：
  1. 离群检测升级为两级分级，消除"整批偏移被误报为多条单点离群"：
     - 单点离群：在<批次内部>按 IQR 识别个别偏离样本
     - 整批偏移：对<各批均值>做跨批 IQR，均值明显偏离整体时提示整批偏移
  2. 单点离群样本从批均值/CV/极值差中剔除，避免统计失真
  3. 样品列名可配置（Params.sample_col），未指定时自动检测常见样品列名
所有字段名、参数名、阈值均由用户输入驱动，本模块不内置任何产线/工艺/数值。
"""
from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass, field, asdict
from statistics import mean, pstdev
from typing import List, Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------- data model

# 常见样品列名（用户未指定 sample_col 时自动检测用，匹配时忽略大小写）
_COMMON_SAMPLE_KEYS = ("sample", "样品", "样品编号", "样本", "样本号", "样本编号", "sid", "样本名")

# 批内单点离群（IQR）所需的最小批量样本数。
# 小样本批（<8）的数据过稀疏、聚集随机，IQR/MAD 的 q1≈q3 会把正常上界样本误判成单点离群
# （如 5 个样本中 4 个几乎相等 + 1 个稍高时），故小样本批不做批内单点判定，仅保留整批偏移检测。
# 仅当批量样本 ≥ 该值时 IQR 分位点才稳定，启用单点离群识别。
_MIN_BATCH_POINTS_FOR_SINGLE = 8


@dataclass
class Params:
    """批次列（batch_col）与待分析参数列（param_cols）。列名由用户提供。"""
    batch_col: str
    param_cols: List[str]
    segment_col: Optional[str] = None      # 用户自定义分段列（可选）
    sample_col: Optional[str] = None       # 样品标识列（用户可配置；None 时自动检测）
    lower_bound: float = 0.0               # 越界剔除下限（用户可配置，仅数据清洗，非规格限）
    upper_bound: float = 100.0             # 越界剔除上限（用户可配置，仅数据清洗，非规格限）
    outlier_std_mult: float = 2.5          # 标准差倍数离群阈值（用户可配置）
    # ---- v1.2 能力指数规格限（可选）。注意：upper_bound/lower_bound 只是数据清洗的越界剔除
    #      阈值，绝非规格限。能力指数 Cp/Ppk/Cpu/Cpl 仅在显式提供规格限时计算。
    spec_usl: Optional[Dict[str, float]] = None   # 参数 -> 规格上限 USL
    spec_lsl: Optional[Dict[str, float]] = None   # 参数 -> 规格下限 LSL


@dataclass
class OutlierHit:
    batch: str
    sample: Optional[str]      # 样品标识，可为空
    param: str
    value: float
    reason: str                # 归因线索（可核查方向，非工艺定性结论）


@dataclass
class BatchStat:
    batch: str
    n: int
    vals: List[float]
    mean: Optional[float] = None
    ptp: Optional[float] = None        # 极值差（range）
    cv: Optional[float] = None         # 变异系数 CV%
    n_used: int = 0                    # 剔除单点离群后参与统计的样本数


@dataclass
class ControlChart:
    """SPC X-bar / R 控制图（单参数，批次=子组）。"""
    param: str
    kind: str                     # "xbar" / "r"
    batches: List[str]            # 批次顺序（与 values 对齐）
    values: List[float]           # xbar: 各批均值；r: 各批极差
    center: Optional[float] = None       # 中心线 CL
    ucl: Optional[float] = None          # 上控制限
    lcl: Optional[float] = None          # 下控制限（R 图 n≤6 时 D3=0 → LCL=0）
    subgroup_size: int = 0        # 子组（批）容量，取各批样本数的众数
    n_batches: int = 0
    signals: List[str] = field(default_factory=list)   # 判异信号（西格玛规则文本）
    outliers: List[int] = field(default_factory=list)  # 超控制限的子组下标


@dataclass
class AnalysisResult:
    batches: List[BatchStat] = field(default_factory=list)
    outliers: List[OutlierHit] = field(default_factory=list)
    correlation: Dict[str, float] = field(default_factory=dict)
    segment_stats: List[Dict[str, Any]] = field(default_factory=list)
    trend: Dict[str, List[float]] = field(default_factory=dict)   # param -> per-batch mean
    # ---- v1.2 新增（不改变上述既有字段的值，仅扩展）----
    control_charts: List[ControlChart] = field(default_factory=list)  # X-bar/R 控制图
    capability: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # param -> 能力指数

    def to_dict(self) -> Dict[str, Any]:
        """JSON 可序列化（含既有字段 + v1.2 新增的控制图与能力指数）。"""
        return {
            "batches": [{
                "batch": bs.batch, "n": bs.n,
                "mean": bs.mean, "ptp": bs.ptp, "cv": bs.cv, "n_used": bs.n_used,
            } for bs in self.batches],
            "outliers": [{
                "batch": o.batch, "sample": o.sample, "param": o.param,
                "value": o.value, "reason": o.reason,
            } for o in self.outliers],
            "correlation": dict(self.correlation),
            "segment_stats": list(self.segment_stats),
            "trend": {k: list(v) for k, v in self.trend.items()},
            "control_charts": [{
                "param": c.param, "kind": c.kind, "batches": list(c.batches),
                "values": list(c.values), "center": c.center, "ucl": c.ucl,
                "lcl": c.lcl, "subgroup_size": c.subgroup_size,
                "n_batches": c.n_batches, "signals": list(c.signals),
                "outliers": list(c.outliers),
            } for c in self.control_charts],
            "capability": {k: dict(v) for k, v in self.capability.items()},
        }


# ---------------------------------------------------------------- core funcs

def _clean_rows(table: List[Dict[str, Any]], p: Params) -> List[Dict[str, Any]]:
    """数据清洗：只保留含批次与全部参数列的整行；越界值剔除；缺失行兜底。"""
    cleaned = []
    for row in table:
        if p.batch_col not in row:
            continue
        vals_ok = all(pc in row and row[pc] is not None for pc in p.param_cols)
        if not vals_ok:
            continue
        try:
            nums = {pc: float(row[pc]) for pc in p.param_cols}
        except (TypeError, ValueError):
            continue
        if any(not (p.lower_bound < v < p.upper_bound) for v in nums.values()):
            continue   # 越界剔除，不参与统计
        cleaned.append({**row, **nums})
    return cleaned


def _resolve_sample_col(p: Params, cleaned: List[Dict[str, Any]]) -> Optional[str]:
    """确定样品列名：优先用用户显式指定，否则从常见样品列名中自动检测（忽略大小写）。"""
    if p.sample_col:
        return p.sample_col
    if not cleaned:
        return None
    keys_lower = {str(k).strip().lower() for k in cleaned[0].keys()}
    for k in _COMMON_SAMPLE_KEYS:
        if k.lower() in keys_lower:
            # 返回表中真实列名（保持原始大小写），便于后续按列取值
            for col in cleaned[0].keys():
                if str(col).strip().lower() == k.lower():
                    return col
    return None


def _detect_single_outliers(cleaned: List[Dict[str, Any]], p: Params,
                            sample_col: Optional[str]) -> Dict[Any, OutlierHit]:
    """单点离群（批内 IQR）：对每个批次内部独立判定，识别个别偏离样本。

    返回 { (batch, param, idx): OutlierHit }，idx 为 cleaned 中该行下标，供剔除统计。
    仅当批次样本数 ≥ _MIN_BATCH_POINTS_FOR_SINGLE 时启用批内 IQR 单点判定——
    小样本批（<8）数据过稀疏、聚集随机，IQR 的 q1≈q3 会把正常上界样本误判成单点离群，
    故小样本批不判单点（避免误报），整批偏移由 _detect_batch_offset 独立负责。
    """
    hits: Dict[Any, OutlierHit] = {}
    # 每批每参数的样本值
    by_batch: Dict[str, Dict[str, List[tuple]]] = {}   # batch -> param -> [(idx, val)]
    for idx, r in enumerate(cleaned):
        b = str(r[p.batch_col])
        by_batch.setdefault(b, {})
        for pc in p.param_cols:
            by_batch[b].setdefault(pc, []).append((idx, float(r[pc])))

    for b, params_map in by_batch.items():
        for pc, sels in params_map.items():
            if len(sels) < _MIN_BATCH_POINTS_FOR_SINGLE:
                continue
            vals = np.array([v for _, v in sels], dtype=float)
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            if iqr < 1e-12:
                m = float(np.mean(vals)); sd = float(np.std(vals))
                if sd < 1e-12:
                    continue
                low, high = m - 3 * sd, m + 3 * sd
            else:
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            for idx, v in sels:
                if v < low or v > high:
                    reason = "单点离群，建议核对该样品数据与来源"
                    if sample_col:
                        sv = cleaned[idx].get(sample_col)
                        if sv is not None:
                            reason += f"（样品 {sv}）"
                    hits[(b, pc, idx)] = OutlierHit(
                        batch=b,
                        sample=str(cleaned[idx].get(sample_col, "")) if sample_col else None,
                        param=pc, value=v, reason=reason)
    return hits


def _batch_stats(cleaned: List[Dict[str, Any]], p: Params,
                 batch: str, exclude_idx: set = None) -> Optional[BatchStat]:
    """批量内统计：剔除单点离群样本后再算均值/极值差/CV，避免统计失真。"""
    exclude_idx = exclude_idx or set()
    rows = [r for i, r in enumerate(cleaned)
            if str(r[p.batch_col]) == batch and i not in exclude_idx]
    if not rows:
        return None
    n = len(rows)
    vals0 = [float(r[p.param_cols[0]]) for r in rows]
    m = mean(vals0)
    ptp = max(vals0) - min(vals0) if n >= 2 else None
    cv = pstdev(vals0) / abs(m) * 100.0 if (n >= 2 and abs(m) > 1e-12) else None
    return BatchStat(batch=batch, n=n, vals=vals0, mean=m, ptp=ptp, cv=cv, n_used=n)


def _detect_batch_offset(cleaned: List[Dict[str, Any]], p: Params,
                         batches: List[BatchStat]) -> List[OutlierHit]:
    """整批偏移（跨批 IQR）：对每批均值做跨批比较，均值明显偏离整体的批次提示整批偏移。

    仅当批数 ≥4 且有足够样本时启用；否则不做整批偏移预警，避免误报。
    """
    hits = []
    if len(batches) < 4:
        return hits
    for pc in p.param_cols:
        # 用剔除后的批均值作为整批水平基准
        means = [bs.mean for bs in batches if bs.mean is not None and abs(bs.mean) > 1e-12]
        if len(means) < 4:
            continue
        arr = np.array(means, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-12:
            m = float(np.mean(arr)); sd = float(np.std(arr))
            if sd < 1e-12:
                continue
            low, high = m - 3 * sd, m + 3 * sd
        else:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for bs in batches:
            if bs.mean is None or abs(bs.mean) < 1e-12:
                continue
            if bs.mean < low or bs.mean > high:
                hits.append(OutlierHit(
                    batch=bs.batch, sample="整批",
                    param=pc, value=bs.mean,
                    reason="整批均值明显偏离整体水平，提示该批制程区间可能变化，建议回查该批对应环节"))
    return hits


# ============ v1.2 SPC 控制图（X-bar / R）与能力指数（Cp/Ppk）============
# 标准常数表（子组容量 n=2..15；A2 用于 X-bar 图，D3/D4 用于 R 图，d2 用于 σ̂=Rbar/d2 估计）
_A2_TABLE = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419,
             8: 0.373, 9: 0.337, 10: 0.308, 11: 0.285, 12: 0.266,
             13: 0.249, 14: 0.235, 15: 0.223}
_D3_TABLE = {2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.076,
             8: 0.136, 9: 0.184, 10: 0.223, 11: 0.256, 12: 0.284,
             13: 0.308, 14: 0.329, 15: 0.348}
_D4_TABLE = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924,
             8: 1.864, 9: 1.816, 10: 1.777, 11: 1.744, 12: 1.716,
             13: 1.692, 14: 1.671, 15: 1.652}
_D2_TABLE = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704,
             8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258,
             13: 3.336, 14: 3.407, 15: 3.472}


def _mode_int(vals: List[int]) -> int:
    """众数；并列时取最小（趋保守，避免过大子组导致查表越界）。"""
    if not vals:
        return 2
    from collections import Counter
    cnt = Counter(vals)
    mx = max(cnt.values())
    return min(k for k, c in cnt.items() if c == mx)


def _subgroup_constants(sg: int):
    """按子组容量取 A2/D3/D4/d2。超大 n（>15）以最接近且 ≤sg 的表内值近似；
    无法查表（sg<2）返回全 None，调用方据此跳过（绝不抛错）。"""
    if sg in _A2_TABLE:
        return (_A2_TABLE[sg], _D3_TABLE[sg], _D4_TABLE[sg], _D2_TABLE.get(sg))
    if sg >= 2:
        key = max(k for k in _A2_TABLE if k <= sg)
        return (_A2_TABLE[key], _D3_TABLE[key], _D4_TABLE[key], _D2_TABLE.get(key))
    return (None, None, None, None)


def _build_control_charts(cleaned: List[Dict[str, Any]],
                          p: Params):
    """以「批次=子组」计算每个参数的 X-bar / R 控制图 + 能力指数。
    返回 (control_charts: List[ControlChart], capability: Dict[str, Dict])。

    - X-bar 图：center = 各批均值均值(grand mean)；R̄ = 各批极差均值；
      UCL/LCL = center ± A2·R̄（A2 按子组容量查表）。
    - R 图：center = R̄；UCL = D4·R̄；LCL = D3·R̄（n≤6 时 D3=0 → LCL=0）。
    - 判异（西格玛规则）：
        ① xbar 值超 UCL/LCL → 失控 + 下标进 outliers（渲染红色高亮）；
        ② xbar 连续 7 点落在中心线同侧 → 偏移趋势信号；
        ③ R 图极差超 UCL → 变异增大信号。
    - 数据不足（有效批 <3、子组 n 全为1、无法查表）时静默跳过该参数，绝不抛错。
    """
    control_charts: List[ControlChart] = []
    capability: Dict[str, Dict[str, Any]] = {}

    # 每参数每批样本值（批次组内在 bounds 内、清洗后的样本）
    batch_order = sorted({str(r[p.batch_col]) for r in cleaned})
    per_param: Dict[str, Dict[str, List[float]]] = {}
    for r in cleaned:
        b = str(r[p.batch_col])
        for pc in p.param_cols:
            try:
                v = float(r[pc])
            except (TypeError, ValueError):
                continue
            per_param.setdefault(pc, {}).setdefault(b, []).append(v)

    for pc in p.param_cols:
        byb = per_param.get(pc, {})
        groups = [(b, byb[b]) for b in batch_order if b in byb and byb[b]]
        if len(groups) < 3:                      # 有效子组过少，跳过
            continue
        # 各子组均值 / 极差（仅 n>=2 的子组可算极差）
        x_means = [sum(vs) / len(vs) for _, vs in groups]
        ranges = {b: (max(vs) - min(vs)) for b, vs in groups if len(vs) >= 2}
        if not ranges:                           # 全为单样本子组，无 R̄，跳过
            continue
        r_bar = sum(ranges.values()) / len(ranges)
        sg = _mode_int([len(vs) for _, vs in groups])
        a2, d3, d4, d2 = _subgroup_constants(sg)
        if a2 is None:                           # sg<2 查不到，跳过
            continue

        # ---- X-bar 图 ----
        center_x = sum(x_means) / len(x_means)
        ucl_x = center_x + a2 * r_bar
        lcl_x = center_x - a2 * r_bar
        signals_x: List[str] = []
        outliers_x: List[int] = []
        for i, bm in enumerate(groups):
            v = x_means[i]
            if v > ucl_x:
                signals_x.append(f"批次 {bm[0]} 均值越上控制限(UCL={ucl_x:.4g})")
                outliers_x.append(i)
            elif v < lcl_x:
                signals_x.append(f"批次 {bm[0]} 均值越下控制限(LCL={lcl_x:.4g})")
                outliers_x.append(i)
        # 连续 7 点同侧
        if len(x_means) >= 7:
            n_up = n_dn = 0
            for m in x_means:
                if m > center_x:
                    n_up += 1; n_dn = 0
                elif m < center_x:
                    n_dn += 1; n_up = 0
                else:
                    n_up = n_dn = 0
                if n_up >= 7 or n_dn >= 7:
                    signals_x.append("连续7点落在中心线同一侧(偏移趋势)")
                    break
        xbar = ControlChart(param=pc, kind="xbar",
                            batches=[b for b, _ in groups], values=x_means,
                            center=center_x, ucl=ucl_x, lcl=lcl_x,
                            subgroup_size=sg, n_batches=len(x_means),
                            signals=signals_x, outliers=outliers_x)
        control_charts.append(xbar)

        # ---- R 图 ----
        r_batches = [b for b in ranges]
        r_vals = [ranges[b] for b in r_batches]
        if len(r_vals) >= 3:
            ucl_r = d4 * r_bar
            lcl_r = d3 * r_bar
            signals_r: List[str] = []
            outliers_r: List[int] = []
            for i, b in enumerate(r_batches):
                if r_vals[i] > ucl_r:
                    signals_r.append(f"批次 {b} 极差越上控制限(UCL={ucl_r:.4g}) 变异增大")
                    outliers_r.append(i)
            control_charts.append(ControlChart(
                param=pc, kind="r", batches=r_batches, values=r_vals,
                center=r_bar, ucl=ucl_r, lcl=lcl_r, subgroup_size=sg,
                n_batches=len(r_vals), signals=signals_r, outliers=outliers_r))

        # ---- 能力指数（σ̂=Rbar/d2）----
        capability[pc] = _capability_for_param(pc, groups, sg, r_bar, d2, p)

    return control_charts, capability


def _capability_for_param(pc: str, groups, sg: int, r_bar: float,
                          d2, p: Params) -> Dict[str, Any]:
    """单参数能力指数。σ̂ = R̄/d2（子组内变异）。Cp/Ppk/Cpu/Cpl 仅在提供规格限时计算。"""
    cap: Dict[str, Any] = {
        "param": pc, "n_batches": len(groups),
        "subgroup_size": sg, "n_samples": 0,
        "mean": None, "sigma_hat": None, "sigma_overall": None,
        "usl": None, "lsl": None,
        "cp": None, "cpu": None, "cpl": None, "cpk": None, "ppk": None,
    }
    all_v = [v for _, vs in groups for v in vs]
    cap["n_samples"] = len(all_v)
    if not all_v:
        return cap
    mean_v = sum(all_v) / len(all_v)
    cap["mean"] = mean_v
    if len(all_v) >= 2:
        cap["sigma_overall"] = float(np.std(all_v, ddof=1))
    if d2 and d2 > 0:
        cap["sigma_hat"] = r_bar / d2
    usl = (p.spec_usl or {}).get(pc)
    lsl = (p.spec_lsl or {}).get(pc)
    cap["usl"], cap["lsl"] = usl, lsl
    sh = cap["sigma_hat"]
    if sh and sh > 0:
        if usl is not None and lsl is not None:
            cap["cp"] = (usl - lsl) / (6 * sh)
            cap["cpu"] = (usl - mean_v) / (3 * sh)
            cap["cpl"] = (mean_v - lsl) / (3 * sh)
            cap["cpk"] = min(cap["cpu"], cap["cpl"])
        elif usl is not None:
            cap["cpu"] = (usl - mean_v) / (3 * sh)
            cap["cpk"] = cap["cpu"]
        elif lsl is not None:
            cap["cpl"] = (mean_v - lsl) / (3 * sh)
            cap["cpk"] = cap["cpl"]
    so = cap["sigma_overall"]
    if so and so > 0 and usl is not None and lsl is not None:
        cap["ppk"] = min((usl - mean_v) / (3 * so), (mean_v - lsl) / (3 * so))
    return cap


def analyze(table: List[Dict[str, Any]], p: Params) -> AnalysisResult:
    cleaned = _clean_rows(table, p)
    res = AnalysisResult()
    if not cleaned:
        return res

    sample_col = _resolve_sample_col(p, cleaned)

    # 1) 单点离群（批内判定）→ 得到需剔除的样本下标
    single_hits = _detect_single_outliers(cleaned, p, sample_col)
    exclude = {k[2] for k in single_hits}

    # 2) 批量内稳定性（剔除单点离群后统计）
    batches_all = sorted({str(r[p.batch_col]) for r in cleaned})
    for batch in batches_all:
        bs = _batch_stats(cleaned, p, batch, exclude)
        if bs:
            res.batches.append(bs)

    # 3) 整批偏移（跨批 IQR，基于剔除后的批均值）
    batch_offset_hits = _detect_batch_offset(cleaned, p, res.batches)

    # 离群表 = 单点离群 + 整批偏移
    seen = set()
    for o in list(single_hits.values()) + batch_offset_hits:
        key = (o.batch, o.sample, o.param)
        if key in seen:
            continue
        seen.add(key)
        res.outliers.append(o)

    # 4) 跨批次趋势：每个参数每批均值（剔除单点离群后的均值）
    for pc in p.param_cols:
        per_batch = []
        for bs in res.batches:
            per_batch.append(bs.mean)
        res.trend[pc] = per_batch

    # 4b) v1.2 控制图（X-bar/R）+ 能力指数（在跨批趋势之后计算）
    res.control_charts, res.capability = _build_control_charts(cleaned, p)

    # 5) 参数相关性（皮尔逊）
    if len(p.param_cols) >= 2:
        for i in range(len(p.param_cols)):
            for j in range(i + 1, len(p.param_cols)):
                a, b = p.param_cols[i], p.param_cols[j]
                x, y = [], []
                for r in cleaned:
                    if a in r and b in r:
                        x.append(float(r[a])); y.append(float(r[b]))
                if len(x) >= 3:
                    res.correlation[f"{a}~{b}"] = float(np.corrcoef(x, y)[0, 1])

    # 6) 用户自定义分段分析（可选）：按分段标记列分组，分段统计各参数均值/极值差/CV
    if p.segment_col:
        seg_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in cleaned:
            sg = r.get(p.segment_col)
            if sg is None:
                continue
            seg_map.setdefault(str(sg), []).append(r)
        for sg in sorted(seg_map, key=str):
            rows = seg_map[sg]
            entry: Dict[str, Any] = {"segment": sg, "n": len(rows)}
            for pc in p.param_cols:
                vals = [float(r[pc]) for r in rows if r.get(pc) is not None]
                if vals:
                    m = sum(vals) / len(vals)
                    sd = pstdev(vals) if len(vals) >= 2 else 0.0
                    entry[f"{pc}_mean"] = m
                    entry[f"{pc}_ptp"] = max(vals) - min(vals)
                    entry[f"{pc}_cv"] = (sd / abs(m) * 100.0) if abs(m) > 1e-12 else None
            res.segment_stats.append(entry)

    return res


# ---------------------------------------------------------------- CLI (sample 驱动)

def _sample_table():
    """构造一份脱敏演示数据： batch / sample / paramA / paramB。仅用于自测，不含真实数据。"""
    rows = []
    batches = [f"lot{i:02d}" for i in range(1, 11)]
    for b in batches:
        base_a = 50.0 + (int(b[3:]) % 5) * 2
        for s in range(1, 6):
            rows.append({
                "batch": b,
                "sample": f"{b}-s{s}",
                "paramA": base_a + (s % 3) * 0.8,
                "paramB": 30.0 + (int(b[3:]) % 4) + s * 0.5,
            })
    # 注入一个单点离群用于自测
    rows[12]["paramA"] = 95.0
    return rows


def _load_table(path: str) -> List[Dict[str, Any]]:
    """从 CSV / XLSX 读取数据为 dict 行列表（列名即键）。"""
    low = path.lower()
    if low.endswith(".csv"):
        import csv
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]
    elif low.endswith(".xlsx") or low.endswith(".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        ws = wb[wb.sheetnames[0]]
        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if not header:
            return []
        rows = []
        for r in rows_iter:
            rows.append({str(h): v for h, v in zip(header, r)})
        return rows
    raise ValueError("仅支持 .csv / .xlsx 数据文件")


def _parse_spec_map(arg: Optional[str]):
    """解析 '参数:上限,参数:上限' 形式的规格限为 dict（参数->浮点）。解析失败返回 None。"""
    if not arg:
        return None
    out: Dict[str, float] = {}
    for part in arg.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out or None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if "--sample" in argv:
        p = Params(batch_col="batch", param_cols=["paramA", "paramB"],
                   spec_usl={"paramA": 60.0, "paramB": 40.0},
                   spec_lsl={"paramA": 40.0, "paramB": 20.0})
        res = analyze(_sample_table(), p)
        print("=== 批次稳定性 ===")
        for bs in res.batches:
            print(f"{bs.batch}: n={bs.n} mean={bs.mean:.2f} "
                  f"ptp={bs.ptp:.2f} cv={bs.cv:.2f}%")
        print("\n=== 离群预警 ===")
        for o in res.outliers:
            print(f"[{o.batch}] {o.param}={o.value:.2f} — {o.reason}")
        print("\n=== 参数相关性 ===")
        for k, v in res.correlation.items():
            print(f"{k}: r={v:.3f}")
        print("\n=== SPC 控制图 (X-bar/R) ===")
        if not res.control_charts:
            print("（无控制图：有效批不足）")
        for cc in res.control_charts:
            kind = "X-bar" if cc.kind == "xbar" else "R   "
            print(f"[{cc.param} {kind}] CL={cc.center:.3f} UCL={cc.ucl:.3f} "
                  f"LCL={cc.lcl:.3f} 判异点={cc.outliers} 信号={cc.signals}")
        print("\n=== 能力指数 (Cp/Ppk) ===")
        for pc, cap in res.capability.items():
            print(f"[{pc}] σ̂={cap.get('sigma_hat') and round(cap['sigma_hat'],4)} "
                  f"Cp={cap.get('cp') and round(cap['cp'],3)} "
                  f"Cpk={cap.get('cpk') and round(cap['cpk'],3)} "
                  f"Cpu={cap.get('cpu') and round(cap['cpu'],3)} "
                  f"Cpl={cap.get('cpl') and round(cap['cpl'],3)} "
                  f"Ppk={cap.get('ppk') and round(cap['ppk'],3)}")
        return 0

    # 真实数据入参：--data 文件 --batch 批次列 --params a,b [--segment 段列] [--sample-col 样品列]
    if "--data" in argv:
        def _get(name):
            try:
                i = argv.index(name)
            except ValueError:
                return None
            return argv[i + 1] if i + 1 < len(argv) else None
        path = _get("--data")
        batch_col = _get("--batch") or "batch"
        params = (_get("--params") or "").split(",")
        params = [x.strip() for x in params if x.strip()]
        seg = _get("--segment")
        sample_col = _get("--sample-col")
        spec_usl = _parse_spec_map(_get("--spec-usl"))
        spec_lsl = _parse_spec_map(_get("--spec-lsl"))
        if not path or not params:
            print("用法：process_spc.py --data <csv|xlsx> --batch <批次列> --params <a,b> "
                  "[--segment <分段列>] [--sample-col <样品列>] "
                  "[--spec-usl <参数:上限,...>] [--spec-lsl <参数:下限,...>]")
            return 2
        table = _load_table(path)
        p = Params(batch_col=batch_col, param_cols=params,
                   segment_col=seg, sample_col=sample_col,
                   spec_usl=spec_usl, spec_lsl=spec_lsl)
        res = analyze(table, p)
        print(f"=== 共 {len(res.batches)} 个批次，离群预警 {len(res.outliers)} 处 ===")
        for o in res.outliers:
            print(f"[{o.batch}] {o.param}={o.value:.2f} — {o.reason}")
        print(f"=== 控制图 {len(res.control_charts)} 张，能力参数 {len(res.capability)} 个 ===")
        return 0

    print("用法：process_spc.py --sample | --data <csv|xlsx> --batch <批次列> --params <a,b>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())