#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qsspc_lifetime.py — QSSPC 有效载流子寿命 (τ_eff) 正向物理仿真模块
=================================================================

纯 Python / numpy / scipy 实现，无外部商用依赖。

目标：给定晶圆参数 + 表面复合参数，输出注入依赖寿命曲线 τ_eff(Δn)，
即 Sinton QSSPC 测法下的正向仿真，并预留与实测数据标定反推膜层参数的接口。

复现的物理过程（准稳态，QSSPC 测量条件 G = .. , τ_eff = Δn / G）：

    G = R_total = R_SRH + R_rad + R_Auger + R_surface
    1/τ_eff = 1/τ_bulk + 1/τ_surface
    τ_bulk  = Δn / (R_SRH + R_rad + R_Auger)          # 体复合
    τ_surface = W / (2 · S_eff)                        # 双面钝化近似

· SRH    : 单能级陷阱 (Shockley-Read-Hall)，注入依赖的关键来源
· 辐射   : band-to-band，B 系数
· Auger  : 三粒子 (nnp) + (npp)，Kerr & Cuevas 2002 参数
           可切换 Richter et al. 2012 高注入增强参数化
· 表面   : 均匀注入近似 + S_eff(Δn) 可注入依赖修正

数值方法：采用「均匀注入近似 + 表面复合修正」的半解析模型 —— 这是 QSSPC
数据常规处理方式。不做完整一维泊松-漂移扩散求解（见 README 中近似假设）。

标定接口：`calibrate()` 输入实测 (Δn, τ_eff) 点，最小二乘反推表面/陷阱参数。

用法：
    python qsspc_lifetime.py --plot            # 默认曲线 + 保存 PNG
    python qsspc_lifetime.py --calibrate       # 演示标定反推
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 物理常数 (SI / cm-g-s 混合，载流子寿命模块惯用 cm 单位)
# --------------------------------------------------------------------------
kB      = 1.380649e-23          # J/K
q       = 1.602176634e-19       # C
h       = 6.62607015e-34        # J·s
m0      = 9.10938370e-31        # kg
VTH_DEF = 1.0e7                 # cm/s, 载流子热速度近似 (~1e7 at 300K)

# --------------------------------------------------------------------------
# 数据类：参数容器
# --------------------------------------------------------------------------
class SiParams:
    """晶圆 + 复合参数容器。缺省为典型 n 型 Cz-Si 占位，可接实测数据标定。"""
    __slots__ = (
        'Nd', 'W', 'T', 'passivation',
        'Et', 'sigma_n', 'sigma_p', 'Nt',
        'Seff', 'Seff_inj_dep', 'auger_model',
    )

    def __init__(self, **kwargs):
        # --- 晶圆 ---
        self.Nd          = kwargs.get('Nd', 4.5e15)        # cm^-3, n 型掺杂 (约 1 Ω·cm)
        self.W           = kwargs.get('W', 165e-4)         # cm, 厚度 150-180 μm
        self.T           = kwargs.get('T', 300.0)          # K
        self.passivation = kwargs.get('passivation', 'double')  # 'double' | 'single'

        # --- SRH 陷阱 ---
        self.Et       = kwargs.get('Et', 0.0)              # eV, 陷阱能级相对本征能级 Ei (>0 偏导带)
        self.sigma_n  = kwargs.get('sigma_n', 1.0e-15)     # cm^2, 电子俘获截面
        self.sigma_p  = kwargs.get('sigma_p', 1.0e-15)     # cm^2, 空穴俘获截面
        self.Nt       = kwargs.get('Nt', 1.0e10)           # cm^-3, 陷阱密度

        # --- 表面 ---
        self.Seff          = kwargs.get('Seff', 1.0)       # cm/s, 有效表面复合速度
        self.Seff_inj_dep  = kwargs.get('Seff_inj_dep', False)  # 表面是否随注入增强

        # --- 体 Auger 模型 ---
        self.auger_model   = kwargs.get('auger_model', 'kc2002')  # 'kc2002' | 'richter2012'


# --------------------------------------------------------------------------
# 基础量
# --------------------------------------------------------------------------
def intrinsic_carrier_density(T=300.0):
    """Green 1990 参数化本征载流子浓度，cm^-3。"""
    # n_i = 5.29e19 (T/300)^2.54 exp(-6726/T)，300K 时 ~9.68e9
    return 5.29e19 * (T / 300.0) ** 2.54 * np.exp(-6726.0 / T)


def thermal_velocity(T=300.0, m_eff=0.26*m0):
    return np.sqrt(3.0 * kB * T / m_eff) * 100.0  # cm/s


# --------------------------------------------------------------------------
# 各复合机制 (均在准稳态，返回复合率 R [cm^-3 s^-1])
# --------------------------------------------------------------------------
def srh_recombination(dn, p, T=300.0, Et=0.0, sigma_n=1e-15, sigma_p=1e-15,
                      Nt=1e11, ni=None, n0=None, p0=None):
    """SRH 单能级陷阱复合率。注入依赖寿命的关键来源。"""
    ni = intrinsic_carrier_density(T) if ni is None else ni
    if n0 is None or p0 is None:
        n0 = n0 or 0.0
        p0 = p0 if p0 is not None else ni**2 / max(n0, 1.0)
    n = n0 + dn
    p_val = p0 + dn
    vth = VTH_DEF
    tau_n0 = 1.0 / (sigma_n * vth * Nt)   # 少子空穴学: 实际 tau_p0 对应空穴
    tau_p0 = 1.0 / (sigma_p * vth * Nt)
    # 陷阱相对本征能级 -> 特征密度
    Efac = np.exp(q * Et / (kB * T))
    n1 = ni * Efac
    p1 = ni / Efac
    np_minus_ni2 = n * p_val - ni**2
    denom = tau_p0 * (n + n1) + tau_n0 * (p_val + p1)
    return np_minus_ni2 / denom


def radiation_recombination(dn, n0, p0, T=300.0, B=4.73e-15, ni=None):
    """band-to-band 辐射复合率。B 系数 (300K ~4.73e-15 cm^3/s)。"""
    ni = intrinsic_carrier_density(T) if ni is None else ni
    n = n0 + dn
    p = p0 + dn
    return B * (n * p - ni**2)


def auger_recombination(dn, n0, p0, T=300.0, ni=None, model='kc2002'):
    """三粒子 Auger 复合率。

    kc2002   : Kerr & Cuevas 2002, C_n=1.83e-31 (nnp), C_p=9.9e-32 (npp)  [cm^6/s]
    richter2012: 高注入带增强的近似参数化（默认同 KC，高注入略增强）.
    说明：标定阶段 kc2002 已足够复现 Sinton 高注入下降形状；
          richter2012 为更高精度选项，可后续用公开参数校准。
    """
    ni = intrinsic_carrier_density(T) if ni is None else ni
    n = n0 + dn
    p = p0 + dn
    C_n = 1.83e-31   # nnp  (2 electron, 1 hole)
    C_p = 9.9e-32    # npp  (1 electron, 2 hole)

    if model == 'richter2012':
        # 严格 Richter 参数化需含能带重正 (BGN)；此处提供经验增强系数占位（见 README）
        factor = 1.0 + 0.9 * np.tanh(dn / 1.0e16)     # 高注入最大 +90%
        C_n = C_n * factor
    R = C_n * n**2 * p + C_p * n * p**2
    return R


# --------------------------------------------------------------------------
# 体复合寿命
# --------------------------------------------------------------------------
def bulk_lifetime(dn, p: SiParams):
    """体复合寿命 τ_bulk = Δn / (R_SRH + R_rad + R_Auger)。"""
    dn = np.atleast_1d(np.asarray(dn, dtype=float))
    T = p.T
    ni = intrinsic_carrier_density(T)
    n0 = p.Nd
    p0 = ni**2 / n0

    R_srh = srh_recombination(dn, p, T=T, Et=p.Et, sigma_n=p.sigma_n,
                              sigma_p=p.sigma_p, Nt=p.Nt, ni=ni, n0=n0, p0=p0)
    R_rad = radiation_recombination(dn, n0, p0, T=T, ni=ni)
    R_aug = auger_recombination(dn, n0, p0, T=T, ni=ni, model=p.auger_model)
    R_tot = R_srh + R_rad + R_aug
    # 准稳态: 1/τ_bulk = R_tot / Δn (但严格应为 Δp=Δn, 用 Δn)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_bulk = np.where(dn > 0, dn / np.where(R_tot > 0, R_tot, np.nan), np.nan)
    return tau_bulk


# --------------------------------------------------------------------------
# 表面复合寿命
# --------------------------------------------------------------------------
def surface_lifetime(dn, p: SiParams):
    """表面复合寿命。均匀注入近似：双面钝化 1/τ_surf = 2·S_eff/W。

    可选 S_eff(Δn) 注入依赖修正（默认关闭——高注入主导由体 Auger 负责）。
    """
    dn = np.atleast_1d(np.asarray(dn, dtype=float))
    W = p.W
    s_eff = p.Seff
    if p.Seff_inj_dep:
        # 高注入扩散系数增强导致的有效 S 增大 (经验型)
        s_eff = s_eff * (1.0 + 5.0 * np.tanh(dn / 3.0e16))
    factor = 2.0 if p.passivation == 'double' else 1.0
    tau_surf = W / (factor * s_eff)
    return np.full_like(dn, tau_surf, dtype=float) if np.ndim(tau_surf) == 0 \
        else np.asarray(tau_surf, dtype=float)


# --------------------------------------------------------------------------
# 有效寿命 (QSSPC 输出曲线)
# --------------------------------------------------------------------------
def effective_lifetime(dn, p: SiParams):
    """有效寿命：1/τ_eff = 1/τ_bulk + 1/τ_surface。"""
    dn = np.atleast_1d(np.asarray(dn, dtype=float))
    tau_bulk = bulk_lifetime(dn, p)
    tau_surf = surface_lifetime(dn, p)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_eff = 1.0 / (1.0 / tau_bulk + 1.0 / tau_surf)
    return tau_eff, tau_bulk, tau_surf


def generation_rate(dn, p: SiParams):
    """QSSPC 正向模型：准稳态产生率 G = Δn / τ_eff  [cm^-3 s^-1]。"""
    tau_eff, _, _ = effective_lifetime(dn, p)
    return np.asarray(dn) / tau_eff


# --------------------------------------------------------------------------
# 标定接口：反推膜层参数
# --------------------------------------------------------------------------
def calibrate(measured_dn, measured_tau, p: SiParams,
              fit=('Seff',), x0=None, **kwargs):
    """用实测 (Δn, τ_eff) 点最小二乘反推膜层参数。

    输入
    ----
    measured_dn  : array, 实测过剩载流子密度 Δn [cm^-3]
    measured_tau : array, 实测有效寿命 τ_eff [s]
    p            : SiParams 基准参数（被拟合量之外的固定）
    fit          : 可拟合参数名（默认只反推 Seff）
                   可选组合: 'Seff', 'Nt', 'Et', 'sigma_n', 'sigma_p'
    x0           : 初值（默认用 p 中当前值/合理对数初值）

    返回
    ----
    (最优 fit_params dict, cov 近似, 残差列表)
    对 log 型参数 (Seff/Nt/sigma) 在 log10 空间拟合以保证正值。
    """
    from scipy.optimize import least_squares

    measured_dn = np.asarray(measured_dn, dtype=float)
    measured_tau = np.asarray(measured_tau, dtype=float)

    log_params = {'Seff', 'Nt', 'sigma_n', 'sigma_p'}   # 需 log10 空间的参数

    # 初值
    if x0 is None:
        x0 = []
        for k in fit:
            v = getattr(p, k)
            x0.append(np.log10(v) if k in log_params else v)
    x0 = np.asarray(x0, float)

    def pack(x):
        pp = SiParams(**{f'{k}': getattr(p, k) for k in SiParams.__slots__})
        for i, k in enumerate(fit):
            v = 10.0**x[i] if k in log_params else x[i]
            setattr(pp, k, v)
        return pp

    def resid(x):
        pp = pack(x)
        tau_eff, _, _ = effective_lifetime(measured_dn, pp)
        return np.log10(tau_eff) - np.log10(measured_tau)  # 对数空间残差更稳

    res = least_squares(resid, x0, method='lm', max_nfev=10000)
    opt = {}
    for i, k in enumerate(fit):
        v = 10.0**res.x[i] if k in log_params else res.x[i]
        opt[k] = float(v)
    pred = effective_lifetime(measured_dn, SiParams(**{
        **{f'{k}': getattr(p, k) for k in SiParams.__slots__}, **opt}))[0]
    r2 = _r2(np.log10(measured_tau), np.log10(pred))
    return opt, res.cost, list(res.fun), r2


def _r2(y, yhat):
    """对数空间决定系数 R²。"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def load_qsspc_data(csv_path):
    """读取实测 QSSPC CSV：期望列名含 dn / tau_eff（含 τ_eff、lifetime 变体），
    或按前两列顺序推断。返回 (dn array, tau_eff array [s])。"""
    import csv
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError('CSV 为空')
        for row in reader:
            if not row or not row[0].strip():
                continue
            rows.append([c.strip() for c in row[:2]])
    if len(rows) < 2:
        raise ValueError(
            f'有效数据点不足（需 ≥2 个 (Δn, τ_eff) 对，实际 {len(rows)} 个）')

    # 列识别
    h = [h.lower().replace('_', '').replace(' ', '') for h in header] if header else []
    dn_col = None
    tau_col = None
    for i, name in enumerate(h):
        if name in ('dn', 'dncm3', 'excesscarrierdensity', 'excessdn', 'delta n', 'deltan'):
            dn_col = i
        if name in ('taueff', 'tau', 'lifetime', 'taueffs', 'lifetimes', 'lifetimeus'):
            tau_col = i
    if dn_col is None or tau_col is None:
        dn_col, tau_col = 0, 1  # 按前两列兜底

    dn = np.array([float(r[dn_col]) for r in rows])
    tau = np.array([float(r[tau_col]) for r in rows])
    # 单位兜底：若 τ 列名含 us/µs 则 ×1e-6
    if tau_col < len(h) and any(k in h[tau_col] for k in ('us', 'microsecond', 'mus')):
        tau = tau * 1e-6
    _validate_qsspc(dn, tau)
    return dn, tau


def _validate_qsspc(dn, tau):
    """边界校验：正注入、正寿命、点数足够。返回告警列表（不抛错降级）。"""
    dn = np.asarray(dn, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n = len(dn)
    warnings = []
    if n < 2:
        raise ValueError('数据点不足 2 个，无法标定')
    if np.any(dn <= 0):
        raise ValueError('存在非正 Δn（须 >0），请检查数据')
    if np.any(tau <= 0):
        raise ValueError('存在非正 τ_eff（须 >0），请检查数据')
    if np.any(np.diff(dn) <= 0):
        raise ValueError('Δn 需单调递增排列')
    if n < 5:
        warnings.append(f'数据点偏少({n}个)，反推参数自由度受限，结果仅供参考')
    return warnings


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _default_dn_grid():
    return np.logspace(13, 17, 129)  # 1e13 - 1e17 cm^-3


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='QSSPC 载流子寿命分析：正向仿真 τ_eff(Δn) + 实测标定反推',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  正向仿真:  python qsspc_lifetime.py --Seff 1.0 --Nt 1e10 --plot\n'
            '  实测标定:  python qsspc_lifetime.py --data run_data.csv --fit Seff,Nt --report\n'
            '  演示标定:  python qsspc_lifetime.py --calibrate\n'
        ))
    ap.add_argument('--dn-min', type=float, default=1e13)
    ap.add_argument('--dn-max', type=float, default=1e17)
    ap.add_argument('--npts', type=int, default=129)
    # 晶圆/复合参数
    ap.add_argument('--Seff', type=float, default=1.0)
    ap.add_argument('--Nt', type=float, default=1e10)
    ap.add_argument('--Et', type=float, default=0.0)
    ap.add_argument('--Nd', type=float, default=4.5e15)
    ap.add_argument('--W', type=float, default=165e-4)
    ap.add_argument('--sigma-n', type=float, default=1e-15)
    ap.add_argument('--sigma-p', type=float, default=1e-15)
    ap.add_argument('--model', choices=['kc2002', 'richter2012'], default='kc2002')
    # 模式
    ap.add_argument('--plot', action='store_true', help='保存正向 τ_eff(Δn) 曲线 PNG')
    ap.add_argument('--data', type=str, help='实测 QSSPC CSV（列 dn, tau_eff），触发标定反推')
    ap.add_argument('--fit', type=str, default='Seff',
                    help='标定反推参数，逗号分隔，可选 Seff/Nt/Et/sigma_n/sigma_p，默认 Seff')
    ap.add_argument('--calibrate', action='store_true', help='演示标定反推 Seff')
    ap.add_argument('--report', type=str, nargs='?', const='qsspc_report',
                    help='生成 Word 报告（含拟合结果），可指定基名（默认 qsspc_report）')
    ap.add_argument('--out', type=str, default=None, help='输出目录（默认当前目录）')
    args = ap.parse_args()

    p = SiParams(Nd=args.Nd, W=args.W, Seff=args.Seff, Nt=args.Nt, Et=args.Et,
                 sigma_n=args.sigma_n, sigma_p=args.sigma_p, auger_model=args.model)

    # ---------- 模式 B：实测标定反推 ----------
    if args.data:
        fit = tuple(x.strip() for x in args.fit.split(',') if x.strip())
        for k in fit:
            if k not in ('Seff', 'Nt', 'Et', 'sigma_n', 'sigma_p'):
                raise SystemExit(f'[error] 非法 fit 参数: {k}（可选 Seff/Nt/Et/sigma_n/sigma_p）')
        try:
            dn_m, tau_m = load_qsspc_data(args.data)
            warn = _validate_qsspc(dn_m, tau_m)
            for w in warn:
                print(f'[warn] {w}')
        except ValueError as e:
            print(f'[error] 数据校验失败: {e}')
            raise SystemExit(1)

        if len(dn_m) < max(2, len(fit)):
            print(f'[error] 数据点({len(dn_m)})需不少于待拟合参数数({len(fit)})')
            raise SystemExit(1)

        # 用正向最优点推断初值：以任意一个实测点附近参数为初值（log 空间合理初值）
        from scipy.optimize import least_squares

        log_params = {'Seff', 'Nt', 'sigma_n', 'sigma_p'}

        def _pack(x):
            pp = SiParams(**{f'{k}': getattr(p, k) for k in SiParams.__slots__})
            for i, k in enumerate(fit):
                setattr(pp, k, 10.0**x[i] if k in log_params else x[i])
            return pp

        def _resid(x):
            pp = _pack(x)
            te, _, _ = effective_lifetime(dn_m, pp)
            return np.log10(te) - np.log10(tau_m)

        x0 = []
        for k in fit:
            v = getattr(p, k)
            x0.append(np.log10(max(v, 1e-30)) if k in log_params else v)
        x0 = np.asarray(x0, float)
        res = least_squares(_resid, x0, method='lm', max_nfev=20000)

        opt = {}
        for i, k in enumerate(fit):
            opt[k] = float(10.0**res.x[i] if k in log_params else res.x[i])
        pred, _, _ = effective_lifetime(dn_m, p)
        pred_opt, _, _ = effective_lifetime(dn_m, _pack(res.x))
        r2 = _r2(np.log10(tau_m), np.log10(pred_opt))

        # 收敛质量检测：R² 过低说明固定参数假设与实际不符，明确告警而非硬出参数
        fit_ok = True
        notes = []
        if r2 < 0.5:
            fit_ok = False
            notes.append(
                f'拟合质量差(R²={r2:.2f})：当前固定参数(仅拟合 {",".join(fit)}) '
                '无法刻画数据曲线形状。可能原因：数据同时受未拟合参数(如 Nt/Et)影响，'
                '或模型假设与实际不符。建议把更多参数加入 --fit（如 --fit Seff,Nt 或 Seff,Nt,Et），'
                '或检查数据是否含噪声/测量误差。'
            )
        if not res.success:
            fit_ok = False
            notes.append(f'最小二乘未收敛：{res.message}')

        print('[标定反推结果]')
        print(f'  数据点   : {len(dn_m)} 个 (Δn: {dn_m.min():.2e}–{dn_m.max():.2e})')
        for k in fit:
            u = 'cm/s' if k == 'Seff' else ('cm^-3' if k in ('Nt',) else 'eV' if k == 'Et' else 'cm^2')
            print(f'  {k:<8}: {opt[k]:.4e} {u}')
        print(f'  R²(log)  : {r2:.4f}  ({"" if fit_ok else "⚠ 质量可疑"})')
        print(f'  残差cost : {res.cost:.3e}')
        for n in notes:
            print(f'  [提示] {n}')

        # ---- 报告 ----
        if args.report is not None:
            # 先画拟合图，再写 Word（Word 内嵌该图）
            _make_fit_plot(dn_m, tau_m, pred_opt, opt, fit, p,
                           args.report, args.out)
            _write_word_report(args.data, dn_m, tau_m, pred_opt, opt, r2, fit,
                               p, args.report, args.out, fit_ok=fit_ok)
        return

    # ---------- 模式 A：正向仿真 ----------
    dn = np.logspace(np.log10(args.dn_min), np.log10(args.dn_max), args.npts)
    tau_eff, tau_b, tau_s = effective_lifetime(dn, p)

    print(f"{'Δn [cm^-3]':>12} {'τ_bulk [μs]':>12} {'τ_surf [μs]':>12} {'τ_eff [μs]':>12}")
    for i in range(0, len(dn), max(1, len(dn)//12)):
        print(f"{dn[i]:12.3e} {tau_b[i]*1e6:12.1f} {tau_s[i]*1e6:12.1f} {tau_eff[i]*1e6:12.1f}")

    if args.plot:
        _make_plot(dn, tau_eff, tau_b, tau_s, p, args.out)

    if args.calibrate:
        _demo_calibrate(p, dn)

    return p, tau_eff, tau_b, tau_s, dn


def _make_fit_plot(dn_m, tau_m, pred_opt, opt, fit, p, base, outdir=None):
    """标定拟合曲线 vs 实测散点 PNG（越标定点越精准看形状吻合）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _try_cn_font(plt)
    except Exception:
        print('[warn] matplotlib 不可用，跳过拟合图')
        return
    import os
    dn_grid = np.logspace(np.log10(dn_m.min()*0.9), np.log10(dn_m.max()*1.1), 120)
    curve, _, _ = effective_lifetime(dn_grid, p)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dn_m, tau_m*1e6, 'o', color='#C0392B', label='measured')
    ax.plot(dn_grid, curve*1e6, '-', color='#2C3E50', lw=2, label='fit')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Excess carrier density $\Delta n$ [cm$^{-3}$]')
    ax.set_ylabel(r'$\tau_{eff}$ [$\mu$s]')
    fit_txt = ', '.join(f'{k}={opt[k]:.2g}' for k in fit)
    ax.set_title(f'QSSPC calibration fit  ({fit_txt})')
    ax.legend(loc='best'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(outdir, f'{base}_fit.png') if outdir else f'{base}_fit.png'
    fig.savefig(fname, dpi=150)
    print(f'[report] 拟合图已保存 {fname}')


def _write_word_report(data_src, dn_m, tau_m, pred_opt, opt, r2, fit, p,
                       base, outdir=None, fit_ok=True):
    """生成 Word 报告：数据概览 + 反推参数表 + R² + 结论。内嵌中文字体。"""
    import os
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    # 标题
    h = doc.add_heading('QSSPC 载流子寿命标定报告', level=1)
    doc.add_paragraph(f'数据源：{data_src} ｜ 拟合参数：{"、".join(fit)}')
    doc.add_paragraph(f'数据点：{len(dn_m)} 个（Δn {dn_m.min():.1e}–{dn_m.max():.1e} cm⁻³）')

    # 参数表
    doc.add_heading('反推膜层参数', level=2)
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Grid Accent 1'
    hdr = table.rows[0].cells
    for i, t in enumerate(['参数', '反推值', '单位']):
        hdr[i].text = t
    units = {'Seff': 'cm/s', 'Nt': 'cm⁻³', 'Et': 'eV', 'sigma_n': 'cm²', 'sigma_p': 'cm²'}
    for k in fit:
        row = table.add_row().cells
        row[0].text = k
        row[1].text = f'{opt[k]:.4e}'
        row[2].text = units.get(k, '')

    doc.add_paragraph('')
    p1 = doc.add_paragraph()
    r = p1.add_run(f'拟合优度 R²（对数空间）= {r2:.4f}')
    r.bold = True
    r.font.size = Pt(12)

    # 结论
    doc.add_heading('结论与建议', level=2)
    if not fit_ok or r2 < 0.5:
        verdict = (f'拟合质量偏低（R²={r2:.2f}），当前固定参数假设无法刻画曲线形状。'
                   f'建议扩大 --fit 参数范围（如同时拟合 Seff/Nt 或 Seff/Nt/Et），'
                   f'或核实实测数据质量后再标定，请勿直接采信本组反推数值。')
    elif r2 >= 0.95:
        verdict = f'拟合优度良好（R²={r2:.2f}），膜层参数可信；当前有效表面复合速度 Seff≈{opt.get("Seff", 0):.2g} cm/s（若拟合了Seff）。'
    elif r2 >= 0.8:
        verdict = f'拟合合理（R²={r2:.2f}），但曲线形状与实测有偏离，建议检查陷阱能级 Et 或界面态假设。'
    else:
        verdict = f'拟合质量偏低（R²={r2:.2f}），当前单能级 SRH 模型可能无法刻画该界面，建议补充更多注入点数据或引入双能级模型。'
    doc.add_paragraph(verdict)

    # 插入拟合图（若已生成）
    fname = os.path.join(outdir, f'{base}_fit.png') if outdir else f'{base}_fit.png'
    if os.path.exists(fname):
        doc.add_paragraph('')
        doc.add_picture(fname, width=docx_emu(5.5))

    out = os.path.join(outdir, f'{base}.docx') if outdir else f'{base}.docx'
    doc.save(out)
    print(f'[report] Word 报告已保存 {out}')


def docx_emu(inches):
    from docx.shared import Inches
    return Inches(inches)


def _try_cn_font(plt):
    """尝试设置中文字体（NotoSansSC），失败静默。"""
    try:
        import matplotlib
        import os
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'NotoSansSC-Regular.ttf')
        if os.path.exists(local):
            from matplotlib import font_manager
            font_manager.fontManager.addfont(local)
            prop = font_manager.FontProperties(fname=local).get_name()
            plt.rcParams['font.sans-serif'] = [prop]
            plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


def _make_plot(dn, tau_eff, tau_b, tau_s, p, outdir=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _try_cn_font(plt)
    except Exception:
        print('[plot] matplotlib 不可用，跳过绘图')
        return
    import os
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dn, tau_eff*1e6, '-', lw=2, color='#C0392B', label=r'$\tau_{eff}$')
    ax.plot(dn, tau_b*1e6, '--', color='#7f8c8d', label=r'$\tau_{bulk}$')
    ax.plot(dn, tau_s*1e6, '--', color='#2980B9', label=r'$\tau_{surface}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Excess carrier density $\Delta n$ [cm$^{-3}$]')
    ax.set_ylabel(r'Lifetime $\tau$ [$\mu$s]')
    ax.set_title('QSSPC effective lifetime (n-type Cz-Si, '
                 f'$S_{{eff}}$={p.Seff:.2f} cm/s, $N_t$={p.Nt:.0e} cm$^{{-3}}$)')
    # 标注高/低注入
    ax.axvspan(1e13, 1e15, color='green', alpha=0.06)
    ax.axvspan(3e15, 1e17, color='orange', alpha=0.06)
    ax.text(2e13, tau_eff.max()*1e6*1.6, 'low-inj\n(SRH/surf)', fontsize=8, va='top')
    ax.text(2e16, tau_eff.max()*1e6*0.25, 'high-inj\n(Auger)', fontsize=8)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, 'qsspc_lifetime_curve.png') if outdir \
        else 'qsspc_lifetime_curve.png'
    fig.savefig(out, dpi=150)
    print(f'[plot] 已保存 {out}')


def _demo_calibrate(p, dn):
    """演示：用"合成实测"+ 高寿命器件目标点，反推 Seff（演示用模拟数据，非真实产线数据）。"""
    # (a) 合成实测：由已知 Seff_true 生成，验证接口可回反
    Seff_true = 1.65
    p_true = SiParams(Nd=p.Nd, W=p.W, Seff=Seff_true, Nt=p.Nt)
    dn_m = np.logspace(14, 15.5, 6)
    tau_m, _, _ = effective_lifetime(dn_m, p_true)
    opt, cost, resid, r2 = calibrate(dn_m, tau_m, p, fit=('Seff',))
    print('\n[标定演示] 合成数据反推 Seff:')
    print(f'  真实 Seff  = {Seff_true:.3f} cm/s')
    print(f'  反推 Seff  = {opt["Seff"]:.3f} cm/s   (对数残差 cost={cost:.2e}, R²={r2:.3f})')

    # (b) 高寿命器件锚点（演示用模拟值）：τ≈5000μs @ Δn=1e15，反推所需 Seff
    target = 5000e-6
    dn_anch = np.array([1e15])
    tau_anch = np.array([target])
    opt2, _, _, _ = calibrate(dn_anch, tau_anch, p, fit=('Seff',))
    te, _, _ = effective_lifetime(dn_anch, SiParams(Nd=p.Nd, W=p.W, Seff=opt2['Seff'], Nt=p.Nt))
    print(f'\n[标定演示] 高寿命器件锚点 τ=5000μs @ Δn=1e15:')
    print(f'  反推 Seff ≈ {opt2["Seff"]:.2f} cm/s')
    print(f'  回代 τ_eff = {te[0]*1e6:.0f} μs')


if __name__ == '__main__':
    main()