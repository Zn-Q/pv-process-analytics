#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_run.py — 载流子寿命分析工具「一键演示」
==============================================
零数据开箱演示，复刻 SPC 工具 demo_run.py 的体验：一条命令跑通三个核心能力，
输出全部产物到 ./demo_output/ 目录。

演示内容：
  1) 正向仿真 τ_eff(Δn)：给定 Seff/Nt，输出注入依赖寿命曲线（终端表格 + PNG）
  2) 实测标定反推：用内置合成"实测"数据（Seff=1.65/Nt=1e11/Et=0.02 生成 + 2%噪声）
     反推膜层参数，验证接口可回收真值
  3) Word 报告：把标定结果 + 拟合图打包成可交付的 docx

用法：
  python3 demo_run.py              # 一条命令全跑
  python3 demo_run.py --out demo_output   # 指定输出目录
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(HERE, 'qsspc_lifetime.py')


def main():
    ap = argparse.ArgumentParser(description='载流子寿命分析工具 - 一键演示')
    ap.add_argument('--out', default='demo_output', help='输出目录（默认 demo_output）')
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    # 生成内置"实测"数据 CSV（高寿命器件锚点 + 合成曲线）
    demo_csv = os.path.join(out, 'demo_measured.csv')
    _gen_demo_csv(demo_csv)

    print('=' * 64)
    print('载流子寿命分析工具 · 一键演示')
    print('=' * 64)

    # 1) 正向仿真
    print('\n【1/3】正向仿真 τ_eff(Δn) 曲线（Seff=1.0 cm/s, Nt=1e10）')
    print('-' * 40)
    _run([sys.executable, SOLVER, '--Seff', '1.0', '--Nt', '1e10',
          '--plot', '--out', out])
    print('   [ok] 正向曲线图已生成')

    # 2) 标定反推（多参数）
    print('\n【2/3】实测标定反推（内置数据，真值 Seff=1.65 / Nt=1e11 / Et=0.02）')
    print('-' * 40)
    _run([sys.executable, SOLVER, '--data', demo_csv,
          '--fit', 'Seff,Nt,Et',
          '--report', 'demo_report', '--out', out])

    print('\n[完成] 演示产物已生成:')
    for f in sorted(os.listdir(out)):
        fp = os.path.join(out, f)
        print(f'  - {f}  ({os.path.getsize(fp)} B)')


def _gen_demo_csv(path):
    """生成合成实测 CSV：高寿命器件锚点量级 + 2% 对数噪声。"""
    import numpy as np
    import csv
    sys.path.insert(0, HERE)
    from qsspc_lifetime import SiParams, effective_lifetime
    rng = np.random.default_rng(2026)
    p_true = SiParams(Nd=4.5e15, W=165e-4, Seff=1.65, Nt=1e11, Et=0.02)
    dn = np.logspace(13.5, 16.5, 25)
    tau, _, _ = effective_lifetime(dn, p_true)
    tau_n = tau * np.exp(rng.normal(0, 0.02, len(tau)))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dn', 'tau_eff'])
        for a, b in zip(dn, tau_n):
            w.writerow([f'{a:.6e}', f'{b:.6e}'])
    print(f'[gen] 合成实测数据: {path} (25点, 锚点~高寿命器件量级)')


def _run(cmd):
    print('  $', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    # 输出关键行（标定结果）
    for line in (r.stdout or '').splitlines():
        if ('R²' in line or 'Seff' in line or 'Nt' in line or 'Et' in line
                or 'cost' in line or '[report]' in line or '[gen]' in line
                or '数据点' in line):
            print('   ', line.strip())
    if r.returncode != 0:
        print('   [error]', (r.stderr or r.stdout or '')[-300:])
        raise SystemExit(1)


if __name__ == '__main__':
    main()