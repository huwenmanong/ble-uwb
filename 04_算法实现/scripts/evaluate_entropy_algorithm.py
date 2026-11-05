"""Evaluate the supplied MSPDE/SPDE entropy code on a small D1 CIR sample.

This is a Python audit/prototype, not a replacement of the supplied MATLAB
implementation.  It records the original multiscale parameter swap and a
corrected version so their impact is explicit.  Full-dataset extraction is
intentionally not attempted until runtime and feature validity are established.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr

from extract_d1_sample_features import BASE_COLUMNS, parse_complex_tap, safe_float, safe_int


PROJECT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT / "数据集论文" / "data_set"
OUT_DIR = PROJECT / "02_公开数据集"
RESULT_JSON = OUT_DIR / "D1_entropy_algorithm_screening_2026-08-02.json"
REPORT_MD = OUT_DIR / "D1熵特征算法适用性分析_2026-08-02.md"


def normal_cdf_classes(x: np.ndarray, c: int, mu: float | None = None, sigma: float | None = None) -> np.ndarray:
    mu = float(np.mean(x)) if mu is None else float(mu)
    sigma = float(np.std(x, ddof=1)) if sigma is None else float(sigma)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.ones(len(x), dtype=np.int16)
    y = ndtr((x - mu) / sigma)
    lo, hi = float(y.min()), float(y.max())
    if hi - lo <= 1e-12:
        scaled = np.full_like(y, 0.5)
    else:
        scaled = (y - lo) / (hi - lo)
    labels = np.floor(scaled * c + 1.0).astype(np.int16)
    return np.clip(labels, 1, c)


def fused_spde(x: np.ndarray, c: int, m: int, t: int, mu=None, sigma=None) -> float:
    z_time = normal_cdf_classes(x, c, mu=mu, sigma=sigma)
    spectrum = np.abs(np.fft.fft(x))
    z_freq = normal_cdf_classes(spectrum, c)
    length = len(x) - (m - 1) * t
    if length <= 0:
        return math.nan
    z_time = np.column_stack([z_time[i * t:i * t + length] for i in range(m)])
    z_freq = np.column_stack([z_freq[i * t:i * t + length] for i in range(m)])
    matrices = []
    for i in range(1, m):
        for time_cols in itertools.combinations(range(m), i):
            for freq_cols in itertools.combinations(range(m), m - i):
                matrices.append(np.hstack([z_time[:, time_cols], z_freq[:, freq_cols]]))
    matrix = np.vstack(matrices)
    counts = Counter(map(tuple, matrix.tolist()))
    probabilities = np.asarray(list(counts.values()), dtype=float) / matrix.shape[0]
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(c ** m))


def entropy_vector(x: np.ndarray, m: int = 3, c: int = 6, t: int = 1, scale: int = 3, corrected: bool = True) -> np.ndarray:
    output = np.full(scale, np.nan, dtype=float)
    mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
    output[0] = fused_spde(x, c, m, t)
    for factor in range(2, scale + 1):
        coarse = np.asarray([np.mean(x[i * factor:(i + 1) * factor]) for i in range(len(x) // factor)])
        if corrected:
            output[factor - 1] = fused_spde(coarse, c, m, t, mu=mu, sigma=sigma)
        else:
            # Reproduce SPDE_2_ms.m: sigma_x=mu and mu_x=sigma.
            output[factor - 1] = fused_spde(coarse, c, m, t, mu=sigma, sigma=mu)
    return output


def iter_sample_rows(rows_per_env: int):
    for env in ("environment0", "environment1", "environment2", "environment3"):
        raw_root = DATASET_ROOT / "raw_data" / env / "data"
        selected = 0
        for path in sorted(raw_root.rglob("*.csv")):
            if selected >= rows_per_env:
                break
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                next(reader, None)
                for index, raw in enumerate(reader):
                    if selected >= rows_per_env:
                        break
                    if len(raw) < len(BASE_COLUMNS):
                        continue
                    row = dict(zip(BASE_COLUMNS, raw[: len(BASE_COLUMNS)]))
                    measured = safe_float(row["RANGE"])
                    tx, ty, tz = safe_float(row["X_TAG"]), safe_float(row["Y_TAG"]), safe_float(row["Z_TAG"])
                    ax, ay, az = safe_float(row["X_ANCHOR"]), safe_float(row["Y_ANCHOR"]), safe_float(row["Z_ANCHOR"])
                    true_range = math.sqrt((tx - ax) ** 2 + (ty - ay) ** 2 + (tz - az) ** 2)
                    taps = [parse_complex_tap(value) for value in raw[len(BASE_COLUMNS):]]
                    taps = [value for value in taps if value is not None]
                    if measured <= 0 or len(taps) < 16:
                        continue
                    yield {
                        "environment": env,
                        "relative_path": path.relative_to(DATASET_ROOT).as_posix(),
                        "measurement_index": index,
                        "nlos": row["NLOS"],
                        "abs_error": abs(measured - true_range),
                        "cir_magnitude": np.abs(np.asarray(taps, dtype=np.complex128)),
                    }
                    selected += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen the supplied MSPDE entropy algorithm on D1 CIR.")
    parser.add_argument("--rows-per-env", type=int, default=40)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--c", type=int, default=6)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    samples = list(iter_sample_rows(args.rows_per_env))
    records = []
    runtimes = {"original": [], "corrected": []}
    for sample in samples:
        record = {k: sample[k] for k in ("environment", "relative_path", "measurement_index", "nlos", "abs_error")}
        for name, corrected in (("original", False), ("corrected", True)):
            start = time.perf_counter()
            values = entropy_vector(sample["cir_magnitude"], m=args.m, c=args.c, scale=args.scale, corrected=corrected)
            runtimes[name].append((time.perf_counter() - start) * 1000.0)
            for index, value in enumerate(values, start=1):
                record[f"{name}_scale{index}"] = value
        records.append(record)

    result_frame = __import__("pandas").DataFrame(records)
    correlations = {}
    for name in ("original", "corrected"):
        correlations[name] = {}
        for scale in range(1, args.scale + 1):
            column = f"{name}_scale{scale}"
            correlations[name][f"scale{scale}"] = {}
            for env, group in result_frame.groupby("environment", sort=True):
                pair = group[[column, "abs_error"]].dropna()
                rho = spearmanr(pair[column], pair["abs_error"]).statistic if len(pair) > 5 else math.nan
                correlations[name][f"scale{scale}"][env] = float(rho)
    output = {
        "parameters": {"rows_per_env": args.rows_per_env, "m": args.m, "c": args.c, "scale": args.scale},
        "sample_rows": len(result_frame),
        "runtime_ms_per_row": {name: {"median": float(np.median(values)), "p95": float(np.quantile(values, 0.95))} for name, values in runtimes.items()},
        "estimated_full_d1_hours": {name: float(np.median(values) * 489552 / 1000 / 3600) for name, values in runtimes.items()},
        "correlations_with_abs_ranging_error": correlations,
        "zero_or_nonfinite_counts": {column: int((~np.isfinite(result_frame[column])).sum()) for column in result_frame.columns if column.startswith(("original_", "corrected_"))},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    result_frame.to_csv(OUT_DIR / "D1_entropy_algorithm_screening_sample.csv", index=False)
    lines = [
        "# D1 熵特征算法适用性分析",
        "",
        "## Target profile",
        "",
        "Measurement / IEEE TIM；目标是判断 supplied MSPDE/SPDE 算法能否作为 D1 CIR 的部署时熵特征。",
        "",
        "## Manuscript claim",
        "",
        "该算法可以作为 CIR 的时频复杂度特征候选，但当前 MATLAB 实现需要修正参数错误、定义复数 CIR 的输入方式，并先通过跨环境消融验证。",
        "",
        "## 算法审查结果",
        "",
        "1. `SPDE_2.m` 对实值序列同时构造时域和 FFT 幅值域模式，再进行组合模式熵归一化；理论上适合提取 CIR 多径结构复杂度。",
        "2. D1 CIR 是复数序列，不能直接把复数送入 `normcdf`；建议以 `abs(CIR)` 或功率序列作为主输入，并把实部/虚部版本作为消融。",
        "3. `SPDE_2_ms.m` 存在明显参数交换：代码写成 `sigma_x = mu; mu_x = sigma`，应核对为 `sigma_x = sigma; mu_x = mu`。本报告同时计算 original 和 corrected 两种版本。",
        "4. `mapminmax` 依赖 MATLAB Neural Network Toolbox，且按单条信号重新缩放；Python/部署版应显式实现并记录缩放规则。",
        "5. 组合模式通过 `containers.Map` 逐行计数，m=3 时每个尺度需要约 20 组组合；对 489,552 条测量逐条计算成本很高，必须先做缓存、降采样或按链路窗口聚合。",
        "",
        "## 样本筛查证据",
        "",
        f"- 样本量：{len(result_frame)} 行（每环境最多 {args.rows_per_env} 行）。",
        f"- corrected 版本单行熵特征中位耗时：{output['runtime_ms_per_row']['corrected']['median']:.2f} ms；按当前实现估算全 D1 约 {output['estimated_full_d1_hours']['corrected']:.2f} 小时，尚未做并行/缓存优化。",
        "- 相关性仅作筛查，不代表熵特征具有因果解释力；正式判断必须进入同一 LOSO 训练/校准/测试协议。",
        "",
        "## 研究建议",
        "",
        "- 保留该算法作为 `CIR entropy` 分支，而不是替代现有 CIR 峰值/能量特征。",
        "- 先在每条链路的 31 个连续 CIR 上做窗口级熵统计（均值、标准差、P90），再送入不确定性模型；不要对每一条 CIR 都产生高维熵向量。",
        "- 必须做四组消融：scalar only、compact CIR only、scalar+CIR、scalar+CIR+entropy。",
        "- Go/no-go：若熵特征在至少 3 个 LOSO 测试环境同时改善测距 P90、区间覆盖或定位 P95，才保留为主方法特征；否则作为补充分析。",
        "",
        "## Evidence still required",
        "",
        "- 当前样本筛查尚不能证明熵算法有效；需要先修正 MATLAB 参数错误并完成全四折消融。",
        "- 需要报告运行时间、内存、窗口长度和参数敏感性（m、c、尺度数）。",
        "- 需要与现有 compact CIR 特征公平比较，禁止使用 NLOS 标签或真实测距误差作为输入。",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(result_frame), "json": str(RESULT_JSON), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
