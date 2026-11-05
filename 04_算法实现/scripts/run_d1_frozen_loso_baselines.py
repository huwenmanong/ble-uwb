"""Run leakage-aware D1 LOSO positioning baselines.

Methods use only measured range and deployment-time diagnostics.  Ground-truth
coordinates and true ranges are retained solely for evaluation and the oracle
reference.  Each position/channel group is solved independently in 2-D with a
known tag height, matching the D1 planar setup.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
INPUT = PROJECT / "data" / "interim" / "d1_stage1_scalar_features.csv"
OUT_DIR = PROJECT / "data" / "results"
AGG_OUT = OUT_DIR / "d1_frozen_anchor_aggregates.csv"
RESULT_OUT = OUT_DIR / "d1_frozen_loso_baseline_results.csv"
SUMMARY_OUT = OUT_DIR / "d1_frozen_loso_baseline_summary.csv"
REPORT_OUT = PROJECT / "03_实验协议" / "D1冻结LOSO定位基线结果_2026-08-02.md"


def solve_position(rows: list[dict], mode: str, robust: bool, max_iter: int = 30):
    if len(rows) < 3:
        return math.nan, math.nan, False, 0
    z_tag = float(rows[0]["tag_z"])
    x = float(np.mean([r["anchor_x"] for r in rows]))
    y = float(np.mean([r["anchor_y"] for r in rows]))
    converged = False
    used_iter = 0
    for iteration in range(max_iter):
        used_iter = iteration + 1
        residuals, jacobians, base_weights = [], [], []
        for row in rows:
            ax, ay, az = float(row["anchor_x"]), float(row["anchor_y"]), float(row["anchor_z"])
            observed = float(row["range_mean"])
            dx, dy, dz = x - ax, y - ay, z_tag - az
            predicted = math.sqrt(dx * dx + dy * dy + dz * dz)
            if predicted < 1e-9:
                continue
            residuals.append(predicted - observed)
            jacobians.append((dx / predicted, dy / predicted))
            variance = max(0.03 ** 2, float(row["range_std"]) ** 2 / max(1.0, float(row["sample_n"])))
            if mode == "equal":
                weight = 1.0
            elif mode == "std":
                weight = 1.0 / variance
            elif mode == "quality":
                weight = float(row["quality_weight"]) / variance
            elif mode == "oracle":
                weight = 1.0
            else:
                raise ValueError(mode)
            base_weights.append(weight)
        if len(residuals) < 3:
            break
        if robust:
            med = float(np.median(np.abs(residuals)))
            delta = max(0.10, 1.345 * med)
        else:
            delta = math.inf
        j00 = j01 = j11 = b0 = b1 = 0.0
        for residual, (j0, j1), base in zip(residuals, jacobians, base_weights):
            rw = 1.0 if abs(residual) <= delta else delta / max(abs(residual), 1e-12)
            weight = base * rw
            j00 += weight * j0 * j0
            j01 += weight * j0 * j1
            j11 += weight * j1 * j1
            b0 += weight * j0 * residual
            b1 += weight * j1 * residual
        det = j00 * j11 - j01 * j01
        if abs(det) < 1e-12:
            break
        step_x = -((j11 * b0 - j01 * b1) / det)
        step_y = -((-j01 * b0 + j00 * b1) / det)
        x += step_x
        y += step_y
        if math.hypot(step_x, step_y) < 1e-6:
            converged = True
            break
    return x, y, converged, used_iter


def quality_weights(group: pd.DataFrame) -> pd.Series:
    """Unsupervised online diagnostic quality; no NLOS/error/true position used."""
    higher = ["rss", "rss_fp", "cir_power", "rxpacc"]
    lower = ["stdev_noise", "max_noise"]
    scores = []
    for col in higher:
        scores.append(group[col].rank(pct=True).fillna(0.5))
    for col in lower:
        scores.append((1.0 - group[col].rank(pct=True)).fillna(0.5))
    score = pd.concat(scores, axis=1).mean(axis=1).clip(0.15, 1.0)
    return score


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    valid = data[data["range_valid"] == 1].copy()
    group_cols = ["environment_id", "raw_folder", "channel", "anchor_label"]
    agg = valid.groupby(group_cols, sort=True).agg(
        range_mean=("range_measured", "mean"),
        range_std=("range_measured", "std"),
        sample_n=("range_measured", "count"),
        tag_x=("tag_x", "first"), tag_y=("tag_y", "first"), tag_z=("tag_z", "first"),
        anchor_x=("anchor_x", "first"), anchor_y=("anchor_y", "first"), anchor_z=("anchor_z", "first"),
        true_range=("range_true_offline", "mean"),
        rss=("rss", "mean"), rss_fp=("rss_fp", "mean"), cir_power=("cir_power", "mean"),
        stdev_noise=("stdev_noise", "mean"), max_noise=("max_noise", "mean"), rxpacc=("rxpacc", "mean"),
        nlos_ratio_eval_only=("nlos_label_eval_only", lambda s: float((s == "NLOS").mean())),
    ).reset_index()
    agg["range_std"] = agg["range_std"].fillna(0.0)
    agg["quality_weight"] = 0.5
    for _, indices in agg.groupby(["environment_id", "raw_folder", "channel"], sort=False).groups.items():
        subset = agg.loc[indices]
        agg.loc[indices, "quality_weight"] = quality_weights(subset).to_numpy()
    return agg


def run(agg: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["environment_id", "raw_folder", "channel"]
    methods = [
        ("LS_equal", "equal", False),
        ("WLS_range_std", "std", False),
        ("Quality_WLS", "quality", False),
        ("Robust_LS_Huber", "equal", True),
        ("Robust_WLS_Huber", "std", True),
        ("Robust_Quality_WLS_Huber", "quality", True),
        ("Oracle_true_range", "oracle", False),
    ]
    output = []
    for (env, folder, channel), group in agg.groupby(group_cols, sort=True):
        rows = group.to_dict("records")
        true_x, true_y, true_z = float(rows[0]["tag_x"]), float(rows[0]["tag_y"]), float(rows[0]["tag_z"])
        for method, mode, robust in methods:
            solve_rows = rows
            if method == "Oracle_true_range":
                solve_rows = [dict(row, range_mean=row["true_range"], range_std=0.03, sample_n=1) for row in rows]
            px, py, converged, iterations = solve_position(solve_rows, mode, robust)
            error = math.hypot(px - true_x, py - true_y) if math.isfinite(px) else math.nan
            output.append({
                "environment_id": env, "raw_folder": folder, "channel": channel,
                "method": method, "true_x": true_x, "true_y": true_y, "true_z": true_z,
                "pred_x": px, "pred_y": py, "error_2d_m": error,
                "anchor_count": len(rows), "converged": int(converged), "iterations": iterations,
                "mean_nlos_ratio_eval_only": float(np.mean([r["nlos_ratio_eval_only"] for r in rows])),
            })
    return pd.DataFrame(output)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, method), group in results.groupby(["environment_id", "method"], sort=True):
        errors = group["error_2d_m"].dropna().to_numpy()
        rows.append({
            "environment_id": env, "method": method, "n": len(errors),
            "converged_rate": float(group["converged"].mean()),
            "mean_m": float(np.mean(errors)), "median_m": float(np.median(errors)),
            "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "p90_m": float(np.quantile(errors, 0.90)), "p95_m": float(np.quantile(errors, 0.95)),
            "fail_at_0_5m": float(np.mean(errors > 0.5)), "fail_at_1m": float(np.mean(errors > 1.0)),
        })
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, input_rows: int, aggregate_rows: int, elapsed: float) -> None:
    lines = [
        "# D1 冻结 LOSO 定位基线结果",
        "",
        "该结果基于冻结位置分组清单；environment3 的已知错位文件夹已排除。真实坐标和真实距离仅用于离线评价，`nlos_label_eval_only` 未参与任何权重。每个位置/信道使用各锚点 31 条有效测距的均值进行二维定位，标签高度视为已知。",
        "",
        f"运行耗时：{elapsed:.2f} s；输入测量行数：{input_rows:,}；聚合锚点链路数：{aggregate_rows:,}。",
        "",
        "## 评价指标",
        "",
        "| environment | method | N | mean (m) | median (m) | RMSE (m) | P90 (m) | P95 (m) | Fail@0.5m | Fail@1m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['environment_id']} | {row['method']} | {row['n']} | {row['mean_m']:.3f} | {row['median_m']:.3f} | {row['rmse_m']:.3f} | {row['p90_m']:.3f} | {row['p95_m']:.3f} | {row['fail_at_0_5m']:.1%} | {row['fail_at_1m']:.1%} |")
    lines += [
        "",
        "## 方法说明",
        "",
        "- `LS_equal`：等权非线性最小二乘。",
        "- `WLS_range_std`：使用同一链路的测距标准差估计观测方差。",
        "- `Quality_WLS`：以 RSS、RSS_FP、CIR_POWER、RXPACC、STDEV_NOISE 和 MAX_NOISE 构造无监督诊断质量权重；只作为透明启发式基线。",
        "- `Robust_*_Huber`：在对应权重上加入 Huber 残差抑制。",
        "- `Oracle_true_range`：用真实距离的理论上限，不得作为部署方法或主结果。",
        "",
        "下一步应检查：质量权重是否在四个未见环境上稳定降低 P90/Fail@1m；如果只改善环境内结果而跨域失效，则进入分层不确定性建模。",
        "",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    agg = aggregate(data)
    results = run(agg)
    summary = summarize(results)
    agg.to_csv(AGG_OUT, index=False)
    results.to_csv(RESULT_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    write_report(summary, len(data), len(agg), time.perf_counter() - start)
    print(json.dumps({"aggregates": len(agg), "position_channel_groups": len(results) // 7, "results": len(results), "summary": str(SUMMARY_OUT), "report": str(REPORT_OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
