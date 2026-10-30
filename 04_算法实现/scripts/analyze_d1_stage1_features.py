"""Analyze stage-1 scalar features without training a deployable model yet.

Outputs exact screening metrics on the frozen, non-excluded feature table and
within-environment Spearman associations.  NLOS is used only for stratified
evaluation, never as a proposed deployment input.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[2]
INPUT = PROJECT / "data" / "interim" / "d1_stage1_scalar_features.csv"
OUT_DIR = PROJECT / "02_公开数据集"
JSON_OUT = OUT_DIR / "D1_stage1_scalar_analysis_2026-08-02.json"
MD_OUT = OUT_DIR / "D1阶段1标量特征分析报告_2026-08-02.md"

FEATURES = [
    "range_measured", "rss", "rss_fp", "fp_index", "fp_point1", "fp_point2", "fp_point3",
    "stdev_noise", "cir_power", "max_noise", "rxpacc",
]


def quantile(series: pd.Series, q: float) -> float:
    values = series.dropna().to_numpy(dtype=float)
    return float(np.quantile(values, q)) if len(values) else float("nan")


def metric_frame(frame: pd.DataFrame) -> dict:
    valid = frame[frame["range_valid"] == 1]
    error = valid["ranging_error_offline"].astype(float)
    absolute = error.abs()
    return {
        "rows": int(len(frame)),
        "valid_range_rate": float((frame["range_valid"] == 1).mean()),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.mean(error.to_numpy() ** 2))),
        "p90_abs_error_m": quantile(absolute, 0.90),
        "p95_abs_error_m": quantile(absolute, 0.95),
        "median_abs_error_m": float(absolute.median()),
    }


def correlations(data: pd.DataFrame) -> dict:
    valid = data[data["range_valid"] == 1]
    target = valid["ranging_error_offline"].abs().astype(float)
    pooled = {}
    for feature in FEATURES:
        pair = pd.concat([valid[feature].astype(float), target], axis=1).dropna()
        pooled[feature] = float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic) if len(pair) > 10 else float("nan")

    within = {feature: [] for feature in FEATURES}
    for _, group in valid.groupby("environment_id", sort=True):
        for feature in FEATURES:
            pair = pd.concat([group[feature].astype(float), group["ranging_error_offline"].abs()], axis=1).dropna()
            if len(pair) > 10 and pair.iloc[:, 0].nunique() > 1:
                within[feature].append(float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic))
    summary = {}
    for feature in FEATURES:
        values = np.asarray(within[feature], dtype=float)
        summary[feature] = {
            "pooled_spearman_abs_error": pooled[feature],
            "within_environment_rhos": values.tolist(),
            "within_environment_median_abs_rho": float(np.median(np.abs(values))) if len(values) else float("nan"),
        }
    return summary


def main() -> None:
    data = pd.read_csv(INPUT)
    result = {
        "input": str(INPUT),
        "rows": int(len(data)),
        "environments": {env: metric_frame(group) for env, group in data.groupby("environment_id", sort=True)},
        "by_nlos_evaluation_only": {
            f"{env}_{label}": metric_frame(group)
            for (env, label), group in data.groupby(["environment_id", "nlos_label_eval_only"], sort=True)
        },
        "feature_error_associations": correlations(data),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# D1 阶段1标量特征分析报告",
        "",
        "本报告基于冻结 LOSO 清单，排除 environment3 的已知错位文件夹。`nlos_label_eval_only` 仅用于分层评价，不是部署输入。",
        "",
        "## 1. 域级测距误差",
        "",
        "| environment | rows | valid RANGE | MAE (m) | RMSE (m) | P90 (m) | P95 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for env, item in result["environments"].items():
        lines.append(f"| {env} | {item['rows']} | {item['valid_range_rate']:.2%} | {item['mae_m']:.4f} | {item['rmse_m']:.4f} | {item['p90_abs_error_m']:.4f} | {item['p95_abs_error_m']:.4f} |")
    lines += ["", "## 2. 可部署标量特征与绝对测距误差的 Spearman 关联", "", "| feature | pooled rho | median within-environment |", "|---|---:|---:|"]
    ranked = sorted(result["feature_error_associations"].items(), key=lambda kv: abs(kv[1]["within_environment_median_abs_rho"]) if np.isfinite(kv[1]["within_environment_median_abs_rho"]) else -1, reverse=True)
    for feature, item in ranked:
        lines.append(f"| {feature} | {item['pooled_spearman_abs_error']:.4f} | {item['within_environment_median_abs_rho']:.4f} |")
    lines += ["", "## 3. 阶段1判断", "", "- 标量特征分析只用于特征筛查，不构成因果结论。", "- 若同一特征在四个环境内关联方向和强度不稳定，应优先进入分层模型或不确定性模型，而不是直接做单变量误差修正。", "- 下一步使用这些特征建立等权 LS/WLS、鲁棒加权和区间预测基线，并在四个 LOSO 折上评估。", ""]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(data), "json": str(JSON_OUT), "markdown": str(MD_OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
