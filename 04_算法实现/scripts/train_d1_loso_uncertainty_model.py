"""Train and evaluate a first cross-environment D1 range-correction model.

The model predicts ranging error from deployment-valid measurements only.  Each
LOSO fold trains on position groups marked ``train``, calibrates an absolute
residual quantile on independent ``calibration`` groups, and evaluates the
held-out environment.  Ground-truth coordinates/range and NLOS are evaluation
labels only.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from run_d1_frozen_loso_baselines import solve_position


PROJECT = Path(__file__).resolve().parents[2]
INPUT = PROJECT / "data" / "interim" / "d1_stage1_scalar_features.csv"
OUT_DIR = PROJECT / "data" / "results"
RANGE_OUT = OUT_DIR / "d1_loso_uncertainty_range_results.csv"
LOC_OUT = OUT_DIR / "d1_loso_uncertainty_localization_results.csv"
SUMMARY_OUT = OUT_DIR / "d1_loso_uncertainty_summary.csv"
REPORT_OUT = PROJECT / "03_实验协议" / "D1_LOSO测距校正与区间校准结果_2026-08-02.md"

BASE_NUMERIC_FEATURES = [
    "range_measured", "rss", "rss_fp", "fp_index", "fp_point1", "fp_point2", "fp_point3",
    "stdev_noise", "cir_power", "max_noise", "rxpacc", "anchor_x", "anchor_y", "anchor_z",
]
CIR_NUMERIC_FEATURES = [
    "cir_len", "cir_peak", "cir_peak_index", "cir_energy", "cir_mean_magnitude", "cir_spread",
    "cir_top5_energy_ratio", "cir_fp_window_energy_ratio",
]
ENTROPY_NUMERIC_FEATURES = [
    "entropy_window_rows", "entropy_scale1_mean", "entropy_scale1_std", "entropy_scale1_p90", "entropy_scale1_max",
    "entropy_scale2_mean", "entropy_scale2_std", "entropy_scale2_p90", "entropy_scale2_max",
    "entropy_scale3_mean", "entropy_scale3_std", "entropy_scale3_p90", "entropy_scale3_max",
]
CATEGORICAL_FEATURES = ["channel", "anchor_label"]
SEED = 20260802


def make_feature_matrix(data: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    numeric = data[numeric_features].apply(pd.to_numeric, errors="coerce")
    categorical = pd.get_dummies(data[CATEGORICAL_FEATURES].astype(str), dtype=float)
    return pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)


def fill_from_train(x: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    medians = train.median(numeric_only=True).fillna(0.0)
    return x.reindex(columns=train.columns, fill_value=0.0).fillna(medians), medians


def q(values: np.ndarray, quantile: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if len(values) else math.nan


def fit_model(x_train: pd.DataFrame, y_train: pd.Series, max_iter: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(x_train, y_train)
    return model


def localization_metrics(test: pd.DataFrame, correction: np.ndarray, fold: str) -> list[dict]:
    work = test.copy()
    work["range_corrected"] = work["range_measured"].to_numpy() - correction
    group_cols = ["environment_id", "raw_folder", "channel", "anchor_label"]
    agg = work.groupby(group_cols, sort=True).agg(
        raw_mean=("range_measured", "mean"), corrected_mean=("range_corrected", "mean"),
        range_std=("range_measured", "std"), sample_n=("range_measured", "count"),
        tag_x=("tag_x", "first"), tag_y=("tag_y", "first"), tag_z=("tag_z", "first"),
        anchor_x=("anchor_x", "first"), anchor_y=("anchor_y", "first"), anchor_z=("anchor_z", "first"),
    ).reset_index()
    agg["range_std"] = agg["range_std"].fillna(0.0)
    rows = []
    for (env, folder, channel), group in agg.groupby(["environment_id", "raw_folder", "channel"], sort=True):
        group_rows = group.to_dict("records")
        if len(group_rows) < 3:
            continue
        true_x, true_y = float(group_rows[0]["tag_x"]), float(group_rows[0]["tag_y"])
        for method, range_key in (("raw_LS", "raw_mean"), ("corrected_LS", "corrected_mean")):
            solve_rows = [dict(row, range_mean=row[range_key], quality_weight=1.0) for row in group_rows]
            px, py, converged, iterations = solve_position(solve_rows, mode="equal", robust=False)
            error = math.hypot(px - true_x, py - true_y) if math.isfinite(px) else math.nan
            rows.append({
                "fold": fold, "environment_id": env, "raw_folder": folder, "channel": channel,
                "method": method, "error_2d_m": error, "converged": int(converged), "iterations": iterations,
            })
    return rows


def run_fold(data: pd.DataFrame, matrix: pd.DataFrame, fold: str, max_train_rows: int, max_iter: int) -> tuple[list[dict], list[dict]]:
    split = data[f"split_{fold}"]
    valid = data["range_valid"].to_numpy() == 1
    train_mask = (split == "train").to_numpy() & valid
    cal_mask = (split == "calibration").to_numpy() & valid
    test_mask = (split == "test").to_numpy() & valid
    train_ids = np.flatnonzero(train_mask)
    if len(train_ids) > max_train_rows:
        rng = np.random.default_rng(SEED + int(fold[-1]))
        train_ids = np.sort(rng.choice(train_ids, size=max_train_rows, replace=False))
    cal_ids = np.flatnonzero(cal_mask)
    test_ids = np.flatnonzero(test_mask)

    x_train, medians = fill_from_train(matrix.iloc[train_ids], matrix.iloc[train_ids])
    x_cal = matrix.iloc[cal_ids].reindex(columns=x_train.columns, fill_value=0.0).fillna(medians)
    x_test = matrix.iloc[test_ids].reindex(columns=x_train.columns, fill_value=0.0).fillna(medians)
    y_train = data.iloc[train_ids]["ranging_error_offline"].astype(float)
    y_cal = data.iloc[cal_ids]["ranging_error_offline"].astype(float)
    y_test = data.iloc[test_ids]["ranging_error_offline"].astype(float)

    model = fit_model(x_train, y_train, max_iter)
    pred_cal = model.predict(x_cal)
    pred_test = model.predict(x_test)
    cal_abs = np.abs(y_cal.to_numpy() - pred_cal)
    q90, q95 = q(cal_abs, 0.90), q(cal_abs, 0.95)
    raw_abs = np.abs(y_test.to_numpy())
    corrected_abs = np.abs(y_test.to_numpy() - pred_test)
    range_result = {
        "fold": fold,
        "train_rows": len(train_ids), "calibration_rows": len(cal_ids), "test_rows": len(test_ids),
        "train_groups": int((split.iloc[train_ids] == "train").sum()),
        "raw_mae_m": float(raw_abs.mean()), "corrected_mae_m": float(corrected_abs.mean()),
        "raw_rmse_m": float(np.sqrt(np.mean(y_test.to_numpy() ** 2))),
        "corrected_rmse_m": float(np.sqrt(np.mean((y_test.to_numpy() - pred_test) ** 2))),
        "raw_p90_m": q(raw_abs, 0.90), "corrected_p90_m": q(corrected_abs, 0.90),
        "raw_p95_m": q(raw_abs, 0.95), "corrected_p95_m": q(corrected_abs, 0.95),
        "calibration_q90_m": q90, "calibration_q95_m": q95,
        "test_coverage_q90": float(np.mean(corrected_abs <= q90)),
        "test_coverage_q95": float(np.mean(corrected_abs <= q95)),
        "test_interval_width_q90_m": 2.0 * q90,
        "test_interval_width_q95_m": 2.0 * q95,
        "predicted_error_mean_m": float(np.mean(pred_test)),
    }
    test_frame = data.iloc[test_ids].copy()
    loc_rows = localization_metrics(test_frame, pred_test, fold)
    for row in loc_rows:
        row["range_correction_fold"] = fold
    return [range_result], loc_rows


def summarize_localization(loc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, method), group in loc.groupby(["fold", "method"], sort=True):
        values = group["error_2d_m"].dropna().to_numpy()
        rows.append({
            "fold": fold, "method": method, "n": len(values),
            "mean_m": float(values.mean()), "median_m": float(np.median(values)),
            "rmse_m": float(np.sqrt(np.mean(values ** 2))), "p90_m": q(values, 0.90), "p95_m": q(values, 0.95),
            "fail_at_0_5m": float(np.mean(values > 0.5)), "fail_at_1m": float(np.mean(values > 1.0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="D1 LOSO range correction and conformal-style interval calibration")
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--cir-input", type=Path, default=None, help="Optional compact CIR feature CSV for ablation.")
    parser.add_argument("--entropy-link-input", type=Path, default=None, help="Optional window-aggregated entropy feature CSV.")
    args = parser.parse_args()
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    data = data[data["range_valid"] == 1].reset_index(drop=True)
    numeric_features = list(BASE_NUMERIC_FEATURES)
    if args.cir_input is not None:
        cir = pd.read_csv(args.cir_input)
        join_keys = ["environment_id", "raw_folder", "relative_path", "channel", "anchor_label", "measurement_index"]
        cir_columns = join_keys + CIR_NUMERIC_FEATURES
        data = data.merge(cir[cir_columns], on=join_keys, how="left", validate="one_to_one")
        numeric_features.extend(CIR_NUMERIC_FEATURES)
    if args.entropy_link_input is not None:
        entropy = pd.read_csv(args.entropy_link_input)
        join_keys = ["environment_id", "raw_folder", "relative_path", "channel", "anchor_label"]
        entropy_columns = join_keys + ENTROPY_NUMERIC_FEATURES
        data = data.merge(entropy[entropy_columns], on=join_keys, how="left", validate="many_to_one")
        numeric_features.extend(ENTROPY_NUMERIC_FEATURES)
    matrix = make_feature_matrix(data, numeric_features)
    range_rows, loc_rows = [], []
    for fold in ("environment0", "environment1", "environment2", "environment3"):
        print(f"running {fold} ...", flush=True)
        fold_range, fold_loc = run_fold(data, matrix, fold, args.max_train_rows, args.max_iter)
        range_rows.extend(fold_range)
        loc_rows.extend(fold_loc)
        print(f"completed {fold}: range_rows={len(fold_range)} localization_rows={len(fold_loc)}", flush=True)
    range_df = pd.DataFrame(range_rows)
    loc_df = pd.DataFrame(loc_rows)
    loc_summary = summarize_localization(loc_df)
    suffix_parts = []
    if args.cir_input is not None:
        suffix_parts.append("cir")
    if args.entropy_link_input is not None:
        suffix_parts.append("entropy")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    range_out = OUT_DIR / f"d1_loso_uncertainty{suffix}_range_results.csv"
    loc_out = OUT_DIR / f"d1_loso_uncertainty{suffix}_localization_results.csv"
    summary_out = OUT_DIR / f"d1_loso_uncertainty{suffix}_summary.csv"
    report_out = PROJECT / "03_实验协议" / f"D1_LOSO测距校正与区间校准结果{suffix}_2026-08-02.md"
    range_df.to_csv(range_out, index=False)
    loc_df.to_csv(loc_out, index=False)
    loc_summary.to_csv(summary_out, index=False)
    elapsed = time.perf_counter() - start
    lines = [
        "# D1 LOSO 测距校正与区间校准结果",
        "",
        "## Target profile",
        "",
        "Measurement / IEEE TIM；本实验是跨环境测距不确定性建模的第一版可复现实验。",
        "",
        "## Manuscript claim",
        "",
        "检验部署可用信号特征能否在未见 environment 上校正测距偏差，并通过独立 calibration 组构造具有覆盖率证据的误差区间。",
        "",
        "## Evidence available",
        "",
        f"- 输入：{len(data):,} 条有效测量；四个 LOSO 折；训练最多 {args.max_train_rows:,} 行/折；运行耗时 {elapsed:.2f} s。",
        f"- 模型输入：Range、RSS、RSS_FP、FP 指标、噪声/接收诊断、锚点坐标、信道和锚点 ID{'，以及 compact CIR 特征' if args.cir_input is not None else ''}{'，以及窗口化熵特征' if args.entropy_link_input is not None else ''}。",
        "- 未使用：真实位置、真实距离、测距误差标签（仅训练目标）、NLOS 标签和错位文件夹。",
        "",
        "### 测距校正与区间覆盖",
        "",
        "| fold | raw MAE | corrected MAE | raw P90 | corrected P90 | q90 (m) | test coverage q90 | q95 (m) | test coverage q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in range_df.to_dict("records"):
        lines.append(f"| {row['fold']} | {row['raw_mae_m']:.4f} | {row['corrected_mae_m']:.4f} | {row['raw_p90_m']:.4f} | {row['corrected_p90_m']:.4f} | {row['calibration_q90_m']:.4f} | {row['test_coverage_q90']:.2%} | {row['calibration_q95_m']:.4f} | {row['test_coverage_q95']:.2%} |")
    lines += ["", "### 定位影响", "", "| fold | method | N | mean (m) | RMSE (m) | P90 (m) | P95 (m) | Fail@0.5m | Fail@1m |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in loc_summary.to_dict("records"):
        lines.append(f"| {row['fold']} | {row['method']} | {row['n']} | {row['mean_m']:.4f} | {row['rmse_m']:.4f} | {row['p90_m']:.4f} | {row['p95_m']:.4f} | {row['fail_at_0_5m']:.2%} | {row['fail_at_1m']:.2%} |")
    lines += [
        "",
        "## Evidence still required",
        "",
        "- 当前是第一版全局误差模型，不等于最终方法；需要与不校正 LS/WLS、分层模型、CIR 特征模型和真正的风险传播模型比较。",
        "- calibration coverage 只在固定 D1 split 上成立；需要报告跨折均值/方差，并进行 bootstrap 或重复种子稳定性检查。",
        "- 后续必须增加 CIR 特征和 uncertainty-aware WLS，验证区间是否能降低定位尾部风险。",
        "",
    ]
    report_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"range_results": str(range_out), "localization_results": str(loc_out), "summary": str(summary_out), "report": str(report_out), "elapsed_s": elapsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
