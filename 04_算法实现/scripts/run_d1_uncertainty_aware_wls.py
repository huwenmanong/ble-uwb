"""Evaluate uncertainty-aware WLS using deployment-valid D1 features.

For each LOSO fold, a bias model is trained on train groups, a scale model is
fit on independent calibration groups, and the resulting per-link uncertainty
controls WLS weights in the held-out environment.  True position/range and NLOS
are used only for evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_d1_frozen_loso_baselines import solve_position
from train_d1_loso_uncertainty_model import (
    BASE_NUMERIC_FEATURES,
    CIR_NUMERIC_FEATURES,
    ENTROPY_NUMERIC_FEATURES,
    fill_from_train,
    fit_model,
    make_feature_matrix,
    q,
)


PROJECT = Path(__file__).resolve().parents[2]
INPUT = PROJECT / "data" / "interim" / "d1_stage1_scalar_features.csv"
ENTROPY_DEFAULT = PROJECT / "data" / "interim" / "d1_stage1_entropy_link_features.csv"
OUT_DIR = PROJECT / "data" / "results"
LOC_OUT = OUT_DIR / "d1_uncertainty_aware_wls_localization.csv"
RISK_OUT = OUT_DIR / "d1_uncertainty_aware_wls_risk.csv"
SUMMARY_OUT = OUT_DIR / "d1_uncertainty_aware_wls_summary.csv"
LINK_OUT = OUT_DIR / "d1_uncertainty_aware_wls_link_diagnostics.csv"
REPORT_OUT = PROJECT / "03_实验协议" / "D1不确定性WLS风险传播结果_2026-08-02.md"
SEED = 20260802


def merge_optional_features(data: pd.DataFrame, entropy_path: Path | None, cir_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    features = list(BASE_NUMERIC_FEATURES)
    if cir_path is not None:
        cir = pd.read_csv(cir_path)
        keys = ["environment_id", "raw_folder", "relative_path", "channel", "anchor_label", "measurement_index"]
        data = data.merge(cir[keys + CIR_NUMERIC_FEATURES], on=keys, how="left", validate="one_to_one")
        features.extend(CIR_NUMERIC_FEATURES)
    if entropy_path is not None:
        entropy = pd.read_csv(entropy_path)
        keys = ["environment_id", "raw_folder", "relative_path", "channel", "anchor_label"]
        data = data.merge(entropy[keys + ENTROPY_NUMERIC_FEATURES], on=keys, how="left", validate="many_to_one")
        features.extend(ENTROPY_NUMERIC_FEATURES)
    return data, features


def localization_rows(test: pd.DataFrame, fold: str) -> list[dict]:
    group_cols = ["environment_id", "raw_folder", "channel", "anchor_label"]
    agg = test.groupby(group_cols, sort=True).agg(
        raw_mean=("range_measured", "mean"), corrected_mean=("range_corrected", "mean"),
        uncertainty=("pred_sigma", "mean"), sample_n=("range_measured", "count"),
        tag_x=("tag_x", "first"), tag_y=("tag_y", "first"), tag_z=("tag_z", "first"),
        anchor_x=("anchor_x", "first"), anchor_y=("anchor_y", "first"), anchor_z=("anchor_z", "first"),
    ).reset_index()
    rows = []
    for (env, folder, channel), group in agg.groupby(["environment_id", "raw_folder", "channel"], sort=True):
        links = group.to_dict("records")
        if len(links) < 3:
            continue
        true_x, true_y = float(links[0]["tag_x"]), float(links[0]["tag_y"])
        methods = {
            "raw_LS": [dict(row, range_mean=row["raw_mean"], range_std=0.03, sample_n=1, quality_weight=1.0) for row in links],
            "corrected_LS": [dict(row, range_mean=row["corrected_mean"], range_std=0.03, sample_n=1, quality_weight=1.0) for row in links],
            "corrected_UWLS": [dict(row, range_mean=row["corrected_mean"], range_std=max(0.03, row["uncertainty"]), sample_n=1, quality_weight=1.0) for row in links],
            "corrected_UWLS_Huber": [dict(row, range_mean=row["corrected_mean"], range_std=max(0.03, row["uncertainty"]), sample_n=1, quality_weight=1.0) for row in links],
        }
        # Selective variants remove only the highest predicted-risk links.
        # D1 normally has eight anchors, so one or two removals leave a
        # well-constrained 2-D geometry while testing a deployable rejection
        # policy rather than an oracle NLOS mask.
        ranked = sorted(links, key=lambda row: row["uncertainty"], reverse=True)
        for drop_n in (1, 2):
            if len(ranked) - drop_n >= 4:
                methods[f"corrected_UWLS_drop{drop_n}"] = [
                    dict(row, range_mean=row["corrected_mean"], range_std=max(0.03, row["uncertainty"]), sample_n=1, quality_weight=1.0)
                    for row in ranked[drop_n:]
                ]
        for method, solve_rows in methods.items():
            mode = "std" if method.startswith("corrected_UWLS") else "equal"
            px, py, converged, iterations = solve_position(solve_rows, mode=mode, robust=method.endswith("Huber"))
            error = math.hypot(px - true_x, py - true_y) if math.isfinite(px) else math.nan
            rows.append({"fold": fold, "environment_id": env, "raw_folder": folder, "channel": channel, "method": method, "error_2d_m": error, "converged": int(converged), "iterations": iterations})
    return rows


def summarize_localization(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, method), group in data.groupby(["fold", "method"], sort=True):
        values = group["error_2d_m"].dropna().to_numpy(dtype=float)
        rows.append({
            "fold": fold, "method": method, "n": len(values), "mean_m": float(values.mean()),
            "median_m": float(np.median(values)), "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "p90_m": q(values, 0.90), "p95_m": q(values, 0.95),
            "fail_at_0_5m": float(np.mean(values > 0.5)), "fail_at_1m": float(np.mean(values > 1.0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 uncertainty-aware WLS risk propagation")
    parser.add_argument("--entropy-link-input", type=Path, default=ENTROPY_DEFAULT)
    parser.add_argument("--cir-input", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()
    start = time.perf_counter()
    data = pd.read_csv(INPUT)
    data = data[data["range_valid"] == 1].reset_index(drop=True)
    data, feature_columns = merge_optional_features(data, args.entropy_link_input, args.cir_input)
    matrix = make_feature_matrix(data, feature_columns)
    loc_rows, risk_rows, link_rows = [], [], []
    for fold in ("environment0", "environment1", "environment2", "environment3"):
        print(f"running {fold} ...", flush=True)
        split = data[f"split_{fold}"]
        valid = data["range_valid"].to_numpy() == 1
        train_ids = np.flatnonzero((split == "train").to_numpy() & valid)
        cal_ids = np.flatnonzero((split == "calibration").to_numpy() & valid)
        test_ids = np.flatnonzero((split == "test").to_numpy() & valid)
        if len(train_ids) > args.max_train_rows:
            rng = np.random.default_rng(SEED + int(fold[-1]))
            train_ids = np.sort(rng.choice(train_ids, size=args.max_train_rows, replace=False))
        x_train, medians = fill_from_train(matrix.iloc[train_ids], matrix.iloc[train_ids])
        x_cal = matrix.iloc[cal_ids].reindex(columns=x_train.columns, fill_value=0.0).fillna(medians)
        x_test = matrix.iloc[test_ids].reindex(columns=x_train.columns, fill_value=0.0).fillna(medians)
        y_train = data.iloc[train_ids]["ranging_error_offline"].astype(float)
        y_cal = data.iloc[cal_ids]["ranging_error_offline"].astype(float)
        y_test = data.iloc[test_ids]["ranging_error_offline"].astype(float)
        bias_model = fit_model(x_train, y_train, args.max_iter)
        pred_cal = bias_model.predict(x_cal)
        pred_test = bias_model.predict(x_test)
        residual_cal = np.abs(y_cal.to_numpy() - pred_cal)
        scale_model = fit_model(x_cal, pd.Series(residual_cal, index=x_cal.index), max(60, args.max_iter // 2))
        scale_cal = np.maximum(0.03, scale_model.predict(x_cal))
        scale_test = np.maximum(0.03, scale_model.predict(x_test))
        ratio_cal = residual_cal / scale_cal
        alpha90, alpha95 = q(ratio_cal, 0.90), q(ratio_cal, 0.95)
        pred_sigma = np.maximum(0.03, scale_test * alpha90)
        corrected_error = y_test.to_numpy() - pred_test
        # Link-level risk diagnostic: aggregate the held-out samples belonging
        # to one anchor/channel file.  This is evaluation-only and is never
        # used by the deployable predictor.
        test_diag = data.iloc[test_ids][["environment_id", "raw_folder", "channel", "anchor_label"]].copy()
        test_diag["abs_raw_error_m"] = np.abs(data.iloc[test_ids]["ranging_error_offline"].to_numpy())
        test_diag["abs_corrected_error_m"] = np.abs(corrected_error)
        test_diag["pred_sigma_m"] = pred_sigma
        link_diag = test_diag.groupby(
            ["environment_id", "raw_folder", "channel", "anchor_label"], sort=True
        ).agg(
            n=("pred_sigma_m", "size"),
            mean_abs_raw_error_m=("abs_raw_error_m", "mean"),
            mean_abs_corrected_error_m=("abs_corrected_error_m", "mean"),
            mean_pred_sigma_m=("pred_sigma_m", "mean"),
            p90_pred_sigma_m=("pred_sigma_m", lambda x: q(x.to_numpy(dtype=float), 0.90)),
        ).reset_index()
        link_diag["fold"] = fold
        link_rows.extend(link_diag.to_dict("records"))
        rank_corr = float(link_diag["mean_pred_sigma_m"].corr(link_diag["mean_abs_corrected_error_m"], method="spearman"))
        risk_rows.append({
            "fold": fold, "train_rows": len(train_ids), "calibration_rows": len(cal_ids), "test_rows": len(test_ids),
            "alpha90": alpha90, "alpha95": alpha95, "raw_mae_m": float(np.mean(np.abs(y_test))),
            "corrected_mae_m": float(np.mean(np.abs(corrected_error))), "raw_p90_m": q(np.abs(y_test.to_numpy()), 0.90),
            "corrected_p90_m": q(np.abs(corrected_error), 0.90), "test_coverage_alpha90": float(np.mean(np.abs(corrected_error) <= pred_sigma)),
            "test_coverage_alpha95": float(np.mean(np.abs(corrected_error) <= scale_test * alpha95)), "median_sigma_m": float(np.median(pred_sigma)),
            "link_spearman_sigma_vs_abs_error": rank_corr,
        })
        test = data.iloc[test_ids].copy()
        test["range_corrected"] = test["range_measured"].to_numpy() - pred_test
        test["pred_sigma"] = pred_sigma
        loc_rows.extend(localization_rows(test, fold))
        print(f"completed {fold}: test_rows={len(test_ids)}", flush=True)
    loc_df = pd.DataFrame(loc_rows)
    risk_df = pd.DataFrame(risk_rows)
    link_df = pd.DataFrame(link_rows)
    summary_df = summarize_localization(loc_df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loc_df.to_csv(LOC_OUT, index=False)
    risk_df.to_csv(RISK_OUT, index=False)
    link_df.to_csv(LINK_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)
    elapsed = time.perf_counter() - start
    lines = [
        "# D1 不确定性 WLS 风险传播结果",
        "",
        "## Target profile",
        "",
        "Measurement / IEEE TIM；验证预测测距不确定性是否能在未见环境中改善定位尾部风险。",
        "",
        "## Manuscript claim",
        "",
        "若测距偏差和尺度均由独立 train/calibration 流程估计，则预测区间应可用于 WLS 权重，并降低定位 P90/P95 或高风险失败率。",
        "",
        "## Evidence available",
        "",
        f"- 输入测量：{len(data):,} 条有效行；窗口熵特征已加入；运行耗时 {elapsed:.2f} s。",
        "- `corrected_UWLS` 使用校准后的每链路预测尺度作为观测方差；未使用真实位置、真实距离或 NLOS 标签。",
        "",
        "### 测距风险覆盖",
        "",
        "| fold | corrected MAE | corrected P90 | q90 coverage | q95 coverage | median sigma (m) | link Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in risk_df.to_dict("records"):
        lines.append(f"| {row['fold']} | {row['corrected_mae_m']:.4f} | {row['corrected_p90_m']:.4f} | {row['test_coverage_alpha90']:.2%} | {row['test_coverage_alpha95']:.2%} | {row['median_sigma_m']:.4f} | {row['link_spearman_sigma_vs_abs_error']:.3f} |")
    lines += ["", "### 定位风险传播", "", "| fold | method | mean (m) | RMSE (m) | P90 (m) | P95 (m) | Fail@0.5m | Fail@1m |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_df.to_dict("records"):
        lines.append(f"| {row['fold']} | {row['method']} | {row['mean_m']:.4f} | {row['rmse_m']:.4f} | {row['p90_m']:.4f} | {row['p95_m']:.4f} | {row['fail_at_0_5m']:.2%} | {row['fail_at_1m']:.2%} |")
    lines += ["", "## Evidence still required", "", "- 当前 uncertainty-aware WLS 是第一版尺度模型，需要与等权/标准差 WLS 和固定鲁棒核做成完整消融。", "- 需要检查尺度预测是否能按链路风险排序，而不仅仅是产生一个全局方差。", "- 需要增加风险预算下的接受/降级/拒绝曲线，以及 bootstrap 稳定性。", ""]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"localization": str(LOC_OUT), "risk": str(RISK_OUT), "link_diagnostics": str(LINK_OUT), "summary": str(SUMMARY_OUT), "report": str(REPORT_OUT), "elapsed_s": elapsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
