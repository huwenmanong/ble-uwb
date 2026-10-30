"""Generate reproducible D1 research figures for visual inspection and papers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT / "data" / "figures" / "d1_stage1"
FEATURES = PROJECT / "data" / "interim" / "d1_stage1_scalar_features.csv"
ANALYSIS_JSON = PROJECT / "02_公开数据集" / "D1_stage1_scalar_analysis_2026-08-02.json"
BASELINE_SUMMARY = PROJECT / "data" / "results" / "d1_frozen_loso_baseline_summary.csv"
UNCERTAINTY_RANGE = PROJECT / "data" / "results" / "d1_loso_uncertainty_range_results.csv"
UNCERTAINTY_LOC = PROJECT / "data" / "results" / "d1_loso_uncertainty_localization_results.csv"
UNCERTAINTY_SUMMARY = PROJECT / "data" / "results" / "d1_loso_uncertainty_summary.csv"
UNCERTAINTY_CIR_RANGE = PROJECT / "data" / "results" / "d1_loso_uncertainty_cir_range_results.csv"
UNCERTAINTY_CIR_SUMMARY = PROJECT / "data" / "results" / "d1_loso_uncertainty_cir_summary.csv"
UNCERTAINTY_ENTROPY_RANGE = PROJECT / "data" / "results" / "d1_loso_uncertainty_entropy_range_results.csv"
UNCERTAINTY_ENTROPY_SUMMARY = PROJECT / "data" / "results" / "d1_loso_uncertainty_entropy_summary.csv"
UWLS_SUMMARY = PROJECT / "data" / "results" / "d1_uncertainty_aware_wls_summary.csv"
UWLS_RISK = PROJECT / "data" / "results" / "d1_uncertainty_aware_wls_risk.csv"
UWLS_LINK = PROJECT / "data" / "results" / "d1_uncertainty_aware_wls_link_diagnostics.csv"
ENTROPY_SAMPLE = PROJECT / "02_公开数据集" / "D1_entropy_algorithm_screening_sample.csv"

ENV_ORDER = ["environment0", "environment1", "environment2", "environment3"]
BLUE = "#3568A8"
ORANGE = "#D77A35"
GOLD = "#B38B2E"
INK = "#202735"
GRID = "#D9DEE8"
PALETTE = {"raw": BLUE, "corrected": ORANGE, "LS_equal": "#5D718E", "WLS_range_std": "#6E9D72", "Quality_WLS": GOLD, "Robust_WLS_Huber": "#B35C5C"}


def style(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color="#667085")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def save(fig: plt.Figure, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def plot_error_distribution(data: pd.DataFrame) -> str:
    values = [data.loc[data["environment_id"] == env, "ranging_error_offline"].abs().dropna().to_numpy() for env in ENV_ORDER]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.55, medianprops={"color": INK, "linewidth": 1.5})
    for patch, color in zip(box["boxes"], [BLUE, "#4D8BBE", GOLD, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK)
    ax.set_xticklabels(ENV_ORDER)
    ax.set_ylabel("Absolute ranging error (m)")
    ax.set_ylim(bottom=0)
    style(ax, "D1 ranging-error distribution by environment", "Valid RANGE rows only; environment3 corrupted folder excluded")
    return save(fig, "01_ranging_error_by_environment.png")


def plot_baseline_heatmap(summary: pd.DataFrame) -> str:
    methods = ["LS_equal", "WLS_range_std", "Quality_WLS", "Robust_WLS_Huber"]
    table = summary[summary["method"].isin(methods)].pivot(index="method", columns="environment_id", values="p90_m").reindex(index=methods, columns=ENV_ORDER)
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    image = ax.imshow(table.to_numpy(dtype=float), cmap="Blues", aspect="auto", vmin=0.5, vmax=max(1.6, float(table.to_numpy().max())))
    ax.set_xticks(range(len(ENV_ORDER)), ENV_ORDER)
    ax.set_yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(len(ENV_ORDER)):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 1.0 else INK, fontsize=9)
    ax.set_xlabel("Held-out environment")
    ax.set_ylabel("Method")
    style(ax, "LOSO localization P90 by baseline", "Lower is better; values are meters")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="P90 error (m)")
    return save(fig, "02_loso_baseline_p90_heatmap.png")


def plot_calibration(range_results: pd.DataFrame, loc_results: pd.DataFrame) -> str:
    range_results = range_results.set_index("fold").reindex(ENV_ORDER)
    loc = loc_results[loc_results["method"].isin(["raw_LS", "corrected_LS"])].pivot(index="fold", columns="method", values="p90_m").reindex(ENV_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    x = np.arange(len(ENV_ORDER))
    width = 0.34
    for ax, metric, ylabel, title in [
        (axes[0, 0], ("raw_mae_m", "corrected_mae_m"), "MAE (m)", "Range correction: MAE"),
        (axes[0, 1], ("raw_p90_m", "corrected_p90_m"), "P90 absolute error (m)", "Range correction: P90"),
    ]:
        raw, corrected = [range_results[col].to_numpy(dtype=float) for col in metric]
        ax.bar(x - width / 2, raw, width, label="Raw", color=PALETTE["raw"])
        ax.bar(x + width / 2, corrected, width, label="Corrected", color=PALETTE["corrected"])
        ax.set_xticks(x, ENV_ORDER)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        style(ax, title, "Held-out environment")
    coverage = range_results["test_coverage_q90"].to_numpy(dtype=float) * 100
    axes[1, 0].bar(x, coverage, color=[BLUE, "#4D8BBE", GOLD, ORANGE])
    axes[1, 0].axhline(90, color=INK, linestyle="--", linewidth=1.1, label="Nominal 90%")
    axes[1, 0].set_xticks(x, ENV_ORDER)
    axes[1, 0].set_ylabel("Test coverage (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend(frameon=False, fontsize=8)
    style(axes[1, 0], "Nominal q90 interval coverage", "Calibration on independent position groups")
    raw, corrected = [loc[col].to_numpy(dtype=float) for col in ("raw_LS", "corrected_LS")]
    axes[1, 1].bar(x - width / 2, raw, width, label="Raw LS", color=PALETTE["raw"])
    axes[1, 1].bar(x + width / 2, corrected, width, label="Corrected LS", color=PALETTE["corrected"])
    axes[1, 1].set_xticks(x, ENV_ORDER)
    axes[1, 1].set_ylabel("Localization P90 (m)")
    axes[1, 1].legend(frameon=False, ncol=2, fontsize=8)
    style(axes[1, 1], "Positioning tail risk after correction", "Lower is better")
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    return save(fig, "03_correction_and_calibration_overview.png")


def plot_feature_associations() -> str:
    payload = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    items = payload["feature_error_associations"]
    ranked = sorted(items.items(), key=lambda kv: kv[1]["within_environment_median_abs_rho"], reverse=True)
    names = [x[0] for x in ranked]
    values = [x[1]["within_environment_median_abs_rho"] for x in ranked]
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    y = np.arange(len(names))
    ax.barh(y, values, color=BLUE, alpha=0.85)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Median absolute within-environment Spearman |rho|")
    ax.set_xlim(0, max(values) * 1.18)
    style(ax, "Signal-feature association with ranging error", "Screening statistic; not a causal effect")
    return save(fig, "04_feature_error_association.png")


def plot_cir_ablation(scalar_range: pd.DataFrame, cir_range: pd.DataFrame, scalar_loc: pd.DataFrame, cir_loc: pd.DataFrame) -> str:
    scalar_range = scalar_range.set_index("fold").reindex(ENV_ORDER)
    cir_range = cir_range.set_index("fold").reindex(ENV_ORDER)
    scalar_loc = scalar_loc[(scalar_loc["method"] == "corrected_LS")].set_index("fold").reindex(ENV_ORDER)
    cir_loc = cir_loc[(cir_loc["method"] == "corrected_LS")].set_index("fold").reindex(ENV_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    x = np.arange(len(ENV_ORDER))
    width = 0.34
    env_labels = ["E0", "E1", "E2", "E3"]
    for ax, metric, ylabel, title in [
        (axes[0], "corrected_mae_m", "MAE (m)", "Range correction MAE"),
        (axes[1], "corrected_p90_m", "P90 error (m)", "Range correction P90"),
        (axes[2], "p90_m", "Localization P90 (m)", "Corrected localization P90"),
    ]:
        if metric == "p90_m":
            scalar = scalar_loc[metric].to_numpy(dtype=float)
            cir = cir_loc[metric].to_numpy(dtype=float)
        else:
            scalar = scalar_range[metric].to_numpy(dtype=float)
            cir = cir_range[metric].to_numpy(dtype=float)
        ax.bar(x - width / 2, scalar, width, label="Scalar", color=BLUE)
        ax.bar(x + width / 2, cir, width, label="Scalar + CIR", color=ORANGE)
        ax.set_xticks(x, env_labels)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        style(ax, title, "E0–E3 = held-out environment0–3; lower is better")
    fig.tight_layout(w_pad=1.8)
    return save(fig, "05_scalar_vs_cir_ablation.png")


def plot_entropy_screening(entropy: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    colors = {"environment0": BLUE, "environment1": "#4D8BBE", "environment2": GOLD, "environment3": ORANGE}
    for ax, scale in zip(axes, (2, 3)):
        for env in ENV_ORDER:
            subset = entropy[entropy["environment"] == env]
            ax.scatter(subset[f"corrected_scale{scale}"], subset["abs_error"], s=26, alpha=0.68, color=colors[env], label=env)
        ax.set_xlabel(f"Corrected MSPDE entropy, scale {scale}")
        ax.set_ylabel("Absolute ranging error (m)")
        style(ax, f"Entropy/error screening: scale {scale}", "Sample only; association is not causal")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(w_pad=1.6)
    return save(fig, "06_entropy_error_screening.png")


def plot_three_way_ablation(scalar_range: pd.DataFrame, cir_range: pd.DataFrame, entropy_range: pd.DataFrame, scalar_loc: pd.DataFrame, cir_loc: pd.DataFrame, entropy_loc: pd.DataFrame) -> str:
    ranges = [x.set_index("fold").reindex(ENV_ORDER) for x in (scalar_range, cir_range, entropy_range)]
    locs = [x[x["method"] == "corrected_LS"].set_index("fold").reindex(ENV_ORDER) for x in (scalar_loc, cir_loc, entropy_loc)]
    labels = ["Scalar", "Scalar+CIR", "Scalar+Entropy"]
    colors = [BLUE, ORANGE, GOLD]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    x = np.arange(len(ENV_ORDER))
    width = 0.24
    for ax, title, ylabel, key, frames in [
        (axes[0], "Corrected range MAE", "MAE (m)", "corrected_mae_m", ranges),
        (axes[1], "Corrected range P90", "P90 error (m)", "corrected_p90_m", ranges),
        (axes[2], "Corrected localization P90", "Localization P90 (m)", "p90_m", locs),
    ]:
        for index, (label, color, frame) in enumerate(zip(labels, colors, frames)):
            values = frame[key].to_numpy(dtype=float)
            ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        ax.set_xticks(x, ["E0", "E1", "E2", "E3"])
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7.5)
        style(ax, title, "E0–E3 = held-out environment; lower is better")
    fig.tight_layout(w_pad=1.8)
    return save(fig, "07_scalar_cir_entropy_ablation.png")


def plot_uncertainty_aware_wls(summary: pd.DataFrame, risk: pd.DataFrame, links: pd.DataFrame) -> str:
    """Show localization tail risk, interval calibration and link risk ranking."""
    summary = summary[summary["method"].isin(["raw_LS", "corrected_LS", "corrected_UWLS", "corrected_UWLS_Huber"])]
    p90 = summary.pivot(index="fold", columns="method", values="p90_m").reindex(ENV_ORDER)
    fail = summary.pivot(index="fold", columns="method", values="fail_at_0_5m").reindex(ENV_ORDER) * 100
    risk = risk.set_index("fold").reindex(ENV_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    x = np.arange(len(ENV_ORDER))
    width = 0.18
    methods = [("raw_LS", BLUE), ("corrected_LS", ORANGE), ("corrected_UWLS", GOLD), ("corrected_UWLS_Huber", "#8E6FB1")]
    for index, (method, color) in enumerate(methods):
        axes[0, 0].bar(x + (index - 1.5) * width, p90[method], width, label=method, color=color)
        axes[0, 1].bar(x + (index - 1.5) * width, fail[method], width, label=method, color=color)
    axes[0, 0].set_xticks(x, ["E0", "E1", "E2", "E3"])
    axes[0, 0].set_ylabel("Localization P90 (m)")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    style(axes[0, 0], "Uncertainty-aware WLS: localization tail", "Lower is better; held-out environment")
    axes[0, 1].set_xticks(x, ["E0", "E1", "E2", "E3"])
    axes[0, 1].set_ylabel("Fail@0.5 m (%)")
    axes[0, 1].legend(frameon=False, fontsize=7.5)
    style(axes[0, 1], "Uncertainty-aware WLS: failure rate", "Lower is better")
    coverage = risk[["test_coverage_alpha90", "test_coverage_alpha95"]].to_numpy(dtype=float) * 100
    axes[1, 0].bar(x - width / 2, coverage[:, 0], width, label="90% interval", color=BLUE)
    axes[1, 0].bar(x + width / 2, coverage[:, 1], width, label="95% interval", color=ORANGE)
    axes[1, 0].axhline(90, color=BLUE, linestyle="--", linewidth=1)
    axes[1, 0].axhline(95, color=ORANGE, linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(x, ["E0", "E1", "E2", "E3"])
    axes[1, 0].set_ylabel("Empirical coverage (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend(frameon=False, fontsize=7.5)
    style(axes[1, 0], "Prediction-interval calibration", "Dashed lines show nominal coverage")
    spearman = risk["link_spearman_sigma_vs_abs_error"].to_numpy(dtype=float)
    axes[1, 1].bar(x, spearman, color=[BLUE, "#4D8BBE", GOLD, ORANGE])
    axes[1, 1].axhline(0, color=INK, linewidth=0.8)
    axes[1, 1].set_xticks(x, ["E0", "E1", "E2", "E3"])
    axes[1, 1].set_ylabel("Spearman rho")
    axes[1, 1].set_ylim(-0.1, 0.8)
    style(axes[1, 1], "Does predicted uncertainty rank risky links?", "Link-level sigma vs. absolute corrected error")
    fig.tight_layout(h_pad=2.0, w_pad=1.7)
    return save(fig, "08_uncertainty_aware_wls_risk.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FEATURES, usecols=["environment_id", "ranging_error_offline"])
    baseline = pd.read_csv(BASELINE_SUMMARY)
    range_results = pd.read_csv(UNCERTAINTY_RANGE)
    loc_summary = pd.read_csv(UNCERTAINTY_SUMMARY)
    cir_range_results = pd.read_csv(UNCERTAINTY_CIR_RANGE)
    cir_loc_summary = pd.read_csv(UNCERTAINTY_CIR_SUMMARY)
    entropy_range_results = pd.read_csv(UNCERTAINTY_ENTROPY_RANGE)
    entropy_loc_summary = pd.read_csv(UNCERTAINTY_ENTROPY_SUMMARY)
    entropy = pd.read_csv(ENTROPY_SAMPLE)
    uwls_summary = pd.read_csv(UWLS_SUMMARY)
    uwls_risk = pd.read_csv(UWLS_RISK)
    uwls_link = pd.read_csv(UWLS_LINK)
    paths = [
        plot_error_distribution(data),
        plot_baseline_heatmap(baseline),
        plot_calibration(range_results, loc_summary),
        plot_feature_associations(),
        plot_cir_ablation(range_results, cir_range_results, loc_summary, cir_loc_summary),
        plot_entropy_screening(entropy),
        plot_three_way_ablation(range_results, cir_range_results, entropy_range_results, loc_summary, cir_loc_summary, entropy_loc_summary),
        plot_uncertainty_aware_wls(uwls_summary, uwls_risk, uwls_link),
    ]
    manifest = {"output_dir": str(FIG_DIR), "figures": paths}
    (FIG_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
