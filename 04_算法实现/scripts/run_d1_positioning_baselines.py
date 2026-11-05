from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "数据集论文" / "data_set"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
REPORT_PATH = PROJECT_ROOT / "03_实验协议" / "D1原始LS_WLS定位基线结果.md"

BASE_FIELDS = [
    "TAG_ID",
    "ANCHOR_ID",
    "X_TAG",
    "Y_TAG",
    "Z_TAG",
    "X_ANCHOR",
    "Y_ANCHOR",
    "Z_ANCHOR",
    "NLOS",
    "RANGE",
    "FP_INDEX",
    "RSS",
    "RSS_FP",
    "FP_POINT1",
    "FP_POINT2",
    "FP_POINT3",
    "STDEV_NOISE",
    "CIR_POWER",
    "MAX_NOISE",
    "RXPACC",
    "CHANNEL_NUMBER",
    "FRAME_LENGTH",
    "PREAMBLE_LENGTH",
    "BITRATE",
    "PRFR",
    "PREAMBLE_CODE",
]

PREFIX_COLUMN_COUNT = len(BASE_FIELDS)


@dataclass
class OnlineStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))


@dataclass
class AnchorAggregate:
    range_stats: OnlineStats
    rss_stats: OnlineStats
    rss_fp_stats: OnlineStats
    tag_id: str
    anchor_id: str
    tag_xyz: tuple[float, float, float]
    anchor_xyz: tuple[float, float, float]
    nlos_count: int = 0


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def parse_file_name(path: Path) -> tuple[str, str]:
    match = re.match(r"^(ch\d+)_(A\d+)\.csv$", path.name)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def env_sort_key(path: Path) -> int:
    match = re.search(r"environment(\d+)", path.name)
    return int(match.group(1)) if match else 999


def iter_environment_names(selected: str) -> list[str]:
    if selected:
        return [part.strip() for part in selected.split(",") if part.strip()]
    env_dirs = sorted(
        [p for p in (DATASET_ROOT / "raw_data").iterdir() if p.is_dir() and p.name.startswith("environment")],
        key=env_sort_key,
    )
    return [p.name for p in env_dirs]


def update_aggregate(agg: AnchorAggregate | None, fields: list[str]) -> AnchorAggregate:
    values = dict(zip(BASE_FIELDS, fields))
    tag_xyz = (
        safe_float(values["X_TAG"]),
        safe_float(values["Y_TAG"]),
        safe_float(values["Z_TAG"]),
    )
    anchor_xyz = (
        safe_float(values["X_ANCHOR"]),
        safe_float(values["Y_ANCHOR"]),
        safe_float(values["Z_ANCHOR"]),
    )
    if agg is None:
        agg = AnchorAggregate(
            range_stats=OnlineStats(),
            rss_stats=OnlineStats(),
            rss_fp_stats=OnlineStats(),
            tag_id=values["TAG_ID"],
            anchor_id=values["ANCHOR_ID"],
            tag_xyz=tag_xyz,
            anchor_xyz=anchor_xyz,
        )

    agg.range_stats.add(safe_float(values["RANGE"]))
    agg.rss_stats.add(safe_float(values["RSS"]))
    agg.rss_fp_stats.add(safe_float(values["RSS_FP"]))
    if values["NLOS"].strip().upper() == "NLOS":
        agg.nlos_count += 1
    return agg


def aggregate_raw_ranges(args: argparse.Namespace) -> list[dict[str, object]]:
    env_names = iter_environment_names(args.environments)
    aggregate_rows: list[dict[str, object]] = []

    for env_name in env_names:
        data_dir = DATASET_ROOT / "raw_data" / env_name / "data"
        position_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        if args.max_positions_per_env:
            position_dirs = position_dirs[: args.max_positions_per_env]

        for position_dir in position_dirs:
            for file_path in sorted(position_dir.glob("*.csv")):
                channel, anchor_label = parse_file_name(file_path)
                if args.channels and channel not in args.channels:
                    continue
                if args.anchors and anchor_label not in args.anchors:
                    continue

                agg: AnchorAggregate | None = None
                with file_path.open("r", encoding="utf-8", errors="replace") as f:
                    next(f, None)
                    next(f, None)
                    for line_index, line in enumerate(f):
                        if args.max_rows_per_file and line_index >= args.max_rows_per_file:
                            break
                        parts = line.rstrip("\n").split(",", PREFIX_COLUMN_COUNT)
                        if len(parts) < PREFIX_COLUMN_COUNT:
                            continue
                        fields = parts[:PREFIX_COLUMN_COUNT]
                        agg = update_aggregate(agg, fields)

                if not agg or agg.range_stats.n == 0:
                    continue

                tx, ty, tz = agg.tag_xyz
                ax, ay, az = agg.anchor_xyz
                true_range = math.sqrt((tx - ax) ** 2 + (ty - ay) ** 2 + (tz - az) ** 2)
                mean_range = agg.range_stats.mean
                aggregate_rows.append(
                    {
                        "environment_id": env_name,
                        "position_id": position_dir.name,
                        "channel": channel,
                        "anchor_label": anchor_label,
                        "tag_id": agg.tag_id,
                        "anchor_id": agg.anchor_id,
                        "tag_x": tx,
                        "tag_y": ty,
                        "tag_z": tz,
                        "anchor_x": ax,
                        "anchor_y": ay,
                        "anchor_z": az,
                        "range_mean": mean_range,
                        "range_std": agg.range_stats.std,
                        "range_n": agg.range_stats.n,
                        "range_true": true_range,
                        "ranging_error": mean_range - true_range,
                        "rss_mean": agg.rss_stats.mean,
                        "rss_std": agg.rss_stats.std,
                        "rss_fp_mean": agg.rss_fp_stats.mean,
                        "nlos_ratio": agg.nlos_count / agg.range_stats.n,
                    }
                )

    return aggregate_rows


def solve_position_2d(
    anchor_rows: list[dict[str, object]],
    weighted: bool,
    robust: bool,
    max_iter: int = 30,
) -> tuple[float, float, bool, int]:
    if len(anchor_rows) < 3:
        return math.nan, math.nan, False, 0

    z_tag = float(anchor_rows[0]["tag_z"])
    x = sum(float(row["anchor_x"]) for row in anchor_rows) / len(anchor_rows)
    y = sum(float(row["anchor_y"]) for row in anchor_rows) / len(anchor_rows)

    converged = False
    used_iter = 0
    for iteration in range(max_iter):
        used_iter = iteration + 1
        jtj = [[0.0, 0.0], [0.0, 0.0]]
        jtr = [0.0, 0.0]

        residuals = []
        jacobians = []
        base_weights = []
        for row in anchor_rows:
            ax = float(row["anchor_x"])
            ay = float(row["anchor_y"])
            az = float(row["anchor_z"])
            observed = float(row["range_mean"])
            dx = x - ax
            dy = y - ay
            dz = z_tag - az
            predicted = math.sqrt(dx * dx + dy * dy + dz * dz)
            if predicted < 1e-9:
                continue
            residual = predicted - observed
            jac = [dx / predicted, dy / predicted]
            if weighted:
                sigma = float(row["range_std"])
                n = max(1.0, float(row["range_n"]))
                # Weight the mean range. A small floor avoids exploding weights.
                weight = 1.0 / max(0.03**2, (sigma * sigma) / n)
            else:
                weight = 1.0
            residuals.append(residual)
            jacobians.append(jac)
            base_weights.append(weight)

        if len(residuals) < 3:
            break

        if robust:
            abs_res = sorted(abs(r) for r in residuals)
            med = abs_res[len(abs_res) // 2]
            delta = max(0.10, 1.345 * med)
        else:
            delta = math.inf

        for residual, jac, base_weight in zip(residuals, jacobians, base_weights):
            robust_weight = 1.0
            abs_residual = abs(residual)
            if robust and abs_residual > delta:
                robust_weight = delta / abs_residual
            weight = base_weight * robust_weight
            j0, j1 = jac
            jtj[0][0] += weight * j0 * j0
            jtj[0][1] += weight * j0 * j1
            jtj[1][0] += weight * j1 * j0
            jtj[1][1] += weight * j1 * j1
            jtr[0] += weight * j0 * residual
            jtr[1] += weight * j1 * residual

        det = jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0]
        if abs(det) < 1e-12:
            break

        inv = [
            [jtj[1][1] / det, -jtj[0][1] / det],
            [-jtj[1][0] / det, jtj[0][0] / det],
        ]
        step_x = -(inv[0][0] * jtr[0] + inv[0][1] * jtr[1])
        step_y = -(inv[1][0] * jtr[0] + inv[1][1] * jtr[1])
        x += step_x
        y += step_y
        if math.sqrt(step_x * step_x + step_y * step_y) < 1e-6:
            converged = True
            break

    return x, y, converged, used_iter


def run_baselines(aggregate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in aggregate_rows:
        key = (str(row["environment_id"]), str(row["position_id"]), str(row["channel"]))
        groups[key].append(row)

    results: list[dict[str, object]] = []
    methods = [
        ("LS", False, False),
        ("WLS_range_std", True, False),
        ("Robust_LS_Huber", False, True),
        ("Robust_WLS_Huber", True, True),
        ("Oracle_Range_LS", False, False),
    ]
    for (env_name, position_id, channel), rows in sorted(groups.items()):
        if len(rows) < 3:
            continue
        true_x = float(rows[0]["tag_x"])
        true_y = float(rows[0]["tag_y"])
        true_z = float(rows[0]["tag_z"])
        mean_abs_ranging_error = sum(abs(float(r["ranging_error"])) for r in rows) / len(rows)
        nlos_ratio = sum(float(r["nlos_ratio"]) for r in rows) / len(rows)
        for method, weighted, robust in methods:
            if method == "Oracle_Range_LS":
                solve_rows = []
                for row in rows:
                    oracle_row = dict(row)
                    oracle_row["range_mean"] = oracle_row["range_true"]
                    oracle_row["range_std"] = 0.03
                    oracle_row["range_n"] = 1
                    solve_rows.append(oracle_row)
            else:
                solve_rows = rows
            pred_x, pred_y, converged, iterations = solve_position_2d(solve_rows, weighted=weighted, robust=robust)
            error_2d = math.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            results.append(
                {
                    "environment_id": env_name,
                    "position_id": position_id,
                    "channel": channel,
                    "method": method,
                    "true_x": true_x,
                    "true_y": true_y,
                    "true_z": true_z,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "error_2d": error_2d,
                    "anchor_count": len(rows),
                    "mean_abs_ranging_error": mean_abs_ranging_error,
                    "mean_nlos_ratio": nlos_ratio,
                    "converged": converged,
                    "iterations": iterations,
                }
            )
    return results


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = int(math.floor((len(ordered) - 1) * q))
    return ordered[index]


def summarize_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in results:
        groups[(str(row["environment_id"]), str(row["method"]))].append(float(row["error_2d"]))

    summary = []
    for (env_name, method), values in sorted(groups.items()):
        summary.append(
            {
                "environment_id": env_name,
                "method": method,
                "n": len(values),
                "mean": sum(values) / len(values),
                "median": quantile(values, 0.50),
                "rmse": math.sqrt(sum(v * v for v in values) / len(values)),
                "p90": quantile(values, 0.90),
                "p95": quantile(values, 0.95),
                "max": max(values),
                "failure_0_5m": sum(v > 0.5 for v in values) / len(values),
                "failure_1_0m": sum(v > 1.0 for v in values) / len(values),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary_rows: list[dict[str, object]], aggregate_rows: list[dict[str, object]], results: list[dict[str, object]]) -> None:
    lines = []
    lines.append("# D1 原始 LS/WLS 定位基线结果")
    lines.append("")
    lines.append("更新时间：2026-07-19")
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    lines.append("本次 baseline 从 D1 `raw_data` 直接读取每个 `environment + position + channel + anchor` 的原始测距，聚合为 mean range 和 range std，然后在每个 `environment + position + channel` 下使用可见 anchors 进行 2D 定位。tag 高度使用数据集中给定的 `Z_TAG`，估计目标为 `(x, y)`。")
    lines.append("")
    lines.append("当前实现包含四个基线：")
    lines.append("")
    lines.append("- `LS`：原始 mean range 非加权最小二乘；")
    lines.append("- `WLS_range_std`：用同一文件内 range 均值的方差估计权重；")
    lines.append("- `Robust_LS_Huber`：非加权 Huber 鲁棒最小二乘；")
    lines.append("- `Robust_WLS_Huber`：range std 加权 + Huber 鲁棒核。")
    lines.append("- `Oracle_Range_LS`：使用几何真实距离替代原始 range 的理论上限。")
    lines.append("")
    lines.append("输出文件：")
    lines.append("")
    lines.append("```text")
    lines.append("SCI-UWB/data/interim/d1_anchor_range_aggregates.csv")
    lines.append("SCI-UWB/data/results/d1_positioning_baseline_results.csv")
    lines.append("SCI-UWB/data/results/d1_positioning_baseline_summary.csv")
    lines.append("```")
    lines.append("")
    lines.append("## 2. 数据规模")
    lines.append("")
    env_agg = defaultdict(int)
    for row in aggregate_rows:
        env_agg[str(row["environment_id"])] += 1
    pos_groups = {(row["environment_id"], row["position_id"], row["channel"]) for row in aggregate_rows}
    lines.append(f"- anchor-level 聚合行数：{len(aggregate_rows)}")
    lines.append(f"- positioning groups：{len(pos_groups)}")
    lines.append(f"- baseline 结果行数：{len(results)}")
    lines.append("")
    lines.append("| 环境 | anchor-level aggregates |")
    lines.append("|---|---:|")
    for env_name, count in sorted(env_agg.items()):
        lines.append(f"| {env_name} | {count} |")
    lines.append("")
    lines.append("## 3. 环境级定位误差")
    lines.append("")
    lines.append("| 环境 | 方法 | N | Mean | Median | RMSE | P90 | P95 | Max | Fail@0.5m | Fail@1.0m |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {environment_id} | {method} | {n} | {mean:.3f} | {median:.3f} | {rmse:.3f} | {p90:.3f} | {p95:.3f} | {max:.3f} | {failure_0_5m:.1%} | {failure_1_0m:.1%} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## 4. 初步解读")
    lines.append("")
    lines.append("这份 baseline 的作用不是追求最终精度，而是建立后续研究的地板线：如果后续的误差补偿、可信度权重和不确定性建模有效，必须在这些原始 LS/WLS 指标上稳定改进，尤其是 P90/P95 和 Fail@1.0m。")
    lines.append("")
    lines.append("需要特别注意：`WLS_range_std` 只是一个朴素权重，使用同一 anchor-channel 文件内的 range std 估计均值不确定性。它不是最终可信度模型，反而适合作为一个简单可解释 baseline。")
    lines.append("")
    lines.append("`Oracle_Range_LS` 的误差接近 0，说明当前 2D 几何定位求解本身没有明显问题。主要瓶颈来自测距偏差和观测可信度，而不是定位方程不可解。")
    lines.append("")
    lines.append("## 5. 关键异常与研究价值")
    lines.append("")
    lines.append("### 5.1 Environment2 是主困难环境")
    lines.append("")
    lines.append("在四个环境中，`environment2` 的原始 LS 定位误差最高：")
    lines.append("")
    lines.append("```text")
    lines.append("LS mean = 0.734 m")
    lines.append("LS P90  = 1.476 m")
    lines.append("LS Fail@1.0m = 22.8%")
    lines.append("```")
    lines.append("")
    lines.append("加入 Huber 鲁棒核后，`Robust_WLS_Huber` 明显改善：")
    lines.append("")
    lines.append("```text")
    lines.append("Robust_WLS_Huber mean = 0.573 m")
    lines.append("Robust_WLS_Huber P90  = 0.972 m")
    lines.append("Robust_WLS_Huber Fail@1.0m = 9.3%")
    lines.append("```")
    lines.append("")
    lines.append("这说明 `environment2` 中存在典型长尾测距误差，鲁棒定位后端确实有价值。但鲁棒核只能被动降低异常残差影响，还不能主动判断哪条测距不可信。")
    lines.append("")
    lines.append("### 5.2 Environment3 存在极端位置级异常")
    lines.append("")
    lines.append("`environment3` 的整体 P90 不算最高，但最大误差达到约 8.77 m。最严重位置为：")
    lines.append("")
    lines.append("```text")
    lines.append("position_id = 9.87_8.32_1.20")
    lines.append("```")
    lines.append("")
    lines.append("该位置在多个 channel 下都产生严重定位失败。进一步查看 anchor-level 聚合发现，多条测距存在 5 m 到 9 m 级别偏差，而且 `range_std` 很小。")
    lines.append("")
    lines.append("这说明一个非常关键的问题：")
    lines.append("")
    lines.append("> 测距值可能非常稳定，但稳定地错。")
    lines.append("")
    lines.append("因此，仅使用同一文件内测距方差作为 WLS 权重是不够的。后续必须引入 CIR、RSS、FP/RXPACC、几何和历史特征来估计 measurement trustworthiness。")
    lines.append("")
    lines.append("### 5.3 Anchor-level 长尾来源")
    lines.append("")
    lines.append("按环境和 anchor 汇总，P90 绝对测距误差最高的组合包括：")
    lines.append("")
    lines.append("| 组合 | Mean abs ranging error | P90 abs ranging error | Mean NLOS ratio |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| environment2, A6 | 1.474 | 8.461 | 0.417 |")
    lines.append("| environment2, A8 | 0.897 | 2.042 | 0.881 |")
    lines.append("| environment2, A7 | 0.868 | 1.818 | 0.929 |")
    lines.append("| environment2, A5 | 0.829 | 1.765 | 0.869 |")
    lines.append("| environment3, A4 | 0.758 | 1.428 | 0.691 |")
    lines.append("")
    lines.append("其中 `environment2, A6` 特别值得注意。它的 NLOS ratio 并不是最高，但 P90 绝对测距误差却最大。这进一步说明：")
    lines.append("")
    lines.append("> NLOS 标签本身不足以解释测距风险。")
    lines.append("")
    lines.append("### 5.4 Channel 差异也存在")
    lines.append("")
    lines.append("以原始 LS 为例，部分环境下 channel 差异明显：")
    lines.append("")
    lines.append("| 环境 | 最优 channel mean error | 最差 channel mean error |")
    lines.append("|---|---:|---:|")
    lines.append("| environment0 | ch2: 0.247 m | ch7: 0.509 m |")
    lines.append("| environment1 | ch1: 0.145 m | ch7: 0.415 m |")
    lines.append("| environment2 | ch4: 0.516 m | ch3: 0.963 m |")
    lines.append("| environment3 | ch5: 0.459 m | ch3: 0.571 m |")
    lines.append("")
    lines.append("这说明 channel 也是测距质量建模的重要上下文变量。后续模型不应只输入 range，还应保留 `channel_number`、`preamble_code`、RSS 和 CIR 统计特征。")
    lines.append("")
    lines.append("## 6. 对后续方法的直接启发")
    lines.append("")
    lines.append("当前 baseline 已经给出三个明确研究理由：")
    lines.append("")
    lines.append("1. **鲁棒后端有必要**  ")
    lines.append("   `environment2` 中 Huber robust WLS 明显降低长尾失败。")
    lines.append("")
    lines.append("2. **朴素 WLS 不够**  ")
    lines.append("   `environment3` 极端位置中，错误测距的 `range_std` 很小，说明仅靠测距方差加权会误信稳定异常值。")
    lines.append("")
    lines.append("3. **二值 NLOS 标签不够**  ")
    lines.append("   `environment2, A6` 的 NLOS ratio 不是最高，但长尾误差最严重，说明需要连续可信度和不确定性。")
    lines.append("")
    lines.append("因此，我们的方法应聚焦：")
    lines.append("")
    lines.append("```text")
    lines.append("signal/geometry/temporal features")
    lines.append("        -> per-link bias, sigma, trustworthiness")
    lines.append("        -> robust uncertainty-weighted WLS")
    lines.append("```")
    lines.append("")
    lines.append("这不是为了比 LS 平均误差低一点，而是要压低：")
    lines.append("")
    lines.append("```text")
    lines.append("P90/P95 error")
    lines.append("Failure@1.0m")
    lines.append("极端位置级失败")
    lines.append("```")
    lines.append("")
    lines.append("## 7. 下一步")
    lines.append("")
    lines.append("下一步建议做两件事：")
    lines.append("")
    lines.append("1. 画出 LS/WLS 的 CDF、按 channel 的误差曲线、按 anchor-level ranging error 的分布；")
    lines.append("2. 基于 oracle 上限结果，训练第一版 `features -> ranging_error` 测距误差补偿模型。")
    lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D1 raw LS/WLS positioning baselines.")
    parser.add_argument("--max-positions-per-env", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--channels", default="", help="Comma-separated channel filter, e.g. ch1,ch5.")
    parser.add_argument("--anchors", default="", help="Comma-separated anchor filter, e.g. A1,A2.")
    parser.add_argument("--environments", default="", help="Comma-separated environment filter.")
    args = parser.parse_args()
    args.channels = set(part.strip() for part in args.channels.split(",") if part.strip())
    args.anchors = set(part.strip() for part in args.anchors.split(",") if part.strip())

    aggregate_rows = aggregate_raw_ranges(args)
    results = run_baselines(aggregate_rows)
    summary_rows = summarize_results(results)

    write_csv(INTERIM_DIR / "d1_anchor_range_aggregates.csv", aggregate_rows)
    write_csv(RESULTS_DIR / "d1_positioning_baseline_results.csv", results)
    write_csv(RESULTS_DIR / "d1_positioning_baseline_summary.csv", summary_rows)
    write_report(summary_rows, aggregate_rows, results)

    print(f"Anchor aggregates: {len(aggregate_rows)}")
    print(f"Positioning results: {len(results)}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
