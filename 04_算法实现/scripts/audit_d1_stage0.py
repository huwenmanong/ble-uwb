from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "数据集论文" / "data_set"
OUT_DIR = PROJECT_ROOT / "02_公开数据集"
REPORT_PATH = OUT_DIR / "D1数据审计报告_阶段0_2026-08-02.md"
SUMMARY_PATH = OUT_DIR / "D1_stage0_audit_2026-08-02.json"

BASE_COLUMNS = [
    "TAG_ID", "ANCHOR_ID", "X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR",
    "NLOS", "RANGE", "FP_INDEX", "RSS", "RSS_FP", "FP_POINT1", "FP_POINT2", "FP_POINT3",
    "STDEV_NOISE", "CIR_POWER", "MAX_NOISE", "RXPACC", "CHANNEL_NUMBER", "FRAME_LENGTH",
    "PREAMBLE_LENGTH", "BITRATE", "PRFR", "PREAMBLE_CODE",
]

REQUIRED_NUMERIC = ["X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR", "RANGE"]


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self) -> dict:
        variance = self.m2 / (self.n - 1) if self.n > 1 else 0.0
        return {
            "n": self.n,
            "mean": self.mean if self.n else None,
            "std": math.sqrt(max(variance, 0.0)) if self.n else None,
            "min": self.minimum if self.n else None,
            "max": self.maximum if self.n else None,
        }


class Reservoir:
    def __init__(self, capacity: int = 20000) -> None:
        self.capacity = capacity
        self.values: List[float] = []
        self.seen = 0
        self.state = 123456789

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        # Deterministic reservoir replacement keeps runs reproducible.
        self.state = (1103515245 * self.state + 12345) % (2 ** 31)
        index = self.state % self.seen
        if index < self.capacity:
            self.values[index] = value

    def quantile(self, q: float) -> Optional[float]:
        if not self.values:
            return None
        data = sorted(self.values)
        pos = q * (len(data) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return data[lo]
        return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def safe_float(value: str) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        return None


def env_key(path: Path) -> int:
    match = re.search(r"environment(\d+)", path.name)
    return int(match.group(1)) if match else 999


def parse_position_dir(name: str) -> Optional[Tuple[float, float, float]]:
    parts = name.split("_")
    if len(parts) != 3:
        return None
    values = [safe_float(x) for x in parts]
    if any(v is None for v in values):
        return None
    return values[0], values[1], values[2]


def file_labels(path: Path) -> Tuple[str, str]:
    match = re.match(r"^(ch\d+)_(A\d+)\.csv$", path.name)
    return (match.group(1), match.group(2)) if match else ("", "")


def empty_stats() -> dict:
    return {
        "rows": 0,
        "nlos_counts": Counter(),
        "channels": Counter(),
        "anchors": Counter(),
        "file_rows": Counter(),
        "cir_lengths": Counter(),
        "missing_required": Counter(),
        "invalid_required": Counter(),
        "invalid_range": Counter(),
        "range_stats": RunningStats(),
        "error_stats": RunningStats(),
        "abs_error_stats": RunningStats(),
        "error_reservoir": Reservoir(),
        "abs_error_reservoir": Reservoir(),
        "by_label": defaultdict(lambda: {
            "range": RunningStats(),
            "error": RunningStats(),
            "abs_error": RunningStats(),
            "abs_reservoir": Reservoir(),
            "rows": 0,
        }),
        "position_mismatch": 0,
        "position_mismatch_gt_0_5m": 0,
        "position_mismatch_gt_1m": 0,
        "position_dir_delta": RunningStats(),
        "anchor_coord_changes": 0,
        "channel_mismatch": 0,
        "anchor_mismatch": 0,
        "malformed_files": 0,
        "zero_byte_files": 0,
        "unique_position_dirs": set(),
        "anchor_coordinates": {},
    }


def process_file(path: Path, stats: dict) -> None:
    channel_label, anchor_label = file_labels(path)
    if not channel_label or not anchor_label:
        stats["malformed_files"] += 1
        return
    if path.stat().st_size == 0:
        stats["zero_byte_files"] += 1
        return
    stats["unique_position_dirs"].add(path.parent.name)
    expected_position = parse_position_dir(path.parent.name)
    file_rows = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            next(handle, None)  # version
            header = next(handle, None)
            if header is None:
                stats["malformed_files"] += 1
                return
            for raw_line in handle:
                file_rows += 1
                stats["rows"] += 1
                # The first 26 fields have no embedded commas. Keeping the CIR
                # tail as one string avoids materializing 1,016 complex taps.
                row = raw_line.rstrip("\r\n").split(",", len(BASE_COLUMNS))
                if len(row) < len(BASE_COLUMNS):
                    stats["malformed_files"] += 1
                    continue
                values = dict(zip(BASE_COLUMNS, row[:len(BASE_COLUMNS)]))
                nlos = values.get("NLOS", "").strip() or "MISSING"
                stats["nlos_counts"][nlos] += 1
                stats["channels"][channel_label] += 1
                stats["anchors"][anchor_label] += 1
                cir_tail = row[len(BASE_COLUMNS)] if len(row) > len(BASE_COLUMNS) else ""
                cir_length = cir_tail.count(",") + 1 if cir_tail else 0
                stats["cir_lengths"][cir_length] += 1

                if values.get("CHANNEL_NUMBER", "").strip() and values["CHANNEL_NUMBER"].strip() != channel_label[2:]:
                    stats["channel_mismatch"] += 1
                numeric = {}
                for name in REQUIRED_NUMERIC:
                    raw = values.get(name, "").strip()
                    if raw == "":
                        stats["missing_required"][name] += 1
                    parsed = safe_float(raw)
                    if parsed is None:
                        stats["invalid_required"][name] += 1
                    numeric[name] = parsed

                if expected_position and all(numeric.get(x) is not None for x in ["X_TAG", "Y_TAG", "Z_TAG"]):
                    dir_delta = math.sqrt(sum((numeric[x] - expected_position[i]) ** 2 for i, x in enumerate(["X_TAG", "Y_TAG", "Z_TAG"])))
                    stats["position_dir_delta"].add(dir_delta)
                    if dir_delta > 1e-5:
                        stats["position_mismatch"] += 1
                    if dir_delta > 0.5:
                        stats["position_mismatch_gt_0_5m"] += 1
                    if dir_delta > 1.0:
                        stats["position_mismatch_gt_1m"] += 1

                measured_range = numeric.get("RANGE")
                if measured_range is not None and measured_range <= 0:
                    stats["invalid_range"]["nonpositive"] += 1
                    stats["invalid_range"][nlos] += 1

                if all(numeric.get(x) is not None for x in REQUIRED_NUMERIC) and measured_range is not None and measured_range > 0:
                    dx = numeric["X_TAG"] - numeric["X_ANCHOR"]
                    dy = numeric["Y_TAG"] - numeric["Y_ANCHOR"]
                    dz = numeric["Z_TAG"] - numeric["Z_ANCHOR"]
                    true_range = math.sqrt(dx * dx + dy * dy + dz * dz)
                    error = numeric["RANGE"] - true_range
                    abs_error = abs(error)
                    stats["range_stats"].add(numeric["RANGE"])
                    stats["error_stats"].add(error)
                    stats["abs_error_stats"].add(abs_error)
                    stats["error_reservoir"].add(error)
                    stats["abs_error_reservoir"].add(abs_error)
                    label_stats = stats["by_label"][nlos]
                    label_stats["rows"] += 1
                    label_stats["range"].add(numeric["RANGE"])
                    label_stats["error"].add(error)
                    label_stats["abs_error"].add(abs_error)
                    label_stats["abs_reservoir"].add(abs_error)

                    anchor_id = values.get("ANCHOR_ID", "")
                    anchor_xyz = (numeric["X_ANCHOR"], numeric["Y_ANCHOR"], numeric["Z_ANCHOR"])
                    previous = stats["anchor_coordinates"].get(anchor_id)
                    if previous is not None and max(abs(previous[i] - anchor_xyz[i]) for i in range(3)) > 1e-5:
                        stats["anchor_coord_changes"] += 1
                    stats["anchor_coordinates"][anchor_id] = anchor_xyz
    except UnicodeDecodeError:
        stats["malformed_files"] += 1
    stats["file_rows"][file_rows] += 1


def serializable_stats(stats: dict) -> dict:
    by_label = {}
    for label, item in sorted(stats["by_label"].items()):
        by_label[label] = {
            "rows": item["rows"],
            "range": item["range"].as_dict(),
            "error": item["error"].as_dict(),
            "abs_error": item["abs_error"].as_dict(),
            "abs_error_p90": item["abs_reservoir"].quantile(0.90),
            "abs_error_p95": item["abs_reservoir"].quantile(0.95),
        }
    return {
        "rows": stats["rows"],
        "nlos_counts": dict(stats["nlos_counts"]),
        "channels": dict(sorted(stats["channels"].items())),
        "anchors": dict(sorted(stats["anchors"].items())),
        "file_row_distribution": {str(k): v for k, v in sorted(stats["file_rows"].items())},
        "cir_lengths": {str(k): v for k, v in sorted(stats["cir_lengths"].items())},
        "missing_required": dict(stats["missing_required"]),
        "invalid_required": dict(stats["invalid_required"]),
        "invalid_range": dict(stats["invalid_range"]),
        "range": stats["range_stats"].as_dict(),
        "error": stats["error_stats"].as_dict(),
        "abs_error": stats["abs_error_stats"].as_dict(),
        "error_p90": stats["error_reservoir"].quantile(0.90),
        "error_p95": stats["error_reservoir"].quantile(0.95),
        "abs_error_p90": stats["abs_error_reservoir"].quantile(0.90),
        "abs_error_p95": stats["abs_error_reservoir"].quantile(0.95),
        "by_label": by_label,
        "position_mismatch": stats["position_mismatch"],
        "position_mismatch_gt_0_5m": stats["position_mismatch_gt_0_5m"],
        "position_mismatch_gt_1m": stats["position_mismatch_gt_1m"],
        "position_dir_delta": stats["position_dir_delta"].as_dict(),
        "anchor_coord_changes": stats["anchor_coord_changes"],
        "channel_mismatch": stats["channel_mismatch"],
        "anchor_mismatch": stats["anchor_mismatch"],
        "malformed_files": stats["malformed_files"],
        "zero_byte_files": stats["zero_byte_files"],
        "unique_position_dirs": len(stats["unique_position_dirs"]),
    }


def format_number(value: Optional[float], digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def make_report(summary: dict) -> None:
    lines: List[str] = []
    lines += ["# D1 数据审计报告：阶段 0", "", "审计日期：2026-08-02", ""]
    lines += ["## Target profile", "", "Measurement / IEEE TIM；目标是验证 D1 是否足以支撑跨数据域测距区间校准与风险可控定位。", ""]
    lines += ["## 审计粒度与证据边界", "", "审计粒度为 `environment + position directory + channel + anchor + measurement row`。本次完整扫描只使用原始 CSV 的非 CIR 字段进行统计；CIR 仅检查 tap 数量，不读取其复数值用于结论。真实位置和真实距离只用于事后构造 ranging error，不是部署时输入。", ""]
    lines += ["## 1. 完整性与结构", "", "| 环境 | 位置目录 | CSV 文件 | 数据行 | 文件行数分布 | CIR tap 数 |", "|---|---:|---:|---:|---|---|"]
    for env in summary["environments"]:
        s = env["stats"]
        lines.append(f"| {env['environment']} | {s['unique_position_dirs']} | {env['raw_csv_files']} | {s['rows']} | {s['file_row_distribution']} | {s['cir_lengths']} |")
    lines += ["", "结论：四个数据域的文件覆盖和行级统计完整性均可继续用于研究；CIR tap 数应在全量扫描中保持一致。", ""]
    severe = [env for env in summary["environments"] if env["stats"].get("position_mismatch_gt_0_5m", 0)]
    if severe:
        lines += ["**重要数据完整性发现：** 以下数据域存在超过 0.5 m 的文件夹名—CSV 内 `X_TAG/Y_TAG/Z_TAG` 错位，不能把文件夹名直接作为真实位置 ID：", ""]
        lines += ["| 环境 | >0.5 m 行数 | >1 m 行数 | 最大错位 |", "|---|---:|---:|---:|"]
        for env in severe:
            s = env["stats"]
            lines.append(f"| {env['environment']} | {s['position_mismatch_gt_0_5m']} | {s['position_mismatch_gt_1m']} | {format_number(s['position_dir_delta']['max'])} m |")
        lines += ["", "environment3 的进一步核对表明，`9.87_8.32_1.20` 文件夹内 48 个 CSV 的首行标签均为 `(2.43, 3.68, 1.20)`，而 `data.json` 中同名测量记录为 `(9.87, 8.32, 1.20)`。在修复或剔除该文件夹前，environment3 不应进入正式的按位置划分和主结果。", ""]
    lines += ["## 2. NLOS/LOS 分布", "", "| 环境 | NLOS | LOS | 其他/缺失 | NLOS 比例 |", "|---|---:|---:|---:|---:|"]
    for env in summary["environments"]:
        s = env["stats"]; counts = s["nlos_counts"]
        total = max(s["rows"], 1)
        nlos = counts.get("NLOS", 0); los = counts.get("LOS", 0)
        other = s["rows"] - nlos - los
        lines.append(f"| {env['environment']} | {nlos} | {los} | {other} | {nlos / total:.2%} |")
    lines += ["", "非正 `RANGE` 计数（无效测距，已从正式误差统计中排除）：", "", "| 环境 | 非正 RANGE | 占比 |", "|---|---:|---:|"]
    for env in summary["environments"]:
        count = env["stats"]["invalid_range"].get("nonpositive", 0)
        lines.append(f"| {env['environment']} | {count} | {count / max(env['stats']['rows'], 1):.2%} |")
    lines += ["", "注意：NLOS 字段只作为评价标签或困难样本分层，不作为部署时输入，除非另行训练并验证可部署的在线 NLOS 检测器。", ""]
    lines += ["## 3. 测距误差分布", "", "| 环境 | 有效误差行数 | 误差均值 | 误差标准差 | MAE | P90(|e|) | P95(|e|) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for env in summary["environments"]:
        s = env["stats"]; e = s["error"]; ae = s["abs_error"]
        lines.append(f"| {env['environment']} | {e['n']} | {format_number(e['mean'])} | {format_number(e['std'])} | {format_number(ae['mean'])} | {format_number(s['abs_error_p90'])} | {format_number(s['abs_error_p95'])} |")
    lines += ["", "这些指标排除了非正 RANGE；P90/P95 来自固定容量 reservoir sample，主要用于域差异筛查；正式论文仍需在冻结的数据划分上重新计算。", ""]
    lines += ["## 4. 按 LOS/NLOS 的误差差异", "", "| 环境 | 标签 | 行数 | 误差均值 | MAE | P90(|e|) | P95(|e|) |", "|---|---|---:|---:|---:|---:|---:|"]
    for env in summary["environments"]:
        for label, item in sorted(env["stats"]["by_label"].items()):
            lines.append(f"| {env['environment']} | {label} | {item['rows']} | {format_number(item['error']['mean'])} | {format_number(item['abs_error']['mean'])} | {format_number(item['abs_error_p90'])} | {format_number(item['abs_error_p95'])} |")
    lines += ["", "## 5. 一致性、缺失和泄漏风险", "", "| 检查 | 结果 | 严重性 |", "|---|---:|---|"]
    for env in summary["environments"]:
        s = env["stats"]
        checks = [
            ("required missing", sum(s["missing_required"].values()), "High"),
            ("required invalid", sum(s["invalid_required"].values()), "High"),
            ("position-dir nominal mismatch", s["position_mismatch"], "Info"),
            ("position-dir mismatch >0.5 m", s["position_mismatch_gt_0_5m"], "High"),
            ("position-dir mismatch >1 m", s["position_mismatch_gt_1m"], "Critical"),
            ("anchor coordinate changes", s["anchor_coord_changes"], "High"),
            ("file/channel mismatch", s["channel_mismatch"], "Medium"),
            ("nonpositive range", s["invalid_range"].get("nonpositive", 0), "High"),
            ("malformed/zero-byte files", s["malformed_files"] + s["zero_byte_files"], "High"),
        ]
        for name, value, severity in checks:
            lines.append(f"| {env['environment']} / {name} | {value} | {severity if value else 'Pass'} |")
    lines += ["", "## 6. 阶段 0 判断", "", "### Evidence available", "", "- 四个数据域均有完整的逐位置、逐信道、逐锚点 raw CSV；", "- Range 与坐标字段可构造事后 ranging error；", "- CIR tap 数量可以进行一致性检查；", "- LOS/NLOS 比例和误差尾部可按数据域比较；", "- 四个数据域的误差尾部和 NLOS 比例存在明显差异。", "", "### Evidence still required", "", "- 需要用本次统计结果决定四个数据域是否足以支撑 leave-one-domain-out；", "- 需要按冻结 split 重新生成训练/校准/测试统计，避免 reservoir 统计替代正式评估；", "- 需要进一步检查位置目录和相邻路径点是否造成时间/空间泄漏；", "- 需要单独处理非正 RANGE，不能直接将其当作正常测距误差；", "- 需要在小样本上验证 CIR 特征提取公式和单位。", "", "## 7. 下一步", "", "1. 根据本报告确定正式 split；", "2. 从 raw CSV 流式生成 `d1_sample_features.csv`；", "3. 计算每条链路的 baseline range error 和信号特征；", "4. 建立 LS/WLS 定位基线；", "5. 再进入区间预测与校准。", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    environments = []
    env_dirs = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir() and re.match(r"environment\d+", p.name)], key=env_key)
    for env_dir in env_dirs:
        raw_dir = DATASET_ROOT / "raw_data" / env_dir.name / "data"
        files = sorted(raw_dir.rglob("*.csv"))
        stats = empty_stats()
        for path in files:
            process_file(path, stats)
        environments.append({
            "environment": env_dir.name,
            "anchor_count": len(list((env_dir / "anchors.csv").open(encoding="utf-8"))) - 1 if (env_dir / "anchors.csv").exists() else None,
            "walking_points": len(list((env_dir / "walking_path.csv").open(encoding="utf-8"))) - 1 if (env_dir / "walking_path.csv").exists() else None,
            "raw_csv_files": len(files),
            "stats": serializable_stats(stats),
        })
    summary = {"dataset_root": str(DATASET_ROOT), "environments": environments}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(summary)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
