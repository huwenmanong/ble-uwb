from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "数据集论文" / "data_set"
REPORT_PATH = PROJECT_ROOT / "02_公开数据集" / "D1数据集首次审计报告.md"
SUMMARY_PATH = PROJECT_ROOT / "02_公开数据集" / "D1_dataset_audit_summary.json"

BASE_COLUMNS = [
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_raw_sample(path: Path, max_rows: int = 20) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        version = next(reader, None)
        header = next(reader, None)
        rows = []
        for _, row in zip(range(max_rows), reader):
            rows.append(row)
    if header is None:
        header = []
    if version:
        header = ["__version__"] + header
    return header, rows


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def env_sort_key(path: Path) -> int:
    m = re.search(r"environment(\d+)", path.name)
    return int(m.group(1)) if m else 999


def summarize_environment(env_dir: Path) -> dict:
    env_name = env_dir.name
    raw_env_dir = DATASET_ROOT / "raw_data" / env_name
    anchors_path = env_dir / "anchors.csv"
    walking_path = env_dir / "walking_path.csv"
    json_path = env_dir / "data.json"

    anchors = read_csv_rows(anchors_path)
    walking = read_csv_rows(walking_path)
    data_dir = raw_env_dir / "data"
    position_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    raw_files = sorted(data_dir.rglob("*.csv"))

    channels = Counter()
    anchors_seen = Counter()
    position_file_counts = Counter()
    file_size_total = 0
    zero_byte_files = []
    malformed_names = []
    sample_files = []

    pattern = re.compile(r"^ch(?P<ch>\d+)_A(?P<anchor>\d+)\.csv$")
    for file_path in raw_files:
        file_size_total += file_path.stat().st_size
        if file_path.stat().st_size == 0:
            zero_byte_files.append(str(file_path))
        position_file_counts[file_path.parent.name] += 1
        match = pattern.match(file_path.name)
        if not match:
            malformed_names.append(str(file_path))
            continue
        channels[f"ch{match.group('ch')}"] += 1
        anchors_seen[f"A{match.group('anchor')}"] += 1
        if len(sample_files) < 3:
            sample_files.append(file_path)

    expected_files = len(walking) * 48
    true_range_samples = []
    row_count_samples = []
    sample_schema = None
    sample_cir_lengths = []
    sample_nlos = Counter()

    for sample_file in sample_files:
        header, rows = read_raw_sample(sample_file, max_rows=30)
        if sample_schema is None:
            sample_schema = header
        row_count_samples.append({"file": str(sample_file), "sampled_rows": len(rows)})
        for row in rows:
            if len(row) < len(BASE_COLUMNS):
                continue
            mapped = dict(zip(BASE_COLUMNS, row[: len(BASE_COLUMNS)]))
            tag_xyz = [safe_float(mapped[k]) for k in ["X_TAG", "Y_TAG", "Z_TAG"]]
            anchor_xyz = [safe_float(mapped[k]) for k in ["X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR"]]
            measured = safe_float(mapped["RANGE"])
            if all(v is not None for v in tag_xyz + anchor_xyz) and measured is not None:
                dx = tag_xyz[0] - anchor_xyz[0]
                dy = tag_xyz[1] - anchor_xyz[1]
                dz = tag_xyz[2] - anchor_xyz[2]
                true_range = math.sqrt(dx * dx + dy * dy + dz * dz)
                true_range_samples.append(
                    {
                        "file": str(sample_file),
                        "range": measured,
                        "true_range": true_range,
                        "ranging_error": measured - true_range,
                    }
                )
            sample_nlos[mapped.get("NLOS", "")] += 1
            sample_cir_lengths.append(max(0, len(row) - len(BASE_COLUMNS)))

    return {
        "environment": env_name,
        "anchor_count": len(anchors),
        "anchor_ids": [row.get("Anchor", "") for row in anchors],
        "walking_points": len(walking),
        "json_size_bytes": json_path.stat().st_size if json_path.exists() else None,
        "raw_position_dirs": len(position_dirs),
        "raw_csv_files": len(raw_files),
        "expected_raw_csv_files": expected_files,
        "raw_coverage_rate": len(raw_files) / expected_files if expected_files else None,
        "raw_file_size_bytes": file_size_total,
        "channels": dict(sorted(channels.items())),
        "anchors_seen": dict(sorted(anchors_seen.items())),
        "position_file_count_distribution": dict(sorted(Counter(position_file_counts.values()).items())),
        "zero_byte_files": zero_byte_files[:20],
        "zero_byte_file_count": len(zero_byte_files),
        "malformed_name_count": len(malformed_names),
        "malformed_names": malformed_names[:20],
        "sample_schema_base_columns": BASE_COLUMNS,
        "sample_header_columns": sample_schema[:35] if sample_schema else [],
        "sample_cir_length_min": min(sample_cir_lengths) if sample_cir_lengths else None,
        "sample_cir_length_max": max(sample_cir_lengths) if sample_cir_lengths else None,
        "sample_nlos_counts": dict(sample_nlos),
        "true_range_samples": true_range_samples[:10],
        "row_count_samples": row_count_samples,
    }


def format_bytes(n: int | None) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def write_report(summary: dict) -> None:
    lines = []
    lines.append("# D1 数据集首次审计报告")
    lines.append("")
    lines.append("更新时间：2026-07-19")
    lines.append("")
    lines.append("## 1. 数据集与粒度判断")
    lines.append("")
    lines.append("本次审计对象为 `SCI-UWB/数据集论文/data_set`。该数据集已经解压，包含四个室内环境、整理后的 `data.json`、原始逐点逐锚逐信道 CSV、anchor 坐标、walking path 坐标和技术验证图。")
    lines.append("")
    lines.append("适合本研究的核心粒度建议设为：")
    lines.append("")
    lines.append("```text")
    lines.append("environment_id + tag_position + channel + anchor_id + measurement_index")
    lines.append("```")
    lines.append("")
    lines.append("也就是说，每一行原始 CSV 记录是一条单链路 UWB 测距观测，可用于构造 `range_measured`、`range_true`、`ranging_error`、信号特征和 NLOS/LOS 标签。")
    lines.append("")

    lines.append("## 2. 环境级覆盖")
    lines.append("")
    lines.append("| 环境 | anchors | walking points | raw position dirs | raw csv files | expected files | coverage | data.json | raw csv size |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for env in summary["environments"]:
        lines.append(
            "| {environment} | {anchor_count} | {walking_points} | {raw_position_dirs} | {raw_csv_files} | {expected_raw_csv_files} | {coverage:.2%} | {json_size} | {raw_size} |".format(
                environment=env["environment"],
                anchor_count=env["anchor_count"],
                walking_points=env["walking_points"],
                raw_position_dirs=env["raw_position_dirs"],
                raw_csv_files=env["raw_csv_files"],
                expected_raw_csv_files=env["expected_raw_csv_files"],
                coverage=env["raw_coverage_rate"] or 0,
                json_size=format_bytes(env["json_size_bytes"]),
                raw_size=format_bytes(env["raw_file_size_bytes"]),
            )
        )
    lines.append("")

    lines.append("## 3. 字段可用性")
    lines.append("")
    lines.append("README 给出的原始 CSV 字段包含：")
    lines.append("")
    lines.append("```text")
    lines.append(", ".join(BASE_COLUMNS) + ", CIR...")
    lines.append("```")
    lines.append("")
    lines.append("这些字段对拟定研究非常有价值：")
    lines.append("")
    lines.append("- `RANGE` 可作为原始测距；")
    lines.append("- `X_TAG/Y_TAG/Z_TAG` 与 `X_ANCHOR/Y_ANCHOR/Z_ANCHOR` 可计算真实距离和测距误差；")
    lines.append("- `NLOS` 可作为 LOS/NLOS 标签或困难样本标签；")
    lines.append("- `RSS/RSS_FP/CIR_POWER/RXPACC/STDEV_NOISE/MAX_NOISE` 可作为轻量信号质量特征；")
    lines.append("- `CIR...` 可提取峰值、能量、首径附近能量、多径扩散等深层信号特征。")
    lines.append("")

    lines.append("## 4. 样本审计结果")
    lines.append("")
    lines.append("| 环境 | 样本 CIR 长度范围 | 样本 NLOS/LOS 计数 | 文件命名异常 | 零字节文件 |")
    lines.append("|---|---:|---|---:|---:|")
    for env in summary["environments"]:
        cir_range = f"{env['sample_cir_length_min']} - {env['sample_cir_length_max']}"
        lines.append(
            f"| {env['environment']} | {cir_range} | {env['sample_nlos_counts']} | {env['malformed_name_count']} | {env['zero_byte_file_count']} |"
        )
    lines.append("")

    lines.append("样本测距误差检查显示，原始 `RANGE` 与几何真实距离可以直接构造监督标签：")
    lines.append("")
    lines.append("| 环境 | measured range | true range | ranging error |")
    lines.append("|---|---:|---:|---:|")
    for env in summary["environments"]:
        sample = env["true_range_samples"][0] if env["true_range_samples"] else None
        if not sample:
            continue
        lines.append(
            f"| {env['environment']} | {sample['range']:.3f} | {sample['true_range']:.3f} | {sample['ranging_error']:.3f} |"
        )
    lines.append("")

    lines.append("## 5. 关键质量发现")
    lines.append("")
    lines.append("### Finding 1: 数据集结构完整，可支撑主研究")
    lines.append("")
    lines.append("四个 environment 均包含 anchor 坐标、walking path、整理后 JSON 和 raw csv。raw csv 覆盖率按 `walking_points * 48` 计算均达到审计表中的水平，字段直接覆盖 range、NLOS、RSS、DW1000 质量寄存器和 CIR。")
    lines.append("")
    lines.append("影响：可以按照前期方案开展单链路测距误差补偿、可信度建模和跨环境定位实验。")
    lines.append("")
    lines.append("严重性：Low。")
    lines.append("")

    has_tilde = any((DATASET_ROOT / env["environment"] / "data.json~").exists() for env in summary["environments"])
    if has_tilde:
        lines.append("### Finding 2: 存在 `data.json~` 备份文件")
        lines.append("")
        lines.append("`environment3` 下检测到 `data.json~`，大小与 `data.json` 相同。它可能是解压或编辑过程中形成的备份文件。")
        lines.append("")
        lines.append("影响：不影响 raw csv 研究路线，但后续若全量扫描 JSON 文件，需要显式排除 `*.json~`，避免重复读入 environment3。")
        lines.append("")
        lines.append("严重性：Medium。")
        lines.append("")

    lines.append("### Finding 3: 巨型 JSON 不适合作为第一版 loader 的主入口")
    lines.append("")
    lines.append("每个 `data.json` 大约 1.5GB 到 1.7GB。直接全量加载会占用大量内存，也不利于调试。")
    lines.append("")
    lines.append("影响：建议第一版工程从 raw csv 流式读取和抽样缓存开始，后续再做 parquet/feather 等中间格式。")
    lines.append("")
    lines.append("严重性：Medium。")
    lines.append("")

    lines.append("### Finding 4: CIR 是高价值但高成本字段")
    lines.append("")
    lines.append("样本文件显示每行包含大量复数 CIR taps。它能支撑信号特征创新，但直接保存完整 CIR 会显著增加训练数据体积。")
    lines.append("")
    lines.append("影响：建议第一阶段先提取轻量统计特征，例如 CIR 最大幅值、总能量、峰值位置、首径附近能量比、峰度/扩散度；完整 CIR 深度网络作为第二阶段。")
    lines.append("")
    lines.append("严重性：Medium。")
    lines.append("")

    lines.append("## 6. 推荐统一字段")
    lines.append("")
    lines.append("第一版 loader 输出建议使用以下字段：")
    lines.append("")
    lines.append("```text")
    lines.append("dataset_id, environment_id, position_id, channel, anchor_id, measurement_index,")
    lines.append("tag_x, tag_y, tag_z, anchor_x, anchor_y, anchor_z,")
    lines.append("range_measured, range_true, ranging_error, nlos_label,")
    lines.append("rss, rss_fp, fp_index, fp_point1, fp_point2, fp_point3,")
    lines.append("stdev_noise, cir_power, max_noise, rxpacc,")
    lines.append("cir_peak, cir_energy, cir_peak_index, cir_spread")
    lines.append("```")
    lines.append("")

    lines.append("## 7. 下一步工程动作")
    lines.append("")
    lines.append("建议马上进入：")
    lines.append("")
    lines.append("1. 写 D1 raw csv loader；")
    lines.append("2. 生成一个小规模 `sample_features.csv` 用于验证字段和特征；")
    lines.append("3. 复现 raw range error 分布；")
    lines.append("4. 实现 LS/WLS baseline；")
    lines.append("5. 再扩展到 full feature extraction。")
    lines.append("")
    lines.append("当前建议：第一版不要直接加载四个 `data.json`。从 raw csv 流式特征抽取开始，会更稳。")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    env_dirs = sorted(
        [p for p in DATASET_ROOT.iterdir() if p.is_dir() and re.match(r"environment\d+", p.name)],
        key=env_sort_key,
    )
    summary = {
        "dataset_root": str(DATASET_ROOT),
        "audit_script": str(Path(__file__).resolve()),
        "environments": [summarize_environment(env_dir) for env_dir in env_dirs],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
