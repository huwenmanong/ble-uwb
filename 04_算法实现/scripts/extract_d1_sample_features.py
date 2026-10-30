from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "数据集论文" / "data_set"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_PATH = OUTPUT_DIR / "d1_sample_features.csv"

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

OUTPUT_COLUMNS = [
    "dataset_id",
    "environment_id",
    "position_id",
    "channel",
    "anchor_label",
    "measurement_index",
    "tag_id",
    "anchor_id",
    "tag_x",
    "tag_y",
    "tag_z",
    "anchor_x",
    "anchor_y",
    "anchor_z",
    "range_measured",
    "range_true",
    "ranging_error",
    "nlos_label",
    "rss",
    "rss_fp",
    "fp_index",
    "fp_point1",
    "fp_point2",
    "fp_point3",
    "stdev_noise",
    "cir_power",
    "max_noise",
    "rxpacc",
    "channel_number",
    "frame_length",
    "preamble_length",
    "bitrate",
    "prfr",
    "preamble_code",
    "cir_len",
    "cir_peak",
    "cir_peak_index",
    "cir_energy",
    "cir_mean_magnitude",
    "cir_spread",
    "cir_top5_energy_ratio",
    "cir_fp_window_energy_ratio",
]


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def safe_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return -1


def parse_complex_tap(value: str) -> complex | None:
    text = value.strip()
    if not text:
        return None
    try:
        return complex(text)
    except Exception:
        return None


def cir_features(cir_values: list[str], fp_index: int) -> dict[str, float | int]:
    magnitudes = []
    for value in cir_values:
        tap = parse_complex_tap(value)
        if tap is not None:
            magnitudes.append(abs(tap))

    if not magnitudes:
        return {
            "cir_len": 0,
            "cir_peak": math.nan,
            "cir_peak_index": -1,
            "cir_energy": math.nan,
            "cir_mean_magnitude": math.nan,
            "cir_spread": math.nan,
            "cir_top5_energy_ratio": math.nan,
            "cir_fp_window_energy_ratio": math.nan,
        }

    energies = [m * m for m in magnitudes]
    total_energy = sum(energies)
    peak_index, peak = max(enumerate(magnitudes), key=lambda item: item[1])
    mean_magnitude = sum(magnitudes) / len(magnitudes)

    if total_energy > 0:
        center = sum(i * energy for i, energy in enumerate(energies)) / total_energy
        spread = math.sqrt(sum(((i - center) ** 2) * energy for i, energy in enumerate(energies)) / total_energy)
        top5_energy_ratio = sum(sorted(energies, reverse=True)[:5]) / total_energy
        if 0 <= fp_index < len(energies):
            left = max(0, fp_index - 5)
            right = min(len(energies), fp_index + 6)
            fp_window_energy_ratio = sum(energies[left:right]) / total_energy
        else:
            fp_window_energy_ratio = math.nan
    else:
        spread = math.nan
        top5_energy_ratio = math.nan
        fp_window_energy_ratio = math.nan

    return {
        "cir_len": len(magnitudes),
        "cir_peak": peak,
        "cir_peak_index": peak_index,
        "cir_energy": total_energy,
        "cir_mean_magnitude": mean_magnitude,
        "cir_spread": spread,
        "cir_top5_energy_ratio": top5_energy_ratio,
        "cir_fp_window_energy_ratio": fp_window_energy_ratio,
    }


def true_range(row: dict[str, str]) -> float:
    tx = safe_float(row["X_TAG"])
    ty = safe_float(row["Y_TAG"])
    tz = safe_float(row["Z_TAG"])
    ax = safe_float(row["X_ANCHOR"])
    ay = safe_float(row["Y_ANCHOR"])
    az = safe_float(row["Z_ANCHOR"])
    return math.sqrt((tx - ax) ** 2 + (ty - ay) ** 2 + (tz - az) ** 2)


def iter_raw_rows(path: Path, max_rows: int):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        for index, raw in enumerate(reader):
            if index >= max_rows:
                break
            if len(raw) < len(BASE_COLUMNS):
                continue
            row = dict(zip(BASE_COLUMNS, raw[: len(BASE_COLUMNS)]))
            cir = raw[len(BASE_COLUMNS) :]
            yield index, row, cir


def parse_file_name(path: Path) -> tuple[str, str]:
    match = re.match(r"^(ch\d+)_(A\d+)\.csv$", path.name)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def selected_files(env_name: str, positions_per_env: int, channels: set[str], anchors: set[str]) -> list[Path]:
    data_dir = DATASET_ROOT / "raw_data" / env_name / "data"
    position_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    files: list[Path] = []
    for position_dir in position_dirs[:positions_per_env]:
        for file_path in sorted(position_dir.glob("*.csv")):
            channel, anchor = parse_file_name(file_path)
            if channels and channel not in channels:
                continue
            if anchors and anchor not in anchors:
                continue
            files.append(file_path)
    return files


def extract(args: argparse.Namespace) -> int:
    channels = set(args.channels.split(",")) if args.channels else set()
    anchors = set(args.anchors.split(",")) if args.anchors else set()
    envs = args.environments.split(",") if args.environments else ["environment0", "environment1", "environment2", "environment3"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for env_name in envs:
            for file_path in selected_files(env_name, args.positions_per_env, channels, anchors):
                channel, anchor_label = parse_file_name(file_path)
                position_id = file_path.parent.name
                for measurement_index, row, cir in iter_raw_rows(file_path, args.rows_per_file):
                    measured = safe_float(row["RANGE"])
                    geom_range = true_range(row)
                    fp_index = safe_int(row["FP_INDEX"])
                    features = cir_features(cir, fp_index)
                    out = {
                        "dataset_id": "D1_indoor_uwb",
                        "environment_id": env_name,
                        "position_id": position_id,
                        "channel": channel,
                        "anchor_label": anchor_label,
                        "measurement_index": measurement_index,
                        "tag_id": row["TAG_ID"],
                        "anchor_id": row["ANCHOR_ID"],
                        "tag_x": row["X_TAG"],
                        "tag_y": row["Y_TAG"],
                        "tag_z": row["Z_TAG"],
                        "anchor_x": row["X_ANCHOR"],
                        "anchor_y": row["Y_ANCHOR"],
                        "anchor_z": row["Z_ANCHOR"],
                        "range_measured": measured,
                        "range_true": geom_range,
                        "ranging_error": measured - geom_range,
                        "nlos_label": row["NLOS"],
                        "rss": row["RSS"],
                        "rss_fp": row["RSS_FP"],
                        "fp_index": fp_index,
                        "fp_point1": row["FP_POINT1"],
                        "fp_point2": row["FP_POINT2"],
                        "fp_point3": row["FP_POINT3"],
                        "stdev_noise": row["STDEV_NOISE"],
                        "cir_power": row["CIR_POWER"],
                        "max_noise": row["MAX_NOISE"],
                        "rxpacc": row["RXPACC"],
                        "channel_number": row["CHANNEL_NUMBER"],
                        "frame_length": row["FRAME_LENGTH"],
                        "preamble_length": row["PREAMBLE_LENGTH"],
                        "bitrate": row["BITRATE"],
                        "prfr": row["PRFR"],
                        "preamble_code": row["PREAMBLE_CODE"],
                    }
                    out.update(features)
                    writer.writerow(out)
                    rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a small D1 feature sample from raw CSV files.")
    parser.add_argument("--positions-per-env", type=int, default=3)
    parser.add_argument("--rows-per-file", type=int, default=10)
    parser.add_argument("--channels", default="", help="Comma-separated channels such as ch1,ch5. Empty means all.")
    parser.add_argument("--anchors", default="", help="Comma-separated anchors such as A1,A2. Empty means all.")
    parser.add_argument("--environments", default="", help="Comma-separated environments. Empty means all four.")
    args = parser.parse_args()
    rows = extract(args)
    print(f"Wrote {rows} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
