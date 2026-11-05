"""Extract deployment-valid scalar D1 features using the frozen LOSO manifest.

The output keeps offline-only labels (true range and ranging error) explicitly
separate from measured/diagnostic inputs.  CIR parsing is intentionally deferred
to a later opt-in script because a full 1,016-tap pass is substantially heavier.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT / "数据集论文" / "data_set"
MANIFEST = PROJECT / "02_公开数据集" / "D1_frozen_LOSO_splits_2026-08-02.json"
DEFAULT_OUTPUT = PROJECT / "data" / "interim" / "d1_stage1_scalar_features.csv"

BASE_COLUMNS = [
    "TAG_ID", "ANCHOR_ID", "X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR",
    "NLOS", "RANGE", "FP_INDEX", "RSS", "RSS_FP", "FP_POINT1", "FP_POINT2", "FP_POINT3",
    "STDEV_NOISE", "CIR_POWER", "MAX_NOISE", "RXPACC", "CHANNEL_NUMBER", "FRAME_LENGTH",
    "PREAMBLE_LENGTH", "BITRATE", "PRFR", "PREAMBLE_CODE",
]

OUTPUT_COLUMNS = [
    "dataset_id", "environment_id", "raw_folder", "relative_path", "channel", "anchor_label",
    "measurement_index", "tag_id", "anchor_id", "tag_x", "tag_y", "tag_z", "anchor_x",
    "anchor_y", "anchor_z", "range_measured", "range_valid", "range_true_offline",
    "ranging_error_offline", "nlos_label_eval_only", "rss", "rss_fp", "fp_index", "fp_point1",
    "fp_point2", "fp_point3", "stdev_noise", "cir_power", "max_noise", "rxpacc",
    "channel_number", "frame_length", "preamble_length", "bitrate", "prfr", "preamble_code",
    "split_environment0", "split_environment1", "split_environment2", "split_environment3",
]


def f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def true_range(row: dict[str, str]) -> float:
    tx, ty, tz = (f(row[k]) for k in ("X_TAG", "Y_TAG", "Z_TAG"))
    ax, ay, az = (f(row[k]) for k in ("X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR"))
    return math.sqrt((tx - ax) ** 2 + (ty - ay) ** 2 + (tz - az) ** 2)


def as_csv(value):
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def load_manifest() -> dict[str, dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {item["relative_path"]: item for item in payload["records"] if not item["excluded"]}


def extract(output_path: Path, max_files: int | None = None) -> tuple[int, int]:
    records = load_manifest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    files_seen = 0
    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for relative_path in sorted(records):
            if max_files is not None and files_seen >= max_files:
                break
            metadata = records[relative_path]
            path = DATASET_ROOT / Path(relative_path)
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                next(reader, None)
                for measurement_index, raw in enumerate(reader):
                    if len(raw) < len(BASE_COLUMNS):
                        continue
                    row = dict(zip(BASE_COLUMNS, raw[: len(BASE_COLUMNS)]))
                    measured = f(row["RANGE"])
                    geometry_range = true_range(row)
                    valid = math.isfinite(measured) and measured > 0
                    item = {
                        "dataset_id": "D1_indoor_uwb",
                        "environment_id": metadata["environment"],
                        "raw_folder": metadata["raw_folder"],
                        "relative_path": relative_path,
                        "channel": metadata["channel"],
                        "anchor_label": metadata["anchor_label"],
                        "measurement_index": measurement_index,
                        "tag_id": row["TAG_ID"],
                        "anchor_id": row["ANCHOR_ID"],
                        "tag_x": f(row["X_TAG"]), "tag_y": f(row["Y_TAG"]), "tag_z": f(row["Z_TAG"]),
                        "anchor_x": f(row["X_ANCHOR"]), "anchor_y": f(row["Y_ANCHOR"]), "anchor_z": f(row["Z_ANCHOR"]),
                        "range_measured": measured,
                        "range_valid": int(valid),
                        "range_true_offline": geometry_range,
                        "ranging_error_offline": measured - geometry_range if valid else math.nan,
                        "nlos_label_eval_only": row["NLOS"],
                        "rss": f(row["RSS"]), "rss_fp": f(row["RSS_FP"]),
                        "fp_index": f(row["FP_INDEX"]), "fp_point1": f(row["FP_POINT1"]),
                        "fp_point2": f(row["FP_POINT2"]), "fp_point3": f(row["FP_POINT3"]),
                        "stdev_noise": f(row["STDEV_NOISE"]), "cir_power": f(row["CIR_POWER"]),
                        "max_noise": f(row["MAX_NOISE"]), "rxpacc": f(row["RXPACC"]),
                        "channel_number": row["CHANNEL_NUMBER"], "frame_length": row["FRAME_LENGTH"],
                        "preamble_length": row["PREAMBLE_LENGTH"], "bitrate": row["BITRATE"],
                        "prfr": row["PRFR"], "preamble_code": row["PREAMBLE_CODE"],
                    }
                    for env in ("environment0", "environment1", "environment2", "environment3"):
                        item[f"split_{env}"] = metadata[f"split_{env}"]
                    writer.writerow({key: as_csv(item.get(key, "")) for key in OUTPUT_COLUMNS})
                    rows_written += 1
            files_seen += 1
            if files_seen % 1000 == 0:
                print(f"processed_files={files_seen} rows={rows_written}", flush=True)
    return files_seen, rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scalar D1 stage-1 features from the frozen manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None, help="Debug mode; omit for all non-excluded files.")
    args = parser.parse_args()
    files, rows = extract(args.output, args.max_files)
    print(json.dumps({"files": files, "rows": rows, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
