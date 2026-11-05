"""Extract compact CIR statistics for all non-excluded D1 rows.

The raw 1,016 complex taps are not copied to the feature table.  Only summary
statistics that can be computed online from one CIR and FP_INDEX are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from extract_d1_sample_features import BASE_COLUMNS, cir_features, safe_int


PROJECT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT / "数据集论文" / "data_set"
MANIFEST = PROJECT / "02_公开数据集" / "D1_frozen_LOSO_splits_2026-08-02.json"
DEFAULT_OUTPUT = PROJECT / "data" / "interim" / "d1_stage1_cir_features.csv"

OUTPUT_COLUMNS = [
    "environment_id", "raw_folder", "relative_path", "channel", "anchor_label", "measurement_index",
    "cir_len", "cir_peak", "cir_peak_index", "cir_energy", "cir_mean_magnitude", "cir_spread",
    "cir_top5_energy_ratio", "cir_fp_window_energy_ratio",
]


def load_manifest() -> dict[str, dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {item["relative_path"]: item for item in payload["records"] if not item["excluded"]}


def extract(output: Path, max_files: int | None) -> tuple[int, int]:
    records = load_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    files_seen = rows_written = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for relative_path in sorted(records):
            if max_files is not None and files_seen >= max_files:
                break
            metadata = records[relative_path]
            path = DATASET_ROOT / Path(relative_path)
            with path.open("r", encoding="utf-8", errors="replace", newline="") as raw_handle:
                reader = csv.reader(raw_handle)
                next(reader, None)
                next(reader, None)
                for measurement_index, raw in enumerate(reader):
                    if len(raw) < len(BASE_COLUMNS):
                        continue
                    row = dict(zip(BASE_COLUMNS, raw[: len(BASE_COLUMNS)]))
                    features = cir_features(raw[len(BASE_COLUMNS):], safe_int(row["FP_INDEX"]))
                    output_row = {
                        "environment_id": metadata["environment"],
                        "raw_folder": metadata["raw_folder"],
                        "relative_path": relative_path,
                        "channel": metadata["channel"],
                        "anchor_label": metadata["anchor_label"],
                        "measurement_index": measurement_index,
                    }
                    output_row.update(features)
                    writer.writerow(output_row)
                    rows_written += 1
            files_seen += 1
            if files_seen % 1000 == 0:
                print(f"processed_files={files_seen} rows={rows_written}", flush=True)
    return files_seen, rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compact D1 CIR features.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    files, rows = extract(args.output, args.max_files)
    print(json.dumps({"files": files, "rows": rows, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
