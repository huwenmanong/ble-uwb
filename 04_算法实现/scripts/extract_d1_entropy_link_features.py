"""Extract window-aggregated entropy features for each D1 link file.

To make the entropy branch deployable, three evenly spaced CIRs (rows 0, 15,
and 30) are summarized per ``environment/folder/channel/anchor`` file.  The
model can therefore use mean, spread, P90, and max entropy as short-window link
quality features without computing the expensive entropy for every historical
row.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from extract_d1_sample_features import BASE_COLUMNS, parse_complex_tap
from extract_d1_entropy_features import entropy_vector, load_manifest


PROJECT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT / "数据集论文" / "data_set"
DEFAULT_OUTPUT = PROJECT / "data" / "interim" / "d1_stage1_entropy_link_features.csv"
WINDOW_INDICES = {0, 15, 30}

OUTPUT_COLUMNS = [
    "environment_id", "raw_folder", "relative_path", "channel", "anchor_label", "entropy_window_rows", "entropy_stride",
    *[f"entropy_scale{scale}_{stat}" for scale in (1, 2, 3) for stat in ("mean", "std", "p90", "max")],
]


def extract(output: Path, stride: int, max_files: int | None) -> tuple[int, int, float]:
    records = load_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    files_seen = rows_used = 0
    start = time.perf_counter()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for relative_path in sorted(records):
            if max_files is not None and files_seen >= max_files:
                break
            metadata = records[relative_path]
            path = DATASET_ROOT / Path(relative_path)
            vectors = []
            with path.open("r", encoding="utf-8", errors="replace", newline="") as raw_handle:
                reader = csv.reader(raw_handle)
                next(reader, None)
                next(reader, None)
                for measurement_index, raw in enumerate(reader):
                    if measurement_index not in WINDOW_INDICES or len(raw) < len(BASE_COLUMNS):
                        continue
                    taps = [parse_complex_tap(value) for value in raw[len(BASE_COLUMNS):]]
                    taps = [value for value in taps if value is not None]
                    if len(taps) >= 16:
                        vectors.append(entropy_vector(np.abs(np.asarray(taps, dtype=np.complex128)), stride))
            if not vectors:
                continue
            matrix = np.vstack(vectors)
            row = {
                "environment_id": metadata["environment"], "raw_folder": metadata["raw_folder"],
                "relative_path": relative_path, "channel": metadata["channel"], "anchor_label": metadata["anchor_label"],
                "entropy_window_rows": matrix.shape[0], "entropy_stride": stride,
            }
            for index, scale in enumerate((1, 2, 3)):
                values = matrix[:, index]
                for stat, value in (("mean", np.mean(values)), ("std", np.std(values, ddof=1) if len(values) > 1 else 0.0), ("p90", np.quantile(values, 0.90)), ("max", np.max(values))):
                    row[f"entropy_scale{scale}_{stat}"] = float(value)
            writer.writerow(row)
            files_seen += 1
            rows_used += matrix.shape[0]
            if files_seen % 1000 == 0:
                print(f"processed_files={files_seen} entropy_rows={rows_used}", flush=True)
    return files_seen, rows_used, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract short-window D1 entropy link features.")
    parser.add_argument("--stride", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    files, rows, elapsed = extract(args.output, args.stride, args.max_files)
    print(json.dumps({"files": files, "entropy_rows": rows, "stride": args.stride, "elapsed_s": elapsed, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
