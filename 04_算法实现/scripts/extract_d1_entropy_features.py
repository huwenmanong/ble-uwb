"""Extract corrected, compact MSPDE entropy features from D1 CIR.

The supplied MATLAB implementation is audited separately.  This extractor uses
the corrected multiscale parameter convention, real-valued abs(CIR) input, and
vectorized base-c encoding.  ``--stride`` controls CIR decimation for a
reproducible speed/structure trade-off (1 preserves all taps; 4 is the default
screening setting).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from extract_d1_sample_features import BASE_COLUMNS, parse_complex_tap, safe_int


PROJECT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT / "数据集论文" / "data_set"
MANIFEST = PROJECT / "02_公开数据集" / "D1_frozen_LOSO_splits_2026-08-02.json"
DEFAULT_OUTPUT = PROJECT / "data" / "interim" / "d1_stage1_entropy_features.csv"
SCALE_COUNT = 3
M = 3
C = 6
T = 1

OUTPUT_COLUMNS = [
    "environment_id", "raw_folder", "relative_path", "channel", "anchor_label", "measurement_index",
    "entropy_input_stride", "entropy_scale1", "entropy_scale2", "entropy_scale3",
]


def load_manifest() -> dict[str, dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {item["relative_path"]: item for item in payload["records"] if not item["excluded"]}


def classes(x: np.ndarray, c: int, mu: float | None = None, sigma: float | None = None) -> np.ndarray:
    mu = float(np.mean(x)) if mu is None else float(mu)
    sigma = float(np.std(x, ddof=1)) if sigma is None else float(sigma)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.ones(len(x), dtype=np.int16)
    y = ndtr((x - mu) / sigma)
    lo, hi = float(y.min()), float(y.max())
    scaled = np.full_like(y, 0.5) if hi - lo <= 1e-12 else (y - lo) / (hi - lo)
    return np.clip(np.floor(scaled * c + 1.0), 1, c).astype(np.int16)


def fused_entropy(x: np.ndarray, c: int = C, m: int = M, t: int = T, mu=None, sigma=None) -> float:
    z_time = classes(x, c, mu=mu, sigma=sigma)
    z_freq = classes(np.abs(np.fft.fft(x)), c)
    length = len(x) - (m - 1) * t
    if length <= 0:
        return math.nan
    z_time = np.column_stack([z_time[i * t:i * t + length] for i in range(m)])
    z_freq = np.column_stack([z_freq[i * t:i * t + length] for i in range(m)])
    counts = np.zeros(c ** m, dtype=np.int64)
    powers = (c ** np.arange(m - 1, -1, -1)).astype(np.int64)
    for n_time in range(1, m):
        for time_cols in itertools.combinations(range(m), n_time):
            for freq_cols in itertools.combinations(range(m), m - n_time):
                pattern = np.hstack([z_time[:, time_cols], z_freq[:, freq_cols]])
                code = ((pattern - 1) * powers).sum(axis=1)
                counts += np.bincount(code, minlength=c ** m)
    probabilities = counts[counts > 0].astype(float)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(c ** m))


def entropy_vector(x: np.ndarray, stride: int) -> np.ndarray:
    x = np.asarray(x[::stride], dtype=float)
    mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
    values = [fused_entropy(x)]
    for scale in range(2, SCALE_COUNT + 1):
        coarse = np.asarray([np.mean(x[i * scale:(i + 1) * scale]) for i in range(len(x) // scale)])
        values.append(fused_entropy(coarse, mu=mu, sigma=sigma))
    return np.asarray(values, dtype=float)


def extract(output: Path, stride: int, max_files: int | None) -> tuple[int, int, float]:
    records = load_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    files_seen = rows_written = 0
    start = time.perf_counter()
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
                    taps = [parse_complex_tap(value) for value in raw[len(BASE_COLUMNS):]]
                    taps = [value for value in taps if value is not None]
                    if len(taps) < 16:
                        continue
                    values = entropy_vector(np.abs(np.asarray(taps, dtype=np.complex128)), stride)
                    writer.writerow({
                        "environment_id": metadata["environment"], "raw_folder": metadata["raw_folder"],
                        "relative_path": relative_path, "channel": metadata["channel"],
                        "anchor_label": metadata["anchor_label"], "measurement_index": measurement_index,
                        "entropy_input_stride": stride, "entropy_scale1": values[0],
                        "entropy_scale2": values[1], "entropy_scale3": values[2],
                    })
                    rows_written += 1
            files_seen += 1
            if files_seen % 1000 == 0:
                print(f"processed_files={files_seen} rows={rows_written}", flush=True)
    return files_seen, rows_written, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract corrected D1 MSPDE entropy features.")
    parser.add_argument("--stride", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    files, rows, elapsed = extract(args.output, args.stride, args.max_files)
    print(json.dumps({"files": files, "rows": rows, "stride": args.stride, "elapsed_s": elapsed, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
