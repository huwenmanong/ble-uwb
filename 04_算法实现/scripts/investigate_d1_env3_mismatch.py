"""Bounded investigation of environment3 folder/row coordinate consistency.

This script deliberately does not parse the 1.7 GB one-line data.json into memory.
It extracts a bounded context around one position key and compares raw CSV labels
with the nominal folder name and walking_path.csv.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(r"E:\陕汽实习\数据集\SCI-UWB\数据集论文\data_set")
ENV = ROOT / "raw_data" / "environment3"
JSON_PATH = ROOT / "environment3" / "data.json"
OUT = Path(r"E:\陕汽实习\数据集\SCI-UWB\02_公开数据集")


def parse_pos(s: str):
    return tuple(float(x) for x in s.split("_"))


def read_first_row(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        row = None
        for candidate in reader:
            # Files start with a version line and a header line.
            if len(candidate) > 10 and candidate[0].strip().replace('.', '', 1).replace('-', '', 1).isdigit():
                row = candidate
                break
        if row is None:
            raise ValueError(f"no data row found: {path}")
    # CSV layout documented by D1: first 26 fields are scalar metadata; CIR follows.
    return {
        "range": float(row[9]),
        "x_tag": float(row[2]),
        "y_tag": float(row[3]),
        "z_tag": float(row[4]),
        "x_anchor": float(row[5]),
        "y_anchor": float(row[6]),
        "z_anchor": float(row[7]),
        "anchor_id": row[1],
        "los": row[8],
        "raw": row[:12],
    }


def bounded_json_context(key: str, before: int = 160, after: int = 900):
    needle = ('"' + key + '"').encode("utf-8")
    overlap = len(needle) + before + after
    with JSON_PATH.open("rb") as f:
        offset = 0
        tail = b""
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                return None
            data = tail + chunk
            idx = data.find(needle)
            if idx >= 0:
                absolute = offset - len(tail) + idx
                lo = max(0, idx - before)
                hi = min(len(data), idx + len(needle) + after)
                return {"offset": absolute, "context": data[lo:hi].decode("utf-8", errors="replace")}
            offset += len(chunk)
            tail = data[-overlap:]


def bounded_measurement_context(key: str, before: int = 80, after: int = 700):
    """Find the measurement object key rather than the earlier walking_path key."""
    needle = ('"' + key + '": {"A1"').encode("utf-8")
    overlap = len(needle) + before + after
    with JSON_PATH.open("rb") as f:
        offset = 0
        tail = b""
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                return None
            data = tail + chunk
            idx = data.find(needle)
            if idx >= 0:
                absolute = offset - len(tail) + idx
                lo = max(0, idx - before)
                hi = min(len(data), idx + len(needle) + after)
                return {"offset": absolute, "context": data[lo:hi].decode("utf-8", errors="replace")}
            offset += len(chunk)
            tail = data[-overlap:]


def main():
    folders = sorted((ENV / "data").iterdir())
    walking = []
    with (ENV / "walking_path.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                walking.append((float(row["x"]), float(row["y"]), float(row["z"])))
            except (KeyError, TypeError, ValueError):
                pass

    mapping = []
    for folder in folders:
        csv_path = folder / "ch1_A1.csv"
        if not csv_path.exists():
            continue
        row = read_first_row(csv_path)
        nominal = parse_pos(folder.name)
        tagged = (row["x_tag"], row["y_tag"], row["z_tag"])
        delta = math.dist(nominal, tagged)
        nearest = min((math.dist(tagged, p), p) for p in walking) if walking else (None, None)
        mapping.append({
            "raw_folder": folder.name,
            "nominal": nominal,
            "first_row_tag": tagged,
            "folder_tag_delta_m": delta,
            "nearest_walking_path_delta_m": nearest[0],
            "nearest_walking_path": nearest[1],
            "first_row_range": row["range"],
            "first_row_anchor": [row["x_anchor"], row["y_anchor"], row["z_anchor"]],
            "first_row_los": row["los"],
        })

    mapping.sort(key=lambda x: x["folder_tag_delta_m"], reverse=True)
    out_csv = OUT / "D1_environment3_folder_mapping.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["raw_folder", "nominal", "first_row_tag", "folder_tag_delta_m", "nearest_walking_path_delta_m", "nearest_walking_path", "first_row_range", "first_row_anchor", "first_row_los"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(mapping)

    key = "9.87_8.32_1.20"
    suspect_folder = ENV / "data" / key
    suspect_files = []
    for csv_path in sorted(suspect_folder.glob("*.csv")):
        row = read_first_row(csv_path)
        tagged = (row["x_tag"], row["y_tag"], row["z_tag"])
        suspect_files.append({
            "file": csv_path.name,
            "tag": tagged,
            "delta_to_folder_m": math.dist(parse_pos(key), tagged),
            "range": row["range"],
            "los": row["los"],
        })
    result = {
        "folder_count": len(mapping),
        "delta_gt_0_5m": sum(x["folder_tag_delta_m"] > 0.5 for x in mapping),
        "delta_gt_1m": sum(x["folder_tag_delta_m"] > 1.0 for x in mapping),
        "max_delta_m": max((x["folder_tag_delta_m"] for x in mapping), default=None),
        "largest_mismatches": mapping[:15],
        "suspect_folder_files": suspect_files,
        "json_path_context": bounded_json_context(key),
        "json_measurement_context": bounded_measurement_context(key),
    }
    out_json = OUT / "D1_environment3_mismatch_investigation.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "folder_count": result["folder_count"],
        "delta_gt_0_5m": result["delta_gt_0_5m"],
        "delta_gt_1m": result["delta_gt_1m"],
        "max_delta_m": result["max_delta_m"],
        "suspect_file_count": len(result["suspect_folder_files"]),
        "output": str(out_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
