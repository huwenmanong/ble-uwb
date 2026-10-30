"""Create leakage-aware, reproducible D1 leave-one-environment-out manifests.

The manifest is file-level, but all assignments are made at the position-group
level.  The known environment3 packaging-corruption folder is excluded by
default.  No true range/error or future sample is used to assign a split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT = Path(r"E:\陕汽实习\数据集\SCI-UWB")
DATASET_ROOT = PROJECT / "数据集论文" / "data_set"
RAW_ROOT = DATASET_ROOT / "raw_data"
OUT_DIR = PROJECT / "02_公开数据集"
OUT_PATH = OUT_DIR / "D1_frozen_LOSO_splits_2026-08-02.json"
SEED = "d1-stage1-v1"
CALIBRATION_FRACTION = 0.20
EXCLUDED = {
    ("environment3", "9.87_8.32_1.20"): "raw CSV tags are (2.43, 3.68, 1.20), while data.json has the folder coordinates"
}


def first_data_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # version
        next(reader, None)  # header
        row = next(reader)
    return {
        "tag": (float(row[2]), float(row[3]), float(row[4])),
        "anchor": (float(row[5]), float(row[6]), float(row[7])),
        "channel": path.stem.split("_")[0],
        "anchor_label": path.stem.split("_")[-1],
    }


def stable_score(text: str) -> int:
    return int.from_bytes(hashlib.sha256((SEED + "|" + text).encode()).digest()[:8], "big")


def rel(path: Path) -> str:
    return path.relative_to(DATASET_ROOT).as_posix()


def main() -> None:
    records = []
    groups = defaultdict(list)
    for env_dir in sorted((p for p in RAW_ROOT.iterdir() if p.is_dir()), key=lambda p: p.name):
        data_dir = env_dir / "data"
        if not data_dir.exists():
            continue
        for folder in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            key = (env_dir.name, folder.name)
            excluded_reason = EXCLUDED.get(key)
            for path in sorted(folder.glob("*.csv"), key=lambda p: p.name):
                row = first_data_row(path)
                item = {
                    "relative_path": rel(path),
                    "environment": env_dir.name,
                    "raw_folder": folder.name,
                    "position_group": f"{env_dir.name}/{folder.name}",
                    "tag_first_row": list(row["tag"]),
                    "anchor_first_row": list(row["anchor"]),
                    "channel": row["channel"],
                    "anchor_label": row["anchor_label"],
                    "excluded": bool(excluded_reason),
                    "exclusion_reason": excluded_reason,
                }
                records.append(item)
                if not excluded_reason:
                    groups[(env_dir.name, folder.name)].append(item)

    envs = sorted({x["environment"] for x in records})
    folds = {}
    for heldout in envs:
        train_groups = sorted([g for g in groups if g[0] != heldout])
        calibration = {g for g in train_groups if stable_score(f"{g[0]}/{g[1]}") % 10000 < int(CALIBRATION_FRACTION * 10000)}
        # Ensure the calibration split is non-empty even for a future tiny subset.
        if not calibration and train_groups:
            calibration = {min(train_groups, key=lambda g: stable_score(f"{g[0]}/{g[1]}"))}
        train = set(train_groups) - calibration
        test = {g for g in groups if g[0] == heldout}
        assignments = {}
        for g in train:
            assignments[f"{g[0]}/{g[1]}"] = "train"
        for g in calibration:
            assignments[f"{g[0]}/{g[1]}"] = "calibration"
        for g in test:
            assignments[f"{g[0]}/{g[1]}"] = "test"
        folds[heldout] = {
            "heldout_environment": heldout,
            "train_groups": sorted(f"{a}/{b}" for a, b in train),
            "calibration_groups": sorted(f"{a}/{b}" for a, b in calibration),
            "test_groups": sorted(f"{a}/{b}" for a, b in test),
            "file_counts": {
                role: sum(len(groups[tuple(g.split("/", 1))]) for g, r in assignments.items() if r == role)
                for role in ("train", "calibration", "test")
            },
        }
        for item in records:
            item[f"split_{heldout}"] = "excluded" if item["excluded"] else assignments.get(item["position_group"], "excluded")

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "seed": SEED,
        "calibration_fraction_target": CALIBRATION_FRACTION,
        "assignment_unit": "environment + raw position folder; no position group is split across roles",
        "excluded_groups": [
            {"environment": e, "raw_folder": f, "reason": reason} for (e, f), reason in EXCLUDED.items()
        ],
        "record_count": len(records),
        "nonexcluded_record_count": sum(not x["excluded"] for x in records),
        "group_count": len(groups),
        "folds": folds,
        "records": records,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps({k: v["file_counts"] for k, v in folds.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
