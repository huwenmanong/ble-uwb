"""Record the Python/runtime dependencies used for reproducible experiments."""
from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "02_公开数据集" / "bluekey_environment_2026-08-02.json"
MODULES = ["numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib", "jupyter", "nbformat"]


def main() -> None:
    result = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "modules_available": {name: bool(importlib.util.find_spec(name)) for name in MODULES},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
