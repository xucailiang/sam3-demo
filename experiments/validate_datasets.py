"""Validate local crack dataset layout and sample counts."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import CrackDataset


EXPECTED = {
    "crackforest": {
        "root": SCRIPT_DIR / "datasets" / "CrackForest-dataset",
        "count": 118,
    },
    "deepcrack": {
        "root": SCRIPT_DIR / "datasets" / "DeepCrack",
        "count": 537,
    },
}


def main() -> int:
    ok = True

    for name, cfg in EXPECTED.items():
        root = cfg["root"]
        expected_count = cfg["count"]
        if not root.exists():
            print(f"[FAIL] {name}: missing directory {root}", file=sys.stderr)
            ok = False
            continue

        dataset = CrackDataset(name, str(root))
        actual_count = len(dataset)
        split_counts = {
            "train": len(dataset.get_split("train")),
            "val": len(dataset.get_split("val")),
            "test": len(dataset.get_split("test")),
        }

        if actual_count != expected_count:
            print(
                f"[FAIL] {name}: expected {expected_count} samples, got {actual_count}",
                file=sys.stderr,
            )
            ok = False
        else:
            print(
                f"[OK] {name}: {actual_count} samples "
                f"(train={split_counts['train']}, val={split_counts['val']}, "
                f"test={split_counts['test']})"
            )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
