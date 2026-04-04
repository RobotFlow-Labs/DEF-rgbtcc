from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    message: str


def _sample_ids(split_dir: Path, limit: int) -> list[str]:
    rgb_files = sorted(split_dir.glob("*_RGB.jpg"))
    ids = [p.name.replace("_RGB.jpg", "") for p in rgb_files[:limit]]
    return ids


def validate_assets(
    dataset_root: Path,
    checkpoint_path: Path | None = None,
    sample_limit: int = 5,
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    if not dataset_root.exists():
        results.append(ValidationResult(False, f"dataset root missing: {dataset_root}"))
        return results

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            results.append(ValidationResult(False, f"missing split directory: {split_dir}"))
            continue

        ids = _sample_ids(split_dir, sample_limit)
        if not ids:
            results.append(ValidationResult(False, f"no *_RGB.jpg files in {split_dir}"))
            continue

        missing_pairs = 0
        for item_id in ids:
            t = split_dir / f"{item_id}_T.jpg"
            gt = split_dir / f"{item_id}_GT.npy"
            if not t.exists() or not gt.exists():
                missing_pairs += 1

        if missing_pairs > 0:
            results.append(
                ValidationResult(
                    False,
                    f"{split}: {missing_pairs}/{len(ids)} sampled ids missing thermal or GT pair",
                )
            )
        else:
            results.append(ValidationResult(True, f"{split}: sampled RGB/T/GT triplets look consistent"))

    if checkpoint_path is not None:
        if checkpoint_path.exists():
            results.append(ValidationResult(True, f"checkpoint found: {checkpoint_path}"))
        else:
            results.append(ValidationResult(False, f"checkpoint missing: {checkpoint_path}"))

    return results
