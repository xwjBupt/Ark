"""Audit metadata and create deterministic, lossless 512px caches."""
import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from data import CLASSES, canonical_image


def cache_image(task):
    source, target = task
    target = Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.png")
        canonical_image(source).save(temporary, compress_level=1)
        temporary.replace(target)
    return target.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"classes": dict(enumerate(CLASSES, 1)), "splits": {}}
    subject_sets, study_sets, tasks = {}, {}, []
    for split in ["train", "valid", "test"]:
        source_csv = args.data_root / f"{split}.csv"
        df = pd.read_csv(source_csv, dtype={"Predict_class": str})
        required = {"Dicom_id", "Subject_id", "Study_id", "ViewPosition"}
        if not required.issubset(df.columns) or df.Dicom_id.duplicated().any():
            raise ValueError(f"Invalid metadata in {source_csv}")
        if split != "test" and df.Predict_class.isna().any():
            raise ValueError(f"Missing labels in {split}")
        subject_sets[split], study_sets[split] = set(df.Subject_id), set(df.Study_id)
        files = {p.stem: p for p in (args.data_root/split).rglob("*")
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        records = []
        for study, group in df.groupby("Study_id", sort=False):
            if group.Subject_id.nunique() != 1:
                raise ValueError(f"Study has multiple subjects: {study}")
            record = {"study_id": study, "subject_id": str(group.Subject_id.iloc[0]), "images": []}
            if split != "test":
                label_sets = [{int(n) for n in re.split(r"[;,\s]+", label.strip())} for label in group.Predict_class]
                if any(labels != label_sets[0] for labels in label_sets):
                    raise ValueError(f"Inconsistent image labels in study {study}")
                labels = label_sets[0]
                if not labels or not labels.issubset(set(range(1, 11))):
                    raise ValueError(f"Invalid class IDs: {labels}")
                record["labels"] = [int(i in labels) for i in range(1, 11)]
            for row in group.itertuples():
                path = files.get(row.Dicom_id)
                if path is None or path.parent.name != study or path.parent.parent.name != row.Subject_id:
                    raise FileNotFoundError(f"Missing/misassigned image {split}/{row.Dicom_id}")
                view = str(row.ViewPosition).upper()
                view_id = 2 if view in {"LATERAL", "LL", "RL"} else 1 if view in {"AP", "PA", "PA LLD"} else 0
                cache = args.output / "cache" / split / f"{row.Dicom_id}.png"
                record["images"].append({"raw": str(path.resolve()), "cache": str(cache.resolve()),
                                          "dicom_id": row.Dicom_id, "view": view_id})
                tasks.append((str(path), str(cache)))
            records.append(record)
        (args.output/f"{split}.json").write_text(json.dumps(records, ensure_ascii=False))
        info = {"images": len(df), "studies": len(records), "subjects": len(subject_sets[split]),
                "views_per_study": group_sizes(df), "csv_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest()}
        if split != "test":
            info["positive_studies"] = np.array([r["labels"] for r in records]).sum(0).tolist()
        audit["splits"][split] = info
        print(split, info, flush=True)
    for first, second in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        if subject_sets[first] & subject_sets[second] or study_sets[first] & study_sets[second]:
            raise ValueError(f"Patient/study leakage between {first} and {second}")
    (args.output/"audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    start, total_bytes = time.monotonic(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, size in enumerate(executor.map(cache_image, tasks, chunksize=16), 1):
            total_bytes += size
            if i % 2000 == 0 or i == len(tasks):
                print(f"cache {i}/{len(tasks)} {total_bytes/1e9:.2f}GB {time.monotonic()-start:.1f}s", flush=True)
    (args.output/"READY").write_text("Deterministic image preprocessing only; test images never used for fitting.\n")


def group_sizes(df):
    return {str(k): int(v) for k, v in df.groupby("Study_id").size().value_counts().items()}


if __name__ == "__main__":
    main()
