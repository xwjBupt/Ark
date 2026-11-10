"""Audit labels and make the multilabel decisions visible without changing submission format."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.raw_root / "train.csv", dtype=str)
    valid = pd.read_csv(args.raw_root / "valid.csv", dtype=str)
    report = {}
    for name, frame in [("train", train), ("valid", valid)]:
        labels = frame.groupby("Study_id", sort=False).Predict_class.first().map(
            lambda x: sorted(int(v) for v in str(x).split(";") if v.strip()))
        report[name] = {
            "studies": len(labels), "images": len(frame),
            "studies_with_2_or_more_labels": int(sum(len(x) >= 2 for x in labels)),
            "label_count_distribution": {str(k): int(v) for k, v in labels.map(len).value_counts().sort_index().items()},
            "positive_by_class": {str(i): int(sum(i in x for x in labels)) for i in range(1, 11)},
        }
    sub = pd.read_csv(args.submission, dtype={"Study_id": str})
    required = ["Subject_id", "Study_id", "Predict_class", "Probability", "Gate"]
    if list(sub.columns) != required:
        raise ValueError(f"Submission columns must be {required}")
    if sub[required].isna().any().any():
        raise ValueError("Missing submission values")
    if sub.duplicated(["Study_id", "Predict_class"]).any():
        raise ValueError("Duplicate study-class predictions")
    coverage = sub.groupby("Study_id").Predict_class.apply(lambda ids: sorted(ids.tolist()))
    if not coverage.map(lambda ids: ids == list(range(1, 11))).all():
        raise ValueError("Every study must contain class IDs 1 through 10 exactly once")
    values = sub[["Probability", "Gate"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Nonfinite or out-of-range probabilities/thresholds")
    test = pd.read_csv(args.raw_root/"test.csv", dtype=str)
    if not test.groupby("Study_id").Subject_id.nunique().eq(1).all():
        raise ValueError("A test study maps to more than one subject")
    subjects = test[["Study_id", "Subject_id"]].drop_duplicates().set_index("Study_id").Subject_id
    if set(sub.Study_id) != set(subjects.index):
        raise ValueError("Submission study IDs differ from official test metadata")
    if not sub.Subject_id.eq(sub.Study_id.map(subjects)).all():
        raise ValueError("Submission subject IDs do not match official test metadata")
    sub["positive"] = sub["Probability"] >= sub["Gate"]
    per_study = sub.groupby("Study_id", sort=False).positive.sum()
    report["submission"] = {
        "format_and_coverage_verified": True,
        "rows": len(sub), "studies": int(sub.Study_id.nunique()),
        "rows_per_study": {str(k): int(v) for k, v in sub.groupby("Study_id").size().value_counts().items()},
        "predicted_positive_count_distribution": {str(k): int(v) for k, v in per_study.value_counts().sort_index().items()},
        "predicted_positive_by_class": {str(i): int(sub.loc[(sub.Predict_class == i) & sub.positive].shape[0]) for i in range(1, 11)},
    }
    report["interpretation"] = "Each Study_id has ten independent class probabilities; positive classes are Probability >= Gate. No softmax or argmax is used."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
