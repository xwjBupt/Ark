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
    if set(sub.columns) < {"Study_id", "Predict_class", "Probability", "Gate"}:
        raise ValueError("Submission must contain ten class probabilities and Gate for every Study_id")
    sub["positive"] = sub["Probability"] >= sub["Gate"]
    per_study = sub.groupby("Study_id", sort=False).positive.sum()
    report["submission"] = {
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
