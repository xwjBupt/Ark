"""Evaluate exported weights, select on validation, freeze, then predict test."""
import argparse
import itertools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from metrics import evaluate, fit_thresholds
from common import sha256_file
from model import PRETRAINS
from predict import predict_checkpoint, predict_bundle, write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--max-model-bytes", type=int, default=500_000_000)
    parser.add_argument("--require-imagenet", action="store_true",
                        help="Reject any candidate without verified ImageNet-only initialization")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    results, candidates, verified_sources = [], [], {}
    labels = None
    for run in sorted(args.runs):
        path = run/"best.pt"
        if not path.exists():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        config = saved["config"]
        del saved
        source = None
        if args.require_imagenet or config.get("init") == "imagenet":
            if config.get("init") != "imagenet":
                raise ValueError(f"Medical/unknown initialization is excluded: {run}")
            initial = Path(config["pretrained"])
            if initial not in verified_sources:
                source_file = initial.parent/f"{config['backbone']}_source.json"
                source = json.loads(source_file.read_text())
                expected = PRETRAINS[config["backbone"]]
                if source["url"] != expected["url"] or initial.name != expected["filename"]:
                    raise ValueError(f"Undocumented initialization: {initial}")
                if sha256_file(initial) != source["sha256"]:
                    raise ValueError(f"ImageNet weight hash does not match its source record: {initial}")
                verified_sources[initial] = source
            source = verified_sources[initial]
        y, probabilities = predict_checkpoint(path, args.data/"valid.json")
        if labels is not None and not np.array_equal(labels, y):
            raise ValueError("Validation study order/labels do not match")
        labels = y
        np.savez_compressed(run/"exported_valid.npz", labels=y, probabilities=probabilities)
        metrics = evaluate(y, probabilities)
        item = {"name": run.name, "path": path, "bytes": path.stat().st_size,
                "config": config, "pretraining_source": source,
                "probabilities": probabilities, "metrics": metrics}
        results.append(item)
        print("EXPORTED VALID", run.name, metrics["macro_auc"], metrics["macro_ap"], flush=True)
        if item["bytes"] <= args.max_model_bytes:
            candidates.append({"members": [(len(results)-1, 1.0)], "probabilities": probabilities,
                               "metrics": metrics, "bytes": item["bytes"]})
    if not candidates:
        raise RuntimeError("No eligible checkpoints under the model size limit")
    for i, j in itertools.combinations(range(len(results)), 2):
        size = results[i]["bytes"]+results[j]["bytes"]
        if size+100_000 > args.max_model_bytes:
            continue
        for weight in [0.25, 0.5, 0.75]:
            probabilities = weight*results[i]["probabilities"]+(1-weight)*results[j]["probabilities"]
            candidates.append({"members": [(i, weight), (j, 1-weight)], "probabilities": probabilities,
                               "metrics": evaluate(labels, probabilities), "bytes": size})
    best = max(candidates, key=lambda c: c["metrics"]["macro_auc"])
    # The competition ranks by Macro-AUC first. Keep the validation-best candidate
    # when its deduplicated inference weights remain under the 500 MB limit.
    thresholds = fit_thresholds(labels, best["probabilities"])
    metrics = evaluate(labels, best["probabilities"], thresholds)
    models = []
    for index, weight in best["members"]:
        item = results[index]
        filename = f"{item['name']}.pt"
        destination = args.output/filename
        shutil.copyfile(item["path"], destination)
        digest = sha256_file(destination)
        models.append({"name": item["name"], "checkpoint": filename, "weight": weight,
                       "sha256": digest, "bytes": destination.stat().st_size,
                       "initialization": item["config"]["init"],
                       "pooling": item["config"].get("pooling", "attention"),
                       "pretraining_source": item["pretraining_source"]})
    medical_initialization = any(m["initialization"] != "imagenet" for m in models)
    bundle = {
        "models": models, "thresholds": thresholds,
        "task": "multi-label; ten independent sigmoid probabilities per Study_id",
        "pooling": sorted({m["pooling"] for m in models}),
        "selection": "Highest Macro-AUC on labeled validation only; pair weights grid {0.25,0.5,0.75}",
        "validation_used_for_training": False, "test_used_for_fitting": False,
        "inference_precision": "fp16_autocast", "random_inference_augmentation": False,
        "medical_pretrained_weights_used": medical_initialization,
        "pretraining_source_verified": all(m["pretraining_source"] is not None for m in models),
        "competition_eligible": False if medical_initialization else None,
        "reason": ("Medical pretraining is prohibited by the saved competition rules" if medical_initialization
                   else "ImageNet initialization only; final T4 latency remains unverified"),
        "t4_latency_verified": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_metrics": metrics,
    }
    (args.output/"bundle.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    np.savez_compressed(args.output/"validation_predictions.npz", labels=labels, probabilities=best["probabilities"])
    report = []
    for candidate in sorted(candidates, key=lambda c: -c["metrics"]["macro_auc"]):
        report.append({"members": [{"name": results[i]["name"], "weight": w} for i, w in candidate["members"]],
                       "bytes": candidate["bytes"], "macro_auc": candidate["metrics"]["macro_auc"],
                       "macro_ap": candidate["metrics"]["macro_ap"], "macro_f1_05": candidate["metrics"]["macro_f1_05"]})
    (args.output/"model_comparison.json").write_text(json.dumps(report, indent=2))
    actual_size = sum(member["bytes"] for member in models)+(args.output/"bundle.json").stat().st_size
    if actual_size > args.max_model_bytes:
        raise ValueError(f"Final inference assets exceed size limit: {actual_size}")
    print("FROZEN SELECTION", json.dumps(bundle, ensure_ascii=False), flush=True)
    # Test inference starts only after model choice and thresholds have been written.
    predict_bundle(args.output/"bundle.json", args.data/"test.json", args.output/"submission.csv", raw=True)
    expected_rows = 10*len(json.loads((args.data/"test.json").read_text()))
    (args.output/"DONE.json").write_text(json.dumps({"finished": True, "rows": expected_rows,
                        "validation_macro_auc": metrics["macro_auc"], "inference_bytes": actual_size}, indent=2))
    print("FINAL SUBMISSION", args.output/"submission.csv", flush=True)


if __name__ == "__main__":
    main()
