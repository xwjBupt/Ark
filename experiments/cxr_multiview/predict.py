"""Frozen inference and study-level submission export."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import StudyDataset, collate_studies
from model import MultiViewModel
from train import ImageBudgetSampler, predict_loader


def load_model(checkpoint, device="cuda"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = MultiViewModel(backbone=config["backbone"], size=config["size"], pooling=config.get("pooling", "attention"))
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), config


def predict_checkpoint(checkpoint, manifest, image_budget=24, workers=4, raw=False):
    model, config = load_model(checkpoint)
    dataset = StudyDataset(manifest, config["size"], raw=raw)
    sampler = ImageBudgetSampler(dataset, image_budget)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        pin_memory=True, collate_fn=collate_studies)
    labels, probabilities = predict_loader(model, loader)
    del model
    torch.cuda.empty_cache()
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid probabilities")
    return labels, probabilities


def write_submission(records, probabilities, thresholds, output):
    if probabilities.shape != (len(records), 10):
        raise ValueError("One probability for every study and each of the 10 classes is required")
    frame = pd.DataFrame({
        "Subject_id": np.repeat([r["subject_id"] for r in records], 10),
        "Study_id": np.repeat([r["study_id"] for r in records], 10),
        "Predict_class": np.tile(np.arange(1, 11), len(records)),
        "Probability": probabilities.reshape(-1),
        "Gate": np.tile(thresholds, len(records)),
    })
    if frame.duplicated(["Study_id", "Predict_class"]).any() or not frame.groupby("Study_id").size().eq(10).all():
        raise ValueError("Duplicate/missing study-class pairs")
    if not np.isfinite(frame.Probability).all() or not frame.Probability.between(0, 1).all():
        raise ValueError("Probability outside [0, 1]")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, encoding="utf-8", float_format="%.8f")
    temporary.replace(output)
    return len(frame)


def predict_bundle(bundle_path, manifest, output, raw=True, workers=4, image_budget=24):
    bundle_path = Path(bundle_path)
    bundle = json.loads(bundle_path.read_text())
    manifest = Path(manifest)
    predictions = []
    for member in bundle["models"]:
        checkpoint = bundle_path.parent/member["checkpoint"]
        _, p = predict_checkpoint(checkpoint, manifest, image_budget, workers, raw=raw)
        predictions.append(member["weight"]*p)
    probabilities = np.sum(predictions, axis=0)
    count = write_submission(json.loads(manifest.read_text()), probabilities, bundle["thresholds"], output)
    print(f"Saved {count} rows to {output}", flush=True)
    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-budget", type=int, default=24)
    parser.add_argument("--cache", action="store_true", help="Use the deterministic cache instead of original images")
    args = parser.parse_args()
    torch.set_num_threads(4)
    predict_bundle(args.bundle, args.manifest, args.output, raw=not args.cache,
                   workers=args.workers, image_budget=args.image_budget)
