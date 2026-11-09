"""One independent multi-view experiment on one explicitly assigned GPU."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from data import StudyDataset, collate_studies, seed_worker
from metrics import evaluate
from model import MultiViewModel


class ImageBudgetSampler(Sampler):
    """Pack studies to a fixed image budget, including studies with missing views."""
    def __init__(self, dataset, budget, training=False, seed=42):
        self.counts = [min(len(r["images"]), dataset.max_views) if training else len(r["images"]) for r in dataset.records]
        if budget < max(self.counts):
            raise ValueError("Image budget is smaller than a complete study")
        self.budget, self.training, self.seed, self.epoch = budget, training, seed, 0

    def batches(self):
        order = np.arange(len(self.counts))
        if self.training:
            np.random.default_rng(self.seed+self.epoch).shuffle(order)
        batch, size = [], 0
        for index in order:
            count = self.counts[index]
            if batch and size+count > self.budget:
                yield batch
                batch, size = [], 0
            batch.append(int(index))
            size += count
        if batch:
            yield batch

    def __iter__(self):
        return self.batches()

    def __len__(self):
        return sum(1 for _ in self.batches())


def atomic_torch_save(value, path):
    temporary = Path(str(path)+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, lr, head_lr):
    groups = {}
    for name, param in model.named_parameters():
        rate = lr if name.startswith("encoder.") else head_lr
        decay = 0.01 if param.ndim > 1 and "embedding" not in name and "relative_position" not in name else 0.0
        groups.setdefault((rate, decay), []).append(param)
    return torch.optim.AdamW([{"params": p, "lr": rate, "base_lr": rate, "weight_decay": decay}
                              for (rate, decay), p in groups.items()])


def tune(args, model, optimizer):
    """Measure real forward/backward/AdamW memory; no dummy memory reservations."""
    free, total = torch.cuda.mem_get_info()
    target = min(total*args.memory_fraction, free-total*0.04)
    safe, upper, samples = 0, None, []
    def probe(candidate):
        images = torch.randn(candidate, 3, args.size, args.size, device="cuda")
        views = torch.ones(candidate, dtype=torch.long, device="cuda")
        bags = torch.arange(candidate, device="cuda") // 2
        labels = torch.zeros((candidate+1)//2, 10, device="cuda")
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, auxiliary = model(images, views, bags, len(labels))
                loss = nn.functional.binary_cross_entropy_with_logits(pred, labels)
                loss = loss+0.15*auxiliary.float().square().mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    candidate = 8
    while candidate <= args.max_images:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            peak = probe(candidate)
            samples.append({"images": candidate, "peak_gib": peak/1024**3, "status": "ok"})
            print(f"AUTOTUNE images={candidate} peak={peak/1024**3:.2f}GiB target={target/1024**3:.2f}GiB", flush=True)
            if peak <= target:
                safe = candidate
            else:
                upper = candidate
        except torch.OutOfMemoryError:
            print(f"AUTOTUNE images={candidate} OOM", flush=True)
            samples.append({"images": candidate, "status": "oom"})
            upper = candidate
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        if upper is not None:
            next_candidate = ((safe+upper)//8)*4
            if next_candidate <= safe or upper-safe <= 4:
                break
            candidate = next_candidate
        else:
            candidate += 16 if candidate >= 24 else 8
    if safe == 0:
        raise RuntimeError("Insufficient memory even for the smallest probe")
    result = {"image_budget": safe, "target_fraction": args.memory_fraction, "gpu": torch.cuda.get_device_name(),
              "total_gib": total/1024**3, "samples": samples, "max_training_views": args.max_views}
    (args.output/"autotune.json").write_text(json.dumps(result, indent=2))
    print("AUTOTUNE RESULT", result, flush=True)


@torch.inference_mode()
def predict_loader(model, loader):
    model.eval()
    probabilities = np.zeros((len(loader.dataset), 10), dtype=np.float32)
    labels = np.zeros_like(probabilities)
    for images, views, bags, targets, indices in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(images.cuda(non_blocking=True), views.cuda(non_blocking=True),
                              bags.cuda(non_blocking=True), len(targets))
        probabilities[indices.numpy()] = logits.float().sigmoid().cpu().numpy()
        labels[indices.numpy()] = targets.numpy()
    return labels, probabilities


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--init", choices=["ark", "imagenet"], default="ark")
    p.add_argument("--backbone", choices=["swin_base", "convnext_small"], default="swin_base")
    p.add_argument("--pooling", choices=["attention", "noisy_or"], default="attention")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-epochs", type=int, default=12)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-views", type=int, default=2)
    p.add_argument("--image-budget", type=int, default=0)
    p.add_argument("--memory-fraction", type=float, default=0.90)
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(backbone=args.backbone, pooling=args.pooling, training_precision="bf16", inference_precision="fp16",
                  competition_eligible=args.init != "ark", visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    (args.output/"config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    model = MultiViewModel(backbone=args.backbone, size=args.size, pretrained_path=args.pretrained,
                           init=args.init, pooling=args.pooling).cuda()
    optimizer = make_optimizer(model, args.lr, args.head_lr)
    if args.tune:
        tune(args, model, optimizer)
        return
    if not args.image_budget:
        args.image_budget = json.loads((args.output/"autotune.json").read_text())["image_budget"]
        config["image_budget"] = args.image_budget
        (args.output/"config.json").write_text(json.dumps(config, indent=2))
    train = StudyDataset(args.data/"train.json", args.size, training=True, max_views=args.max_views)
    valid = StudyDataset(args.data/"valid.json", args.size)
    train_sampler = ImageBudgetSampler(train, args.image_budget, True, args.seed)
    valid_sampler = ImageBudgetSampler(valid, args.image_budget)
    loader_kwargs = dict(num_workers=args.workers, pin_memory=True, collate_fn=collate_studies,
                         persistent_workers=args.workers > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(train, batch_sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid, batch_sampler=valid_sampler, **loader_kwargs)
    train_y = torch.tensor([r["labels"] for r in train.records], dtype=torch.float32)
    prevalence = train_y.mean(0).clamp(1e-4, 1-1e-4)
    with torch.no_grad():
        model.classifier.bias.copy_(torch.logit(prevalence).cuda())
    pos_weight = ((1-prevalence)/prevalence).sqrt().clamp(1, 3).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    start_epoch, best_auc, best_epoch, updates = 0, -1., -1, 0
    if args.resume and (args.output/"last.pt").exists():
        state = torch.load(args.output/"last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_auc, best_epoch, updates = state["epoch"]+1, state["best_auc"], state["best_epoch"], state["updates"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        print(f"Resumed after epoch {start_epoch}, best AUC={best_auc:.6f}", flush=True)
        del state
    total_steps = args.epochs*len(train_sampler)
    warmup_steps = 2*len(train_sampler)
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.epoch = epoch
        model.train()
        torch.cuda.reset_peak_memory_stats()
        running_loss, studies, image_count = 0., 0, 0
        epoch_start = time.monotonic()
        for step, (images, views, bags, labels, indices) in enumerate(train_loader):
            progress = epoch*len(train_sampler)+step
            if progress < warmup_steps:
                multiplier = 0.05+0.95*progress/max(1, warmup_steps)
            else:
                multiplier = 0.05+0.95*0.5*(1+math.cos(math.pi*(progress-warmup_steps)/max(1, total_steps-warmup_steps)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"]*multiplier
            images, views, bags, labels = [x.cuda(non_blocking=True) for x in (images, views, bags, labels)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, image_logits = model(images, views, bags, len(labels))
                loss = loss_fn(logits, labels)+0.15*loss_fn(image_logits.float(), labels[bags])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
            decay = min(0.995, (1+updates)/(10+updates))
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            running_loss += float(loss.detach())*len(labels)
            studies += len(labels)
            image_count += len(images)
            if step % 50 == 0 or step+1 == len(train_sampler):
                print(f"TRAIN epoch={epoch+1}/{args.epochs} step={step+1}/{len(train_sampler)} "
                      f"loss={running_loss/studies:.5f} images/s={image_count/(time.monotonic()-epoch_start):.1f} "
                      f"peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GiB lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
        y, probabilities = predict_loader(ema, valid_loader)
        metrics = evaluate(y, probabilities)
        improved = metrics["macro_auc"] > best_auc+1e-5
        if improved:
            best_auc, best_epoch = metrics["macro_auc"], epoch
            state = {"state_dict": {k: v.detach().cpu().half() if v.is_floating_point() else v.cpu()
                                      for k, v in ema.state_dict().items()},
                     "config": config, "epoch": epoch+1, "metrics": metrics}
            atomic_torch_save(state, args.output/"best.pt")
            np.savez_compressed(args.output/"best_valid.npz", labels=y, probabilities=probabilities)
            (args.output/"best_metrics.json").write_text(json.dumps(metrics, indent=2))
            del state
        record = dict(epoch=epoch+1, loss=running_loss/studies, seconds=time.monotonic()-epoch_start,
                      peak_gib=torch.cuda.max_memory_allocated()/1024**3, best_auc=best_auc,
                      best_epoch=best_epoch+1, **{k: v for k, v in metrics.items() if k != "per_class"})
        with (args.output/"history.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        print("VALID", json.dumps(record), flush=True)
        atomic_torch_save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch, "best_auc": best_auc, "best_epoch": best_epoch, "updates": updates,
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, args.output/"last.pt")
        if epoch+1 >= args.min_epochs and epoch-best_epoch >= args.patience:
            print(f"EARLY STOP best_epoch={best_epoch+1} best_auc={best_auc:.6f}", flush=True)
            break
    summary = {"best_auc": best_auc, "best_epoch": best_epoch+1, "seconds": time.monotonic()-started,
               "finished": True, "competition_eligible": config["competition_eligible"]}
    (args.output/"DONE.json").write_text(json.dumps(summary, indent=2))
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
