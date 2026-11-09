"""Train only from public ImageNet initialization; no medical checkpoint is referenced."""
import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = [
    (4, "imagenet_swin_384_s42", 384, 42, 3e-5),
    (5, "imagenet_swin_448_s2026", 448, 2026, 2e-5),
    (6, "imagenet_swin_384_s3407", 384, 3407, 3e-5),
    (7, "imagenet_swin_448_s17", 448, 17, 2e-5),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/cxr_imagenet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = ROOT/"artifacts/cxr_multiview/data"
    weight = ROOT/"pretrained_ckpts/imagenet/swin_base_patch4_window7_224_22kto1k.pth"
    if not (data/"READY").exists() or not weight.exists():
        raise RuntimeError("Missing prepared data or ImageNet-only weight")
    lock, statuses = threading.Lock(), {}

    def status(name, **values):
        with lock:
            statuses[name] = dict(statuses.get(name, {}), **values, updated_utc=datetime.now(timezone.utc).isoformat())
            tmp = args.output/"status.tmp"
            tmp.write_text(json.dumps(statuses, indent=2))
            tmp.replace(args.output/"status.json")

    def run_one(spec):
        gpu, name, size, seed, lr = spec
        output = args.output/"runs"/name
        output.mkdir(parents=True, exist_ok=True)
        free = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                                            "--format=csv,noheader,nounits"], text=True).strip())
        if free < 40000:
            status(name, state="failed", gpu=gpu, error=f"GPU {gpu} has only {free}MiB free")
            return name, 1
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = [sys.executable, "-u", str(ROOT/"experiments/cxr_multiview/train.py"),
                   "--data", str(data), "--output", str(output), "--init", "imagenet",
                   "--backbone", "swin_base", "--pooling", "attention", "--pretrained", str(weight),
                   "--size", str(size), "--seed", str(seed), "--lr", str(lr), "--epochs", str(args.epochs),
                   "--memory-fraction", "0.92"]
        for phase in ["tune", "train"]:
            if args.resume and phase == "tune" and (output/"autotune.json").exists():
                continue
            extra = ["--tune"] if phase == "tune" else (["--resume"] if args.resume else [])
            with (output/f"{phase}.log").open("a") as log:
                child = subprocess.Popen(command+extra, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                status(name, state=phase, gpu=gpu, pid=child.pid)
                code = child.wait()
            if code:
                status(name, state="failed", phase=phase, returncode=code)
                return name, code
        status(name, state="done", gpu=gpu)
        return name, 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [f.result() for f in as_completed([pool.submit(run_one, e) for e in EXPERIMENTS])]
    successful = [args.output/"runs"/name for name, code in results if code == 0]
    if not successful:
        raise RuntimeError("All ImageNet-only runs failed")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    command = [sys.executable, "-u", str(ROOT/"experiments/cxr_multiview/finalize.py"), "--data", str(data),
               "--output", str(args.output/"final"), "--runs", *map(str, successful)]
    with (args.output/"finalize.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status("finalize", state="running", gpu=4, pid=child.pid)
        code = child.wait()
    status("finalize", state="done" if code == 0 else "failed", returncode=code)
    if code:
        raise RuntimeError("ImageNet finalization failed")
    print("ImageNet-only submission:", args.output/"final/submission.csv", flush=True)


if __name__ == "__main__":
    main()
