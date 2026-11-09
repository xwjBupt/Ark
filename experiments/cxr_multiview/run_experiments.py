"""Durable supervisor: four independent experiments, then validation-only selection."""
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
    (4, "ark6_media_384_s42", "ark6_swinbase_224_ep200.pth.tar", 384, 42, 3e-5),
    (5, "ark6_media_448_s2026", "ark6_swinbase_224_ep200.pth.tar", 448, 2026, 2e-5),
    (6, "ark5_media_384_s3407", "ark5_swinbase_224_ep200.pth.tar", 384, 3407, 3e-5),
    (7, "ark6_miccai_448_s17", "ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar", 448, 17, 2e-5),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/cxr_multiview")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output/"data/READY").exists():
        raise RuntimeError("Run prepare.py to completion first")
    lock, statuses = threading.Lock(), {}
    def status(name, **values):
        with lock:
            statuses[name] = dict(statuses.get(name, {}), **values, updated_utc=datetime.now(timezone.utc).isoformat())
            temporary = args.output/"status.tmp"
            temporary.write_text(json.dumps(statuses, indent=2))
            temporary.replace(args.output/"status.json")
    def run_one(experiment):
        gpu, name, filename, size, seed, lr = experiment
        output = args.output/"runs"/name
        output.mkdir(parents=True, exist_ok=True)
        if args.resume and (output/"DONE.json").exists():
            status(name, state="done", gpu=gpu)
            return name, 0
        free = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if free < 40000:
            status(name, state="failed", gpu=gpu, error=f"GPU {gpu} has only {free}MiB free; did not touch other processes")
            return name, 1
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = [sys.executable, "-u", str(ROOT/"experiments/cxr_multiview/train.py"),
                   "--data", str(args.output/"data"), "--output", str(output), "--init", "ark",
                   "--pretrained", str(ROOT/"pretrained_ckpts"/filename), "--size", str(size),
                   "--seed", str(seed), "--lr", str(lr), "--epochs", str(args.epochs), "--memory-fraction", "0.92"]
        for phase in ["tune", "train"]:
            if phase == "tune" and args.resume and (output/"autotune.json").exists():
                continue
            phase_command = command + (["--tune"] if phase == "tune" else ["--resume"] if args.resume else [])
            with (output/f"{phase}.log").open("a") as log:
                child = subprocess.Popen(phase_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                status(name, state=phase, gpu=gpu, pid=child.pid, command=phase_command)
                code = child.wait()
            if code:
                status(name, state="failed", phase=phase, returncode=code)
                return name, code
        status(name, state="done", returncode=0)
        return name, 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = [future.result() for future in as_completed([executor.submit(run_one, e) for e in EXPERIMENTS])]
    failed = [name for name, code in results if code]
    if failed:
        print("Some runs failed:", failed, flush=True)
    successful = [args.output/"runs"/name for name, code in results if not code]
    if not successful:
        raise RuntimeError("All experiments failed; see runs/*/train.log and tune.log")
    # Fixed parameters are selected with labeled validation data before looking at test predictions.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    command = [sys.executable, "-u", str(ROOT/"experiments/cxr_multiview/finalize.py"),
               "--data", str(args.output/"data"), "--output", str(args.output/"final"),
               "--runs", *map(str, successful)]
    with (args.output/"finalize.log").open("a") as log:
        child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status("finalize", state="running", gpu=4, pid=child.pid)
        code = child.wait()
    status("finalize", state="done" if code == 0 else "failed", returncode=code)
    if code:
        raise RuntimeError("Finalization failed; see finalize.log")
    print("All completed. Submission:", args.output/"final/submission.csv", flush=True)


if __name__ == "__main__":
    main()
