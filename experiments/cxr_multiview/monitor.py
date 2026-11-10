"""Print current experiments and latest validation measurements."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2]/"artifacts/cxr_multiview")
    args = parser.parse_args()
    status_file = args.output/"status.json"
    statuses = json.loads(status_file.read_text()) if status_file.exists() else {}
    print(f"{'GPU':<5} {'experiment':<30} {'state':<10} {'epoch':>6} {'best AUC':>10} {'best AP':>9}")
    for name, status in sorted(statuses.items(), key=lambda x: (x[1].get("gpu", 99), x[0])):
        history = args.output/"runs"/name/"history.jsonl"
        rows = history.read_text().splitlines() if history.exists() else []
        latest = json.loads(rows[-1]) if rows else {}
        best_path = args.output/"runs"/name/"best_metrics.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        print(f"{status.get('gpu','-'):<5} {name:<30} {status['state']:<10} {latest.get('epoch','-'):>6} "
              f"{best.get('macro_auc',0):>10.5f} {best.get('macro_ap',0):>9.5f}")
    final = args.output/"final/submission.csv"
    print("Submission:", str(final) if final.exists() else "not generated yet; waiting for training/finalization")


if __name__ == "__main__":
    main()
