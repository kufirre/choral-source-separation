#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


METRIC_RE = re.compile(r"Metric avg sdr\s*:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
EPOCH_RE = re.compile(r"Train epoch:\s*(\d+)")
NONFINITE_RE = re.compile(r"(?i)(?:^|[^a-z])(nan|inf)(?:[^a-z]|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse train.log and enforce a basic SDR quality gate.")
    parser.add_argument("--train-log", required=True, help="Path to train log file.")
    parser.add_argument("--out-json", default="", help="Optional output summary JSON path.")
    parser.add_argument("--min-best-sdr", type=float, default=0.8)
    parser.add_argument("--max-drop-from-best", type=float, default=1.0)
    parser.add_argument("--min-evals", type=int, default=2)
    return parser.parse_args()


def load_summary(log_path: Path) -> dict:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    avg_sdr = []
    epoch_ids = []
    nonfinite_lines = []

    for idx, line in enumerate(lines, start=1):
        metric_match = METRIC_RE.search(line)
        if metric_match:
            value = float(metric_match.group(1))
            avg_sdr.append(value)

        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            epoch_ids.append(int(epoch_match.group(1)))

        if NONFINITE_RE.search(line):
            nonfinite_lines.append({"line": idx, "text": line.strip()[:300]})

    best_sdr = max(avg_sdr) if avg_sdr else None
    last_sdr = avg_sdr[-1] if avg_sdr else None
    drop_from_best = (best_sdr - last_sdr) if (best_sdr is not None and last_sdr is not None) else None

    summary = {
        "log_path": str(log_path),
        "num_lines": len(lines),
        "num_eval_points": len(avg_sdr),
        "num_epochs_seen": len(set(epoch_ids)),
        "epoch_ids": epoch_ids,
        "avg_sdr_values": avg_sdr,
        "best_sdr": best_sdr,
        "last_sdr": last_sdr,
        "drop_from_best": drop_from_best,
        "nonfinite_count": len(nonfinite_lines),
        "nonfinite_examples": nonfinite_lines[:10],
    }
    return summary


def evaluate_gate(summary: dict, min_best_sdr: float, max_drop_from_best: float, min_evals: int) -> tuple[bool, list[str]]:
    reasons = []
    passed = True

    if summary["num_eval_points"] < min_evals:
        passed = False
        reasons.append(f"insufficient eval points ({summary['num_eval_points']} < {min_evals})")

    if summary["nonfinite_count"] > 0:
        passed = False
        reasons.append(f"non-finite tokens found in log ({summary['nonfinite_count']})")

    best_sdr = summary["best_sdr"]
    last_sdr = summary["last_sdr"]
    drop = summary["drop_from_best"]

    if best_sdr is None or not math.isfinite(best_sdr):
        passed = False
        reasons.append("best_sdr is missing or non-finite")
    elif best_sdr < min_best_sdr:
        passed = False
        reasons.append(f"best_sdr below threshold ({best_sdr:.4f} < {min_best_sdr:.4f})")

    if drop is None or not math.isfinite(drop):
        passed = False
        reasons.append("drop_from_best is missing or non-finite")
    elif drop > max_drop_from_best:
        passed = False
        reasons.append(f"drop_from_best exceeds threshold ({drop:.4f} > {max_drop_from_best:.4f})")

    if last_sdr is None or not math.isfinite(last_sdr):
        passed = False
        reasons.append("last_sdr is missing or non-finite")

    return passed, reasons


def main() -> int:
    args = parse_args()
    log_path = Path(args.train_log)
    if not log_path.exists():
        raise FileNotFoundError(f"train log not found: {log_path}")

    summary = load_summary(log_path)
    passed, reasons = evaluate_gate(
        summary=summary,
        min_best_sdr=args.min_best_sdr,
        max_drop_from_best=args.max_drop_from_best,
        min_evals=args.min_evals,
    )
    summary["gate"] = {
        "passed": passed,
        "min_best_sdr": args.min_best_sdr,
        "max_drop_from_best": args.max_drop_from_best,
        "min_evals": args.min_evals,
        "reasons": reasons,
    }

    payload = json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True)
    print(payload)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
