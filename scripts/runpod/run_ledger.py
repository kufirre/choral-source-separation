#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append RunPod/Vertex run metadata to a JSONL ledger.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    append = subparsers.add_parser("append", help="Append a single ledger record.")
    append.add_argument("--ledger", required=True, help="Path to JSONL ledger file.")
    append.add_argument("--run-id", required=True)
    append.add_argument("--platform", default="runpod")
    append.add_argument("--phase", default="")
    append.add_argument("--tier", default="")
    append.add_argument("--status", default="submitted")
    append.add_argument("--job-id", default="")
    append.add_argument("--config-path", default="")
    append.add_argument("--train-data-paths", default="")
    append.add_argument("--valid-data-paths", default="")
    append.add_argument("--dataset-gcs-paths", default="")
    append.add_argument("--results-path", default="")
    append.add_argument("--image-uri", default="")
    append.add_argument("--gpu-type", default="")
    append.add_argument("--gpu-count", default="")
    append.add_argument("--cloud-type", default="")
    append.add_argument("--repo-url", default="")
    append.add_argument("--git-ref", default="")
    append.add_argument("--start-checkpoint", default="")
    append.add_argument("--use-checkpoint", default="")
    append.add_argument("--notes", default="")
    append.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra key=value field; can be passed multiple times.",
    )
    return parser.parse_args()


def parse_extra(extra_items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in extra_items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def split_paths(raw: str) -> list[str]:
    return [part for part in raw.split() if part]


def split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def append_record(args: argparse.Namespace) -> None:
    ledger_path = Path(args.ledger).expanduser().resolve()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "platform": args.platform,
        "phase": args.phase,
        "tier": args.tier,
        "status": args.status,
        "job_id": args.job_id,
        "config_path": args.config_path,
        "train_data_paths_raw": args.train_data_paths,
        "valid_data_paths_raw": args.valid_data_paths,
        "dataset_gcs_paths_raw": args.dataset_gcs_paths,
        "train_data_paths": split_paths(args.train_data_paths),
        "valid_data_paths": split_paths(args.valid_data_paths),
        "dataset_gcs_paths": split_csv(args.dataset_gcs_paths),
        "results_path": args.results_path,
        "image_uri": args.image_uri,
        "gpu_type": args.gpu_type,
        "gpu_count": args.gpu_count,
        "cloud_type": args.cloud_type,
        "repo_url": args.repo_url,
        "git_ref": args.git_ref,
        "start_checkpoint": args.start_checkpoint,
        "use_checkpoint": args.use_checkpoint,
        "notes": args.notes,
        "host_user": os.getenv("USER", ""),
        "host_name": os.getenv("HOSTNAME", ""),
    }
    record.update(parse_extra(args.extra))

    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.command == "append":
        append_record(args)


if __name__ == "__main__":
    main()
