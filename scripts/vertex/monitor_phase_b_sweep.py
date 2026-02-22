#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ACTIVE_STATES = {
    "JOB_STATE_PENDING",
    "JOB_STATE_QUEUED",
    "JOB_STATE_RUNNING",
    "JOB_STATE_CANCELLING",
}

EPOCH_RE = re.compile(r"Train epoch:\s*(\d+)\s+Learning rate:\s*([0-9.eE+-]+)")
AVG_SISDR_RE = re.compile(r"Metric avg si_sdr\s*:\s*([-+]?\d+(?:\.\d+)?)")
AVG_SIR_RE = re.compile(r"Metric avg sir\s*:\s*([-+]?\d+(?:\.\d+)?)")
AVG_SAR_RE = re.compile(r"Metric avg sar\s*:\s*([-+]?\d+(?:\.\d+)?)")
AVG_ADJ_SIR_RE = re.compile(r"Metric avg adj_sir\s*:\s*([-+]?\d+(?:\.\d+)?)")
AVG_ADJ_BLEED_RE = re.compile(r"Metric avg adj_bleed_db\s*:\s*([-+]?\d+(?:\.\d+)?)")
INSTR_SISDR_RE = re.compile(
    r"Instr\s+([SABT]|Soprano|Alto|Tenor|Bass)\s+si_sdr:\s*([-+]?\d+(?:\.\d+)?)"
)


@dataclass
class JobSpec:
    job_id: str
    label: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "<no stderr>"
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)} :: {stderr}")
    return result.stdout


def describe_job_state(*, project: str, region: str, job_id: str) -> str:
    job_name = job_id
    if not job_name.startswith("projects/"):
        job_name = f"projects/{project}/locations/{region}/customJobs/{job_id}"
    output = run_command(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "describe",
            job_name,
            "--project",
            project,
            "--region",
            region,
            "--format=value(state)",
        ]
    )
    return output.strip() or "JOB_STATE_UNKNOWN"


def fetch_job_logs(*, project: str, job_id: str, limit: int) -> list[str]:
    output = run_command(
        [
            "gcloud",
            "logging",
            "read",
            (
                f'(resource.type="aiplatform.googleapis.com/CustomJob" AND labels.job_id="{job_id}") '
                f'OR (resource.type="ml_job" AND resource.labels.job_id="{job_id}")'
            ),
            "--project",
            project,
            "--limit",
            str(limit),
            "--order",
            "desc",
            "--format",
            "value(textPayload)",
        ]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def extract_latest_metrics(lines: list[str]) -> dict[str, float | int | None]:
    epoch: int | None = None
    learning_rate: float | None = None
    avg_si_sdr: float | None = None
    avg_sir: float | None = None
    avg_sar: float | None = None
    avg_adj_sir: float | None = None
    avg_adj_bleed_db: float | None = None
    part_scores: dict[str, float] = {}

    for line in lines:
        if epoch is None:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                learning_rate = float(epoch_match.group(2))
                continue
        if epoch is None:
            continue

        if avg_si_sdr is None:
            avg_match = AVG_SISDR_RE.search(line)
            if avg_match:
                avg_si_sdr = float(avg_match.group(1))
                continue
        if avg_sir is None:
            avg_sir_match = AVG_SIR_RE.search(line)
            if avg_sir_match:
                avg_sir = float(avg_sir_match.group(1))
                continue
        if avg_sar is None:
            avg_sar_match = AVG_SAR_RE.search(line)
            if avg_sar_match:
                avg_sar = float(avg_sar_match.group(1))
                continue
        if avg_adj_sir is None:
            avg_adj_sir_match = AVG_ADJ_SIR_RE.search(line)
            if avg_adj_sir_match:
                avg_adj_sir = float(avg_adj_sir_match.group(1))
                continue
        if avg_adj_bleed_db is None:
            avg_adj_bleed_match = AVG_ADJ_BLEED_RE.search(line)
            if avg_adj_bleed_match:
                avg_adj_bleed_db = float(avg_adj_bleed_match.group(1))
                continue

        instr_match = INSTR_SISDR_RE.search(line)
        if instr_match:
            raw_name = instr_match.group(1).upper()
            if raw_name.startswith("SOPRANO"):
                name = "S"
            elif raw_name.startswith("ALTO"):
                name = "A"
            elif raw_name.startswith("TENOR"):
                name = "T"
            elif raw_name.startswith("BASS"):
                name = "B"
            else:
                name = raw_name
            part_scores[name] = float(instr_match.group(2))
            if avg_si_sdr is not None and all(part in part_scores for part in ("S", "A", "T", "B")):
                break

    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "avg_si_sdr": avg_si_sdr,
        "avg_sir": avg_sir,
        "avg_sar": avg_sar,
        "avg_adj_sir": avg_adj_sir,
        "avg_adj_bleed_db": avg_adj_bleed_db,
        "a_si_sdr": part_scores.get("A"),
        "t_si_sdr": part_scores.get("T"),
    }


def cancel_job(*, project: str, region: str, job_id: str) -> None:
    run_command(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "cancel",
            job_id,
            "--project",
            project,
            "--region",
            region,
            "--quiet",
        ]
    )


def load_monitor_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_monitor_state(path: Path, state: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def monitor(
    *,
    project: str,
    region: str,
    jobs: list[JobSpec],
    interval_seconds: int,
    log_path: Path,
    state_path: Path,
    logs_limit: int,
    min_epoch: int,
    alto_threshold: float,
    tenor_threshold: float,
    consecutive_bad: int,
    run_once: bool,
) -> None:
    monitor_state = load_monitor_state(state_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[START] {utc_now()}\n")
        log_file.write(f"[JOBS] {' '.join(f'{job.job_id}:{job.label}' for job in jobs)}\n")
        log_file.flush()

        while True:
            check_time = utc_now()
            log_file.write(f"[CHECK] {check_time}\n")
            any_active = False

            for job in jobs:
                state = describe_job_state(project=project, region=region, job_id=job.job_id)
                log_file.write(f"{job.job_id} {job.label} {state}\n")

                record = monitor_state.setdefault(
                    job.job_id,
                    {
                        "label": job.label,
                        "bad_streak": 0,
                        "last_epoch_evaluated": None,
                        "canceled_by_monitor": False,
                        "history": [],
                    },
                )

                if state in ACTIVE_STATES:
                    any_active = True

                if state != "JOB_STATE_RUNNING":
                    continue

                metrics = extract_latest_metrics(
                    fetch_job_logs(project=project, job_id=job.job_id, limit=logs_limit)
                )
                log_file.write(
                    "metrics "
                    f"epoch={metrics['epoch']} "
                    f"lr={metrics['learning_rate']} "
                    f"avg_si_sdr={metrics['avg_si_sdr']} "
                    f"avg_sir={metrics['avg_sir']} "
                    f"avg_sar={metrics['avg_sar']} "
                    f"avg_adj_sir={metrics['avg_adj_sir']} "
                    f"avg_adj_bleed_db={metrics['avg_adj_bleed_db']} "
                    f"a_si_sdr={metrics['a_si_sdr']} "
                    f"t_si_sdr={metrics['t_si_sdr']}\n"
                )

                epoch = metrics["epoch"]
                a_si_sdr = metrics["a_si_sdr"]
                t_si_sdr = metrics["t_si_sdr"]
                if epoch is None or a_si_sdr is None or t_si_sdr is None:
                    continue

                history_item = {
                    "ts": check_time,
                    "epoch": epoch,
                    "avg_si_sdr": metrics["avg_si_sdr"],
                    "avg_sir": metrics["avg_sir"],
                    "avg_sar": metrics["avg_sar"],
                    "avg_adj_sir": metrics["avg_adj_sir"],
                    "avg_adj_bleed_db": metrics["avg_adj_bleed_db"],
                    "a_si_sdr": a_si_sdr,
                    "t_si_sdr": t_si_sdr,
                }
                record["history"].append(history_item)

                last_epoch = record.get("last_epoch_evaluated")
                if epoch <= (last_epoch if isinstance(last_epoch, int) else -1):
                    continue

                record["last_epoch_evaluated"] = epoch
                if epoch < min_epoch:
                    record["bad_streak"] = 0
                    continue

                is_bad = (a_si_sdr < alto_threshold) or (t_si_sdr < tenor_threshold)
                if is_bad:
                    record["bad_streak"] = int(record.get("bad_streak", 0)) + 1
                else:
                    record["bad_streak"] = 0

                log_file.write(
                    f"decision epoch={epoch} bad={is_bad} bad_streak={record['bad_streak']} "
                    f"thresholds(A<{alto_threshold}, T<{tenor_threshold})\n"
                )

                if record["bad_streak"] >= consecutive_bad and not bool(
                    record.get("canceled_by_monitor", False)
                ):
                    cancel_job(project=project, region=region, job_id=job.job_id)
                    record["canceled_by_monitor"] = True
                    log_file.write(
                        f"[ACTION] canceled job {job.job_id} at epoch={epoch} "
                        f"(A={a_si_sdr:.4f}, T={t_si_sdr:.4f})\n"
                    )

            save_monitor_state(state_path, monitor_state)
            log_file.write("\n")
            log_file.flush()

            if run_once:
                break

            if not any_active:
                log_file.write(f"[DONE] {utc_now()}\n")
                log_file.flush()
                break

            time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Phase B sweep and apply stop rules.")
    parser.add_argument("--project", default="ai-training-484220")
    parser.add_argument("--region", default="europe-west4")
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        help="Job spec as <job_id>:<label>, for example 1234567890:lambda0.05",
    )
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--logs-limit", type=int, default=1200)
    parser.add_argument("--min-epoch", type=int, default=12)
    parser.add_argument("--alto-threshold", type=float, default=-0.8)
    parser.add_argument("--tenor-threshold", type=float, default=0.1)
    parser.add_argument("--consecutive-bad", type=int, default=3)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase_b_sweep_decision_monitor.log"),
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("logs/phase_b_sweep_decision_state.json"),
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs: list[JobSpec] = []
    for spec in args.job:
        if ":" not in spec:
            raise ValueError(f"Invalid --job value '{spec}'. Use <job_id>:<label>.")
        job_id, label = spec.split(":", 1)
        jobs.append(JobSpec(job_id=job_id.strip(), label=label.strip()))

    monitor(
        project=args.project,
        region=args.region,
        jobs=jobs,
        interval_seconds=args.interval_seconds,
        log_path=args.log_path,
        state_path=args.state_path,
        logs_limit=args.logs_limit,
        min_epoch=args.min_epoch,
        alto_threshold=args.alto_threshold,
        tenor_threshold=args.tenor_threshold,
        consecutive_bad=args.consecutive_bad,
        run_once=args.once,
    )


if __name__ == "__main__":
    main()
