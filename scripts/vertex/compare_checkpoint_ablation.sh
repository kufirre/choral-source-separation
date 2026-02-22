#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ai-training-484220}"
REGION="${REGION:-europe-west4}"
LOG_LIMIT="${LOG_LIMIT:-2000}"

usage() {
    echo "Usage: $0 <TRANSFER_JOB_ID> <SCRATCH_JOB_ID>"
    echo ""
    echo "Compares checkpoint-transfer vs scratch-control runs from Vertex logs."
    echo "Optional env vars: PROJECT_ID, REGION, LOG_LIMIT"
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
fi

TRANSFER_JOB_ID="$1"
SCRATCH_JOB_ID="$2"

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required." >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

fetch_logs() {
    local job_id="$1"
    local out_file="$2"
    gcloud logging read \
        "resource.type=ml_job AND resource.labels.job_id=\"${job_id}\" AND (textPayload:\"Train epoch:\" OR textPayload:\"Metric avg\" OR textPayload:\"Instr \")" \
        --project "${PROJECT_ID}" \
        --limit "${LOG_LIMIT}" \
        --order asc \
        --format "value(timestamp,textPayload)" >"${out_file}"
}

fetch_logs "${TRANSFER_JOB_ID}" "${tmp_dir}/transfer.log"
fetch_logs "${SCRATCH_JOB_ID}" "${tmp_dir}/scratch.log"

python3 - "${tmp_dir}/transfer.log" "${tmp_dir}/scratch.log" "${TRANSFER_JOB_ID}" "${SCRATCH_JOB_ID}" <<'PY'
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EPOCH_RE = re.compile(r"Train epoch:\s*(\d+)")
AVG_RE = re.compile(r"Metric avg (\w+)\s*:\s*([-\d.]+)")
INSTR_RE = re.compile(r"Instr ([A-Z]) (\w+):\s*([-\d.]+)")


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class MetricPoint:
    ts: datetime
    epoch_marker: Optional[int] = None
    avg: Dict[str, float] = field(default_factory=dict)
    instr: Dict[Tuple[str, str], float] = field(default_factory=dict)


@dataclass
class JobSummary:
    job_id: str
    latest_epoch: Optional[int]
    latest_metrics: Optional[MetricPoint]
    best_sdr: Optional[MetricPoint]
    best_si_sdr: Optional[MetricPoint]
    by_epoch_marker: Dict[int, MetricPoint]


def parse_job(job_id: str, log_path: Path) -> JobSummary:
    epoch_events: List[Tuple[datetime, int]] = []
    grouped: Dict[datetime, MetricPoint] = {}

    for raw in log_path.read_text().splitlines():
        if "\t" not in raw:
            continue
        ts_raw, payload = raw.split("\t", 1)
        ts = parse_ts(ts_raw)

        epoch_match = EPOCH_RE.search(payload)
        if epoch_match:
            epoch_events.append((ts, int(epoch_match.group(1))))
            continue

        avg_match = AVG_RE.search(payload)
        if avg_match:
            metric_name, value = avg_match.groups()
            point = grouped.setdefault(ts, MetricPoint(ts=ts))
            point.avg[metric_name] = float(value)
            continue

        instr_match = INSTR_RE.search(payload)
        if instr_match:
            instr, metric_name, value = instr_match.groups()
            point = grouped.setdefault(ts, MetricPoint(ts=ts))
            point.instr[(instr, metric_name)] = float(value)

    epoch_events.sort(key=lambda pair: pair[0])
    metric_points = sorted(grouped.values(), key=lambda point: point.ts)

    epoch_idx = 0
    for point in metric_points:
        while epoch_idx < len(epoch_events) and epoch_events[epoch_idx][0] < point.ts:
            epoch_idx += 1
        if epoch_idx < len(epoch_events):
            point.epoch_marker = epoch_events[epoch_idx][1]

    by_epoch_marker: Dict[int, MetricPoint] = {}
    for point in metric_points:
        if point.epoch_marker is not None and "sdr" in point.avg and "si_sdr" in point.avg:
            by_epoch_marker[point.epoch_marker] = point

    latest_epoch = epoch_events[-1][1] if epoch_events else None
    metric_points_core = [
        point for point in metric_points if "sdr" in point.avg and "si_sdr" in point.avg
    ]
    latest_metrics = metric_points_core[-1] if metric_points_core else (metric_points[-1] if metric_points else None)
    best_sdr = max(
        metric_points_core,
        key=lambda point: point.avg["sdr"],
        default=None,
    )
    best_si_sdr = max(
        metric_points_core,
        key=lambda point: point.avg["si_sdr"],
        default=None,
    )

    return JobSummary(
        job_id=job_id,
        latest_epoch=latest_epoch,
        latest_metrics=latest_metrics,
        best_sdr=best_sdr,
        best_si_sdr=best_si_sdr,
        by_epoch_marker=by_epoch_marker,
    )


def fmt_point(point: Optional[MetricPoint]) -> str:
    if point is None:
        return "n/a"
    marker = f"ep{point.epoch_marker}" if point.epoch_marker is not None else "ep?"
    sdr = point.avg.get("sdr")
    si = point.avg.get("si_sdr")
    sdr_text = "n/a" if sdr is None else f"{sdr:.4f}"
    si_text = "n/a" if si is None else f"{si:.4f}"
    return f"{marker} sdr={sdr_text} si_sdr={si_text}"


def fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{a - b:+.4f}"


transfer_log = Path(sys.argv[1])
scratch_log = Path(sys.argv[2])
transfer_id = sys.argv[3]
scratch_id = sys.argv[4]

transfer = parse_job(transfer_id, transfer_log)
scratch = parse_job(scratch_id, scratch_log)

print("Checkpoint Ablation Summary")
print(f"- transfer job: {transfer.job_id} (latest epoch: {transfer.latest_epoch})")
print(f"- scratch  job: {scratch.job_id} (latest epoch: {scratch.latest_epoch})")
print("")

print("Latest metrics")
print(f"- transfer: {fmt_point(transfer.latest_metrics)}")
print(f"- scratch : {fmt_point(scratch.latest_metrics)}")
print("")

print("Best-so-far metrics")
print(f"- transfer best sdr   : {fmt_point(transfer.best_sdr)}")
print(f"- transfer best si_sdr: {fmt_point(transfer.best_si_sdr)}")
print(f"- scratch  best sdr   : {fmt_point(scratch.best_sdr)}")
print(f"- scratch  best si_sdr: {fmt_point(scratch.best_si_sdr)}")
print("")

common_epochs = sorted(set(transfer.by_epoch_marker) & set(scratch.by_epoch_marker))
if not common_epochs:
    print("Matched-epoch comparison")
    print("- no common epoch markers yet (scratch run likely not far enough).")
    sys.exit(0)

epoch = common_epochs[-1]
t_point = transfer.by_epoch_marker[epoch]
s_point = scratch.by_epoch_marker[epoch]

print("Matched-epoch comparison")
print(f"- epoch marker: {epoch}")
print(
    f"- transfer sdr={t_point.avg['sdr']:.4f} si_sdr={t_point.avg['si_sdr']:.4f}"
)
print(
    f"- scratch  sdr={s_point.avg['sdr']:.4f} si_sdr={s_point.avg['si_sdr']:.4f}"
)
print(
    f"- delta (transfer - scratch): "
    f"sdr={fmt_delta(t_point.avg.get('sdr'), s_point.avg.get('sdr'))} "
    f"si_sdr={fmt_delta(t_point.avg.get('si_sdr'), s_point.avg.get('si_sdr'))}"
)

for stem in ("S", "A", "T", "B"):
    t_stem = t_point.instr.get((stem, "sdr"))
    s_stem = s_point.instr.get((stem, "sdr"))
    if t_stem is None or s_stem is None:
        continue
    print(f"- {stem} SDR delta: {t_stem - s_stem:+.4f}")
PY
