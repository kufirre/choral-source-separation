#!/usr/bin/env python3
"""
Sample MIM pair curriculum from precomputed HOS CSV.

Implements P(a,b) ∝ exp(kappa * HOS(a,b)) with stage-wise kappa schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MIM curriculum pairs from HOS scores.")
    parser.add_argument("--hos_csv", required=True, help="CSV produced by compute_hos.py")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--pairs_per_stage", type=int, default=200)
    parser.add_argument("--kappa_start", type=float, default=5.0)
    parser.add_argument("--kappa_end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_pairs(path: str) -> List[Tuple[str, str, float]]:
    pairs: List[Tuple[str, str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["track_a"], row["track_b"], float(row["hos"])))
    if not pairs:
        raise RuntimeError(f"No pairs found in {path}")
    return pairs


def _sample_pairs(
    pairs: List[Tuple[str, str, float]],
    kappa: float,
    num_samples: int,
    rng: random.Random,
) -> List[Tuple[str, str, float]]:
    scores = [p[2] for p in pairs]
    max_score = max(scores)
    weights = [math.exp(kappa * (s - max_score)) for s in scores]
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = list(range(len(pairs)))
    sampled_idx = rng.choices(idx, weights=probs, k=num_samples)
    return [pairs[i] for i in sampled_idx]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pairs = _load_pairs(args.hos_csv)

    stages = max(int(args.stages), 1)
    curriculum = []
    for stage_idx in range(stages):
        if stages == 1:
            progress = 0.0
        else:
            progress = stage_idx / float(stages - 1)
        kappa = args.kappa_start + progress * (args.kappa_end - args.kappa_start)
        sampled = _sample_pairs(
            pairs=pairs,
            kappa=kappa,
            num_samples=int(args.pairs_per_stage),
            rng=rng,
        )
        curriculum.append(
            {
                "stage": stage_idx,
                "kappa": kappa,
                "pairs": [
                    {"track_a": a, "track_b": b, "hos": hos}
                    for a, b, hos in sampled
                ],
            }
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"stages": curriculum}, f, indent=2)
    print(f"Wrote MIM curriculum: {args.output_json}")


if __name__ == "__main__":
    main()
