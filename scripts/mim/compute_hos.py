#!/usr/bin/env python3
"""
Compute Musically Informed Mixing (MIM) Harmonic Overlap Scores (HOS).

For each pair of tracks, this script computes:
  - HOS_chroma: beat-synchronous chroma cosine overlap
  - HOS_harm: harmonic-mask overlap
  - HOS: lambda_h * HOS_harm + (1 - lambda_h) * HOS_chroma
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import librosa
import numpy as np


def _cosine_overlap(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    t = min(a.shape[1], b.shape[1])
    if t <= 0:
        return 0.0
    a = a[:, :t]
    b = b[:, :t]
    num = np.sum(a * b, axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + eps
    return float(np.mean(num / den))


def _harmonic_overlap(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    t = min(a.shape[1], b.shape[1])
    if t <= 0:
        return 0.0
    a = a[:, :t]
    b = b[:, :t]
    num = np.sum(a * b)
    den = np.sum(a) + np.sum(b) + eps
    return float(num / den)


@dataclass
class TrackFeatures:
    track_id: str
    chroma: np.ndarray
    harmonic_mask: np.ndarray


def _extract_features(path: str, sample_rate: int, n_fft: int, hop_length: int) -> TrackFeatures:
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    if y.size == 0:
        return TrackFeatures(track_id=os.path.basename(os.path.dirname(path)), chroma=np.zeros((12, 1)), harmonic_mask=np.zeros((1, 1)))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sample_rate)
    mag = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    harm, perc = librosa.decompose.hpss(mag)
    harmonic_mask = harm / np.clip(harm + perc, 1e-8, None)
    track_id = os.path.basename(os.path.dirname(path))
    return TrackFeatures(track_id=track_id, chroma=chroma.astype(np.float32), harmonic_mask=harmonic_mask.astype(np.float32))


def _iter_track_paths(roots: Iterable[str], mixture_name: str) -> List[str]:
    paths: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for song in sorted(os.listdir(root)):
            song_dir = os.path.join(root, song)
            if not os.path.isdir(song_dir):
                continue
            candidate = os.path.join(song_dir, mixture_name)
            if os.path.isfile(candidate):
                paths.append(candidate)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MIM HOS pair scores.")
    parser.add_argument("--roots", nargs="+", required=True, help="Dataset roots with {song}/mixture.wav structure")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--mixture_name", default="mixture.wav")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--lambda_h", type=float, default=0.6, help="Weight for harmonic overlap term")
    parser.add_argument("--max_tracks", type=int, default=0, help="Optional cap for quick runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track_paths = _iter_track_paths(args.roots, args.mixture_name)
    if args.max_tracks > 0:
        track_paths = track_paths[: args.max_tracks]
    if len(track_paths) < 2:
        raise RuntimeError("Need at least two tracks to compute pair HOS.")

    feats: List[TrackFeatures] = []
    for path in track_paths:
        feats.append(_extract_features(path, args.sample_rate, args.n_fft, args.hop_length))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_a", "track_b", "hos_chroma", "hos_harm", "hos"])
        for fa, fb in itertools.combinations(feats, 2):
            hos_chroma = _cosine_overlap(fa.chroma, fb.chroma)
            hos_harm = _harmonic_overlap(fa.harmonic_mask, fb.harmonic_mask)
            hos = args.lambda_h * hos_harm + (1.0 - args.lambda_h) * hos_chroma
            writer.writerow([fa.track_id, fb.track_id, f"{hos_chroma:.6f}", f"{hos_harm:.6f}", f"{hos:.6f}"])

    print(f"Wrote HOS pairs: {args.output_csv}")


if __name__ == "__main__":
    main()
