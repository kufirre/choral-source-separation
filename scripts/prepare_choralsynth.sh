#!/usr/bin/env bash
set -euo pipefail
#
# Download and process the ChoralSynth dataset for VCIN SATB training.
#
# ChoralSynth (MTG, 2023): 20 multitrack SATB choral pieces, ~3.8 hours.
# Source: https://zenodo.org/records/10137883
# License: CC BY-NC-SA 4.0
#
# Stem mapping:
#   CANTUS   -> S (Soprano)
#   ALTUS    -> A (Alto)
#   TENOR I  -> T (Tenor, or merged TENOR I + TENOR II)
#   BASSUS   -> B (Bass)
#
# Output format:
#   {song_name}/S.wav, A.wav, T.wav, B.wav, mixture.wav
#   44100 Hz, stereo (or mono duplicated to stereo), peak-normalized
#
# Usage:
#   ./scripts/prepare_choralsynth.sh [--upload]
#
# Options:
#   --upload   Upload processed output to GCS after processing.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
ZENODO_URL="https://zenodo.org/records/10137883/files/Dataset.zip"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/data/choralsynth_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/ChoralSynth_satb}"
TARGET_SR=44100
GCS_DEST="${GCS_DEST:-gs://csmamba2-484220-ew4-20260206-500242/datasets/processed/ChoralSynth_satb/}"
DO_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload) DO_UPLOAD=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Check dependencies
for cmd in python3 ffmpeg; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: ${cmd} is required but not found." >&2
        exit 1
    fi
done

echo "=== ChoralSynth Processing Pipeline ==="
echo "Work dir:   ${WORK_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Target SR:  ${TARGET_SR}"

# Step 1: Download
mkdir -p "${WORK_DIR}"
ZIP_FILE="${WORK_DIR}/Dataset.zip"
if [[ ! -f "${ZIP_FILE}" ]]; then
    echo "[1/5] Downloading ChoralSynth from Zenodo..."
    curl -L -o "${ZIP_FILE}" "${ZENODO_URL}"
else
    echo "[1/5] Dataset.zip already exists, skipping download."
fi

# Step 2: Extract
EXTRACTED_DIR="${WORK_DIR}/Dataset"
if [[ ! -d "${EXTRACTED_DIR}" ]]; then
    echo "[2/5] Extracting..."
    unzip -q "${ZIP_FILE}" -d "${WORK_DIR}"
else
    echo "[2/5] Already extracted, skipping."
fi

# Step 3: Process each song
echo "[3/5] Processing songs..."
mkdir -p "${OUTPUT_DIR}"

python3 - "${EXTRACTED_DIR}" "${OUTPUT_DIR}" "${TARGET_SR}" <<'PYEOF'
import sys
import os
import json
import subprocess
import re
import numpy as np

extracted_dir = sys.argv[1]
output_dir = sys.argv[2]
target_sr = int(sys.argv[3])

def convert_audio(input_path, output_path, target_sr):
    """Convert audio to WAV at target sample rate, mono->stereo if needed."""
    # First convert to target SR mono/stereo WAV
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', str(target_sr),
        '-ac', '2',  # force stereo
        '-acodec', 'pcm_f32le',
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

def peak_normalize(wav_path):
    """Peak-normalize a WAV file in place."""
    import soundfile as sf
    data, sr = sf.read(wav_path)
    peak = np.abs(data).max()
    if peak > 0 and peak != 1.0:
        data = data / peak * 0.95  # leave small headroom
        sf.write(wav_path, data, sr, subtype='FLOAT')

IGNORE_PATTERNS = (
    'accomp',
    'piano',
    'organ',
    'practice',
)

def normalize_voice_name(filename: str) -> str:
    """Normalize raw voice filenames for robust SATB matching."""
    name = os.path.splitext(filename)[0].lower().strip()
    # Collapse punctuation and repeated spaces to simplify pattern checks.
    name = re.sub(r'[^a-z0-9]+', ' ', name)
    return re.sub(r'\s+', ' ', name).strip()

def infer_satb_label(normalized_name: str):
    """Infer SATB part from a normalized voice track name."""
    if any(pattern in normalized_name for pattern in IGNORE_PATTERNS):
        return None, "auxiliary"

    if normalized_name in {'voice', 'voice 2', 'voice 3', 'voice 4'}:
        return {
            'voice': 'S',
            'voice 2': 'A',
            'voice 3': 'T',
            'voice 4': 'B',
        }[normalized_name], "voice-numbered"

    if 'tenor ii' in normalized_name:
        # Keep single tenor by default; prefer tenor i / generic tenor.
        return None, "auxiliary"

    if 'cantus' in normalized_name or 'soprano' in normalized_name:
        return 'S', "named"
    if 'altus' in normalized_name or 'alto' in normalized_name:
        return 'A', "named"
    if 'tenor i' in normalized_name or 'tenor' in normalized_name:
        return 'T', "named"
    if 'bassus' in normalized_name or 'bass' in normalized_name:
        return 'B', "named"

    return None, "unknown"

processed = 0
skipped = 0

for song_name in sorted(os.listdir(extracted_dir)):
    song_dir = os.path.join(extracted_dir, song_name)
    if not os.path.isdir(song_dir):
        continue

    voices_dir = os.path.join(song_dir, 'voices')
    if not os.path.isdir(voices_dir):
        print(f"  SKIP {song_name}: no voices/ directory")
        skipped += 1
        continue

    out_song_dir = os.path.join(output_dir, song_name)
    os.makedirs(out_song_dir, exist_ok=True)

    # Find voice files
    voice_files = {}
    for fname in sorted(os.listdir(voices_dir)):
        if not fname.lower().endswith('.mp3') and not fname.lower().endswith('.wav'):
            continue
        normalized_name = normalize_voice_name(fname)
        satb_letter, reason = infer_satb_label(normalized_name)

        if reason == "unknown":
            print(f"  WARNING: Unknown voice '{fname}' in {song_name}, skipping")
            continue
        if satb_letter is None:
            print(f"  NOTE: Ignoring auxiliary voice '{fname}' in {song_name}")
            continue

        # If we already have this letter, skip duplicates (e.g., Tenor II)
        if satb_letter in voice_files:
            print(f"  NOTE: Duplicate {satb_letter} in {song_name}, keeping first")
            continue

        voice_files[satb_letter] = os.path.join(voices_dir, fname)

    # Need at least S, A, T, B
    missing = [p for p in ['S', 'A', 'T', 'B'] if p not in voice_files]
    if missing:
        print(f"  SKIP {song_name}: missing parts {missing}")
        skipped += 1
        continue

    # Convert each voice
    print(f"  Processing {song_name}...")
    for part in ['S', 'A', 'T', 'B']:
        out_path = os.path.join(out_song_dir, f'{part}.wav')
        if not os.path.exists(out_path):
            convert_audio(voice_files[part], out_path, target_sr)
            peak_normalize(out_path)

    # Generate mixture as sum of stems
    import soundfile as sf
    mixture_path = os.path.join(out_song_dir, 'mixture.wav')
    if not os.path.exists(mixture_path):
        stems = []
        min_len = float('inf')
        for part in ['S', 'A', 'T', 'B']:
            data, sr = sf.read(os.path.join(out_song_dir, f'{part}.wav'))
            stems.append(data)
            min_len = min(min_len, len(data))
        # Trim all to same length
        stems = [s[:min_len] for s in stems]
        mixture = sum(stems)
        # Peak-normalize mixture
        peak = np.abs(mixture).max()
        if peak > 0:
            mixture = mixture / peak * 0.95
        sf.write(mixture_path, mixture, target_sr, subtype='FLOAT')

    processed += 1

print(f"\nDone: {processed} songs processed, {skipped} skipped")
PYEOF

# Step 4: Generate summary
echo "[4/5] Generating summary..."
python3 - "${OUTPUT_DIR}" <<'PYEOF2'
import os
import json
import sys
import soundfile as sf

output_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

summary = {"dataset": "ChoralSynth_satb", "songs": []}
total_duration = 0.0

for song_name in sorted(os.listdir(output_dir)):
    song_dir = os.path.join(output_dir, song_name)
    mix_path = os.path.join(song_dir, 'mixture.wav')
    if not os.path.isfile(mix_path):
        continue
    info = sf.info(mix_path)
    duration = info.duration
    total_duration += duration
    has_parts = {p: os.path.exists(os.path.join(song_dir, f'{p}.wav')) for p in ['S', 'A', 'T', 'B']}
    summary["songs"].append({
        "name": song_name,
        "duration_s": round(duration, 2),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "parts": has_parts,
    })

summary["total_songs"] = len(summary["songs"])
summary["total_duration_min"] = round(total_duration / 60, 1)

with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary: {summary['total_songs']} songs, {summary['total_duration_min']} min")
PYEOF2

# Step 5: Upload to GCS
if [[ "${DO_UPLOAD}" == "true" ]]; then
    echo "[5/5] Uploading to GCS: ${GCS_DEST}"
    gsutil -m rsync -r "${OUTPUT_DIR}" "${GCS_DEST}"
    echo "Upload complete."
else
    echo "[5/5] Skipping GCS upload (use --upload to enable)."
fi

echo "=== Done ==="
