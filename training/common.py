"""Shared helpers for fine-tuning Basic Pitch on guitar.

Paths are configurable through environment variables so the multi-GB data
never has to live inside the repo:
  TABFORGE_DATA     raw downloads (GuitarSet zips)          default C:/tftrain/data
  TABFORGE_CACHE    decoded 22.05 kHz audio + labels        default C:/tftrain/cache
  BPTORCH           checkout of gudgud96/basic-pitch-torch  default C:/tftrain/bptorch
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

DATA = Path(os.environ.get("TABFORGE_DATA", "C:/tftrain/data"))
CACHE = Path(os.environ.get("TABFORGE_CACHE", "C:/tftrain/cache"))
BPTORCH = Path(os.environ.get("BPTORCH", "C:/tftrain/bptorch"))
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BPTORCH))

from basic_pitch_torch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP  # noqa: E402
from basic_pitch_torch.model import BasicPitchTorch  # noqa: E402
from scoring import decode, note_f1  # noqa: E402,F401  (re-exported)

SR = AUDIO_SAMPLE_RATE
HOP = FFT_HOP
FPS = SR / HOP  # true frame rate of the network (86.13 Hz)
N_FRAMES = 172  # frames per 2 s window
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # GuitarSet is in standard tuning
TEST_PLAYER = "05"  # held out entirely: never seen during fine-tuning


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(weights=None):
    model = BasicPitchTorch()
    path = weights or BPTORCH / "assets/basic_pitch_pytorch_icassp_2022.pth"
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    return model.to(device()).eval()


def tracks(split=None):
    """GuitarSet track ids from the cache; split = 'train' | 'test' | None."""
    ids = sorted(p.stem for p in (CACHE / "labels").glob("*.json"))
    if split == "test":
        return [t for t in ids if t.startswith(TEST_PLAYER + "_")]
    if split == "train":
        return [t for t in ids if not t.startswith(TEST_PLAYER + "_")]
    return ids


def load_audio(track, source):
    if track.startswith("synth_"):
        return np.load(CACHE / "synth" / "audio" / f"{track}.npy").astype(np.float32)
    return np.load(CACHE / source / f"{track}.npy").astype(np.float32)


def load_notes(track):
    """[(start, end, midi_float, string)] from the GuitarSet / synthetic annotation."""
    if track.startswith("synth_"):
        return json.load(open(CACHE / "synth" / "labels" / f"{track}.json"))
    return json.load(open(CACHE / "labels" / f"{track}.json"))


def synth_ids(split):
    """Synthetic riffs (synth_riffs.py): the last 200 are held out."""
    ids = sorted(p.stem for p in (CACHE / "synth" / "labels").glob("*.json"))
    return ids[-200:] if split == "val" else ids[:-200]


@torch.no_grad()
def run_model(model, audio):
    """Sliding-window inference exactly like the browser (30-frame overlap)."""
    overlap_frames = 30
    overlap = overlap_frames * HOP
    hop = AUDIO_N_SAMPLES - overlap
    x = np.concatenate([np.zeros(overlap // 2, np.float32), audio])
    n = int(np.ceil((len(x) - AUDIO_N_SAMPLES) / hop)) + 1
    x = np.pad(x, (0, (n - 1) * hop + AUDIO_N_SAMPLES - len(x)))
    windows = np.stack([x[i * hop: i * hop + AUDIO_N_SAMPLES] for i in range(n)])
    outs = {"note": [], "onset": [], "contour": []}
    for i in range(0, n, 32):
        o = model(torch.from_numpy(windows[i:i + 32]).to(device()))
        for k in outs:
            outs[k].append(o[k].cpu().numpy())
    n_keep = int(np.floor(len(audio) * (86 / SR)))
    res = {}
    for k, v in outs.items():
        v = np.concatenate(v)[:, overlap_frames // 2: -overlap_frames // 2]
        res[k] = v.reshape(-1, v.shape[-1])[:n_keep]
    return res

