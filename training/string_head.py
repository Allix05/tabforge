"""String head: predicts which of the six strings plays each note, from timbre.

The same pitch sounds different on different strings (string gauge, wound vs
plain, where along the string it's fretted), and GuitarSet's hexaphonic
annotation tells us the true string of every note. A small CNN on top of Basic
Pitch's harmonic CQT learns that; the browser then mixes its probabilities with
the playability-based fingering search.

  python string_head.py --base guitar6 --steps 8000        # train
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from augment import condition, random_training_condition
from common import AUDIO_N_SAMPLES, FPS, N_FRAMES, OPEN_STRINGS, SR, device, load_audio, load_model, load_notes, synth_ids
from basic_pitch_torch.model import get_cqt
from scoring import note_creation
from train import split_ids, targets

MAX_FRET = 24


def string_mask():
    """(6, 88) True where pitch p can be played on string s."""
    m = np.zeros((6, 88), bool)
    for s, open_ in enumerate(OPEN_STRINGS):
        for p in range(88):
            m[s, p] = 0 <= p + 21 - open_ <= MAX_FRET
    return torch.from_numpy(m)


class TabModel(nn.Module):
    """Basic Pitch (frozen) + string head. Outputs note/onset/contour like
    Basic Pitch plus `string`: (batch, time, 88, 6) string probabilities."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 3), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(48, 6, kernel_size=(5, 1), padding=(2, 0)),
        )
        self.register_buffer("mask", string_mask())

    def forward(self, audio):
        out = self.base(audio)
        with torch.no_grad():
            cqt = get_cqt(audio, len(self.base.stack_harmonics), True, self.base.bn_layer)
            hcqt = self.base.hs(cqt)  # (B, 8, T, 264)
        logits = self.head(hcqt)  # (B, 6, T, 88)
        logits = logits.masked_fill(~self.mask[None, :, None, :], -1e4)
        out["string_logits"] = logits
        out["string"] = torch.softmax(logits, dim=1).permute(0, 2, 3, 1)  # (B, T, 88, 6)
        return out


def string_targets(notes, t0, ratio=1.0, shift=0):
    """(T, 88) int: true string index for every active (frame, pitch), else -1."""
    lab = np.full((N_FRAMES, 88), -1, np.int64)
    span = AUDIO_N_SAMPLES / SR * ratio
    for s, e, m, string in notes:
        if e < t0 or s > t0 + span:
            continue
        p = int(round(m + shift)) - 21
        if not 0 <= p < 88 or not 0 <= p + 21 - OPEN_STRINGS[string] <= MAX_FRET:
            continue
        a = max(0, int(np.ceil((s - t0) / ratio * FPS)))
        b = min(N_FRAMES, int(np.ceil((e - t0) / ratio * FPS)))
        lab[a:b, p] = string
    return lab


class StringWindows(IterableDataset):
    def __init__(self, ids, synth=(), p_synth=0.3):
        self.ids = ids
        self.synth = list(synth)
        self.p_synth = p_synth

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng((info.id if info else 0) + int(time.time()))
        cache = {}
        while True:
            synthetic = bool(self.synth) and rng.random() < self.p_synth
            pool = self.synth if synthetic else self.ids
            t = pool[rng.integers(len(pool))]
            src = "synth" if synthetic else "pickup" if rng.random() < 0.6 else "mic"
            if (t, src) not in cache:
                if len(cache) > 400:
                    cache.clear()
                cache[(t, src)] = load_audio(t, src)
            x = cache[(t, src)]
            notes = load_notes(t)
            shift = int(rng.integers(-2, 3)) if rng.random() < 0.4 else 0
            ratio = 2 ** (shift / 12)
            need = int(np.ceil(AUDIO_N_SAMPLES * ratio)) + 2
            i0 = int(rng.integers(0, len(x) - need))
            seg = np.interp(np.arange(AUDIO_N_SAMPLES) * ratio, np.arange(need), x[i0:i0 + need]).astype(np.float32)
            t0 = i0 / SR
            local = [((s - t0) / ratio, (e - t0) / ratio, m + shift, st) for s, e, m, st in notes]
            y = random_training_condition(seg, local, 0, rng, synthetic)
            lab = string_targets(notes, t0, ratio, shift)
            if (lab >= 0).sum() < 5:
                continue
            yield torch.from_numpy(y), torch.from_numpy(lab)


@torch.no_grad()
def string_accuracy(model, items):
    """Accuracy of the most likely string at each true note's onset frame."""
    model.eval()
    ok = total = 0
    for y, notes in items:
        out = run_tab_model(model, y)
        # windows overlap by a non-integer number of frames, so map time -> frame
        # with Basic Pitch's own (piecewise) frame clock
        times = note_creation.model_frames_to_time(len(out))
        for s, _e, m, string in notes:
            p = int(round(m)) - 21
            k = int(np.searchsorted(times, s))
            if 0 <= p < 88 and k + 3 < len(out):
                probs = out[k:k + 4, p].mean(0)  # first few frames of the note
                ok += int(np.argmax(probs) == string)
                total += 1
    model.head.train()
    return ok / max(1, total)


def run_tab_model(model, audio):
    """Sliding-window inference returning only the string probabilities (T, 88, 6)."""
    from common import AUDIO_N_SAMPLES as N, HOP
    overlap_frames = 30
    hop = N - overlap_frames * HOP
    x = np.concatenate([np.zeros(overlap_frames * HOP // 2, np.float32), audio])
    n = int(np.ceil((len(x) - N) / hop)) + 1
    x = np.pad(x, (0, (n - 1) * hop + N - len(x)))
    windows = np.stack([x[i * hop: i * hop + N] for i in range(n)])
    outs = []
    for i in range(0, n, 16):
        outs.append(model(torch.from_numpy(windows[i:i + 16]).to(device()))["string"].cpu().numpy())
    v = np.concatenate(outs)[:, overlap_frames // 2: -overlap_frames // 2]
    return v.reshape(-1, 88, 6)[: int(np.floor(len(audio) * 86 / SR))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="guitar6", help="note model run to build on")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = Path("C:/tftrain/runs/strings")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_model(f"C:/tftrain/runs/{args.base}/best.pth")
    model = TabModel(base).to(device())
    model.base.eval()
    train_ids, val_ids = split_ids()
    rng = np.random.default_rng(5)
    val = []
    for t in val_ids[:24]:
        notes = load_notes(t)
        val.append((load_audio(t, "mic"), notes))
        val.append((condition(load_audio(t, "pickup"), notes, 0, rng, "highgain"), notes))
    best = string_accuracy(model, val)
    print(f"untrained string accuracy {best:.3f}", flush=True)

    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    loader = DataLoader(StringWindows(train_ids, synth=synth_ids("train")), batch_size=args.batch,
                        num_workers=args.workers, persistent_workers=True)
    step, running, t0 = 0, [], time.time()
    for audio, lab in loader:
        audio, lab = audio.to(device()), lab.to(device())
        logits = model(audio)["string_logits"]  # (B, 6, T, 88)
        loss = F.cross_entropy(logits, lab, ignore_index=-1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        step += 1
        if step % 200 == 0:
            print(f"step {step}  loss {np.mean(running):.4f}  {time.time() - t0:.0f}s", flush=True)
            running = []
        if step % 1000 == 0 or step == args.steps:
            acc = string_accuracy(model, val)
            mark = ""
            if acc > best:
                best = acc
                torch.save(model.head.state_dict(), out_dir / "head.pth")
                mark = "  *best*"
            print(f"step {step}  val string accuracy {acc:.3f}{mark}", flush=True)
        if step >= args.steps:
            break
    print(f"done. best string accuracy {best:.3f}")


if __name__ == "__main__":
    main()
