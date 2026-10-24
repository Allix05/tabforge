"""Fine-tune Basic Pitch on GuitarSet (+ synthetic metal riffs) with amp / mix
augmentations.

Players 00-03 train, 04 validates (checkpoint selection), 05 is the untouched
test set reported in the README. The shipped model was trained in stages:
  python train.py --run guitar  --steps 6000
  python train.py --run guitar2 --init guitar  --steps 15000 --lr 1.5e-4
  python train.py --run guitar3 --init guitar2 --steps 8000  --lr 1e-4
  python train.py --run guitar4 --init guitar3 --steps 12000 --lr 1.5e-4 --p-synth 0.4
  python train.py --run guitar5 --init guitar4 --steps 8000  --lr 8e-5  --p-synth 0.35
  python train.py --run guitar6 --init guitar5 --steps 6000  --lr 8e-5  --p-synth 0.3 --p-palm 0.3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from augment import condition, palm_mute, random_training_condition
from common import (AUDIO_N_SAMPLES, FPS, HOP, N_FRAMES, SR, TEST_PLAYER, decode, device,
                    load_audio, load_model, load_notes, note_f1, run_model, synth_ids, tracks)

VAL_PLAYER = "04"


def split_ids():
    ids = tracks()
    train = [t for t in ids if t[:2] not in (TEST_PLAYER, VAL_PLAYER)]
    val = [t for t in ids if t.startswith(VAL_PLAYER + "_")]
    return train, val


def targets(notes, t0, ratio=1.0, shift=0):
    """Frame / onset / contour targets for a 2 s window starting at t0 (source time)."""
    note = np.zeros((N_FRAMES, 88), np.float32)
    onset = np.zeros((N_FRAMES, 88), np.float32)
    contour = np.zeros((N_FRAMES, 264), np.float32)
    span = AUDIO_N_SAMPLES / SR * ratio
    for s, e, m, _string in notes:
        if e < t0 or s > t0 + span:
            continue
        m = m + shift
        p = int(round(m)) - 21
        if not 0 <= p < 88:
            continue
        fs = (s - t0) / ratio * FPS
        fe = (e - t0) / ratio * FPS
        a, b = max(0, int(np.ceil(fs))), min(N_FRAMES, int(np.ceil(fe)))
        note[a:b, p] = 1
        k = int(round(fs))
        if 0 <= k < N_FRAMES:
            onset[k, p] = 1
        c = int(round((m - 21) * 3))
        for dc, v in ((0, 1.0), (-1, 0.5), (1, 0.5)):
            if 0 <= c + dc < 264:
                contour[a:b, c + dc] = np.maximum(contour[a:b, c + dc], v)
    return note, onset, contour


class Windows(IterableDataset):
    def __init__(self, ids, seed=0, synth=(), p_synth=0.0, p_palm=0.0):
        self.ids = ids
        self.seed = seed
        self.synth = list(synth)
        self.p_synth = p_synth
        self.p_palm = p_palm

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) + int(time.time()))
        audio = {}
        while True:
            synthetic = bool(self.synth) and rng.random() < self.p_synth
            pool = self.synth if synthetic else self.ids
            t = pool[rng.integers(len(pool))]
            src = "synth" if synthetic else "pickup" if rng.random() < 0.75 else "mic"
            key = (t, src)
            if key not in audio:
                if len(audio) > 400:
                    audio.clear()
                audio[key] = load_audio(t, src)
            x = audio[key]
            notes = load_notes(t)
            shift = int(rng.integers(-2, 3)) if rng.random() < 0.5 else 0
            ratio = 2 ** (shift / 12)  # read `ratio` x more source audio, squeeze it into the window
            need = int(np.ceil(AUDIO_N_SAMPLES * ratio)) + 2
            if len(x) <= need:
                continue
            i0 = int(rng.integers(0, len(x) - need))
            seg = x[i0:i0 + need]
            seg = np.interp(np.arange(AUDIO_N_SAMPLES) * ratio, np.arange(len(seg)), seg).astype(np.float32)
            t0 = i0 / SR
            # notes in window time (after the pitch shift) drive the synthetic bass
            local = [((s - t0) / ratio, (e - t0) / ratio, m + shift, st) for s, e, m, st in notes]
            if not synthetic and rng.random() < self.p_palm:
                seg, local = palm_mute(seg, local, rng)
                seg = seg.astype(np.float32)
                # back to source time for the targets
                notes = [(t0 + s * ratio, t0 + e * ratio, m - shift, st) for s, e, m, st in local]
            y = random_training_condition(seg, local, 0, rng, synthetic)
            yield (torch.from_numpy(y), *map(torch.from_numpy, targets(notes, t0, ratio, shift)))


def weighted_bce(pred, target, pos_weight):
    pred = pred.clamp(1e-6, 1 - 1e-6)
    loss = -(pos_weight * target * torch.log(pred) + (1 - pos_weight) * (1 - target) * torch.log(1 - pred))
    return loss.mean()


def build_val(val_ids, n=20, synth_val=()):
    """Fixed augmented validation audio: the conditions we care about most,
    on real GuitarSet takes and (if given) held-out synthetic metal riffs."""
    rng = np.random.default_rng(1234)
    items = []
    for t in val_ids[:n]:
        x = load_audio(t, "pickup")
        notes = load_notes(t)
        for kind in ("highgain", "metalmix"):
            items.append((t, kind, condition(x, notes, 0, rng, kind), notes))
    for t in list(synth_val)[:40]:
        x = load_audio(t, "synth")
        notes = load_notes(t)
        kind = "highgain" if len(items) % 2 else "metalmix"
        items.append((t, "synth-" + kind, condition(x, notes, 0, rng, kind), notes))
    return items


def validate(model, items):
    """Mean of (real, synthetic) F1, each at its best threshold pair."""
    model.eval()
    grid = [(o, f) for o in (0.4, 0.6, 0.8) for f in (0.25, 0.4, 0.55)]
    groups = {}
    for t, _kind, y, notes in items:
        out = run_model(model, y)
        g = groups.setdefault("synth" if t.startswith("synth_") else "real", {k: [] for k in grid})
        for k in grid:
            g[k].append(note_f1(decode(out, *k), notes)["f"])
    parts = {name: max(float(np.mean(s[k])) for k in grid) for name, s in groups.items()}
    best = max(grid, key=lambda k: np.mean(groups["real"][k]))
    model.train()
    freeze_bn(model)
    print("   val " + "  ".join(f"{k} {v:.3f}" for k, v in parts.items()), flush=True)
    return float(np.mean(list(parts.values()))), best


def freeze_bn(model):
    # small batches: keep the pretrained running statistics
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--run", default="guitar", help="run name; checkpoints go to C:/tftrain/runs/<run>")
    ap.add_argument("--init", default=None, help="start from another run's best checkpoint")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--p-synth", type=float, default=0.0, help="share of synthetic metal riffs in each batch")
    ap.add_argument("--p-palm", type=float, default=0.0, help="share of real takes turned into palm-muted playing")
    args = ap.parse_args()
    out = Path("C:/tftrain/runs") / args.run
    out.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = split_ids()
    synth_train, synth_val = (synth_ids("train"), synth_ids("val")) if args.p_synth > 0 else ([], [])
    print(f"train tracks {len(train_ids)} (+{len(synth_train)} synthetic)  val tracks {len(val_ids)}  device {device()}")
    model = load_model(f"C:/tftrain/runs/{args.init}/best.pth" if args.init else None)
    val_items = build_val(val_ids, synth_val=synth_val)
    best_f, best_th = validate(model, val_items)
    print(f"pretrained  val F1 {best_f:.3f} @ {best_th}", flush=True)
    log = [{"step": 0, "val_f1": best_f, "thresholds": best_th}]
    torch.save(model.state_dict(), out / "best.pth")

    model.train()
    freeze_bn(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    loader = DataLoader(Windows(train_ids, synth=synth_train, p_synth=args.p_synth, p_palm=args.p_palm), batch_size=args.batch,
                        num_workers=args.workers, persistent_workers=True)
    step, t_start, running = 0, time.time(), []
    for audio, note, onset, contour in loader:
        audio, note, onset, contour = (a.to(device()) for a in (audio, note, onset, contour))
        o = model(audio)
        loss = (F.binary_cross_entropy(o["note"].clamp(1e-6, 1 - 1e-6), note)
                + weighted_bce(o["onset"], onset, 0.95)
                + 0.5 * F.binary_cross_entropy(o["contour"].clamp(1e-6, 1 - 1e-6), contour))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        step += 1
        if step % 100 == 0:
            print(f"step {step}  loss {np.mean(running):.4f}  {time.time() - t_start:.0f}s", flush=True)
            running = []
        if step % args.val_every == 0 or step == args.steps:
            f, th = validate(model, val_items)
            log.append({"step": step, "val_f1": f, "thresholds": th})
            mark = ""
            if f > best_f:
                best_f, best_th = f, th
                torch.save(model.state_dict(), out / "best.pth")
                mark = "  *best*"
            print(f"step {step}  val F1 {f:.3f} @ {th}{mark}", flush=True)
            json.dump(log, open(out / "log.json", "w"), indent=1)
        if step >= args.steps:
            break
    print(f"done. best val F1 {best_f:.3f} @ {best_th}")


if __name__ == "__main__":
    main()
