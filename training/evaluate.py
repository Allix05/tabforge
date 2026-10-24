"""Measure transcription accuracy on held-out GuitarSet players and calibrate
the automatic sensitivity.

For each model and condition it reports onset-note F1 at:
  - the browser's old fixed thresholds,
  - the best single fixed thresholds,
  - the automatic per-song thresholds (rule fitted on the validation player).

It also dumps predicted + reference notes as JSON so the JavaScript cleanup
and fret assignment can be scored on the same real data (eval_js.mjs).

  python evaluate.py                              # stock Basic Pitch
  python evaluate.py --run guitar5 --name guitar5   # a fine-tuned run
"""

# Heavy imports (torch) live inside main(): on Windows every pool worker
# re-imports this file, and 16 copies of PyTorch don't fit in RAM.
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scoring import decode, grid_scores

CONDITIONS = ["clean", "highgain", "metalmix", "stem"]
REPORT = CONDITIONS + ["synthmetal"]
ONSETS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FRAMES = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
OLD_DEFAULT = (0.53, 0.35)  # what TabForge used before (sensitivity 0.5)


def stats(out):
    """Per-song activation levels the auto rule keys on (mirrored in js/sensitivity.js)."""
    peaks = out["onset"].max(axis=1)
    return {"onset_p95": float(np.percentile(peaks, 95)), "note_p99": float(np.percentile(out["note"], 99.5))}


def auto_thresholds(st, rule):
    a, b = rule
    return float(np.clip(a * st["onset_p95"], 0.2, 0.9)), float(np.clip(b * st["note_p99"], 0.15, 0.65))


def nearest(o, f):
    return min(ONSETS, key=lambda v: abs(v - o)), min(FRAMES, key=lambda v: abs(v - f))


def items_for(player, seed):
    from augment import condition
    from common import load_audio, load_notes, synth_ids, tracks
    rng = np.random.default_rng(seed)
    out = []
    for t in [t for t in tracks() if t.startswith(player + "_")]:
        notes = load_notes(t)
        for kind in CONDITIONS:
            x = load_audio(t, "mic" if kind == "clean" else "pickup")
            out.append((t, kind, condition(x, notes, 0, rng, kind), notes))
    # held-out synthetic metal riffs (chugs, palm mutes, fast riffs): first half
    # for calibration, second half for the test report
    from train import VAL_PLAYER
    synth = synth_ids("val")
    synth = synth[:100] if player == VAL_PLAYER else synth[100:]
    for i, t in enumerate(synth):
        notes = load_notes(t)
        kind = "metalmix" if i % 2 else "stem"
        out.append((t, "synthmetal", condition(load_audio(t, "synth"), notes, 0, rng, kind), notes))
    return out


def score_model(model, items):
    from common import run_model
    outs = [run_model(model, y) for _t, _k, y, _n in items]  # GPU
    jobs = [({"note": o["note"], "onset": o["onset"]}, it[3], ONSETS, FRAMES) for o, it in zip(outs, items)]
    with ProcessPoolExecutor(max_workers=8) as pool:  # note decoding is pure Python
        grids = list(pool.map(grid_scores, jobs, chunksize=4))
    return [{"track": t, "kind": kind, "grid": g, "stats": stats(o), "out": o, "notes": notes}
            for (t, kind, _y, notes), o, g in zip(items, outs, grids)]


def fit_rule(rows):
    best, best_f = None, -1
    for a in np.arange(0.3, 1.31, 0.05):
        for b in np.arange(0.3, 1.31, 0.05):
            fs = [r["grid"][nearest(*auto_thresholds(r["stats"], (a, b)))]["f"] for r in rows]
            if np.mean(fs) > best_f:
                best, best_f = (round(float(a), 2), round(float(b), 2)), float(np.mean(fs))
    return best, best_f


def report(rows, rule):
    res = {}
    fixed = max(((o, f) for o in ONSETS for f in FRAMES), key=lambda g: np.mean([r["grid"][g]["f"] for r in rows]))
    old = nearest(*OLD_DEFAULT)
    mean = lambda xs, k: float(np.mean([x[k] for x in xs]))  # noqa: E731
    for kind in REPORT + ["all"]:
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        autos = [r["grid"][nearest(*auto_thresholds(r["stats"], rule))] for r in sel]
        res[kind] = {
            "old_default": {k: mean([r["grid"][old] for r in sel], k) for k in "prf"},
            "best_fixed": {k: mean([r["grid"][fixed] for r in sel], k) for k in "prf"},
            "auto": {k: mean(autos, k) for k in "prf"},
        }
    return res, fixed


def dump_notes(rows, rule, path):
    data = []
    for r in rows:
        pred = decode(r["out"], *auto_thresholds(r["stats"], rule))
        data.append({
            "track": r["track"], "kind": r["kind"],
            "pred": [[float(s), float(e), int(p), float(a)] for s, e, p, a, *_ in pred],
            "ref": [[s, e, m, st] for s, e, m, st in r["notes"]],
        })
    json.dump(data, open(path, "w"))


def main():
    from common import TEST_PLAYER, load_model
    from train import VAL_PLAYER

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="training run name under C:/tftrain/runs (omit for the stock model)")
    ap.add_argument("--name", default="pretrained")
    ap.add_argument("--out", default="C:/tftrain/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(f"C:/tftrain/runs/{args.run}/best.pth" if args.run else None)

    val_rows = score_model(model, items_for(VAL_PLAYER, 7))
    rule, val_f = fit_rule(val_rows)
    print(f"[{args.name}] auto rule {rule} (val F1 {val_f:.3f})", flush=True)

    test_rows = score_model(model, items_for(TEST_PLAYER, 11))
    res, fixed = report(test_rows, rule)
    print(f"[{args.name}] test player {TEST_PLAYER}, best fixed thresholds {fixed}")
    for kind, r in res.items():
        print(f"  {kind:9s} old {r['old_default']['f']:.3f} | fixed {r['best_fixed']['f']:.3f} | auto {r['auto']['f']:.3f}"
              f"  (auto P {r['auto']['p']:.3f} R {r['auto']['r']:.3f})")
    json.dump({"rule": rule, "fixed": fixed, "test": res}, open(out / f"{args.name}.json", "w"), indent=1)
    dump_notes(test_rows, rule, out / f"{args.name}_notes.json")


if __name__ == "__main__":
    main()
