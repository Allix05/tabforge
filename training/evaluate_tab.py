"""Dump the full guitar model's notes *with string probabilities* on the
held-out player, so eval_js.mjs can score end-to-end tabs and fit how much the
fret search should trust the string head.

  python evaluate_tab.py --base guitar7      # -> C:/tftrain/results/tab_notes.json
"""

import argparse
import json

import numpy as np
import torch

from common import AUDIO_N_SAMPLES, HOP, SR, TEST_PLAYER, device, load_model
from evaluate import auto_thresholds, items_for, stats
from scoring import decode, note_f1
from string_head import TabModel


@torch.no_grad()
def run_all(model, audio):
    overlap_frames = 30
    hop = AUDIO_N_SAMPLES - overlap_frames * HOP
    x = np.concatenate([np.zeros(overlap_frames * HOP // 2, np.float32), audio])
    n = int(np.ceil((len(x) - AUDIO_N_SAMPLES) / hop)) + 1
    x = np.pad(x, (0, (n - 1) * hop + AUDIO_N_SAMPLES - len(x)))
    windows = np.stack([x[i * hop: i * hop + AUDIO_N_SAMPLES] for i in range(n)])
    outs = {"note": [], "onset": [], "string": []}
    for i in range(0, n, 16):
        o = model(torch.from_numpy(windows[i:i + 16]).to(device()))
        for k in outs:
            outs[k].append(o[k].cpu().numpy())
    keep = int(np.floor(len(audio) * 86 / SR))
    res = {}
    for k, v in outs.items():
        v = np.concatenate(v)[:, overlap_frames // 2: -overlap_frames // 2]
        res[k] = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])[:keep]
    return res


# the automatic-sensitivity rule shipped in js/sensitivity.js (guitar model)
RULE = (0.75, 0.55, 0.4, 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="guitar7")
    args = ap.parse_args()
    tab = TabModel(load_model(f"C:/tftrain/runs/{args.base}/best.pth"))
    tab.head.load_state_dict(torch.load("C:/tftrain/runs/strings/head.pth", map_location="cpu"))
    tab = tab.to(device()).eval()
    rule = RULE

    data = []
    for t, kind, y, notes in items_for(TEST_PLAYER, 11):
        out = run_all(tab, y)
        o, fr = auto_thresholds(stats(out), rule)
        pred = decode(out, o, fr)
        rows = []
        for s, e, p, a, k, dur in pred:
            # same as js/guitar-model.js stringProbsFor: mean over the attack frames
            probs = out["string"][k:k + max(1, min(4, dur)), p - 21].mean(0) if 0 <= p - 21 < 88 else np.full(6, 1 / 6)
            rows.append([float(s), float(e), int(p), float(a), [round(float(v), 4) for v in probs]])
        data.append({"track": t, "kind": kind, "pred": rows, "ref": notes, "f1": note_f1(pred, notes)["f"]})
    json.dump({"rule": rule, "items": data}, open("C:/tftrain/results/tab_notes.json", "w"))
    for kind in ("clean", "highgain", "metalmix", "stem", "synthmetal", "idmt"):
        print(f"  {kind:9s} note F1 {np.mean([d['f1'] for d in data if d['kind'] == kind]):.3f}")


if __name__ == "__main__":
    main()
