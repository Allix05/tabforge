"""Procedural metal / rock riff generator with exact tab labels.

GuitarSet has no palm mutes, chugs or riffs at 200 BPM, which is exactly what
metal is made of, so the model had never heard them. This renders thousands of
labelled riffs (palm-muted chugs, power chords, gallops, pedal-tone riffs,
chromatic runs, tremolo leads) with a plucked-string model (darker wound
strings, palm-mute damping, pick attack); the training loop then puts them
through the high-gain amp / band-mix augmentations like the real recordings.

  python synth_riffs.py --n 3000      # -> C:/tftrain/cache/synth/{audio,labels}
"""

import argparse
import json

import numpy as np
from scipy.signal import lfilter, resample_poly

from common import CACHE, OPEN_STRINGS

SR_SYN = 44100
CLIP = 6.0


def pluck(midi, string, dur, palm, vel, rng):
    """Karplus-Strong string via lfilter (comb + averaging), 44.1 kHz."""
    f0 = 440 * 2 ** ((midi - 69) / 12)
    N = max(2, int(round(SR_SYN / f0 - 0.5)))
    wound = string <= 2
    bright = rng.uniform(0.35, 0.55) if wound else rng.uniform(0.55, 0.8)
    exc = rng.standard_normal(N)
    exc = lfilter([bright], [1, -(1 - bright)], exc)  # darker excitation for wound strings
    pick_pos = rng.uniform(0.08, 0.25)  # pick position comb
    d = max(1, int(pick_pos * N))
    exc[d:] -= exc[:-d]
    if palm:
        g = rng.uniform(0.86, 0.95)
        ring = min(dur, rng.uniform(0.08, 0.2))
    else:
        g = rng.uniform(0.993, 0.999)
        ring = dur
    length = int((ring + 0.03) * SR_SYN)
    x = np.zeros(length)
    x[:N] = exc
    # y[n] = x[n] + g/2 (y[n-N] + y[n-N-1])
    a = np.zeros(N + 2)
    a[0] = 1
    a[N] = a[N + 1] = -g / 2
    y = lfilter([1.0], a, x)
    if palm:
        y = lfilter([0.35], [1, -0.65], y)  # the palm also kills the highs
    attack = rng.standard_normal(int(0.004 * SR_SYN)) * 0.3 * np.exp(-np.arange(int(0.004 * SR_SYN)) / 40)
    y[:len(attack)] += attack
    fade = int(0.01 * SR_SYN)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y / (np.max(np.abs(y)) + 1e-9) * vel, ring


SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": list(range(12)),
    "blues": [0, 3, 5, 6, 7, 10],
}


def riff(rng):
    """A list of note events (start_s, dur_s, string, fret, palm, vel)."""
    bpm = rng.uniform(80, 230)
    s16 = 60 / bpm / 4
    scale = SCALES[rng.choice(list(SCALES))]
    events = []
    t = rng.uniform(0, 0.2)
    style = rng.choice(["chug", "power", "pedal", "gallop", "lead", "mixed"], p=[0.15, 0.2, 0.25, 0.1, 0.15, 0.15])
    root_string = int(rng.choice([0, 0, 1]))

    def fret_in_scale(lo, hi):
        f = int(rng.integers(lo, hi + 1))
        while (f - 0) % 12 not in scale:
            f = int(rng.integers(lo, hi + 1))
        return f

    while t < CLIP - 0.3:
        mode = style if style != "mixed" else rng.choice(["chug", "power", "pedal", "gallop", "lead"])
        if mode == "chug":
            step = s16 * rng.choice([1, 2])
            for _ in range(int(rng.integers(4, 9))):
                events.append((t, step, root_string, 0 if rng.random() < 0.8 else fret_in_scale(1, 5), True, rng.uniform(0.6, 1)))
                t += step
        elif mode == "power":
            f = fret_in_scale(0, 12)
            s = int(rng.choice([0, 1, 2]))
            length = s16 * rng.choice([2, 4, 8])
            palm = rng.random() < 0.3
            vel = rng.uniform(0.7, 1)
            events.append((t, length, s, f, palm, vel))
            events.append((t + rng.uniform(0, 0.01), length, s + 1, f + 2, palm, vel))
            if rng.random() < 0.5:
                events.append((t + rng.uniform(0, 0.015), length, s + 2, f + 2, palm, vel * 0.9))
            t += length
        elif mode == "pedal":  # Master-of-Puppets style: open E chug between fretted notes
            run = sorted({fret_in_scale(1, 12) for _ in range(6)}, reverse=rng.random() < 0.5)
            for f in run:
                for _ in range(int(rng.integers(1, 3))):
                    events.append((t, s16, 0, 0, True, rng.uniform(0.6, 0.9)))
                    t += s16
                s = int(rng.choice([0, 1])) if f > 5 else 0
                events.append((t, s16 * rng.choice([1, 2]), s, f if s == 0 else max(0, f - 5), rng.random() < 0.5, rng.uniform(0.7, 1)))
                t += s16 * 2
        elif mode == "gallop":
            for _ in range(int(rng.integers(2, 5))):
                for step in (2, 1, 1):
                    events.append((t, s16 * step, root_string, 0, True, rng.uniform(0.6, 1)))
                    t += s16 * step
        else:  # lead / tremolo line on the high strings
            s = int(rng.integers(2, 6))
            f = fret_in_scale(3, 15)
            for _ in range(int(rng.integers(6, 16))):
                if rng.random() < 0.4:
                    f = max(0, min(20, f + int(rng.choice([-3, -2, -1, 1, 2, 3]))))
                if rng.random() < 0.15:
                    s = max(0, min(5, s + int(rng.choice([-1, 1]))))
                step = s16 * rng.choice([1, 1, 2])
                events.append((t, step, s, f, False, rng.uniform(0.5, 1)))
                t += step
        t += s16 * rng.choice([0, 0, 1, 2])
    return [e for e in events if e[0] < CLIP - 0.1]


def render(events, rng):
    out = np.zeros(int((CLIP + 1) * SR_SYN))
    labels = []
    by_string = {}
    for e in sorted(events):
        by_string.setdefault(e[2], []).append(e)
    for string, evs in by_string.items():
        for i, (t, dur, s, f, palm, vel) in enumerate(evs):
            midi = OPEN_STRINGS[s] + f
            nxt = evs[i + 1][0] if i + 1 < len(evs) else CLIP + 1
            dur = min(dur * rng.uniform(1.0, 1.6), nxt - t)  # a string stops when it's played again
            y, ring = pluck(midi, s, dur, palm, vel, rng)
            ring = min(ring, nxt - t)
            i0 = int(t * SR_SYN)
            n = min(len(y), int(ring * SR_SYN) + int(0.01 * SR_SYN), len(out) - i0)
            out[i0:i0 + n] += y[:n]
            labels.append([round(t, 5), round(t + max(0.05, ring), 5), float(midi), int(s)])
    out = out[: int(CLIP * SR_SYN)]
    out = resample_poly(out, 1, 2)
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.8).astype(np.float16), sorted(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    adir, ldir = CACHE / "synth" / "audio", CACHE / "synth" / "labels"
    adir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        audio, labels = render(riff(rng), rng)
        np.save(adir / f"synth_{i:05d}.npy", audio)
        json.dump(labels, open(ldir / f"synth_{i:05d}.json", "w"))
        if i % 500 == 0:
            print(i, len(labels), "notes", flush=True)
    print("done", args.n)


if __name__ == "__main__":
    main()
