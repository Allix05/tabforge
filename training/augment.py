"""Audio augmentations that turn clean GuitarSet takes into the conditions
TabForge actually meets: high-gain amps, double-tracked rhythm guitars, and
drums / bass leaking through stem separation (or a full mix).

Every function takes and returns float32 mono audio at 22.05 kHz.
"""

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

from common import SR


def _sos(kind, freq, order=2):
    return butter(order, np.clip(freq, 20, SR / 2 - 100), btype=kind, fs=SR, output="sos")


def peak(x, freq, gain_db, q=1.0):
    """RBJ peaking EQ."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * freq / SR
    alpha = np.sin(w) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w), 1 - alpha / a]
    return lfilter(b, den, x)


def cabinet(x, rng):
    """4x12-ish speaker: steep low cut, mid hump, harsh roll-off above ~5 kHz.
    The low cut sometimes reaches 250 Hz: in a dense mix (and after stem
    separation) the 82 Hz fundamental of a palm-muted low E is mostly gone,
    and the model must still call it E2, not the E3 harmonic that remains."""
    hp = rng.uniform(70, 130) if rng.random() < 0.6 else rng.uniform(130, 250)
    x = sosfilt(_sos("highpass", hp, order=4 if hp > 130 else 2), x)
    x = peak(x, rng.uniform(1500, 3200), rng.uniform(2, 7), q=1.2)
    x = peak(x, rng.uniform(300, 600), rng.uniform(-4, 2), q=0.8)
    return sosfilt(_sos("lowpass", rng.uniform(3800, 6500), order=4), x)


def amp(x, rng, gain=None):
    """Tube-ish amp: tight pre-EQ, two asymmetric clipping stages, tone stack, cab."""
    gain = gain if gain is not None else rng.choice([2, 8, 30, 120, 400])
    x = x / (np.max(np.abs(x)) + 1e-6)
    x = sosfilt(_sos("highpass", rng.uniform(80, 700)), x)  # tube-screamer style tightening
    x = peak(x, rng.uniform(600, 1200), rng.uniform(0, 8), q=0.7)
    bias = rng.uniform(0, 0.3)
    x = np.tanh(gain ** 0.5 * x + bias) - np.tanh(bias)
    x = sosfilt(_sos("lowpass", rng.uniform(6000, 9000)), x)
    x = np.tanh(gain ** 0.5 * x * rng.uniform(0.5, 1.5))
    x = peak(x, rng.uniform(400, 800), rng.uniform(-9, 0), q=0.7)  # scooped mids
    x = peak(x, rng.uniform(80, 150), rng.uniform(0, 5), q=0.8)
    return cabinet(x, rng)


def palm_mute(x, notes, rng):
    """Palm-mute a real take: after every attack the sound dies within ~0.1 s
    and loses its highs, like a palm resting on the strings near the bridge.
    Returns the muted audio and notes with matching (shortened) durations."""
    tau = rng.uniform(0.04, 0.12)
    onsets = sorted({round(s, 3) for s, *_ in notes if 0 <= s < len(x) / SR})
    if not onsets:
        return x, notes
    t = np.arange(len(x)) / SR
    idx = np.searchsorted(onsets, t, side="right") - 1
    since = np.where(idx >= 0, t - np.array(onsets)[np.clip(idx, 0, None)], 10.0)
    env = np.exp(-np.maximum(since - 0.01, 0) / tau)
    muted = sosfilt(_sos("lowpass", rng.uniform(1200, 3000)), x) * env
    keep = 3 * tau + 0.01
    return muted, [(s, min(e, s + keep), m, *rest) for s, e, m, *rest in notes]


def double_track(x, rng):
    """Second take of the same part: slightly late, slightly detuned."""
    delay = int(rng.uniform(0.006, 0.03) * SR)
    ratio = 2 ** (rng.uniform(-6, 6) / 1200)
    idx = np.arange(len(x)) * ratio
    y = np.interp(idx, np.arange(len(x)), x, right=0)
    y = np.concatenate([np.zeros(delay), y[: len(x) - delay]])
    return 0.5 * (x + y * rng.uniform(0.6, 1.0))


def drums(n, rng):
    """Synthetic kit on a random tempo: kick, snare, hats."""
    out = np.zeros(n)
    bpm = rng.uniform(70, 220)
    step = 60 / bpm / 2
    t = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 18)
    snare = (rng.standard_normal(len(t)) * np.exp(-t * 25) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30))
    snare = sosfilt(_sos("highpass", 300), snare)
    hat = sosfilt(_sos("highpass", 7000), rng.standard_normal(len(t))) * np.exp(-t * 80)
    pattern_kick = rng.random(8) < 0.4
    pattern_kick[0] = True
    pos, k = rng.uniform(0, step), 0
    while pos < n / SR:
        i = int(pos * SR)
        seg = slice(i, min(n, i + len(t)))
        L = seg.stop - seg.start
        if pattern_kick[k % 8]:
            out[seg] += kick[:L]
        if k % 4 == 2:
            out[seg] += 0.7 * snare[:L]
        out[seg] += 0.25 * hat[:L]
        pos += step
        k += 1
    return out


def bass(n, notes, offset, rng):
    """Bass guitar doubling the lowest guitar note an octave down (not labelled)."""
    out = np.zeros(n)
    t0 = offset / SR
    for s, e, m, *_ in notes:
        if e < t0 or s > t0 + n / SR:
            continue
        f = 440 * 2 ** ((round(m) - 12 - 69) / 12)
        i0, i1 = max(0, int((s - t0) * SR)), min(n, int((e - t0) * SR))
        if i1 <= i0:
            continue
        tt = np.arange(i1 - i0) / SR
        wave = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt) + 0.1 * np.sign(np.sin(2 * np.pi * f * tt))
        out[i0:i1] += wave * np.exp(-tt * 2)
    return sosfilt(_sos("lowpass", 900), out)


def reverb(x, rng):
    length = int(rng.uniform(0.2, 1.2) * SR)
    ir = rng.standard_normal(length) * np.exp(-np.arange(length) / SR * rng.uniform(3, 10))
    ir[0] = 1 / rng.uniform(0.05, 0.3)
    wet = fftconvolve(x, ir)[: len(x)]
    return x + wet / (np.max(np.abs(wet)) + 1e-6) * np.max(np.abs(x)) * rng.uniform(0.05, 0.3)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def condition(x, notes, offset, rng, kind):
    """Apply one evaluation/training condition by name."""
    if kind == "clean":
        y = x
    elif kind == "crunch":
        y = amp(x, rng, gain=rng.choice([4, 12]))
    elif kind == "highgain":
        y = amp(x, rng, gain=rng.choice([120, 300, 600]))
    elif kind == "metalmix":
        # double-tracked high gain guitars + drums + bass at mix-like levels
        y = amp(double_track(x, rng), rng, gain=rng.choice([120, 300, 600]))
        d = drums(len(x), rng)
        b = bass(len(x), notes, offset, rng)
        y = y / rms(y) + d / rms(d) * rng.uniform(0.3, 1.0) + (b / rms(b) * rng.uniform(0.2, 0.7) if rms(b) > 1e-5 else 0)
    elif kind == "stem":
        # what a separated guitar stem sounds like: guitar plus faint bleed
        y = amp(x, rng) if rng.random() < 0.7 else x
        d = drums(len(x), rng)
        b = bass(len(x), notes, offset, rng)
        y = y / rms(y) + d / rms(d) * rng.uniform(0.03, 0.15) + (b / rms(b) * rng.uniform(0.03, 0.15) if rms(b) > 1e-5 else 0)
    else:
        raise ValueError(kind)
    y = y / (np.max(np.abs(y)) + 1e-6) * rng.uniform(0.3, 0.95)
    return y.astype(np.float32)


def random_training_condition(x, notes, offset, rng, synthetic=False):
    if synthetic:  # the synth DI sounds fake on its own: always through an amp
        kind = rng.choice(["crunch", "highgain", "highgain", "highgain", "metalmix", "metalmix", "stem"])
    else:
        kind = rng.choice(["clean", "clean", "crunch", "highgain", "highgain", "metalmix", "metalmix", "stem", "stem"])
    y = condition(x, notes, offset, rng, kind)
    if rng.random() < 0.3:
        y = reverb(y, rng)
    if rng.random() < 0.5:
        y = y + rng.standard_normal(len(y)) * rms(y) * 10 ** (rng.uniform(-50, -25) / 20)
    y = y / (np.max(np.abs(y)) + 1e-6) * rng.uniform(0.2, 0.95)
    return y.astype(np.float32)
