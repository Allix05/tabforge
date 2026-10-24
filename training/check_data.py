"""Check that training targets line up with the audio: the pretrained model's
frame activations on a clean window should agree well with our labels, with
and without pitch-shift augmentation. Misaligned labels would show up as a
collapse in agreement."""

import numpy as np
import torch

from common import AUDIO_N_SAMPLES, SR, device, load_audio, load_model, load_notes
from train import split_ids, targets

model = load_model()
train_ids, _ = split_ids()
rng = np.random.default_rng(0)
for shift in (0, 2, -2):
    agree = []
    for t in train_ids[:30]:
        x = load_audio(t, "pickup")
        notes = load_notes(t)
        ratio = 2 ** (shift / 12)
        need = int(np.ceil(AUDIO_N_SAMPLES * ratio)) + 2
        i0 = int(rng.integers(0, len(x) - need))
        seg = x[i0:i0 + need]
        seg = np.interp(np.arange(AUDIO_N_SAMPLES) * ratio, np.arange(len(seg)), seg).astype(np.float32)
        note, onset, _ = targets(notes, i0 / SR, ratio, shift)
        if note.sum() < 20:
            continue
        with torch.no_grad():
            out = model(torch.from_numpy(seg[None]).to(device()))["note"][0].cpu().numpy() > 0.3
        tp = (out & (note > 0)).sum()
        agree.append(2 * tp / (out.sum() + (note > 0).sum()))
    print(f"shift {shift:+d}: frame-level agreement {np.mean(agree):.3f} over {len(agree)} windows")
    # a deliberately wrong label (one semitone off) to show what misalignment looks like
    if shift == 0:
        wrong = []
        for t in train_ids[:10]:
            x = load_audio(t, "pickup")
            notes = load_notes(t)
            i0 = int(rng.integers(0, len(x) - AUDIO_N_SAMPLES))
            note, _, _ = targets(notes, i0 / SR, 1.0, 1)
            with torch.no_grad():
                out = model(torch.from_numpy(x[None, i0:i0 + AUDIO_N_SAMPLES]).to(device()))["note"][0].cpu().numpy() > 0.3
            if note.sum() > 20:
                wrong.append(2 * (out & (note > 0)).sum() / (out.sum() + (note > 0).sum()))
        print(f"  (control, labels off by a semitone: {np.mean(wrong):.3f})")
