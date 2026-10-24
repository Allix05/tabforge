"""Note decoding + scoring without importing PyTorch, so process pools stay light."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("BPTORCH", "C:/tftrain/bptorch"))))

from basic_pitch_torch import note_creation  # noqa: E402


def decode(out, onset_thresh=0.5, frame_thresh=0.3, min_note_len=6):
    """Same note decoding as the browser (outputToNotesPoly)."""
    events = note_creation.output_to_notes_polyphonic(
        out["note"], out["onset"], onset_thresh=onset_thresh, frame_thresh=frame_thresh,
        min_note_len=min_note_len, infer_onsets=True, min_freq=None, max_freq=None, melodia_trick=True,
    )
    times = note_creation.model_frames_to_time(out["note"].shape[0])
    # (start s, end s, midi, amplitude, start frame, duration in frames)
    return [(times[s], times[e], p, a, s, e - s) for s, e, p, a in events]


def note_f1(pred, ref, onset_tol=0.05):
    """Onset-only note F1 (mir_eval 'no offset' convention): pitch must match
    to the semitone and the onset fall within 50 ms."""
    ref = sorted((s, round(m)) for s, _e, m, *_ in ref)
    pred = sorted((s, int(p)) for s, _e, p, *_ in pred)
    used = [False] * len(pred)
    tp = 0
    j0 = 0
    for s, m in ref:
        while j0 < len(pred) and pred[j0][0] < s - onset_tol:
            j0 += 1
        j = j0
        while j < len(pred) and pred[j][0] <= s + onset_tol:
            if not used[j] and pred[j][1] == m:
                used[j] = True
                tp += 1
                break
            j += 1
    p = tp / len(pred) if pred else 0.0
    r = tp / len(ref) if ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f": f, "tp": tp, "n_pred": len(pred), "n_ref": len(ref)}


def grid_scores(args):
    """Worker: F1 for every (onset, frame) threshold pair on one track."""
    out, notes, onsets, frames = args
    return {(o, f): note_f1(decode(out, o, f), notes) for o in onsets for f in frames}
