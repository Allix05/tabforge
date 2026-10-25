"""Decode IDMT-SMT-Guitar (Kehling et al., Fraunhofer IDMT, CC BY-NC-ND 4.0)
into the training cache. Real electric guitars recorded direct, every note
annotated with string, fret and playing technique, including palm mutes
("MU"), which GuitarSet doesn't have.

Expects C:/tftrain/data_idmt.zip (https://zenodo.org/records/7544110).
Dead notes and harmonics are left out: their annotated pitch is not the
pitch that sounds. Every 10th file is held out for validation.
"""

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import CACHE, OPEN_STRINGS

ZIP = Path("C:/tftrain/data_idmt.zip")
MIN_LEN = 2.2  # seconds: training windows are ~2 s


def events(xml):
    for ev in re.findall(r"<event>(.*?)</event>", xml, re.S):
        get = lambda tag: (re.search(f"<{tag}>(.*?)</{tag}>", ev) or [None, None])[1]  # noqa: E731
        pitch, on, off, string, fret = get("pitch"), get("onsetSec"), get("offsetSec"), get("stringNumber"), get("fretNumber")
        if None in (pitch, on, off, string):
            continue
        style, expr = get("excitationStyle") or "", get("expressionStyle") or ""
        if expr in ("DN", "HA"):
            continue
        s = int(string) - 1  # 1 = low E
        midi = float(pitch)
        if not 0 <= s < 6 or not 0 <= round(midi) - OPEN_STRINGS[s] <= 24:
            continue
        yield [float(on), max(float(off), float(on) + 0.05), midi, s, style == "MU"]


def main():
    adir, ldir = CACHE / "idmt" / "audio", CACHE / "idmt" / "labels"
    adir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    names = set(z.namelist())
    n_files = n_notes = n_muted = 0
    for xml_name in sorted(n for n in names if n.endswith(".xml") and "/dataset4/" not in n):
        xml = z.read(xml_name).decode("utf-8", "replace")
        audio_file = (re.search(r"<audioFileName>(.*?)</audioFileName>", xml) or [None, None])[1]
        folder = xml_name.rsplit("/annotation/", 1)[0]
        wav = f"{folder}/audio/{audio_file}" if audio_file else None
        if wav not in names:
            wav = xml_name.replace("/annotation/", "/audio/")[:-4] + ".wav"
        if wav not in names:
            continue
        notes = list(events(xml))
        if not notes:
            continue
        audio, sr = sf.read(io.BytesIO(z.read(wav)), dtype="float32", always_2d=True)
        audio = audio.mean(1)
        if sr != 44100:
            continue
        audio = resample_poly(audio, 1, 2)
        if len(audio) < MIN_LEN * 22050:
            audio = np.pad(audio, (0, int(MIN_LEN * 22050) - len(audio)))
        tid = f"idmt_{n_files:04d}"
        np.save(adir / f"{tid}.npy", (audio / (np.max(np.abs(audio)) + 1e-9) * 0.8).astype(np.float16))
        json.dump([n[:4] for n in notes], open(ldir / f"{tid}.json", "w"))
        n_files += 1
        n_notes += len(notes)
        n_muted += sum(n[4] for n in notes)
    print(f"{n_files} files, {n_notes} notes, {n_muted} palm-muted")


if __name__ == "__main__":
    main()
