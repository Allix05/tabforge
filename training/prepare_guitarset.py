"""Decode GuitarSet (Xi et al., ISMIR 2018, CC BY 4.0) into a fast cache.

Expects in TABFORGE_DATA: annotation.zip, pickup.zip (audio_mono-pickup_mix)
and mic.zip (audio_mono-mic) from https://zenodo.org/records/3371780.
Writes 22.05 kHz float16 audio and per-track note/string/fret labels.
"""

import io
import json
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import CACHE, DATA, OPEN_STRINGS


def notes_from_jams(jams):
    notes = []
    for ann in jams["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        string = int(ann["annotation_metadata"]["data_source"])
        for obs in ann["data"]:
            midi = float(obs["value"])
            fret = int(round(midi)) - OPEN_STRINGS[string]
            if fret < 0:
                continue
            notes.append([obs["time"], obs["time"] + obs["duration"], midi, string])
    return sorted(notes)


def main():
    (CACHE / "labels").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA / "annotation.zip") as z:
        for name in z.namelist():
            if name.endswith(".jams"):
                track = name.split("/")[-1][:-5]
                notes = notes_from_jams(json.loads(z.read(name)))
                json.dump(notes, open(CACHE / "labels" / f"{track}.json", "w"))
    print("labels:", len(list((CACHE / "labels").glob("*.json"))))

    for source, zname, suffix in [("pickup", "pickup.zip", "_mix.wav"), ("mic", "mic.zip", "_mic.wav")]:
        out = CACHE / source
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(DATA / zname) as z:
            for name in z.namelist():
                if not name.endswith(suffix):
                    continue
                track = name.split("/")[-1][: -len(suffix)]
                audio, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
                audio = audio.mean(1)
                assert sr == 44100, sr
                audio = resample_poly(audio, 1, 2).astype(np.float16)
                np.save(out / f"{track}.npy", audio)
        print(source, len(list(out.glob("*.npy"))))


if __name__ == "__main__":
    main()
