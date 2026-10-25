# TabForge

[![CI](https://github.com/Allix05/tabforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Allix05/tabforge/actions/workflows/ci.yml)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](LICENSE)

Turn any song into playable **6-string guitar tabs**, with a Songsterr-style synced player.

Drop an audio file, paste a **YouTube / Spotify / SoundCloud / Deezer / Apple Music** link, record from your mic, and choose what should become the tab:

| Mode | What you get |
|---|---|
| 🎸 **Guitar part** | The electric or acoustic guitar that's in the song: riffs, chords, solos |
| 🎤 **Vocal melody** | The singer's melody, so you can play the lyrics' tune on guitar |
| 🎹 **Lead melody** | The main line from *any* instrument (piano, keys, synth, sax, violin…), moved to a comfortable guitar range |
| ✨ **Fingerstyle arrangement** | Melody on top plus bass notes from the detected chords: the whole song on one guitar |

**[Open TabForge](https://allix05.github.io/tabforge/)**: free, and the AI runs in your browser. Audio files never leave your device.

<!-- SCREENSHOT_PLACEHOLDER -->

## Features

- **Synced playback**: the original recording and a plucked-string rendition of the tab play together; a mix slider blends them. Clean, overdrive and acoustic tones.
- Moving cursor, highlighted notes and a **live fretboard**; click anywhere in the tab to jump there, or click a note to hear it.
- **Practice tools**: 50–125 % speed (pitch preserved), A/B loop, metronome, auto-follow.
- Chord names above the bars, key and tempo detection, editable BPM (½× / 2× fixes).
- 7 tunings (Standard, Eb, Drop D, D, Drop C, Open G, DADGAD), capo 0–9, transpose ±7.
- **Export** as plain-text ASCII tab or **MIDI** (opens in Guitar Pro, TuxGuitar, MuseScore, any DAW).

## Run it

### In the browser (files, mic, demo)

Open **https://allix05.github.io/tabforge/**. Nothing to install.

### With links + AI stem separation (one click, free)

Browsers aren't allowed to download from YouTube or Spotify, and stem separation needs real compute, so TabForge ships a small helper that runs **on your own machine**:

1. Install [Python 3.10+](https://www.python.org/downloads/).
2. [Download this repo](https://github.com/Allix05/tabforge/archive/refs/heads/main.zip) and unzip it.
3. Double-click **`start-windows.bat`** (Windows) or run **`./start-mac-linux.sh`** (macOS / Linux).

It installs its dependencies, asks once whether you want AI stem separation (~2 GB, optional), and opens TabForge at `http://127.0.0.1:8765` with everything enabled. The hosted page also detects a running helper automatically.

Manual alternative:

```bash
pip install -r helper/requirements.txt
python helper/tabforge_helper.py
```

## How it works

```
 file / mic ─────────────────────────────┐
 YouTube / SoundCloud ─▶ yt-dlp ─────────┤
 Spotify / Deezer / Apple ─▶ track name ─▶ YouTube search
                                          │
                     (optional, local) Demucs htdemucs_6s
                vocals · guitar · piano · bass · drums · other
                                          │
                                          ▼
   guitar mode: TabForge guitar model (Basic Pitch fine-tuned on real tabs
                + high-gain / metal augmentation, + a string head), ONNX
   melody modes: stock Basic Pitch (TF.js), better on voice & keys
                                          │
     automatic sensitivity: per-song thresholds from activation levels
                                          │
     beat tracking (Ellis DP) ▶ downbeats ▶ 16th-note grid ▶ key + chords
                                          │
    ┌──────────── guitar ─────────────┬──── melody / arrangement ────┐
    │ all notes                       │ Viterbi melody tracking       │
    │                                 │ (+ legato hold, octave fit,   │
    │                                 │  bass roots from chords)      │
    └────────────────┬────────────────┴───────────────────────────────┘
                     ▼
   fingering: per-chord playable shapes (one note per string, ≤4-fret span)
   + Viterbi over the song; costs fitted on real tabs, plus what the
     string head heard (-log P(string | timbre))
                     ▼
            SVG tab · fretboard · synced Web Audio player · ASCII / MIDI
```

| File | Role |
|---|---|
| `js/transcribe.js` | Model selection, 22.05 kHz mono resampling, note decoding |
| `js/guitar-model.js` | The fine-tuned guitar model + string head on onnxruntime-web |
| `js/decode.js` | Basic Pitch's note decoder, rewritten to run in linear time (identical output) |
| `js/sensitivity.js` | Automatic per-song thresholds |
| `js/notes.js` | Melody extraction, tempo/beat tracking, key & chords, quantization, arranging |
| `js/tab.js` | Tunings and the string/fret assignment (candidate shapes + Viterbi) |
| `js/pipeline.js` | Glues the steps together for each mode |
| `js/render.js` | SVG tab notation, rhythm stems, cursor, fretboard |
| `js/synth.js`, `js/ks.js` | Karplus-Strong guitar synth (allpass-tuned), synced player |
| `js/export.js` | ASCII tab and Standard MIDI File writers |
| `helper/tabforge_helper.py` | Local server: static app, yt-dlp downloads, streaming-link lookup, Demucs |
| `training/` | Fine-tuning, synthetic riff generator, evaluation and ONNX export (Python / PyTorch) |

## Training on real tabs

The guitar model is [Basic Pitch](https://github.com/spotify/basic-pitch) fine-tuned (via the [PyTorch port](https://github.com/gudgud96/basic-pitch-torch)) on **[GuitarSet](https://guitarset.weebly.com/)**: 360 real performances by 6 guitarists, every note annotated with its string and fret. Players 00–03 train, player 04 picks checkpoints and calibrates, and **player 05 is never seen**: all numbers below come from player 05.

Basic Pitch had never heard a high-gain amp, a palm mute or a 200 BPM riff, so training also uses:

- **Amp / mix augmentation**: tube-style high-gain amps and cabinets (including heavy low cuts, so a palm-muted low E is still read as E2 when its fundamental is gone), double-tracked guitars, synthetic drums and bass bleed, reverb, noise, pitch shifts.
- **Synthetic metal riffs**: 3,000 procedurally generated riffs with exact tab labels (palm-muted chugs, power chords, gallops, pedal-tone riffs, chromatic runs, tremolo leads, 80–230 BPM), rendered with a string model (darker wound strings, palm damping, pick attack) and pushed through the amps.
- **Real palm mutes**: real GuitarSet takes damped after every attack, as a palm on the bridge would, plus **[IDMT-SMT-Guitar](https://zenodo.org/records/7544110)**: real electric guitars recorded direct, 4,776 annotated notes including 1,197 genuine palm mutes, bends, slides and vibrato (10% of it held out for validation and 10% for test).

A small **string head** on top of the model learns *which string* plays each note from timbre (GuitarSet's hexaphonic labels), and the fret search combines it with playability costs that were themselves fitted to the real tabs.

**Note accuracy** (onset F1: right pitch within 50 ms; automatic sensitivity; held-out player):

| Condition | Stock Basic Pitch | TabForge guitar model |
|---|---|---|
| Clean acoustic | **0.856** | 0.836 |
| High-gain amp | 0.659 | **0.713** |
| Separated guitar stem (with bleed) | 0.751 | **0.829** |
| Full metal mix (2 guitars, drums, bass) | 0.384 | **0.660** |
| Metal riffs: chugs, palm mutes, fast picking (synthetic, held out) | 0.593 | **0.811** |
| Real electric takes with palm mutes, bends, slides (IDMT, held out, through an amp) | 0.508 | **0.743** |

Stock Basic Pitch was trained on GuitarSet itself, so its clean-acoustic score is probably flattered; everywhere distortion is involved the fine-tuned model is far ahead. Both columns use automatic sensitivity calibrated for that model.

**Tab accuracy**: right note, right time *and* the same string as the guitarist played (a harder bar: many notes can sensibly be played in two or three places). Both columns use the fret search fitted to real tabs; the right one adds the guitar model and its string head:

| Condition | Stock Basic Pitch | TabForge |
|---|---|---|
| Clean acoustic | 0.532 | **0.604** |
| High-gain amp | 0.311 | **0.412** |
| Separated guitar stem | 0.439 | **0.527** |
| Full metal mix | 0.186 | **0.350** |
| Metal riffs (synthetic, held out) | 0.355 | **0.584** |
| IDMT electric takes | 0.177 | **0.206** |

On the real notes alone, the fitted fret search picks the guitarist's string 69.9% of the time (57.7% with the original hand-set costs). The IDMT tab score is low for every method because much of that set is positional exercises (every fret of one string), not song-like playing; for the same reason the string head is trained without it.

**What it means in practice**: on clean playing or a separated guitar stem, most notes are right and most land where a guitarist would play them; on a dense, distorted full mix, many notes are still missed or wrong. Use the local helper's stem separation, set the right tuning, and check by ear with the synced player: no automatic transcriber (this one included) is perfect yet.

## Tests

```bash
node --test
```

The unit tests cover the beat tracker, melody extraction, chord detection, fret assignment (playability and the hand-span limits), the fast note decoder, automatic sensitivity, the four modes on the demo, the ASCII/MIDI writers and synth tuning.

To retrain the guitar model (needs an NVIDIA GPU; data stays outside the repo):

```bash
python training/prepare_guitarset.py      # GuitarSet zips -> cache
python training/prepare_idmt.py           # IDMT-SMT-Guitar zip -> cache
python training/synth_riffs.py --n 3000   # synthetic metal riffs
python training/train.py --help           # staged runs are listed in its docstring
python training/string_head.py
python training/evaluate.py --run guitar7 --name guitar7
python training/export_onnx.py            # -> model-guitar/tabforge-guitar.onnx
node training/eval_js.mjs fingering       # fit fret costs to the real tabs
```

## Credits

- Note transcription: [Basic Pitch](https://github.com/spotify/basic-pitch) by Spotify, Apache-2.0. `model/` is the original; `model-guitar/` is TabForge's fine-tuned derivative (Apache-2.0 terms apply to it).
- Training data: [GuitarSet](https://zenodo.org/records/3371780) (Xi et al., ISMIR 2018), CC BY 4.0, and [IDMT-SMT-Guitar](https://zenodo.org/records/7544110) (Kehling et al., Fraunhofer IDMT), CC BY-NC-ND 4.0. Because of the latter, the guitar model is for non-commercial use only; TabForge is free and non-commercial.
- PyTorch port used for training: [basic-pitch-torch](https://github.com/gudgud96/basic-pitch-torch).
- Stem separation: [Demucs](https://github.com/facebookresearch/demucs) by Meta AI, MIT.
- Downloads: [yt-dlp](https://github.com/yt-dlp/yt-dlp), Unlicense.

Only download and tab music you have the right to use.

## License

All Rights Reserved, see [LICENSE](LICENSE). Third-party components keep their own licenses.
