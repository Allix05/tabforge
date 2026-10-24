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
          Basic Pitch (Spotify, TF.js in the browser): notes + onsets
                                          │
     cleanup: overtone ghosts, ringing-note echoes, masked bleed
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
   + Viterbi over the song minimising hand movement and stretch
                     ▼
            SVG tab · fretboard · synced Web Audio player · ASCII / MIDI
```

| File | Role |
|---|---|
| `js/transcribe.js` | Basic Pitch inference (TensorFlow.js), 22.05 kHz mono resampling |
| `js/notes.js` | Cleanup, melody extraction, tempo/beat tracking, key & chords, quantization, arranging |
| `js/tab.js` | Tunings and the string/fret assignment (candidate shapes + Viterbi) |
| `js/pipeline.js` | Glues the steps together for each mode |
| `js/render.js` | SVG tab notation, rhythm stems, cursor, fretboard |
| `js/synth.js`, `js/ks.js` | Karplus-Strong guitar synth (allpass-tuned), synced player |
| `js/export.js` | ASCII tab and Standard MIDI File writers |
| `helper/tabforge_helper.py` | Local server: static app, yt-dlp downloads, streaming-link lookup, Demucs |

### Accuracy

Measured on the built-in demo (synthesized guitar, known ground truth, full mix, no separation): **guitar mode recovers 96 % of the notes at 82 % precision**, with tempo within 0.5 BPM and the chord progression (Am–F–C–G) exact. Real recordings vary: a clean guitar or a separated stem gives tabs close to hand-made ones, while dense, heavily distorted full mixes give more missed or extra notes. Turn on stem separation, adjust *Sensitivity*, and check by ear with the synced player.

## Tests

```bash
node --test
```

The unit tests cover the beat tracker, melody extraction, chord detection, fret assignment (playability and the hand-span limits), the four modes on the demo, the ASCII/MIDI writers and synth tuning.

## Credits

- Note transcription: [Basic Pitch](https://github.com/spotify/basic-pitch) by Spotify, Apache-2.0 (the model in `model/` keeps its license).
- Stem separation: [Demucs](https://github.com/facebookresearch/demucs) by Meta AI, MIT.
- Downloads: [yt-dlp](https://github.com/yt-dlp/yt-dlp), Unlicense.

Only download and tab music you have the right to use.

## License

All Rights Reserved, see [LICENSE](LICENSE). Third-party components keep their own licenses.
