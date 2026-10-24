// Ties the pure steps together for each tab mode. Input is decoded notes
// (from the main stem, and optionally an accompaniment stem used for beats and
// chords); output is everything the UI needs to draw and play the tab.
import {
  cleanNotes, removeOvertones, dropContinuations, dropMasked, extractMelody, fixOctaveJumps, onsetEnvelope, estimateTempo, trackBeats,
  quantize, detectKey, detectChords, fitRange, arrange, NOTE_NAMES,
} from './notes.js';
import { assignFrets, playableRange, TUNINGS } from './tab.js';

export const MODES = {
  guitar: { label: 'Guitar part', stem: 'guitar', accomp: null },
  vocals: { label: 'Vocal melody', stem: 'vocals', accomp: 'accomp' },
  lead: { label: 'Lead melody', stem: 'instruments', accomp: 'accomp' },
  arrange: { label: 'Fingerstyle arrangement', stem: 'vocals', accomp: 'accomp' },
};

const FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
// keys conventionally written with flats (F Bb Eb Ab Db major, D G C F Bb Eb minor)
const FLAT_MAJOR = new Set([5, 10, 3, 8, 1]);
const FLAT_MINOR = new Set([2, 7, 0, 5, 10, 3]);

// transpose a chord/note name and spell it to suit the key
const spell = (name, semis, flats) => {
  const m = /^([A-G]#?)(.*)$/.exec(name);
  if (!m) return name;
  const i = (((NOTE_NAMES.indexOf(m[1]) + semis) % 12) + 12) % 12;
  return (flats ? FLAT_NAMES : NOTE_NAMES)[i] + m[2];
};

export function buildTab({
  mainNotes, accompNotes = null, duration, mode = 'guitar', tuning = 'standard', capo = 0,
  transpose = 0, cleanOvertones = true, bpm: bpmOverride = 0, sensitivity = 0.5, isolated = false,
}) {
  const strings = TUNINGS[tuning].strings;
  const [lo, hi] = playableRange(strings, capo);
  const rhythmSrc = accompNotes?.length ? accompNotes : mainNotes;

  const env = onsetEnvelope(rhythmSrc, duration);
  const bpm = bpmOverride || estimateTempo(env);
  const beats = trackBeats(env, 100, bpm, duration, rhythmSrc);
  const harmonySrc = mode === 'guitar' ? mainNotes : rhythmSrc;
  const chords = detectChords(cleanNotes(harmonySrc, { minDur: 0.08 }), beats);
  const key = detectKey(harmonySrc);

  let quantized;
  if (mode === 'guitar') {
    let notes = dropContinuations(cleanNotes(mainNotes, { minDur: 0.05 })).map((n) => ({ ...n, midi: n.midi + transpose }));
    notes = cleanNotes(notes, { minMidi: lo, maxMidi: hi });
    if (cleanOvertones) notes = dropMasked(removeOvertones(notes, 0.5), 0.7 - 0.3 * sensitivity);
    quantized = quantize(notes, beats);
  } else {
    const vocal = mode === 'vocals' || mode === 'arrange';
    // ghosts above the line would pull the melody tracker upwards, so clean first
    // (no loudness masking here: a melody is often quieter than the bass under it)
    let pool = dropContinuations(cleanNotes(mainNotes, { minMidi: vocal ? 45 : 36, maxMidi: 88, minDur: 0.06 }));
    // An isolated stem holds a single voice, so anything above it is mostly
    // harmonics: clean harder and don't favour high notes. In a full mix the
    // melody is usually the top line, so lean upwards there.
    if (cleanOvertones) pool = removeOvertones(pool, isolated ? 0.95 : 0.5);
    const bias = isolated ? (vocal ? -0.05 : 0.1) : vocal ? 0.2 : 0.3;
    const melody = fixOctaveJumps(extractMelody(pool, { pitchBias: bias }));
    if (mode === 'arrange') {
      const all = arrange(melody, chords, beats, { transpose, bassLo: Math.max(40, strings[0] + capo) });
      quantized = [
        ...quantize(all.filter((n) => !n.bass), beats, { mono: true }),
        ...quantize(all.filter((n) => n.bass), beats),
      ];
    } else {
      // centre around E4: G/B/e strings in the first 10 frets, easy to play and sing along
      const line = fitRange(melody, Math.max(lo, 50), Math.min(hi, 88), 64, transpose);
      quantized = quantize(line, beats, { mono: true });
    }
  }

  const tab = assignFrets(quantized, { strings, capo });
  const tonic = (((key.tonic + transpose) % 12) + 12) % 12;
  const flats = (key.mode === 'minor' ? FLAT_MINOR : FLAT_MAJOR).has(tonic);
  return {
    mode,
    strings,
    capo,
    bpm,
    beats,
    key: { ...key, tonic, name: (flats ? FLAT_NAMES : NOTE_NAMES)[tonic] + ' ' + key.mode },
    chords: chords.map((c) => ({ ...c, name: spell(c.name, transpose, flats) })),
    notes: tab.notes,
    dropped: tab.dropped,
    duration,
  };
}
