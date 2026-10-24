import test from 'node:test';
import assert from 'node:assert/strict';
import {
  removeOvertones, dropContinuations, dropMasked, extractMelody, fixOctaveJumps, onsetEnvelope, estimateTempo, trackBeats, timeToBeat, beatToTime,
  quantize, detectKey, nameChord, detectChords, fitRange, arrange, midiName,
} from '../js/notes.js';
import { assignFrets, positionsFor, TUNINGS } from '../js/tab.js';
import { toAscii, toMidi } from '../js/export.js';
import { demoNotes, DEMO_BPM } from '../js/demo.js';
import { pluck, midiToFreq } from '../js/ks.js';

const std = TUNINGS.standard.strings;

test('midi names', () => {
  assert.equal(midiName(40), 'E2');
  assert.equal(midiName(69), 'A4');
});

test('positions on a standard guitar', () => {
  assert.deepEqual(positionsFor(40, std), [{ string: 0, fret: 0 }]);
  const a4 = positionsFor(69, std).map((p) => `${p.string}:${p.fret}`);
  assert.ok(a4.includes('5:5') && a4.includes('4:10'));
  assert.deepEqual(positionsFor(42, std, 2), [{ string: 0, fret: 0 }]);
});

test('overtone ghosts are removed, real octaves kept', () => {
  const notes = [
    { start: 0, dur: 1, midi: 45, vel: 0.9 },
    { start: 0.01, dur: 1, midi: 57, vel: 0.3 },
    { start: 1, dur: 1, midi: 45, vel: 0.8 },
    { start: 1, dur: 1, midi: 57, vel: 0.8 },
  ];
  const out = removeOvertones(notes);
  assert.equal(out.length, 3);
  assert.ok(!out.some((n) => n.start === 0.01));
});

test('ringing-note echoes and masked ghosts are dropped', () => {
  const notes = [
    { start: 0, dur: 0.5, midi: 45, vel: 0.8 },
    { start: 0.3, dur: 0.1, midi: 45, vel: 0.5 }, // echo while 45 still rings
    { start: 0.3, dur: 0.3, midi: 52, vel: 0.8 },
    { start: 0.31, dur: 0.1, midi: 70, vel: 0.2 }, // quiet ghost
    { start: 1, dur: 0.3, midi: 45, vel: 0.8 }, // genuine repeat after a gap
  ];
  const out = dropMasked(dropContinuations(notes));
  assert.deepEqual(out.map((n) => `${n.midi}@${n.start}`), ['45@0', '52@0.3', '45@1']);
});

test('a strummed chord shares one tab column, fast runs stay apart', () => {
  const beats = Array.from({ length: 20 }, (_, i) => i * 0.3); // 200 BPM
  const chord = [50, 57, 62].map((m, i) => ({ start: 1.2 + i * 0.02, dur: 0.3, midi: m, vel: 0.8 }));
  const steps = new Set(quantize(chord, beats, { strum: 0.05 }).map((n) => n.step));
  assert.equal(steps.size, 1);
  const run = [64, 65, 67, 69].map((m, i) => ({ start: 2.4 + i * 0.075, dur: 0.07, midi: m, vel: 0.8 }));
  assert.equal(new Set(quantize(run, beats, { strum: 0.05 }).map((n) => n.step)).size, 4);
});

test('melody extraction follows the top voice and skips accompaniment', () => {
  const notes = [];
  for (let i = 0; i < 8; i++) {
    notes.push({ start: i * 0.5, dur: 0.45, midi: 72 + (i % 3), vel: 0.8 });
    notes.push({ start: i * 0.5, dur: 0.45, midi: 48, vel: 0.85 });
    notes.push({ start: i * 0.5 + 0.25, dur: 0.2, midi: 55, vel: 0.5 });
  }
  const mel = extractMelody(notes);
  assert.equal(mel.length, 8);
  assert.ok(mel.every((n) => n.midi >= 72));
});

test('isolated octave errors in a melody are folded back', () => {
  const line = [64, 65, 79, 67, 65].map((m, i) => ({ start: i * 0.4, dur: 0.35, midi: m, vel: 0.8 }));
  assert.deepEqual(fixOctaveJumps(line).map((n) => n.midi), [64, 65, 67, 67, 65]);
  const leap = [60, 72, 72, 72].map((m, i) => ({ start: i * 0.4, dur: 0.35, midi: m, vel: 0.8 }));
  assert.deepEqual(fixOctaveJumps(leap).map((n) => n.midi), [72, 72, 72, 72]);
  // a real stepwise leap of a fifth is left alone
  const fifth = [60, 67, 69, 67].map((m, i) => ({ start: i * 0.4, dur: 0.35, midi: m, vel: 0.8 }));
  assert.deepEqual(fixOctaveJumps(fifth).map((n) => n.midi), [60, 67, 69, 67]);
});

test('tempo and beats recovered from the demo', () => {
  const notes = demoNotes();
  const dur = Math.max(...notes.map((n) => n.start + n.dur));
  const env = onsetEnvelope(notes, dur);
  const bpm = estimateTempo(env);
  assert.ok(Math.abs(bpm - DEMO_BPM) < 3, `bpm ${bpm}`);
  const beats = trackBeats(env, 100, bpm, dur, notes);
  // first note (bar 1 beat 1) lands on a downbeat
  const b = timeToBeat(beats, notes[0].start);
  assert.ok(Math.abs(b - Math.round(b)) < 0.1);
  assert.equal(Math.round(b), 0, 'first bar is not an empty pickup');
  assert.ok(Math.abs(beatToTime(beats, timeToBeat(beats, 3.3)) - 3.3) < 1e-9);
});

test('key and chords of the demo', () => {
  const notes = demoNotes();
  assert.match(detectKey(notes).name, /^(A minor|C major)$/);
  const bpm = DEMO_BPM, beat = 60 / bpm;
  const beats = Array.from({ length: 40 }, (_, i) => 0.5 + i * beat);
  const chords = detectChords(notes, beats).map((c) => c.name);
  assert.deepEqual(chords.slice(0, 4), ['Am', 'F', 'C', 'G']);
  assert.equal(nameChord([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).name, 'C');
});

test('frets: the demo is playable and every pitch is preserved', () => {
  const notes = demoNotes();
  const beats = Array.from({ length: 40 }, (_, i) => 0.5 + i * 0.6);
  const q = quantize(notes, beats);
  const { notes: tab, dropped } = assignFrets(q, { strings: std });
  assert.equal(dropped, 0);
  assert.equal(tab.length, notes.length);
  for (const n of tab) assert.equal(std[n.string] + n.fret, n.midi);
  // chords on one column never share a string and fit the hand
  const cols = new Map();
  for (const n of tab) { if (!cols.has(n.step)) cols.set(n.step, []); cols.get(n.step).push(n); }
  for (const col of cols.values()) {
    assert.equal(new Set(col.map((n) => n.string)).size, col.length);
    const fr = col.filter((n) => n.fret > 0).map((n) => n.fret);
    if (fr.length) assert.ok(Math.max(...fr) - Math.min(...fr) <= 4);
  }
  // arpeggios stay in open / low position
  assert.ok(tab.slice(0, 32).every((n) => n.fret <= 5));
});

test('impossible chords drop the weakest non-melody note', () => {
  const col = [40, 41, 42, 43, 64].map((m, i) => ({ step: 0, start: 0, dur: 1, midi: m, vel: 0.5 + i * 0.1 }));
  const { notes, dropped } = assignFrets(col, { strings: std });
  assert.ok(dropped > 0);
  assert.ok(notes.some((n) => n.midi === 64));
});

test('an impossible unison does not hang the fret assignment', () => {
  // E2 exists only on the low E string, so two of them can't share a column
  const col = [40, 40].map((m) => ({ step: 0, start: 0, dur: 1, midi: m, vel: 0.8 }));
  const { notes, dropped } = assignFrets(col, { strings: std });
  assert.equal(notes.length, 1);
  assert.equal(dropped, 1);
});

test('fitRange moves a vocal line into guitar range', () => {
  const line = [84, 86, 88, 91].map((m, i) => ({ start: i, dur: 1, midi: m, vel: 1 }));
  const out = fitRange(line, 52, 88, 69);
  assert.ok(out.every((n) => n.midi >= 52 && n.midi <= 88));
  assert.deepEqual(out.map((n) => n.midi), [60, 62, 64, 67]);
});

test('arrangement adds bass roots under the melody', () => {
  const beats = Array.from({ length: 20 }, (_, i) => i * 0.5);
  const mel = [{ start: 0, dur: 1, midi: 72, vel: 0.9 }];
  const out = arrange(mel, [{ beat: 0, beats: 4, name: 'Am', root: 9 }], beats);
  const bass = out.filter((n) => n.bass);
  assert.deepEqual(bass.map((n) => n.midi), [45, 52]);
});

test('ascii tab has six lines per system with frets in place', () => {
  const tab = [{ step: 0, string: 0, fret: 0 }, { step: 4, string: 5, fret: 12 }];
  const txt = toAscii(tab, std, { chords: [{ beat: 0, name: 'Em' }] });
  const lines = txt.split('\n');
  assert.ok(lines[0].includes('Em'));
  assert.ok(lines[1].startsWith('e|') && lines[1].includes('12'));
  assert.ok(lines[6].startsWith('E|0'));
});

test('midi file is well formed', () => {
  const bytes = toMidi([{ start: 0, dur: 0.5, midi: 64, vel: 0.8 }], { bpm: 120 });
  assert.equal(String.fromCharCode(...bytes.slice(0, 4)), 'MThd');
  assert.equal(String.fromCharCode(...bytes.slice(14, 18)), 'MTrk');
  const len = (bytes[18] << 24) | (bytes[19] << 16) | (bytes[20] << 8) | bytes[21];
  assert.equal(len, bytes.length - 22);
});

test('plucked string is in tune', () => {
  const sr = 44100, midi = 76;
  const x = pluck(midi, sr, 1);
  // autocorrelation pitch estimate with parabolic interpolation
  const a = Math.round(sr * 0.2), W = 4096, f0 = midiToFreq(midi);
  const ac = (l) => { let s = 0; for (let i = 0; i < W; i++) s += x[a + i] * x[a + i + l]; return s; };
  let best = -Infinity, bl = 0;
  for (let lag = Math.floor(sr / f0 * 0.8); lag < sr / f0 * 1.25; lag++) { const s = ac(lag); if (s > best) { best = s; bl = lag; } }
  const y0 = ac(bl - 1), y1 = ac(bl), y2 = ac(bl + 1);
  const lag = bl + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2);
  const cents = 1200 * Math.log2(sr / lag / f0);
  assert.ok(Math.abs(cents) < 10, `off by ${cents} cents`);
});
