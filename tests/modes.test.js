import test from 'node:test';
import assert from 'node:assert/strict';
import { buildTab } from '../js/pipeline.js';
import { demoNotes, DEMO_BPM } from '../js/demo.js';

const notes = demoNotes();
const duration = Math.max(...notes.map((n) => n.start + n.dur)) + 1;
const playable = (r) => r.notes.every((n) => r.strings[n.string] + r.capo + n.fret === n.midi && n.fret >= 0 && n.fret <= 22);

test('guitar mode reproduces the demo note-for-note', () => {
  const r = buildTab({ mainNotes: notes, duration, mode: 'guitar' });
  assert.ok(Math.abs(r.bpm - DEMO_BPM) < 3);
  assert.equal(r.notes.length, notes.length);
  assert.ok(playable(r));
  assert.deepEqual(r.chords.slice(0, 4).map((c) => c.name), ['Am', 'F', 'C', 'G']);
});

test('melody modes are monophonic and in guitar range', () => {
  for (const mode of ['vocals', 'lead']) {
    const r = buildTab({ mainNotes: notes, duration, mode });
    const steps = r.notes.map((n) => n.step);
    assert.equal(new Set(steps).size, steps.length, mode);
    assert.ok(r.notes.length > 10, mode);
    assert.ok(playable(r), mode);
  }
});

test('lead mode picks the top line of the second half', () => {
  const r = buildTab({ mainNotes: notes, duration, mode: 'lead' });
  const secondHalf = r.notes.filter((n) => n.start > 0.5 + 16 * 0.6 - 0.05).map((n) => n.midi % 12);
  const expected = [64, 62, 60, 57, 60, 62, 64, 65, 67, 64, 60, 64, 62, 59, 55, 59].map((m) => m % 12);
  assert.deepEqual(secondHalf, expected);
});

test('arrangement keeps melody on top with bass under it', () => {
  const r = buildTab({ mainNotes: notes, duration, mode: 'arrange' });
  assert.ok(r.notes.some((n) => n.bass));
  assert.ok(playable(r));
});

test('capo, alternate tuning and transpose stay playable', () => {
  const r = buildTab({ mainNotes: notes, duration, mode: 'guitar', tuning: 'dropD', capo: 2, transpose: 2 });
  assert.ok(playable(r));
  assert.equal(r.chords[0].name, 'Bm');
});

test('flat keys are spelled with flats', () => {
  const r = buildTab({ mainNotes: notes, duration, mode: 'guitar', transpose: 1 });
  assert.ok(r.chords.slice(0, 4).map((c) => c.name).join(' ') === 'Bbm Gb Db Ab', r.chords.map((c) => c.name).join(' '));
  assert.match(r.key.name, /^(Bb minor|Db major)$/);
});

test('isolated vocal stems are not pulled up to harmonics', () => {
  const line = [60, 62, 64, 62].map((m, i) => ({ start: i * 0.5, dur: 0.45, midi: m, vel: 0.8 }));
  const ghosts = line.map((n) => ({ ...n, midi: n.midi + 12, vel: 0.7 }));
  const r = buildTab({ mainNotes: [...line, ...ghosts], duration: 3, mode: 'vocals', isolated: true });
  assert.deepEqual(r.notes.map((n) => n.midi % 12), [0, 2, 4, 2]);
  assert.ok(Math.max(...r.notes.map((n) => n.fret)) <= 12);
});
