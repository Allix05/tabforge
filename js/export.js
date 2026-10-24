// Exports: plain-text ASCII tab (paste anywhere) and a Standard MIDI File
// (imports into Guitar Pro, TuxGuitar, MuseScore, any DAW).
import { STRING_NAMES } from './tab.js';

export function toAscii(tabNotes, strings, { stepsPerMeasure = 16, measuresPerLine = 4, chords = [], stepsPerBeat = 4, title = '' } = {}) {
  if (!tabNotes.length) return '';
  const names = STRING_NAMES(strings);
  // the classic lowercase "e" for the high string when both E strings exist
  if (names[names.length - 1] === names[0]) names[names.length - 1] = names[0].toLowerCase();
  const width = Math.max(...names.map((n) => n.length));
  const lastStep = Math.max(...tabNotes.map((n) => n.step));
  const nMeasures = Math.floor(lastStep / stepsPerMeasure) + 1;
  const byStep = new Map();
  for (const n of tabNotes) {
    if (!byStep.has(n.step)) byStep.set(n.step, []);
    byStep.get(n.step).push(n);
  }
  const chordAt = new Map(chords.map((c) => [c.beat * stepsPerBeat, c.name]));

  const blocks = [];
  if (title) blocks.push(title, '');
  for (let m0 = 0; m0 < nMeasures; m0 += measuresPerLine) {
    const lines = strings.map(() => '');
    let chordLine = '';
    for (let m = m0; m < Math.min(nMeasures, m0 + measuresPerLine); m++) {
      lines.forEach((_, s) => { lines[s] += '|'; });
      for (let k = 0; k < stepsPerMeasure; k++) {
        const step = m * stepsPerMeasure + k;
        const col = byStep.get(step) || [];
        const cellW = Math.max(1, ...col.map((n) => String(n.fret).length)) + 1;
        const ch = chordAt.get(step);
        if (ch) chordLine = chordLine.padEnd(lines[0].length, ' ') + ch;
        lines.forEach((_, s) => {
          const note = col.find((n) => n.string === s);
          const txt = note ? String(note.fret) : '';
          lines[s] += txt + '-'.repeat(cellW - txt.length);
        });
      }
    }
    const pad = ' '.repeat(width);
    blocks.push((pad + chordLine).trimEnd());
    // print high e on top, like every tab
    for (let s = strings.length - 1; s >= 0; s--) blocks.push(names[s].padEnd(width) + lines[s] + '|');
    blocks.push('');
  }
  return blocks.join('\n');
}

function vlq(n) {
  const bytes = [n & 0x7f];
  while ((n >>= 7)) bytes.unshift((n & 0x7f) | 0x80);
  return bytes;
}

export function toMidi(notes, { bpm = 120, program = 27, ppq = 480 } = {}) {
  const usPerBeat = Math.round(60000000 / bpm);
  const tick = (sec) => Math.max(0, Math.round(sec * (bpm / 60) * ppq));
  const events = [];
  for (const n of notes) {
    const vel = Math.max(1, Math.min(127, Math.round(30 + n.vel * 97)));
    events.push({ t: tick(n.start), on: true, midi: n.midi, vel });
    events.push({ t: Math.max(tick(n.start) + 1, tick(n.start + n.dur)), on: false, midi: n.midi, vel: 0 });
  }
  events.sort((a, b) => a.t - b.t || (a.on === b.on ? 0 : a.on ? 1 : -1));

  const track = [
    0x00, 0xff, 0x51, 0x03, (usPerBeat >> 16) & 0xff, (usPerBeat >> 8) & 0xff, usPerBeat & 0xff,
    0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08,
    0x00, 0xc0, program & 0x7f,
  ];
  let last = 0;
  for (const e of events) {
    track.push(...vlq(e.t - last), e.on ? 0x90 : 0x80, e.midi & 0x7f, e.vel);
    last = e.t;
  }
  track.push(0x00, 0xff, 0x2f, 0x00);

  const len = track.length;
  return new Uint8Array([
    0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, (ppq >> 8) & 0xff, ppq & 0xff,
    0x4d, 0x54, 0x72, 0x6b, (len >>> 24) & 0xff, (len >>> 16) & 0xff, (len >>> 8) & 0xff, len & 0xff,
    ...track,
  ]);
}
