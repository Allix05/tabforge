// An original 8-bar demo (Am - F - C - G, 100 BPM): four bars of fingerpicked
// arpeggios, then four bars of melody over a bass line. Rendered with the
// plucked-string synth so the app can be tried without uploading anything,
// and used by the tests as ground truth.
import { renderNotes } from './ks.js';

export const DEMO_BPM = 100;
const BEAT = 60 / DEMO_BPM;
const LEAD_IN = 0.5;

const CHORDS = [
  [45, 52, 57, 60, 64], // Am
  [41, 48, 53, 57, 60], // F
  [48, 55, 60, 64, 67], // C
  [43, 50, 55, 59, 62], // G
];
const ARP = [0, 1, 2, 3, 4, 3, 2, 1];
const MELODY = [
  [64, 62, 60, 57], [60, 62, 64, 65], [67, 64, 60, 64], [62, 59, 55, 59],
];

export function demoNotes() {
  const notes = [];
  CHORDS.forEach((ch, bar) => {
    ARP.forEach((k, i) => {
      notes.push({ start: LEAD_IN + (bar * 4 + i / 2) * BEAT, dur: BEAT / 2, midi: ch[k], vel: i === 0 ? 0.9 : 0.7 });
    });
  });
  CHORDS.forEach((ch, bar) => {
    const b0 = 16 + bar * 4;
    notes.push({ start: LEAD_IN + b0 * BEAT, dur: 2 * BEAT, midi: ch[0], vel: 0.8 });
    notes.push({ start: LEAD_IN + (b0 + 2) * BEAT, dur: 2 * BEAT, midi: ch[1], vel: 0.7 });
    MELODY[bar].forEach((m, i) => {
      notes.push({ start: LEAD_IN + (b0 + i) * BEAT, dur: BEAT, midi: m, vel: 0.95 });
    });
  });
  return notes.sort((a, b) => a.start - b.start);
}

export function renderDemo(sampleRate) {
  return renderNotes(demoNotes(), sampleRate, { brightness: 0.6, sustain: 0.997 });
}
