// Fast port of Basic Pitch's outputToNotesPoly (Apache-2.0, Spotify).
// Same algorithm and results, on flat typed arrays. The original "melodia"
// stage rescans the whole (frames x 88) matrix for its global maximum after
// every note, which is quadratic: minutes of frozen page on an 8-minute song.
// Remaining energy only ever drops to zero, so visiting the above-threshold
// cells once in descending order picks exactly the same maxima.

const P = 88;
const MIDI_OFFSET = 21;
const MAX_FREQ_IDX = 87;
const hzToMidi = (hz) => 12 * (Math.log2(hz) - Math.log2(440)) + 69;

function flatten(rows) {
  const out = new Float32Array(rows.length * P);
  rows.forEach((r, t) => out.set(r, t * P));
  return out;
}

function inferOnsetMatrix(O, F, T, nDiff = 2) {
  // frame-energy rises (min over 1..nDiff-frame differences), scaled to the onset range
  const diff = new Float32Array(T * P);
  let diffMax = 0, onsetMax = 0;
  for (let i = 0; i < T * P; i++) onsetMax = Math.max(onsetMax, O[i]);
  for (let t = nDiff; t < T; t++) {
    for (let p = 0; p < P; p++) {
      let d = Infinity;
      for (let n = 1; n <= nDiff; n++) d = Math.min(d, F[t * P + p] - F[(t - n) * P + p]);
      d = Math.max(d, 0);
      diff[t * P + p] = d;
      diffMax = Math.max(diffMax, d);
    }
  }
  const I = new Float32Array(T * P);
  for (let i = 0; i < T * P; i++) I[i] = Math.max(O[i], diffMax > 0 ? (onsetMax * diff[i]) / diffMax : 0);
  return I;
}

export function outputToNotesPoly(frames, onsets, onsetThresh = 0.5, frameThresh = 0.3, minNoteLen = 5,
  inferOnsets = true, maxFreq = null, minFreq = null, melodiaTrick = true, energyTolerance = 11) {
  const T = frames.length;
  if (!T) return [];
  const F = flatten(frames);
  let O = flatten(onsets);
  const lo = minFreq ? Math.trunc(hzToMidi(minFreq) - MIDI_OFFSET) : 0;
  const hi = maxFreq ? Math.trunc(hzToMidi(maxFreq) - MIDI_OFFSET) : P;
  if (lo > 0 || hi < P) {
    for (let t = 0; t < T; t++) {
      for (let p = 0; p < P; p++) if (p < lo || p >= hi) { F[t * P + p] = 0; O[t * P + p] = 0; }
    }
  }
  if (inferOnsets) O = inferOnsetMatrix(O, F, T);

  // onset peaks (local maxima in time) above threshold, latest first
  const starts = [];
  for (let t = 0; t < T; t++) {
    for (let p = 0; p < P; p++) {
      const v = O[t * P + p];
      if (v <= onsetThresh) continue;
      if ((t > 0 && !(v > O[(t - 1) * P + p])) || (t < T - 1 && !(v > O[(t + 1) * P + p]))) continue;
      starts.push(t * P + p);
    }
  }
  starts.reverse();

  const E = F.slice(); // remaining energy
  const clear = (t, p) => {
    E[t * P + p] = 0;
    if (p < MAX_FREQ_IDX) E[t * P + p + 1] = 0;
    if (p > 0) E[t * P + p - 1] = 0;
  };
  const amplitude = (a, b, p) => {
    let s = 0;
    for (let t = a; t < b; t++) s += F[t * P + p];
    return s / (b - a);
  };
  const notes = [];
  for (const cell of starts) {
    const t0 = Math.floor(cell / P), p = cell % P;
    if (t0 >= T - 1) continue;
    let i = t0 + 1, k = 0;
    while (i < T - 1 && k < energyTolerance) {
      k = E[i * P + p] < frameThresh ? k + 1 : 0;
      i++;
    }
    i -= k;
    if (i - t0 <= minNoteLen) continue;
    for (let j = t0; j < i; j++) clear(j, p);
    notes.push({ startFrame: t0, durationFrames: i - t0, pitchMidi: p + MIDI_OFFSET, amplitude: amplitude(t0, i, p) });
  }

  if (melodiaTrick) {
    const cand = [];
    for (let i = 0; i < T * P; i++) if (E[i] > frameThresh) cand.push(i);
    cand.sort((a, b) => E[b] - E[a]);
    for (const cell of cand) {
      if (!(E[cell] > frameThresh)) continue; // already absorbed by an earlier note
      const mid = Math.floor(cell / P), p = cell % P;
      E[cell] = 0;
      let i = mid + 1, k = 0;
      while (i < T - 1 && k < energyTolerance) {
        k = E[i * P + p] < frameThresh ? k + 1 : 0;
        clear(i, p);
        i++;
      }
      const iEnd = i - 1 - k;
      i = mid - 1;
      k = 0;
      while (i > 0 && k < energyTolerance) {
        k = E[i * P + p] < frameThresh ? k + 1 : 0;
        clear(i, p);
        i--;
      }
      const iStart = i + 1 + k;
      if (iEnd - iStart <= minNoteLen) continue;
      notes.push({ startFrame: iStart, durationFrames: iEnd - iStart, pitchMidi: p + MIDI_OFFSET, amplitude: amplitude(iStart, iEnd, p) });
    }
  }
  return notes;
}
