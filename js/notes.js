// Pure note-processing: cleanup, melody extraction, tempo / beat tracking,
// key + chord detection, quantization and guitar-friendly arrangement.
// No DOM access, so everything here runs (and is tested) under Node too.

export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export const pc = (midi) => ((midi % 12) + 12) % 12;
export const midiName = (midi) => NOTE_NAMES[pc(midi)] + (Math.floor(midi / 12) - 1);
const end = (n) => n.start + n.dur;

// basic-pitch note events -> our compact format
export function fromBasicPitch(events) {
  return events
    .map((e) => ({
      start: e.startTimeSeconds,
      dur: e.durationSeconds,
      midi: e.pitchMidi,
      vel: e.amplitude,
      ...(e.stringProbs ? { stringProbs: e.stringProbs } : {}),
    }))
    .sort((a, b) => a.start - b.start || a.midi - b.midi);
}

export function cleanNotes(notes, { minMidi = 0, maxMidi = 127, minDur = 0.05, minVel = 0 } = {}) {
  return notes.filter((n) => n.midi >= minMidi && n.midi <= maxMidi && n.dur >= minDur && n.vel >= minVel);
}

// Polyphonic transcribers often report a ghost note an octave / twelfth above
// a strong note. Drop those when they are clearly quieter than the fundamental.
const OVERTONE_INTERVALS = new Set([12, 19, 24, 28, 31]);
export function removeOvertones(notes, ratio = 0.75, window = 0.06) {
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  const keep = sorted.map(() => true);
  for (let i = 0; i < sorted.length; i++) {
    const n = sorted[i];
    for (let j = i - 1; j >= 0 && n.start - sorted[j].start < window; j--) check(i, j);
    for (let j = i + 1; j < sorted.length && sorted[j].start - n.start < window; j++) check(i, j);
  }
  function check(i, j) {
    const n = sorted[i], m = sorted[j];
    if (OVERTONE_INTERVALS.has(n.midi - m.midi) && n.vel < ratio * m.vel) keep[i] = false;
  }
  return sorted.filter((_, i) => keep[i]);
}

// A note that is still ringing often gets "re-detected" when another note is
// struck (the attack's broadband energy re-triggers it). Drop those echoes.
export function dropContinuations(notes, gap = 0.03) {
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  const lastByPitch = new Map();
  return sorted.filter((n) => {
    const prev = lastByPitch.get(n.midi);
    const echo = prev && prev.start < n.start - 0.02 && end(prev) >= n.start - gap && n.vel <= prev.vel * 1.05;
    if (!echo || end(n) > end(prev)) lastByPitch.set(n.midi, echo ? { ...prev, dur: end(n) - prev.start } : n);
    return !echo;
  });
}

// Quiet notes struck together with much louder ones are mostly harmonics or
// bleed; keep them only if they are at least `ratio` of the loudest attack.
export function dropMasked(notes, ratio = 0.5, window = 0.05) {
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  return sorted.filter((n, i) => {
    let loud = 0;
    for (let j = i - 1; j >= 0 && n.start - sorted[j].start < window; j--) loud = Math.max(loud, sorted[j].vel);
    for (let j = i + 1; j < sorted.length && sorted[j].start - n.start < window; j++) loud = Math.max(loud, sorted[j].vel);
    return n.vel >= ratio * loud;
  });
}

// Viterbi over 20 ms frames picking at most one sounding note per frame:
// favours loud notes, slightly favours high notes (melody usually sits on top),
// and penalises switching / large leaps so the line stays singable.
export function extractMelody(input, {
  hop = 0.02, pitchBias = 0.06, switchCost = 0.08, leapCost = 0.12, restLevel = 0.08, minDur = 0.07, hold = 0.9,
} = {}) {
  if (!input.length) return [];
  // Legato hold: transcribers often cut melody notes short, which would let a
  // sustained bass note "win" the gap. Treat each note as held (slightly
  // weaker) until the next onset in its register, if one comes within `hold` s.
  const sorted = [...input].sort((a, b) => a.start - b.start);
  const notes = sorted.map((n, i) => {
    let next = end(n);
    for (let j = i + 1; j < sorted.length && sorted[j].start < n.start + hold; j++) {
      if (sorted[j].start > n.start + 0.03 && sorted[j].midi >= n.midi - 7) { next = sorted[j].start; break; }
    }
    return { ...n, held: Math.max(end(n), next) };
  });
  const total = Math.max(...notes.map((n) => n.held));
  const nFrames = Math.ceil(total / hop) + 1;
  const active = Array.from({ length: nFrames }, () => []);
  notes.forEach((n, idx) => {
    const a = Math.floor(n.start / hop), b = Math.min(nFrames - 1, Math.ceil(n.held / hop) - 1);
    for (let f = a; f <= b; f++) active[f].push(idx);
  });
  const emit = (idx, f) => {
    const n = notes[idx];
    const fading = f * hop > end(n) ? 0.8 : 1;
    return n.vel * fading + pitchBias * (n.midi - 60) / 12;
  };
  const trans = (from, to) => {
    if (from === to) return 0;
    if (from < 0 || to < 0) return -0.02;
    return -switchCost - leapCost * Math.min(Math.abs(notes[from].midi - notes[to].midi), 24) / 12;
  };

  // states: -1 = rest, otherwise note index
  let prevStates = [-1], prevScores = [0];
  const back = [];
  for (let f = 0; f < nFrames; f++) {
    const states = [-1, ...active[f]];
    const scores = new Array(states.length), ptr = new Array(states.length);
    states.forEach((s, k) => {
      let best = -Infinity, arg = 0;
      prevStates.forEach((p, q) => {
        const v = prevScores[q] + trans(p, s);
        if (v > best) { best = v; arg = q; }
      });
      scores[k] = best + (s < 0 ? restLevel : emit(s, f));
      ptr[k] = arg;
    });
    back.push({ states, ptr });
    prevStates = states; prevScores = scores;
  }
  let k = prevScores.indexOf(Math.max(...prevScores));
  const path = new Array(nFrames);
  for (let f = nFrames - 1; f >= 0; f--) {
    path[f] = back[f].states[k];
    k = back[f].ptr[k];
  }

  const out = [];
  for (let f = 0; f < nFrames;) {
    const s = path[f];
    let g = f;
    while (g + 1 < nFrames && path[g + 1] === s) g++;
    if (s >= 0) {
      const n = notes[s];
      const start = Math.max(n.start, f * hop), stop = Math.min(n.held, (g + 1) * hop);
      if (stop - start >= minDur) out.push({ start, dur: stop - start, midi: n.midi, vel: n.vel });
    }
    f = g + 1;
  }
  return out;
}

// Pitch trackers (especially on voice) sometimes jump an octave for a note or
// two. Fold an isolated note back when an octave move puts it next to both
// neighbours, which is what the singer actually did.
export function fixOctaveJumps(line, maxGap = 1.0) {
  const out = line.map((n) => ({ ...n }));
  for (let i = 0; i < out.length; i++) {
    const prev = out[i - 1], next = out[i + 1];
    const near = (m) => m && Math.abs(m.start - out[i].start) < maxGap + m.dur ? m : null;
    const ns = [near(prev), near(next)].filter(Boolean);
    if (!ns.length) continue;
    const dist = (m) => Math.max(...ns.map((x) => Math.abs(x.midi - m)));
    const m = out[i].midi;
    for (const cand of [m - 12, m + 12]) {
      if (dist(m) >= 9 && dist(cand) <= 5) { out[i].midi = cand; break; }
    }
  }
  return out;
}

// ---------- rhythm ----------

export function onsetEnvelope(notes, duration, fps = 100) {
  const n = Math.ceil(duration * fps) + 2;
  const raw = new Float32Array(n);
  for (const note of notes) {
    const i = Math.round(note.start * fps);
    if (i >= 0 && i < n) raw[i] += note.vel * (note.midi < 52 ? 1.5 : 1);
  }
  // gaussian smoothing, sigma = 2 frames
  const k = [0.054, 0.242, 0.399, 0.242, 0.054];
  const env = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let j = -2; j <= 2; j++) if (i + j >= 0 && i + j < n) v += raw[i + j] * k[j + 2];
    env[i] = v;
  }
  return env;
}

export function estimateTempo(env, fps = 100, { min = 60, max = 190, prior = 115 } = {}) {
  const lo = Math.floor(fps * 60 / max), hi = Math.ceil(fps * 60 / min);
  const ac = new Float32Array(hi * 2 + 2);
  for (let lag = 1; lag < ac.length; lag++) {
    let s = 0;
    for (let i = lag; i < env.length; i++) s += env[i] * env[i - lag];
    ac[lag] = s;
  }
  let best = 0, bestLag = Math.round(fps * 60 / prior);
  for (let lag = lo; lag <= hi; lag++) {
    const bpm = fps * 60 / lag;
    const w = Math.exp(-0.5 * (Math.log2(bpm / prior) / 0.9) ** 2);
    const score = w * (ac[lag] + 0.5 * ac[2 * lag] + 0.25 * (ac[Math.round(lag / 2)] || 0));
    if (score > best) { best = score; bestLag = lag; }
  }
  // parabolic refinement for sub-frame precision
  const y0 = ac[bestLag - 1], y1 = ac[bestLag], y2 = ac[bestLag + 1];
  const denom = y0 - 2 * y1 + y2;
  const offset = denom !== 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (y0 - y2) / denom)) : 0;
  return Math.round((fps * 60 / (bestLag + offset)) * 10) / 10;
}

// Ellis (2007) dynamic-programming beat tracker; returns beat times (s)
// covering the whole song, aligned so beats[0] is a downbeat (4/4).
export function trackBeats(env, fps, bpm, duration, notes = []) {
  const period = fps * 60 / bpm;
  let beats = [];
  const energy = env.reduce((a, b) => a + b, 0);
  if (energy > 0) {
    const mean = energy / env.length;
    let sd = 0;
    for (const v of env) sd += (v - mean) ** 2;
    sd = Math.sqrt(sd / env.length) || 1;
    const local = env.map((v) => v / sd);
    const tightness = 100;
    const cum = new Float32Array(env.length), prev = new Int32Array(env.length).fill(-1);
    const wFrom = Math.round(2 * period), wTo = Math.max(1, Math.round(period / 2));
    for (let i = 0; i < env.length; i++) {
      let best = -Infinity, arg = -1;
      for (let d = wTo; d <= wFrom; d++) {
        const j = i - d;
        if (j < 0) break;
        const v = cum[j] - tightness * Math.log(d / period) ** 2;
        if (v > best) { best = v; arg = j; }
      }
      cum[i] = local[i] + (arg >= 0 ? Math.max(0, best) : 0);
      prev[i] = best > 0 ? arg : -1;
    }
    let i = env.length - 1, bestEnd = i;
    for (let j = Math.max(0, env.length - Math.round(period)); j < env.length; j++) if (cum[j] > cum[bestEnd]) bestEnd = j;
    i = bestEnd;
    while (i >= 0) { beats.push(i / fps); i = prev[i]; }
    beats.reverse();
  }
  const step = 60 / bpm;
  if (beats.length < 2) {
    beats = [];
    const first = notes.length ? Math.min(...notes.map((n) => n.start)) % step : 0;
    for (let t = first; t <= duration + step; t += step) beats.push(t);
  }
  while (beats[0] > 0.02) beats.unshift(beats[0] - step);
  while (beats[beats.length - 1] < duration + step) beats.push(beats[beats.length - 1] + step);

  // choose the downbeat phase: strong (esp. low) onsets land on beat 1
  const strength = (t) => {
    let s = 0;
    for (const n of notes) if (Math.abs(n.start - t) < 0.07) s += n.vel * (n.midi < 52 ? 2 : 1);
    return s;
  };
  const phaseScore = [0, 0, 0, 0];
  beats.forEach((t, i) => { phaseScore[i % 4] += strength(t); });
  const phase = phaseScore.indexOf(Math.max(...phaseScore));
  for (let k = 0; k < (4 - phase) % 4; k++) beats.unshift(beats[0] - step);
  // never start with a completely empty bar
  const firstNote = notes.length ? Math.min(...notes.map((n) => n.start)) : 0;
  while (beats.length > 8 && beats[4] <= firstNote + 0.05) beats.splice(0, 4);
  return beats;
}

export function timeToBeat(beats, t) {
  if (t <= beats[0]) return (t - beats[0]) / (beats[1] - beats[0]);
  const last = beats.length - 1;
  if (t >= beats[last]) return last + (t - beats[last]) / (beats[last] - beats[last - 1]);
  let lo = 0, hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (beats[mid] <= t) lo = mid; else hi = mid;
  }
  return lo + (t - beats[lo]) / (beats[lo + 1] - beats[lo]);
}

export function beatToTime(beats, b) {
  const last = beats.length - 1;
  if (b <= 0) return beats[0] + b * (beats[1] - beats[0]);
  if (b >= last) return beats[last] + (b - last) * (beats[last] - beats[last - 1]);
  const i = Math.floor(b);
  return beats[i] + (b - i) * (beats[i + 1] - beats[i]);
}

// A strummed or picked chord spreads its notes over a few tens of ms; tabs
// write them in one column. Returns the chord-onset time for each note.
export function strumOnsets(notes, window = 0.05) {
  const order = notes.map((n, i) => i).sort((a, b) => notes[a].start - notes[b].start);
  const at = new Array(notes.length);
  let head = null, pitches = new Set();
  for (const i of order) {
    const n = notes[i];
    if (head === null || n.start - head > window || pitches.has(n.midi) || pitches.size >= 6) {
      head = n.start;
      pitches = new Set();
    }
    pitches.add(n.midi);
    at[i] = head;
  }
  return at;
}

// Snap to a 16th-note grid (4 steps per beat). Real times are kept for playback.
export function quantize(notes, beats, { stepsPerBeat = 4, mono = false, strum = 0 } = {}) {
  const onsets = strum > 0 ? strumOnsets(notes, strum) : notes.map((n) => n.start);
  let out = notes.map((n, i) => {
    const step = Math.max(0, Math.round(timeToBeat(beats, onsets[i]) * stepsPerBeat));
    const stop = Math.round(timeToBeat(beats, end(n)) * stepsPerBeat);
    return { ...n, step, durSteps: Math.max(1, stop - step) };
  });
  if (mono) {
    const byStep = new Map();
    for (const n of out) {
      const cur = byStep.get(n.step);
      if (!cur || n.vel * n.dur > cur.vel * cur.dur) byStep.set(n.step, n);
    }
    out = [...byStep.values()];
  } else {
    // the same pitch twice in one column is a duplicate
    const seen = new Set();
    out = out.filter((n) => {
      const key = n.step + ':' + n.midi;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }
  return out.sort((a, b) => a.step - b.step || a.midi - b.midi);
}

// ---------- harmony ----------

const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function correlate(a, b) {
  const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

export function chroma(notes, from = -Infinity, to = Infinity) {
  const c = new Array(12).fill(0);
  for (const n of notes) {
    const overlap = Math.min(end(n), to) - Math.max(n.start, from);
    if (overlap > 0) c[pc(n.midi)] += overlap * n.vel;
  }
  return c;
}

export function detectKey(notes) {
  const c = chroma(notes);
  let best = { score: -2, tonic: 0, mode: 'major' };
  for (let t = 0; t < 12; t++) {
    const rot = c.map((_, i) => c[(i + t) % 12]);
    for (const [mode, prof] of [['major', MAJOR_PROFILE], ['minor', MINOR_PROFILE]]) {
      const score = correlate(rot, prof);
      if (score > best.score) best = { score, tonic: t, mode };
    }
  }
  return { ...best, name: NOTE_NAMES[best.tonic] + (best.mode === 'minor' ? ' minor' : ' major') };
}

const CHORD_TYPES = [
  { suffix: '', tones: [0, 4, 7], bonus: 0.04 },
  { suffix: 'm', tones: [0, 3, 7], bonus: 0.04 },
  { suffix: '7', tones: [0, 4, 7, 10], bonus: 0 },
  { suffix: 'm7', tones: [0, 3, 7, 10], bonus: 0 },
  { suffix: 'maj7', tones: [0, 4, 7, 11], bonus: 0 },
  { suffix: 'sus4', tones: [0, 5, 7], bonus: -0.02 },
  { suffix: 'dim', tones: [0, 3, 6], bonus: -0.03 },
  { suffix: '5', tones: [0, 7], bonus: -0.01 },
];

export function nameChord(c, bassPc = -1) {
  const norm = Math.sqrt(c.reduce((s, v) => s + v * v, 0));
  if (norm === 0) return null;
  let best = null;
  for (let root = 0; root < 12; root++) {
    for (const type of CHORD_TYPES) {
      const tpl = new Array(12).fill(0);
      type.tones.forEach((t) => { tpl[(root + t) % 12] = 1; });
      const tplNorm = Math.sqrt(type.tones.length);
      let dot = 0;
      for (let i = 0; i < 12; i++) dot += c[i] * tpl[i];
      const score = dot / (norm * tplNorm) + type.bonus + (root === bassPc ? 0.06 : 0);
      if (!best || score > best.score) best = { score, root, suffix: type.suffix };
    }
  }
  if (best.score < 0.55) return null;
  return { ...best, name: NOTE_NAMES[best.root] + best.suffix };
}

// One chord guess per half-measure, merged when it repeats.
export function detectChords(notes, beats, beatsPerChord = 2) {
  const out = [];
  for (let b = 0; b + beatsPerChord < beats.length; b += beatsPerChord) {
    const from = beats[b], to = beats[b + beatsPerChord];
    const seg = notes.filter((n) => n.start < to && end(n) > from);
    if (!seg.length) continue;
    const c = chroma(seg, from, to);
    const bass = seg.reduce((lo, n) => (n.midi < lo.midi ? n : lo), seg[0]);
    const chord = nameChord(c, pc(bass.midi));
    if (!chord) continue;
    const last = out[out.length - 1];
    if (last && last.name === chord.name && last.beat + last.beats === b) last.beats += beatsPerChord;
    else out.push({ beat: b, beats: beatsPerChord, name: chord.name, root: chord.root, suffix: chord.suffix });
  }
  return out;
}

// ---------- arranging for guitar ----------

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

// Shift a line by whole octaves so its centre sits at `target`, then fold
// any stragglers into [lo, hi]. Also applies an optional transpose.
export function fitRange(notes, lo, hi, target, transpose = 0) {
  if (!notes.length) return [];
  const med = median(notes.map((n) => n.midi + transpose));
  let shift = Math.round((target - med) / 12) * 12;
  const out = notes.map((n) => {
    let m = n.midi + transpose + shift;
    while (m < lo) m += 12;
    while (m > hi) m -= 12;
    return { ...n, midi: m };
  });
  return out.filter((n) => n.midi >= lo && n.midi <= hi);
}

// Solo-guitar arrangement: melody on top, chord roots (and fifths on long
// chords) in the bass, so a single guitar carries the whole song.
export function arrange(melody, chords, beats, { bassLo = 40, transpose = 0 } = {}) {
  // centred around E4-F4 so melody and bass fit under one hand in low positions
  const top = fitRange(melody, 55, 84, 65, transpose);
  const bass = [];
  for (const ch of chords) {
    const root = bassLo + ((ch.root + transpose - bassLo) % 12 + 12) % 12;
    const fifth = root + 7 <= bassLo + 14 ? root + 7 : root - 5;
    const parts = ch.beats >= 4 ? [[0, root], [2, fifth]] : [[0, root]];
    for (let k = 0; k < ch.beats; k += 4) {
      for (const [off, m] of parts) {
        const b = ch.beat + k + off;
        if (b >= ch.beat + ch.beats) continue;
        const start = beatToTime(beats, b), stop = beatToTime(beats, Math.min(b + 2, ch.beat + ch.beats));
        bass.push({ start, dur: stop - start, midi: m, vel: 0.45, bass: true });
      }
    }
  }
  return [...top, ...bass].sort((a, b) => a.start - b.start);
}
