// Karplus-Strong plucked string with an allpass fractional delay so every
// pitch is in tune (plain KS rounds the delay line and drifts up to ~25 cents
// on high notes). Pure DSP, shared by the live synth and the demo renderer.

export const midiToFreq = (m) => 440 * 2 ** ((m - 69) / 12);

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return ((s >>> 0) / 4294967296) * 2 - 1;
  };
}

export function pluck(midi, sampleRate, seconds, { brightness = 0.55, sustain = 0.996, pickPos = 0.18, seed = 7 } = {}) {
  const freq = midiToFreq(midi);
  const L = sampleRate / freq - 0.5;
  const N = Math.max(2, Math.floor(L - 0.1));
  const frac = L - N;
  const C = (1 - frac) / (1 + frac);

  const rand = rng(seed + midi * 131);
  const buf = new Float32Array(N);
  let lp = 0;
  for (let i = 0; i < N; i++) {
    lp += brightness * (rand() - lp);
    buf[i] = lp;
  }
  // pick-position comb: plucking near the bridge thins out some harmonics
  const d = Math.max(1, Math.round(pickPos * N));
  const exc = Float32Array.from(buf, (v, i) => v - (i >= d ? buf[i - d] : 0));
  buf.set(exc);

  // higher strings ring shorter, as on a real guitar
  const damp = Math.min(0.9995, sustain + (1 - sustain) * 0.6 * (1 - Math.min(1, (midi - 40) / 50)));
  const out = new Float32Array(Math.ceil(seconds * sampleRate));
  let idx = 0, prev = 0, apX = 0, apY = 0;
  for (let i = 0; i < out.length; i++) {
    const cur = buf[idx];
    out[i] = cur;
    const avg = damp * 0.5 * (cur + prev);
    prev = cur;
    const y = C * avg + apX - C * apY;
    apX = avg; apY = y;
    buf[idx] = y;
    idx = idx + 1 === N ? 0 : idx + 1;
  }
  // normalise so all pitches come out at similar loudness
  let peak = 0;
  for (let i = 0; i < Math.min(out.length, 4096); i++) peak = Math.max(peak, Math.abs(out[i]));
  if (peak > 0) for (let i = 0; i < out.length; i++) out[i] /= peak;
  return out;
}

// Mix notes ({start, dur, midi, vel}) into a mono buffer.
export function renderNotes(notes, sampleRate, { tail = 1.5, ...opts } = {}) {
  const total = Math.max(0, ...notes.map((n) => n.start + n.dur)) + tail;
  const out = new Float32Array(Math.ceil(total * sampleRate));
  const cache = new Map();
  for (const n of notes) {
    const ring = n.dur + 0.08;
    if (!cache.has(n.midi)) cache.set(n.midi, pluck(n.midi, sampleRate, 4, opts));
    const src = cache.get(n.midi);
    const off = Math.round(n.start * sampleRate);
    const len = Math.min(src.length, Math.round(ring * sampleRate));
    const rel = Math.round(0.06 * sampleRate);
    for (let i = 0; i < len && off + i < out.length; i++) {
      const env = i > len - rel ? (len - i) / rel : 1;
      out[off + i] += src[i] * n.vel * env * 0.3;
    }
  }
  let peak = 0;
  for (const v of out) peak = Math.max(peak, Math.abs(v));
  if (peak > 0.95) for (let i = 0; i < out.length; i++) out[i] *= 0.95 / peak;
  return out;
}
