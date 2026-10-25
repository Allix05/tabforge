// Automatic sensitivity: pick decoding thresholds per song from how strongly
// the network fires on it. Quiet or busy recordings (full mixes, heavy
// distortion) get lower thresholds, clean solo guitar higher ones.
// The rule constants come from training/evaluate.py, fitted on GuitarSet
// player 04 and reported on the untouched player 05.

// floors keep sparse recordings (isolated notes, lots of silence) from getting
// thresholds so low that noise turns into notes
export const AUTO_RULES = {
  general: { a: 0.85, b: 0.75, onsetFloor: 0.5, frameFloor: 0.25 },
  guitar: { a: 0.75, b: 0.55, onsetFloor: 0.4, frameFloor: 0.25 },
};

// numpy-style percentile (linear interpolation)
export function percentile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function activationStats({ onsets, frames }) {
  const peaks = Float32Array.from(onsets, (row) => Math.max(...row)).sort();
  const all = new Float32Array(frames.length * (frames[0]?.length || 0));
  let k = 0;
  for (const row of frames) for (const v of row) all[k++] = v;
  all.sort();
  return { onsetP95: percentile(peaks, 95), noteP99: percentile(all, 99.5) };
}

const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function autoThresholds(stats, model = 'general') {
  const { a, b, onsetFloor, frameFloor } = AUTO_RULES[model] || AUTO_RULES.general;
  return { onset: clip(a * stats.onsetP95, onsetFloor, 0.9), frame: clip(b * stats.noteP99, frameFloor, 0.65) };
}
