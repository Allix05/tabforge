// The guitar model: Basic Pitch fine-tuned on GuitarSet (high-gain amp, band
// mix and stem-bleed augmentation) plus a string head that hears which string
// each note is played on. Exported from training/export_onnx.py and run with
// onnxruntime-web. Windowing mirrors training/common.py run_model exactly.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const MODEL_URL = new URL('../model-guitar/tabforge-guitar.onnx', import.meta.url).href;
const N_SAMPLES = 43844; // 2 s at 22.05 kHz minus one hop
const HOP = 256;
const OVERLAP = 30; // frames shared by neighbouring windows
const KEEP = 172 - OVERLAP; // frames kept per window
const BATCH = 4;
let session = null;

export async function analyzeGuitar(mono, onProgress = () => {}) {
  session ??= await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
  const pre = (OVERLAP * HOP) / 2;
  const hop = N_SAMPLES - OVERLAP * HOP;
  const n = Math.ceil((mono.length + pre - N_SAMPLES) / hop) + 1;
  const x = new Float32Array((n - 1) * hop + N_SAMPLES);
  x.set(mono, pre);

  const frames = [], onsets = [], contours = [], strings = [];
  for (let i = 0; i < n; i += BATCH) {
    const b = Math.min(BATCH, n - i);
    const input = new Float32Array(b * N_SAMPLES);
    for (let k = 0; k < b; k++) input.set(x.subarray((i + k) * hop, (i + k) * hop + N_SAMPLES), k * N_SAMPLES);
    const out = await session.run({ audio: new ort.Tensor('float32', input, [b, N_SAMPLES]) });
    for (let k = 0; k < b; k++) {
      for (let f = OVERLAP / 2; f < OVERLAP / 2 + KEEP; f++) {
        const t = k * 172 + f;
        frames.push(Array.from(out.note.data.subarray(t * 88, t * 88 + 88)));
        onsets.push(Array.from(out.onset.data.subarray(t * 88, t * 88 + 88)));
        contours.push(Array.from(out.contour.data.subarray(t * 264, t * 264 + 264)));
        strings.push(out.string.data.slice(t * 528, t * 528 + 528));
      }
    }
    onProgress(Math.min(1, (i + b) / n));
    await new Promise((r) => setTimeout(r, 0)); // keep the page responsive
  }
  const keep = Math.floor(mono.length * (86 / 22050));
  return { frames: frames.slice(0, keep), onsets: onsets.slice(0, keep), contours: contours.slice(0, keep), strings: strings.slice(0, keep) };
}

// Mean string probabilities over the first frames of a note (its attack is
// the most telling part of the timbre).
export function stringProbsFor(strings, startFrame, durationFrames, pitchMidi) {
  const p = pitchMidi - 21;
  const probs = new Array(6).fill(0);
  const n = Math.max(1, Math.min(4, durationFrames));
  let used = 0;
  for (let f = startFrame; f < startFrame + n && f < strings.length; f++) {
    for (let s = 0; s < 6; s++) probs[s] += strings[f][p * 6 + s];
    used++;
  }
  return used ? probs.map((v) => v / used) : null;
}
