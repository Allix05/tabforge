// Polyphonic note transcription with Spotify's Basic Pitch (Apache-2.0),
// running in the browser on TensorFlow.js (WebGL). The raw network output is
// kept so sensitivity changes re-run only the cheap note-decoding step.
//
// Two models: the stock "general" one (voice, piano, anything) on TF.js, and
// the "guitar" one (js/guitar-model.js, ONNX): fine-tuned by training/train.py
// on GuitarSet with high-gain amp and band-mix augmentation, plus a string head.
import {
  BasicPitch, addPitchBendsToNoteEvents, noteFramesToTime,
} from 'https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/+esm';
import { outputToNotesPoly } from './decode.js';
import { fromBasicPitch } from './notes.js';
import { autoThresholds, activationStats } from './sensitivity.js';

const GENERAL_URL = new URL('../model/model.json', import.meta.url).href;
const models = {};
let guitar = null; // onnxruntime-web module, loaded on first use

export async function toMono22k(audioBuffer) {
  const sr = 22050;
  const ctx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * sr), sr);
  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0);
}

export async function analyze(audioBuffer, onProgress = () => {}, { model = 'general' } = {}) {
  const mono = await toMono22k(audioBuffer);
  let raw;
  if (model === 'guitar') {
    guitar ??= await import('./guitar-model.js');
    raw = await guitar.analyzeGuitar(mono, onProgress);
  } else {
    models.general ??= new BasicPitch(GENERAL_URL);
    const frames = [], onsets = [], contours = [];
    await models.general.evaluateModel(mono, (f, o, c) => {
      for (const row of f) frames.push(row);
      for (const row of o) onsets.push(row);
      for (const row of c) contours.push(row);
    }, onProgress);
    raw = { frames, onsets, contours };
  }
  Object.assign(raw, { duration: audioBuffer.duration, model });
  raw.stats = activationStats(raw);
  return raw;
}

// sensitivity: 'auto' (per-song thresholds learned on GuitarSet) or 0..1,
// where higher catches quieter notes (and more false positives)
export function thresholdsFor(raw, sensitivity = 'auto') {
  if (sensitivity === 'auto') return autoThresholds(raw.stats, raw.model);
  return { onset: 0.78 - 0.5 * sensitivity, frame: 0.5 - 0.3 * sensitivity };
}

export function decodeNotes(raw, { sensitivity = 'auto', minFreq = null, maxFreq = null, minLenFrames = 6, inferOnsets = true } = {}) {
  const { onset, frame } = thresholdsFor(raw, sensitivity);
  const events = outputToNotesPoly(raw.frames, raw.onsets, onset, frame, minLenFrames, inferOnsets, maxFreq, minFreq, true);
  const timed = noteFramesToTime(addPitchBendsToNoteEvents(raw.contours, events));
  if (raw.strings) {
    // noteFramesToTime keeps the order, so frame indices line up with `events`
    timed.forEach((n, i) => {
      n.stringProbs = guitar.stringProbsFor(raw.strings, events[i].startFrame, events[i].durationFrames, events[i].pitchMidi);
    });
  }
  return fromBasicPitch(timed);
}
