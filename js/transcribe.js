// Polyphonic note transcription with Spotify's Basic Pitch (Apache-2.0),
// running in the browser on TensorFlow.js (WebGL). The raw network output is
// kept so sensitivity changes re-run only the cheap note-decoding step.
import {
  BasicPitch, outputToNotesPoly, addPitchBendsToNoteEvents, noteFramesToTime,
} from 'https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/+esm';
import { fromBasicPitch } from './notes.js';

const MODEL_URL = new URL('../model/model.json', import.meta.url).href;
let basicPitch = null;

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

export async function analyze(audioBuffer, onProgress = () => {}) {
  basicPitch ??= new BasicPitch(MODEL_URL);
  const mono = await toMono22k(audioBuffer);
  const frames = [], onsets = [], contours = [];
  await basicPitch.evaluateModel(mono, (f, o, c) => {
    for (const row of f) frames.push(row);
    for (const row of o) onsets.push(row);
    for (const row of c) contours.push(row);
  }, onProgress);
  return { frames, onsets, contours, duration: audioBuffer.duration };
}

// sensitivity 0..1: higher catches quieter notes (and more false positives)
export function decodeNotes(raw, { sensitivity = 0.5, minFreq = null, maxFreq = null, minLenFrames = 6, inferOnsets = true } = {}) {
  // tuned on the demo: 0.5 gives ~0.9 precision at ~0.95 recall after cleanup
  const onset = 0.78 - 0.5 * sensitivity;
  const frame = 0.5 - 0.3 * sensitivity;
  const events = outputToNotesPoly(raw.frames, raw.onsets, onset, frame, minLenFrames, inferOnsets, maxFreq, minFreq, true);
  return fromBasicPitch(noteFramesToTime(addPitchBendsToNoteEvents(raw.contours, events)));
}
