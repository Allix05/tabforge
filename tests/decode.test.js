import test from 'node:test';
import assert from 'node:assert/strict';
import { outputToNotesPoly } from '../js/decode.js';

const empty = (T) => Array.from({ length: T }, () => new Float32Array(88));

test('an onset peak followed by sustained energy becomes one note', () => {
  const frames = empty(60), onsets = empty(60);
  for (let t = 10; t < 40; t++) frames[t][40] = 0.8; // E4 (midi 61), 30 frames
  onsets[10][40] = 0.9;
  const notes = outputToNotesPoly(frames, onsets, 0.5, 0.3, 6, false, null, null, false);
  assert.equal(notes.length, 1);
  assert.equal(notes[0].pitchMidi, 61);
  assert.equal(notes[0].startFrame, 10);
  assert.equal(notes[0].durationFrames, 30);
  assert.ok(Math.abs(notes[0].amplitude - 0.8) < 1e-6);
});

test('melodia recovers a note with no onset peak, and skips short blips', () => {
  const frames = empty(80), onsets = empty(80);
  for (let t = 20; t < 50; t++) frames[t][30] = 0.7; // sustained, never "struck"
  for (let t = 60; t < 63; t++) frames[t][50] = 0.9; // 3 frames: too short
  const notes = outputToNotesPoly(frames, onsets, 0.5, 0.3, 6, false, null, null, true);
  assert.equal(notes.length, 1);
  assert.equal(notes[0].pitchMidi, 51);
  assert.equal(notes[0].startFrame, 20);
  assert.equal(notes[0].durationFrames, 29); // Basic Pitch's melodia end is exclusive of the last frame
});

test('frequency limits zero out notes outside the range', () => {
  const frames = empty(40), onsets = empty(40);
  for (let t = 5; t < 30; t++) frames[t][10] = 0.8;
  onsets[5][10] = 0.9;
  assert.equal(outputToNotesPoly(frames, onsets, 0.5, 0.3, 6, false, null, 100, true).length, 0); // below 100 Hz
});
