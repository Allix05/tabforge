import test from 'node:test';
import assert from 'node:assert/strict';
import { percentile, activationStats, autoThresholds, AUTO_RULES } from '../js/sensitivity.js';

test('percentile matches numpy linear interpolation', () => {
  const xs = Float32Array.from([0, 1, 2, 3, 4]);
  assert.equal(percentile(xs, 50), 2);
  assert.ok(Math.abs(percentile(xs, 95) - 3.8) < 1e-9);
  assert.equal(percentile(new Float32Array(0), 90), 0);
});

test('auto thresholds follow how strongly the network fires', () => {
  const quiet = { onsets: Array.from({ length: 100 }, (_, i) => [i % 10 === 0 ? 0.3 : 0.01, 0]), frames: Array.from({ length: 100 }, () => [0.2, 0.01]) };
  const loud = { onsets: Array.from({ length: 100 }, (_, i) => [i % 10 === 0 ? 0.95 : 0.02, 0]), frames: Array.from({ length: 100 }, () => [0.9, 0.02]) };
  const q = autoThresholds(activationStats(quiet), 'guitar');
  const l = autoThresholds(activationStats(loud), 'guitar');
  assert.ok(q.onset < l.onset && q.frame < l.frame);
  for (const t of [q, l]) {
    assert.ok(t.onset >= 0.2 && t.onset <= 0.9);
    assert.ok(t.frame >= 0.15 && t.frame <= 0.65);
  }
  assert.ok(AUTO_RULES.general && AUTO_RULES.guitar);
});

test('sparse recordings never get thresholds below the floor', () => {
  const sparse = { onsets: Array.from({ length: 100 }, () => [0.01, 0]), frames: Array.from({ length: 100 }, () => [0.01, 0]) };
  const t = autoThresholds(activationStats(sparse), 'guitar');
  assert.equal(t.onset, AUTO_RULES.guitar.onsetFloor);
  assert.equal(t.frame, AUTO_RULES.guitar.frameFloor);
});
