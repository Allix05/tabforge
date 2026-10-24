// Scores TabForge's JavaScript stages on real GuitarSet data.
//
//   node eval_js.mjs fingering [labelsDir]   fit fret-assignment weights to real tabs
//   node eval_js.mjs cleanup [name]          score the note cleanup chain on evaluate.py's dump
//
// Fingering: players 00-04 fit, player 05 is reported. A note counts as
// correct when TabForge puts it on the same string (hence fret) as the player.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { assignFrets, DEFAULT_WEIGHTS } from '../js/tab.js';
import { dropContinuations, removeOvertones, dropMasked, cleanNotes, strumOnsets } from '../js/notes.js';

const STD = [40, 45, 50, 55, 59, 64];

function loadTracks(dir) {
  return readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => {
    const raw = JSON.parse(readFileSync(join(dir, f), 'utf8'));
    // snap to the 16th-note grid of the take's tempo (in the file name, e.g.
    // "BN1-129-Eb"), exactly like the app's quantization; strums share a column
    const bpm = Number(/-(\d+)-/.exec(f)[1]);
    const grid = 60 / bpm / 4;
    const seen = new Set();
    const notes = raw.map(([s, e, m, string]) => ({ start: s, dur: e - s, midi: Math.round(m), vel: 0.8, string, step: Math.round(s / grid) }))
      .filter((n) => { const k = `${n.step}:${n.midi}`; if (seen.has(k)) return false; seen.add(k); return true; })
      .sort((a, b) => a.start - b.start);
    return { id: f.slice(0, -5), notes };
  });
}

function stringAccuracy(tracks, weights) {
  let ok = 0, total = 0;
  for (const t of tracks) {
    const { notes } = assignFrets(t.notes, { strings: STD, weights });
    const truth = new Map(t.notes.map((n) => [`${n.step}:${n.midi}`, n.string]));
    for (const n of notes) {
      total++;
      if (truth.get(`${n.step}:${n.midi}`) === n.string) ok++;
    }
  }
  return ok / total;
}

function fingering(dir) {
  const all = loadTracks(dir);
  const fit = all.filter((t) => !t.id.startsWith('05_'));
  const test = all.filter((t) => t.id.startsWith('05_'));
  let best = { ...DEFAULT_WEIGHTS }, bestAcc = stringAccuracy(fit, best);
  console.log(`default weights: fit ${bestAcc.toFixed(4)}  test ${stringAccuracy(test, best).toFixed(4)}`);
  // coordinate search in log space, then random restarts around the best
  let rng = 12345;
  const rand = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648);
  const keys = Object.keys(DEFAULT_WEIGHTS);
  const additive = new Set(['open', 'stringIdx']); // may be negative (a bonus)
  for (let round = 0; round < 4; round++) {
    for (const k of keys) {
      for (const mult of [0, 0.25, 0.5, 0.75, 1.5, 2, 4]) {
        const cand = { ...best, [k]: additive.has(k) ? best[k] + (mult - 1) * 0.2 : best[k] * mult };
        const acc = stringAccuracy(fit, cand);
        if (acc > bestAcc + 1e-4) { best = cand; bestAcc = acc; }
      }
    }
    for (let i = 0; i < 25; i++) {
      const cand = Object.fromEntries(keys.map((k) => [k, additive.has(k) ? best[k] + (rand() - 0.5) * 0.2 : best[k] * Math.exp((rand() - 0.5) * 0.8)]));
      const acc = stringAccuracy(fit, cand);
      if (acc > bestAcc + 1e-4) { best = cand; bestAcc = acc; }
    }
    console.log(`round ${round}: fit ${bestAcc.toFixed(4)}  test ${stringAccuracy(test, best).toFixed(4)}`);
  }
  const rounded = Object.fromEntries(Object.entries(best).map(([k, v]) => [k, +v.toFixed(3)]));
  console.log('best weights', JSON.stringify(rounded));
}

function f1(pred, ref, tol = 0.05) {
  const r = ref.map(([s, , m]) => [s, Math.round(m)]).sort((a, b) => a[0] - b[0]);
  const p = pred.map((n) => [n.start, n.midi]).sort((a, b) => a[0] - b[0]);
  const used = new Array(p.length).fill(false);
  let tp = 0, j0 = 0;
  for (const [s, m] of r) {
    while (j0 < p.length && p[j0][0] < s - tol) j0++;
    for (let j = j0; j < p.length && p[j][0] <= s + tol; j++) {
      if (!used[j] && p[j][1] === m) { used[j] = true; tp++; break; }
    }
  }
  const P = p.length ? tp / p.length : 0, R = r.length ? tp / r.length : 0;
  return { p: P, r: R, f: P + R ? (2 * P * R) / (P + R) : 0 };
}

function cleanup(file) {
  const raw = JSON.parse(readFileSync(file, 'utf8'));
  const items = Array.isArray(raw) ? raw : raw.items;
  const chains = {
    none: (n) => n,
    range: (n) => cleanNotes(n, { minMidi: 40, maxMidi: 88, minDur: 0.05 }),
    'range+echo': (n) => dropContinuations(cleanNotes(n, { minMidi: 40, maxMidi: 88, minDur: 0.05 })),
  };
  for (const ratio of [0.5, 0.75]) {
    for (const mask of [0.3, 0.45, 0.55]) {
      chains[`range+echo+ovt${ratio}+mask${mask}`] = (n) => dropMasked(removeOvertones(chains['range+echo'](n), ratio), mask);
    }
  }
  const kinds = [...new Set(items.map((i) => i.kind))];
  for (const [name, fn] of Object.entries(chains)) {
    const row = kinds.map((k) => {
      const sel = items.filter((i) => i.kind === k);
      const fs = sel.map((i) => f1(fn(i.pred.map(([s, e, p, a]) => ({ start: s, dur: e - s, midi: p, vel: a }))), i.ref).f);
      return `${k} ${(fs.reduce((a, b) => a + b, 0) / fs.length).toFixed(3)}`;
    });
    console.log(name.padEnd(30), row.join('  '));
  }
}

// End to end: model notes -> range filter -> 16th grid -> fret assignment,
// scored against the real tab. "tab F1" needs pitch, onset AND string right.
function tabs(file) {
  const raw = JSON.parse(readFileSync(file, 'utf8'));
  const items = Array.isArray(raw) ? raw : raw.items;
  const hasStrings = items.some((i) => i.pred.some((p) => p[4]));
  const timbres = hasStrings ? [0, 0.3, 0.5, 0.8] : [0];
  const kinds = [...new Set(items.map((i) => i.kind))];
  for (const strum of [0, 0.03, 0.05, 0.07]) {
    for (const timbre of timbres) {
      const row = kinds.map((k) => {
        const r = scoreTabs(items.filter((i) => i.kind === k), { timbre }, strum);
        return `${k} ${r.tab.toFixed(3)}`;
      });
      console.log(`strum ${String(strum).padEnd(4)} timbre ${String(timbre).padEnd(4)} tab F1: ${row.join('  ')}`);
    }
  }
}

function scoreTabs(items, weights, strum = 0) {
  let tp = 0, tpString = 0, nPred = 0, nRef = 0;
  {
    for (const it of items) {
      // GuitarSet names carry the tempo; synthetic riffs fall back to a 50 ms grid
      const m = /-(\d+)-/.exec(it.track);
      const grid = m ? 60 / Number(m[1]) / 4 : 0.05;
      const ranged = cleanNotes(it.pred.map(([s, e, p, a, probs]) => ({ start: s, dur: e - s, midi: p, vel: a, ...(probs ? { stringProbs: probs } : {}) })), { minMidi: 40, maxMidi: 86, minDur: 0.05 });
      const onsets = strum ? strumOnsets(ranged, strum) : ranged.map((n) => n.start);
      const pred = ranged.map((n, i) => ({ ...n, step: Math.round(onsets[i] / grid) }));
      const { notes } = assignFrets(pred, { strings: STD, weights });
      const ref = it.ref.map(([s, , m, string]) => ({ start: s, midi: Math.round(m), string }));
      const used = new Array(notes.length).fill(false);
      for (const r of ref) {
        const j = notes.findIndex((n, i) => !used[i] && n.midi === r.midi && Math.abs(n.start - r.start) <= 0.05);
        if (j >= 0) {
          used[j] = true;
          tp++;
          if (notes[j].string === r.string) tpString++;
        }
      }
      nPred += notes.length;
      nRef += ref.length;
    }
  }
  const f = (t) => (2 * t) / (nPred + nRef);
  return { note: f(tp), tab: f(tpString) };
}

// Where do notes get lost between the model and the tab?
function losses(file) {
  const { items } = JSON.parse(readFileSync(file, 'utf8'));
  const stats = {};
  for (const it of items) {
    const s = (stats[it.kind] ??= { model: 0, range: 0, tab: 0, dropped: 0, sameStepPitch: 0 });
    const m = /-(\d+)-/.exec(it.track);
    const grid = m ? 60 / Number(m[1]) / 4 : 0.05;
    const all = it.pred.map(([st, e, p, a, probs]) => ({ start: st, dur: e - st, midi: p, vel: a, stringProbs: probs }));
    const ranged = cleanNotes(all, { minMidi: 40, maxMidi: 86, minDur: 0.05 }).map((n) => ({ ...n, step: Math.round(n.start / grid) }));
    const keys = new Set(ranged.map((n) => `${n.step}:${n.midi}`));
    const { notes, dropped } = assignFrets(ranged, { strings: STD });
    s.model += all.length; s.range += ranged.length; s.tab += notes.length; s.dropped += dropped;
    s.sameStepPitch += ranged.length - keys.size;
  }
  for (const [k, s] of Object.entries(stats)) console.log(k.padEnd(10), JSON.stringify(s));
}

const [mode, arg] = process.argv.slice(2);
if (mode === 'losses') losses(`C:/tftrain/results/${arg || 'tab'}_notes.json`);
else if (mode === 'tabs') tabs(`C:/tftrain/results/${arg || 'tab'}_notes.json`);
else if (mode === 'fingering') fingering(arg || 'C:/tftrain/cache/labels');
else if (mode === 'cleanup') cleanup(`C:/tftrain/results/${arg || 'finetuned'}_notes.json`);
else console.log('usage: node eval_js.mjs fingering|cleanup <path>');
