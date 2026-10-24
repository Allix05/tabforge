// Fret assignment: turns pitched notes into (string, fret) positions a human
// can actually play. Each column of simultaneous notes gets a set of candidate
// fingerings; a Viterbi pass then picks the sequence with the least hand
// movement, stretch and awkward high-fret playing.

// strings listed low (6th) -> high (1st), as MIDI numbers
export const TUNINGS = {
  standard: { label: 'Standard (E A D G B E)', strings: [40, 45, 50, 55, 59, 64] },
  halfDown: { label: 'Half step down (Eb)', strings: [39, 44, 49, 54, 58, 63] },
  dropD: { label: 'Drop D (D A D G B E)', strings: [38, 45, 50, 55, 59, 64] },
  fullDown: { label: 'Full step down (D G C F A D)', strings: [38, 43, 48, 53, 57, 62] },
  dropC: { label: 'Drop C (C G C F A D)', strings: [36, 43, 48, 53, 57, 62] },
  openG: { label: 'Open G (D G D G B D)', strings: [38, 43, 50, 55, 59, 62] },
  dadgad: { label: 'DADGAD', strings: [38, 45, 50, 55, 57, 62] },
};

export const STRING_NAMES = (strings) => strings.map((m) => ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][m % 12]);

export function playableRange(strings, capo = 0, maxFret = 22) {
  return [strings[0] + capo, strings[strings.length - 1] + maxFret];
}

export function positionsFor(midi, strings, capo = 0, maxFret = 22) {
  const out = [];
  strings.forEach((open, s) => {
    const fret = midi - open - capo;
    if (fret >= 0 && fret <= maxFret - capo) out.push({ string: s, fret });
  });
  return out;
}

const MAX_SPAN = 4;
const MAX_CANDIDATES = 24;

function staticCost(fingering) {
  const fretted = fingering.filter((p) => p.fret > 0).map((p) => p.fret);
  if (!fretted.length) return 0;
  const lo = Math.min(...fretted), hi = Math.max(...fretted);
  const mean = fretted.reduce((a, b) => a + b, 0) / fretted.length;
  return mean * 0.04 + (hi - lo) * 0.35 + Math.max(0, hi - 12) * 0.12 + fretted.length * 0.03;
}

// all assignments of the column's notes to distinct strings within a hand span
export function candidatesFor(column, strings, capo, maxFret) {
  const notes = [...column].sort((a, b) => b.midi - a.midi);
  const options = notes.map((n) => positionsFor(n.midi, strings, capo, maxFret));
  const found = [];
  const used = new Set();
  const chosen = [];
  let budget = 4000;
  (function dfs(i, lo, hi) {
    if (budget-- <= 0) return;
    if (i === notes.length) {
      const f = chosen.map((p, k) => ({ ...p, note: notes[k] }));
      found.push({ fingering: f, cost: staticCost(f) });
      return;
    }
    for (const p of options[i]) {
      if (used.has(p.string)) continue;
      let nlo = lo, nhi = hi;
      if (p.fret > 0) { nlo = Math.min(lo, p.fret); nhi = Math.max(hi, p.fret); }
      if (nhi - nlo > MAX_SPAN) continue;
      used.add(p.string); chosen.push(p);
      dfs(i + 1, nlo, nhi);
      used.delete(p.string); chosen.pop();
    }
  })(0, Infinity, -Infinity);
  return found.sort((a, b) => a.cost - b.cost).slice(0, MAX_CANDIDATES);
}

const handPos = (fingering) => {
  const fretted = fingering.filter((p) => p.fret > 0).map((p) => p.fret);
  return fretted.length ? Math.min(...fretted) : null;
};
const meanString = (fingering) => fingering.reduce((a, p) => a + p.string, 0) / fingering.length;

export function assignFrets(notes, { strings = TUNINGS.standard.strings, capo = 0, maxFret = 22 } = {}) {
  const [lo, hi] = playableRange(strings, capo, maxFret);
  const columns = new Map();
  for (const n of notes) {
    if (n.midi < lo || n.midi > hi) continue;
    if (!columns.has(n.step)) columns.set(n.step, []);
    columns.get(n.step).push(n);
  }
  const steps = [...columns.keys()].sort((a, b) => a - b);
  let dropped = 0;

  const layers = steps.map((step) => {
    // strongest six at most; melody (top) notes are protected, bass/soft notes go first
    let col = columns.get(step).sort((a, b) => b.vel - a.vel).slice(0, strings.length);
    dropped += columns.get(step).length - col.length;
    let cands = candidatesFor(col, strings, capo, maxFret);
    while (!cands.length && col.length > 1) {
      const top = Math.max(...col.map((n) => n.midi));
      const removable = col.filter((n) => n.midi !== top);
      const weakest = removable.reduce((w, n) => (n.vel < w.vel ? n : w), removable[0]);
      col = col.filter((n) => n !== weakest);
      dropped++;
      cands = candidatesFor(col, strings, capo, maxFret);
    }
    return { step, time: col[0].start, cands };
  }).filter((l) => l.cands.length);

  // Viterbi over columns; each state remembers its effective hand position
  let prev = null;
  const history = [];
  for (const layer of layers) {
    const states = layer.cands.map((c) => ({ c, pos: handPos(c.fingering), score: c.cost, from: -1 }));
    if (prev) {
      const gap = layer.time - prev.time;
      const ease = gap > 0.6 ? 0.4 : 1;
      for (const st of states) {
        let best = Infinity, arg = 0, bestPos = st.pos;
        prev.states.forEach((ps, k) => {
          const pos = st.pos ?? ps.pos;
          const move = pos != null && ps.pos != null ? Math.abs(pos - ps.pos) : 0;
          const v = ps.score + ease * (move * 0.3 + Math.abs(meanString(st.c.fingering) - meanString(ps.c.fingering)) * 0.05);
          if (v < best) { best = v; arg = k; bestPos = pos; }
        });
        st.score += best; st.from = arg; st.pos = bestPos;
      }
    }
    history.push(states);
    prev = { states, time: layer.time };
  }

  const out = [];
  if (history.length) {
    const lastStates = history[history.length - 1];
    let k = lastStates.reduce((bi, s, i) => (s.score < lastStates[bi].score ? i : bi), 0);
    for (let i = history.length - 1; i >= 0; i--) {
      const st = history[i][k];
      for (const p of st.c.fingering) out.push({ ...p.note, string: p.string, fret: p.fret });
      k = st.from;
    }
  }
  return { notes: out.sort((a, b) => a.step - b.step || a.string - b.string), dropped };
}
