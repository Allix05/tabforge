// SVG tab notation: measures wrapped into rows, fret numbers on six strings,
// chord names on top, rhythm stems/beams below, a moving playback cursor and
// highlighted notes. Plus a fretboard that lights up what is sounding.
import { STRING_NAMES } from './tab.js';
import { timeToBeat, beatToTime } from './notes.js';

const NS = 'http://www.w3.org/2000/svg';
const STEPS = 16;
const STRING_GAP = 15;
const TOP = 34;
const ROW_H = TOP + STRING_GAP * 5 + 58;
const LEFT = 30;

function el(name, attrs = {}, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent?.appendChild(e);
  return e;
}

export class TabView {
  constructor(container, { onSeek, onNoteClick } = {}) {
    this.container = container;
    this.onSeek = onSeek;
    this.onNoteClick = onNoteClick;
    this.data = null;
    this.lastRow = -1;
    new ResizeObserver(() => {
      const w = this.container.clientWidth;
      if (this.data && Math.abs(w - this.width) > 30) this.render(this.data);
    }).observe(container);
  }

  render(data) {
    this.data = data;
    const { notes, strings, chords = [], beats } = data;
    this.width = this.container.clientWidth;
    const names = STRING_NAMES(strings);
    if (names[names.length - 1] === names[0]) names[names.length - 1] = names[0].toLowerCase();
    const lastStep = notes.length ? Math.max(...notes.map((n) => n.step + n.durSteps)) : STEPS;
    const nMeasures = Math.max(1, Math.ceil(lastStep / STEPS));
    const perRow = Math.max(1, Math.floor((this.width - LEFT - 8) / 300));
    const measureW = (this.width - LEFT - 8) / perRow;
    const stepW = (measureW - 16) / STEPS;
    const rows = Math.ceil(nMeasures / perRow);
    Object.assign(this, { perRow, measureW, stepW, nMeasures, rows });

    const svg = el('svg', { width: this.width, height: rows * ROW_H + 10, class: 'tab-svg' });
    const stringY = (s) => TOP + (strings.length - 1 - s) * STRING_GAP;
    this.stringY = stringY;

    for (let r = 0; r < rows; r++) {
      const g = el('g', { transform: `translate(0 ${r * ROW_H})` }, svg);
      const mCount = Math.min(perRow, nMeasures - r * perRow);
      const right = LEFT + mCount * measureW;
      names.forEach((nm, s) => {
        el('text', { x: 12, y: stringY(s) + 4, class: 'string-name' }, g).textContent = nm;
        el('line', { x1: LEFT, x2: right, y1: stringY(s), y2: stringY(s), class: 'string' }, g);
      });
      for (let m = 0; m <= mCount; m++) {
        const x = LEFT + m * measureW;
        el('line', { x1: x, x2: x, y1: stringY(strings.length - 1), y2: stringY(0), class: 'bar' }, g);
        if (m < mCount) el('text', { x: x + 3, y: TOP - 20, class: 'measure-num' }, g).textContent = r * perRow + m + 1;
      }
    }

    const pos = (step) => {
      const m = Math.floor(step / STEPS), r = Math.floor(m / perRow);
      return { x: LEFT + (m % perRow) * measureW + 10 + (step % STEPS) * stepW + stepW / 2, row: r, y0: r * ROW_H };
    };
    this.pos = pos;

    for (const ch of chords) {
      const p = pos(ch.beat * 4);
      el('text', { x: p.x - stepW / 2, y: p.y0 + TOP - 8, class: 'chord-name' }, svg).textContent = ch.name;
    }

    // rhythm: one stem per column, beamed within each beat
    const columns = [...new Set(notes.map((n) => n.step))].sort((a, b) => a - b);
    const stemTop = TOP + STRING_GAP * 5 + 12, stemBot = stemTop + 18;
    const byBeat = new Map();
    columns.forEach((step, i) => {
      const next = columns[i + 1] ?? step + 4;
      const beatEnd = (Math.floor(step / 4) + 1) * 4;
      const ioi = Math.min(next, beatEnd) - step;
      const p = pos(step);
      el('line', { x1: p.x, x2: p.x, y1: p.y0 + stemTop, y2: p.y0 + stemBot, class: 'stem' }, svg);
      const key = Math.floor(step / 4);
      if (!byBeat.has(key)) byBeat.set(key, []);
      byBeat.get(key).push({ step, ioi, p });
    });
    for (const group of byBeat.values()) {
      if (group.length === 1) {
        const { ioi, p } = group[0];
        const flags = ioi <= 1 ? 2 : ioi <= 2 ? 1 : 0;
        for (let f = 0; f < flags; f++) {
          el('line', { x1: p.x, x2: p.x + 6, y1: p.y0 + stemBot - f * 5, y2: p.y0 + stemBot - f * 5 - 4, class: 'beam' }, svg);
        }
        continue;
      }
      const a = group[0].p, b = group[group.length - 1].p;
      if (a.row !== b.row) continue;
      el('line', { x1: a.x, x2: b.x, y1: a.y0 + stemBot, y2: b.y0 + stemBot, class: 'beam' }, svg);
      group.forEach((c, i) => {
        if (c.ioi > 1) return;
        const nb = group[i + 1];
        const x2 = nb && nb.ioi <= 1 ? nb.p.x : nb ? c.p.x + (nb.p.x - c.p.x) / 2 : c.p.x - 6;
        el('line', { x1: c.p.x, x2, y1: c.p.y0 + stemBot - 5, y2: c.p.y0 + stemBot - 5, class: 'beam' }, svg);
      });
    }

    // notes: sustain bar along the string, then the fret number
    this.noteEls = [];
    for (const n of notes) {
      const p = pos(n.step);
      const y = p.y0 + stringY(n.string);
      const endInMeasure = (Math.floor(n.step / STEPS) + 1) * STEPS;
      const len = Math.min(n.durSteps, endInMeasure - n.step) * stepW;
      el('rect', { x: p.x, y: y - 2, width: Math.max(0, len - 4), height: 4, rx: 2, class: 'sustain' }, svg);
      const txt = String(n.fret);
      const w = txt.length * 7 + 4;
      const g = el('g', { class: 'note' + (n.bass ? ' bass' : '') }, svg);
      el('rect', { x: p.x - w / 2, y: y - 7, width: w, height: 14, rx: 3, class: 'note-bg' }, g);
      el('text', { x: p.x, y: y + 4, class: 'fret' }, g).textContent = txt;
      g.addEventListener('click', (e) => { e.stopPropagation(); this.onNoteClick?.(n); });
      this.noteEls.push({ n, g });
    }

    this.cursor = el('rect', { x: 0, y: 0, width: 3, height: STRING_GAP * 5 + 20, rx: 1.5, class: 'cursor' }, svg);
    this.loopRect = el('rect', { x: 0, y: 0, width: 0, height: 0, class: 'loop-band' }, svg);
    svg.insertBefore(this.loopRect, svg.firstChild);
    svg.addEventListener('click', (e) => {
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const row = Math.floor(y / ROW_H);
      const m = Math.floor((x - LEFT) / measureW);
      if (m < 0 || m >= perRow) return;
      const measure = row * perRow + m;
      const within = Math.max(0, Math.min(STEPS - 1, Math.floor((x - LEFT - m * measureW - 10) / stepW)));
      this.onSeek?.(beatToTime(beats, (measure * STEPS + within) / 4));
    });

    this.container.replaceChildren(svg);
    this.svg = svg;
    this.active = new Set();
    this.lastRow = -1;
    this.setLoop(data.loop);
  }

  xyForTime(t) {
    const step = timeToBeat(this.data.beats, t) * 4;
    if (step < 0) return this.pos(0);
    const p = this.pos(Math.floor(step));
    return { ...p, x: p.x - this.stepW / 2 + (step % 1) * this.stepW };
  }

  setCursor(t, follow = true) {
    if (!this.data) return;
    const p = this.xyForTime(t);
    this.cursor.setAttribute('x', p.x - 1.5);
    this.cursor.setAttribute('y', p.y0 + TOP - 10);
    for (const item of this.noteEls) {
      const on = t >= item.n.start - 0.02 && t < item.n.start + Math.max(0.12, item.n.dur);
      if (on !== this.active.has(item)) {
        item.g.classList.toggle('active', on);
        on ? this.active.add(item) : this.active.delete(item);
      }
    }
    if (follow && p.row !== this.lastRow) {
      this.lastRow = p.row;
      const scroller = this.container.parentElement;
      scroller.scrollTo({ top: Math.max(0, p.row * ROW_H - 20), behavior: 'smooth' });
    }
    return [...this.active].map((a) => a.n);
  }

  setLoop(loop) {
    if (!this.loopRect) return;
    if (!loop) { this.loopRect.setAttribute('width', 0); return; }
    const a = this.xyForTime(loop.a), b = this.xyForTime(loop.b);
    if (a.row !== b.row) {
      // spans rows: shade from A to the end of its row
      this.loopRect.setAttribute('x', a.x);
      this.loopRect.setAttribute('width', LEFT + this.perRow * this.measureW - a.x);
    } else {
      this.loopRect.setAttribute('x', a.x);
      this.loopRect.setAttribute('width', Math.max(2, b.x - a.x));
    }
    this.loopRect.setAttribute('y', a.y0 + TOP - 12);
    this.loopRect.setAttribute('height', STRING_GAP * 5 + 24);
  }
}

export class Fretboard {
  constructor(container) {
    this.container = container;
  }

  render(strings, maxFret = 15) {
    this.strings = strings;
    this.maxFret = maxFret;
    const w = Math.max(this.container.clientWidth, 320), h = 120;
    const left = 26, right = 10, top = 14, gap = (h - top * 2) / 5;
    const fretX = (f) => left + (w - left - right) * (1 - 2 ** (-f / 12)) / (1 - 2 ** (-maxFret / 12));
    this.geom = { fretX, top, gap, left };
    const svg = el('svg', { width: w, height: h, class: 'fretboard-svg' });
    el('rect', { x: left, y: top - 6, width: w - left - right, height: gap * 5 + 12, rx: 4, class: 'fb-wood' }, svg);
    for (let f = 0; f <= maxFret; f++) {
      el('line', { x1: fretX(f), x2: fretX(f), y1: top - 6, y2: top + gap * 5 + 6, class: f === 0 ? 'fb-nut' : 'fb-fret' }, svg);
    }
    for (const f of [3, 5, 7, 9, 15, 17, 19, 21].filter((f) => f <= maxFret)) {
      el('circle', { cx: (fretX(f - 1) + fretX(f)) / 2, cy: top + gap * 2.5, r: 4, class: 'fb-inlay' }, svg);
    }
    if (maxFret >= 12) {
      const cx = (fretX(11) + fretX(12)) / 2;
      el('circle', { cx, cy: top + gap * 1.5, r: 4, class: 'fb-inlay' }, svg);
      el('circle', { cx, cy: top + gap * 3.5, r: 4, class: 'fb-inlay' }, svg);
    }
    const names = STRING_NAMES(strings);
    strings.forEach((_, s) => {
      const y = top + (strings.length - 1 - s) * gap;
      el('line', { x1: left, x2: w - right, y1: y, y2: y, class: 'fb-string', 'stroke-width': 0.8 + s * -0.1 + (5 - s) * 0.25 }, svg);
      el('text', { x: 8, y: y + 4, class: 'string-name' }, svg).textContent = s === strings.length - 1 && names[s] === names[0] ? names[s].toLowerCase() : names[s];
    });
    this.dots = el('g', {}, svg);
    this.container.replaceChildren(svg);
  }

  show(notes) {
    if (!this.dots) return;
    const { fretX, top, gap } = this.geom;
    this.dots.replaceChildren();
    for (const n of notes) {
      const f = Math.min(n.fret, this.maxFret);
      const x = f === 0 ? fretX(0) - 9 : (fretX(f - 1) + fretX(f)) / 2;
      const y = top + (this.strings.length - 1 - n.string) * gap;
      el('circle', { cx: x, cy: y, r: 8, class: 'fb-dot' + (f === 0 ? ' open' : '') }, this.dots);
      el('text', { x, y: y + 3.5, class: 'fb-dot-label' }, this.dots).textContent = n.fret;
    }
  }
}
