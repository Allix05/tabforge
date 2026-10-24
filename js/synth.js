// Playback engine: plays the original audio and a plucked-string rendition of
// the tab in sync (Songsterr-style), with speed, mix, tone, loop and metronome.
import { pluck } from './ks.js';

const TONES = {
  clean: { brightness: 0.5, sustain: 0.997 },
  drive: { brightness: 0.7, sustain: 0.998 },
  acoustic: { brightness: 0.8, sustain: 0.995, pickPos: 0.12 },
};
const LOOKAHEAD = 0.15;

function driveCurve(amount = 3) {
  const n = 2048, curve = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    curve[i] = Math.tanh(amount * x) / Math.tanh(amount);
  }
  return curve;
}

export class Player {
  constructor() {
    this.notes = [];
    this.beats = [];
    this.duration = 0;
    this.rate = 1;
    this.playing = false;
    this.loop = null;
    this.metronome = false;
    this.tone = 'clean';
    this.cache = new Map();
    this.live = new Set();
    this.songStart = 0;
    this.ctxStart = 0;
    this.pausedAt = 0;
    this.onEnd = () => {};
  }

  ensureContext() {
    if (this.ctx) return;
    const ctx = this.ctx = new AudioContext();
    this.master = ctx.createGain();
    this.master.connect(ctx.destination);
    this.synthIn = ctx.createGain();
    this.synthOut = ctx.createGain();
    this.synthOut.connect(this.master);
    this.origGain = ctx.createGain();
    this.origGain.connect(this.master);
    this.clickGain = ctx.createGain();
    this.clickGain.gain.value = 0.35;
    this.clickGain.connect(this.master);
    this.buildToneChain();
    if (this.audioEl) this.attachAudio();
  }

  buildToneChain() {
    const ctx = this.ctx;
    this.synthIn.disconnect();
    this.chain?.forEach((n) => n.disconnect());
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18; comp.ratio.value = 3;
    let nodes;
    if (this.tone === 'drive') {
      const pre = ctx.createGain(); pre.gain.value = 5;
      const shaper = ctx.createWaveShaper(); shaper.curve = driveCurve(3.5); shaper.oversample = '4x';
      const mid = ctx.createBiquadFilter(); mid.type = 'peaking'; mid.frequency.value = 900; mid.gain.value = 5;
      const cab = ctx.createBiquadFilter(); cab.type = 'lowpass'; cab.frequency.value = 4200; cab.Q.value = 0.8;
      const post = ctx.createGain(); post.gain.value = 0.28;
      nodes = [pre, shaper, mid, cab, post, comp];
    } else {
      const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = this.tone === 'acoustic' ? 9000 : 5500;
      nodes = [lp, comp];
    }
    let prev = this.synthIn;
    for (const n of nodes) { prev.connect(n); prev = n; }
    prev.connect(this.synthOut);
    this.chain = nodes;
  }

  setTone(tone) {
    this.tone = tone;
    if (this.ctx) this.buildToneChain();
  }

  setOriginal(url) {
    if (this.audioEl) { this.audioEl.pause(); this.audioEl.src = ''; }
    const el = this.audioEl = new Audio();
    el.src = url;
    el.preservesPitch = true;
    el.preload = 'auto';
    this.mediaNode = null;
    if (this.ctx) this.attachAudio();
  }

  attachAudio() {
    if (!this.mediaNode) {
      this.mediaNode = this.ctx.createMediaElementSource(this.audioEl);
      this.mediaNode.connect(this.origGain);
    }
  }

  setTrack({ notes, beats, duration }) {
    this.stop();
    this.notes = [...notes].sort((a, b) => a.start - b.start);
    this.beats = beats;
    this.duration = duration;
  }

  // mix: 0 = only original, 1 = only tab synth
  setMix(mix) {
    this.mix = mix;
    if (!this.ctx) return;
    this.origGain.gain.value = Math.cos(mix * Math.PI / 2);
    this.synthOut.gain.value = Math.sin(mix * Math.PI / 2) * 1.2;
  }

  setRate(rate) {
    const t = this.time;
    this.rate = rate;
    if (this.audioEl) this.audioEl.playbackRate = rate;
    if (this.playing) this.seek(t);
  }

  get time() {
    if (!this.playing) return this.pausedAt;
    return this.songStart + (this.ctx.currentTime - this.ctxStart) * this.rate;
  }

  buffer(midi) {
    const key = this.tone + midi;
    if (!this.cache.has(key)) {
      const data = pluck(midi, this.ctx.sampleRate, 4, TONES[this.tone]);
      const buf = this.ctx.createBuffer(1, data.length, this.ctx.sampleRate);
      buf.copyToChannel(data, 0);
      this.cache.set(key, buf);
    }
    return this.cache.get(key);
  }

  playNote(n, when, held = n.dur / this.rate) {
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer(n.midi);
    const g = this.ctx.createGain();
    const vol = 0.25 + 0.55 * Math.min(1, n.vel);
    const stopAt = when + Math.min(3.8, held + 0.12);
    g.gain.setValueAtTime(vol, when);
    g.gain.setValueAtTime(vol, stopAt - 0.1);
    g.gain.linearRampToValueAtTime(0.0001, stopAt);
    src.connect(g).connect(this.synthIn);
    src.start(when);
    src.stop(stopAt + 0.02);
    this.live.add(src);
    src.onended = () => this.live.delete(src);
  }

  // audition a single note (clicking a fret number)
  preview(n) {
    this.ensureContext();
    this.ctx.resume();
    this.playNote({ ...n, dur: Math.min(n.dur, 1.5) }, this.ctx.currentTime + 0.01, Math.min(n.dur, 1.5));
  }

  click(when, accent) {
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.frequency.value = accent ? 1600 : 1000;
    g.gain.setValueAtTime(1, when);
    g.gain.exponentialRampToValueAtTime(0.001, when + 0.04);
    osc.connect(g).connect(this.clickGain);
    osc.start(when);
    osc.stop(when + 0.05);
    this.live.add(osc);
    osc.onended = () => this.live.delete(osc);
  }

  firstIndexAfter(arr, t, key = (x) => x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (key(arr[mid]) < t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  async play(from = this.pausedAt) {
    this.ensureContext();
    await this.ctx.resume();
    if (from >= this.duration - 0.05) from = 0;
    this.playing = true;
    this.songStart = from;
    this.ctxStart = this.ctx.currentTime + 0.05;
    this.noteIdx = this.firstIndexAfter(this.notes, from - 0.001, (n) => n.start);
    this.beatIdx = this.firstIndexAfter(this.beats, from - 0.001);
    this.setMix(this.mix ?? 0.5);
    if (this.audioEl) {
      this.audioEl.playbackRate = this.rate;
      this.audioEl.currentTime = Math.max(0, from);
      this.audioEl.play().catch(() => {});
    }
    clearInterval(this.timer);
    this.timer = setInterval(() => this.schedule(), 25);
    this.schedule();
  }

  schedule() {
    if (!this.playing) return;
    const now = this.time;
    if (this.loop && now >= this.loop.b) { this.seek(this.loop.a); return; }
    if (now >= this.duration) { this.stop(); this.onEnd(); return; }
    const horizon = now + LOOKAHEAD * this.rate;
    const toCtx = (t) => this.ctxStart + (t - this.songStart) / this.rate;
    const limit = this.loop ? Math.min(horizon, this.loop.b) : horizon;
    while (this.noteIdx < this.notes.length && this.notes[this.noteIdx].start < limit) {
      const n = this.notes[this.noteIdx++];
      if (n.start >= now - 0.03) this.playNote(n, Math.max(this.ctx.currentTime, toCtx(n.start)));
    }
    while (this.beatIdx < this.beats.length && this.beats[this.beatIdx] < limit) {
      const i = this.beatIdx++;
      if (this.metronome && this.beats[i] >= now - 0.03) this.click(Math.max(this.ctx.currentTime, toCtx(this.beats[i])), i % 4 === 0);
    }
    // keep the original recording locked to our clock
    if (this.audioEl && !this.audioEl.paused && Math.abs(this.audioEl.currentTime - now) > 0.12) {
      this.audioEl.currentTime = now;
    }
  }

  silence() {
    for (const s of this.live) { try { s.stop(); } catch { /* already stopped */ } }
    this.live.clear();
  }

  pause() {
    if (!this.playing) return;
    this.pausedAt = this.time;
    this.playing = false;
    clearInterval(this.timer);
    this.silence();
    this.audioEl?.pause();
  }

  stop() {
    this.pause();
    this.pausedAt = 0;
  }

  seek(t) {
    t = Math.max(0, Math.min(this.duration, t));
    if (this.playing) {
      this.playing = false;
      clearInterval(this.timer);
      this.silence();
      this.play(t);
    } else {
      this.pausedAt = t;
    }
  }
}
