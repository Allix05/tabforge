import { buildTab, MODES } from './pipeline.js';
import { TUNINGS } from './tab.js';
import { TabView, Fretboard } from './render.js';
import { Player } from './synth.js';
import { toAscii, toMidi } from './export.js';
import { renderDemo } from './demo.js';
import { helperStatus, jobFromUrl, jobFromFile, waitForJob, fetchStem } from './helper.js';

const $ = (id) => document.getElementById(id);
const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

const state = {
  pane: 'file',
  source: null, // { id, kind, name, file?, url?, blob? }
  helper: null,
  job: null, // { key, id, stems, title }
  mix: null, // { buffer, url }
  raw: new Map(), // stem -> basic-pitch output
  decoded: new Map(), // stem|sensitivity -> notes
  inputs: null, // { main, accomp } stem names used for the current result
  result: null,
  bpmOverride: 0,
  busy: false,
};

window.tabforge = state; // handy for poking at results from the devtools console

const player = new Player();
const tabView = new TabView($('tabView'), {
  onSeek: (t) => { player.seek(t); tick(true); },
  onNoteClick: (n) => player.preview(n),
});
const fretboard = new Fretboard($('fretboard'));
let transcriber = null; // lazily imported (pulls TensorFlow.js from the CDN)

// ---------- setup ----------

for (const [key, t] of Object.entries(TUNINGS)) $('tuning').add(new Option(t.label, key));
for (let c = 0; c <= 9; c++) $('capo').add(new Option(c ? `Fret ${c}` : 'None', c));

document.querySelectorAll('.tab-btn').forEach((b) => b.addEventListener('click', () => {
  state.pane = b.dataset.pane;
  document.querySelectorAll('.tab-btn').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.pane').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== state.pane));
  if (state.pane === 'link') refreshHelper();
  restorePaneSource();
}));

function setSource(src) {
  if (src && state.source && src.id === state.source.id) return;
  state.source = src;
  state.job = null;
  state.mix = null;
  state.raw.clear();
  state.decoded.clear();
  state.bpmOverride = 0;
  const box = $('sourceStatus');
  box.classList.toggle('hidden', !src);
  if (src) box.innerHTML = `<span class="dot"></span> Ready: <strong>${escapeHtml(src.name)}</strong>`;
  updateGenerate();
}

let recorded = null;
function restorePaneSource() {
  const f = $('fileInput').files[0];
  const url = $('linkInput').value.trim();
  if (state.pane === 'file') setSource(f ? { id: `file:${f.name}:${f.size}:${f.lastModified}`, kind: 'file', name: f.name, file: f } : null);
  else if (state.pane === 'link') setSource(url ? { id: `link:${url}`, kind: 'link', name: url, url } : null);
  else if (state.pane === 'record') setSource(recorded);
  else updateGenerate();
}

function updateGenerate() {
  const link = state.pane === 'link' && $('linkInput').value.trim();
  $('generateBtn').disabled = state.busy || !(state.source || link);
}

const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

// file input + drag & drop
$('fileInput').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (f) setSource({ id: `file:${f.name}:${f.size}:${f.lastModified}`, kind: 'file', name: f.name, file: f });
});
const dz = $('dropzone');
['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove('over')));
dz.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) setSource({ id: `file:${f.name}:${f.size}:${f.lastModified}`, kind: 'file', name: f.name, file: f });
});

$('linkInput').addEventListener('input', () => {
  const url = $('linkInput').value.trim();
  setSource(url ? { id: `link:${url}`, kind: 'link', name: url, url } : null);
});
$('linkInput').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !$('generateBtn').disabled) generate(); });

$('demoBtn').addEventListener('click', () => {
  setSource({ id: 'demo', kind: 'demo', name: 'TabForge demo (Am – F – C – G)' });
  generate();
});

// microphone recording
let recorder = null, recTimer = null;
$('recordBtn').addEventListener('click', async () => {
  if (recorder) { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recTimer);
      recorder = null;
      $('recordBtn').textContent = '● Record again';
      $('recordBtn').classList.remove('recording');
      const blob = new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' });
      recorded = { id: `rec:${Date.now()}`, kind: 'record', name: `Recording (${$('recordTime').textContent})`, blob };
      setSource(recorded);
    };
    recorder.start();
    const t0 = Date.now();
    recTimer = setInterval(() => { $('recordTime').textContent = fmt((Date.now() - t0) / 1000); }, 250);
    $('recordBtn').textContent = '■ Stop';
    $('recordBtn').classList.add('recording');
  } catch (e) {
    showError('Microphone unavailable: ' + e.message);
  }
});

// ---------- local helper ----------

async function refreshHelper() {
  state.helper = await helperStatus();
  const box = $('helperBox');
  if (state.helper) {
    box.className = 'helper-box ok';
    box.innerHTML = `<span class="dot"></span> Local helper connected. Links work${state.helper.separation
      ? `, and AI stem separation is on (${state.helper.device.toUpperCase()}).`
      : '. Stem separation is not installed (optional): <code>pip install -r helper/requirements-separation.txt</code>'}`;
  } else {
    box.className = 'helper-box';
    box.innerHTML = `
      <strong>One-time setup for links (free, about 1 minute):</strong> browsers aren't allowed to download from YouTube or Spotify,
      so TabForge uses a tiny helper that runs on <em>your</em> computer.
      <ol>
        <li>Install <a href="https://www.python.org/downloads/" target="_blank" rel="noopener">Python 3.10+</a>.</li>
        <li><a href="https://github.com/Allix05/tabforge/archive/refs/heads/main.zip">Download TabForge</a> and unzip it.</li>
        <li>Double-click <code>start-windows.bat</code> (or run <code>./start-mac-linux.sh</code>). It installs what it needs and opens TabForge with links and stem separation enabled.</li>
      </ol>
      <button class="btn small ghost" id="recheckHelper">I started it, check again</button>`;
    $('recheckHelper').onclick = refreshHelper;
  }
}
refreshHelper();

// ---------- audio plumbing ----------

async function decodeBlob(blob) {
  const ctx = new OfflineAudioContext(1, 1, 44100);
  return ctx.decodeAudioData(await blob.arrayBuffer());
}

function wavBlob(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 32767, true));
  return new Blob([buf], { type: 'audio/wav' });
}

async function ensureMix() {
  if (state.mix) return;
  const src = state.source;
  let blob;
  if (src.kind === 'demo') blob = wavBlob(renderDemo(44100), 44100);
  else if (state.job) blob = await fetchStem(state.job.id, 'mix');
  else blob = src.file || src.blob;
  progress(null, 'Decoding audio…');
  let buffer;
  try {
    buffer = await decodeBlob(blob);
  } catch {
    throw new Error("This file couldn't be decoded by your browser. Try MP3, WAV or M4A.");
  }
  state.mix = { buffer, url: URL.createObjectURL(blob) };
}

async function ensureJob(wantSep) {
  const src = state.source;
  const key = `${src.id}|${wantSep}`;
  if (state.job && (state.job.key === key || (state.job.separated && state.job.sourceId === src.id))) return;
  if (!state.helper) {
    await refreshHelper();
    if (!state.helper) throw new Error('Links need the free local helper. Follow the setup steps under the link box, then try again.');
  }
  progress(0, 'Sending to the local helper…');
  const created = src.kind === 'link'
    ? await jobFromUrl(src.url, wantSep)
    : await jobFromFile(src.file || src.blob || new File([wavBlob(renderDemo(44100), 44100)], 'demo.wav'), wantSep);
  const job = await waitForJob(created.id, (j) => {
    const extra = j.elapsed ? ` ${j.elapsed}s` : '';
    progress(j.status === 'separating' ? null : j.progress, j.message + extra);
  });
  state.job = { key, id: job.id, stems: job.stems || [], title: job.title, artist: job.artist, separated: wantSep, sourceId: src.id };
  state.mix = null;
  state.raw.clear();
  state.decoded.clear();
}

function stemsForMode(mode) {
  const has = (s) => state.job?.stems?.includes(s);
  const choice = $('stem').value;
  const m = MODES[mode];
  const main = choice !== 'auto' ? (choice === 'mix' || has(choice) ? choice : 'mix') : has(m.stem) ? m.stem : 'mix';
  const accomp = m.accomp && has(m.accomp) && main !== 'mix' ? m.accomp : null;
  return { main, accomp };
}

async function rawFor(stem, label) {
  if (state.raw.has(stem)) return state.raw.get(stem);
  transcriber ??= await import('./transcribe.js');
  let buffer;
  if (stem === 'mix') {
    buffer = state.mix.buffer;
  } else {
    progress(null, `Loading the ${stem} track…`);
    buffer = await decodeBlob(await fetchStem(state.job.id, stem));
  }
  progress(0, `Listening for notes (${label})…`);
  const raw = await transcriber.analyze(buffer, (p) => progress(p, `Listening for notes (${label})… ${Math.round(p * 100)}%`));
  state.raw.set(stem, raw);
  return raw;
}

function notesFor(stem) {
  const sens = Number($('sensitivity').value);
  const key = `${stem}|${sens}`;
  if (!state.decoded.has(key)) state.decoded.set(key, transcriber.decodeNotes(state.raw.get(stem), { sensitivity: sens }));
  return state.decoded.get(key);
}

// ---------- generate ----------

const mode = () => document.querySelector('input[name=mode]:checked').value;

async function generate() {
  if (state.busy || !state.source) return;
  state.busy = true;
  updateGenerate();
  hideError();
  try {
    const wantSep = $('separate').checked && !!state.helper?.separation;
    if (state.source.kind === 'link' || wantSep) await ensureJob(wantSep);
    await ensureMix();
    const { main, accomp } = stemsForMode(mode());
    await rawFor(main, main === 'mix' ? 'full mix' : `${main} track`);
    if (accomp) await rawFor(accomp, 'accompaniment, for beats & chords');
    state.inputs = { main, accomp };
    progress(1, 'Placing notes on the fretboard…');
    await new Promise((r) => setTimeout(r, 20));
    rebuild(true);
    $('progress').classList.add('hidden');
  } catch (e) {
    console.error(e);
    $('progress').classList.add('hidden');
    showError(e.message || String(e));
  } finally {
    state.busy = false;
    updateGenerate();
  }
}
$('generateBtn').addEventListener('click', generate);

function rebuild(fresh = false) {
  if (!state.inputs) return;
  const { main, accomp } = state.inputs;
  const keepTime = fresh ? 0 : player.time;
  const result = buildTab({
    mainNotes: notesFor(main),
    accompNotes: accomp ? notesFor(accomp) : null,
    duration: state.mix.buffer.duration,
    mode: mode(),
    tuning: $('tuning').value,
    capo: Number($('capo').value),
    transpose: Number($('transpose').value),
    cleanOvertones: $('overtones').checked,
    bpm: state.bpmOverride,
    sensitivity: Number($('sensitivity').value),
    isolated: main !== 'mix' && main !== 'instruments',
  });
  result.loop = player.loop;
  state.result = result;
  showResult(result, main, accomp);
  const wasPlaying = player.playing;
  if (fresh || player.audioEl?.src !== state.mix.url) player.setOriginal(state.mix.url);
  player.setTrack({ notes: result.notes, beats: result.beats, duration: result.duration });
  if (wasPlaying) player.play(keepTime); else player.seek(keepTime);
  tick(true);
}

function showResult(r, main, accomp) {
  $('result').classList.remove('hidden');
  $('player').classList.remove('hidden');
  document.body.classList.add('has-player');
  const title = state.job?.title || state.source.name;
  $('songTitle').textContent = state.job?.artist && !title.includes(state.job.artist) ? `${title} · ${state.job.artist}` : title;
  const tuning = TUNINGS[$('tuning').value];
  const chips = [
    MODES[r.mode].label,
    `Key: ${r.key.name}`,
    `${r.bpm} BPM`,
    tuning.label.split(' (')[0] + (r.capo ? `, capo ${r.capo}` : ''),
    `${r.notes.length} notes`,
    main === 'mix' ? 'from full mix' : `from ${main} stem`,
  ];
  $('chips').innerHTML = chips.map((c) => `<span class="chip">${escapeHtml(c)}</span>`).join('');
  $('bpmInput').value = r.bpm;
  tabView.render({ notes: r.notes, strings: r.strings, chords: r.chords, beats: r.beats, loop: r.loop });
  const maxFret = Math.max(12, ...r.notes.map((n) => n.fret));
  fretboard.render(r.strings, Math.min(24, maxFret + 2));
  const notes = [];
  if (r.dropped) notes.push(`${r.dropped} note${r.dropped > 1 ? 's' : ''} couldn't fit under one hand and were left out.`);
  if (main === 'mix' && r.mode !== 'lead') notes.push('Tip: run the local helper with stem separation for cleaner results on full songs.');
  if (!r.notes.length) notes.push('No notes found. Try raising the sensitivity or another mode.');
  $('resultNote').textContent = notes.join(' ');
  if (!r.notes.length) return;
  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// options that only need a rebuild
['tuning', 'capo', 'overtones'].forEach((id) => $(id).addEventListener('change', () => rebuild()));
$('transpose').addEventListener('input', () => { $('transposeVal').textContent = (+$('transpose').value > 0 ? '+' : '') + $('transpose').value; });
$('transpose').addEventListener('change', () => rebuild());
$('sensitivity').addEventListener('input', () => { $('sensVal').textContent = Math.round($('sensitivity').value * 100) + '%'; });
$('sensitivity').addEventListener('change', () => rebuild());
// options that may need another transcription
document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', () => { if (state.inputs) generate(); }));
$('stem').addEventListener('change', () => { if (state.inputs) generate(); });
$('separate').addEventListener('change', () => { if (state.inputs && $('separate').checked) generate(); });

$('bpmInput').addEventListener('change', () => { state.bpmOverride = Number($('bpmInput').value) || 0; rebuild(); });
$('halfBtn').addEventListener('click', () => { state.bpmOverride = Math.round(state.result.bpm * 5) / 10; rebuild(); });
$('doubleBtn').addEventListener('click', () => { state.bpmOverride = Math.round(state.result.bpm * 20) / 10; rebuild(); });
$('autoBpmBtn').addEventListener('click', () => { state.bpmOverride = 0; rebuild(); });

// ---------- progress / errors ----------

function progress(p, text) {
  $('progress').classList.remove('hidden');
  $('progressText').textContent = text;
  const fill = $('progressFill');
  fill.classList.toggle('indeterminate', p == null);
  if (p != null) fill.style.width = `${Math.round(p * 100)}%`;
}
function showError(msg) { $('errorBox').textContent = msg; $('errorBox').classList.remove('hidden'); }
function hideError() { $('errorBox').classList.add('hidden'); }

// ---------- player ----------

$('playBtn').addEventListener('click', togglePlay);
function togglePlay() {
  if (!state.result) return;
  if (player.playing) player.pause(); else player.play();
  syncPlayButton();
}
function syncPlayButton() {
  $('playBtn').textContent = player.playing ? '❚❚' : '▶';
  $('playBtn').setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
}
player.onEnd = syncPlayButton;
$('stopBtn').addEventListener('click', () => { player.seek(0); tick(true); });
$('speed').addEventListener('change', () => player.setRate(Number($('speed').value)));
$('mix').addEventListener('input', () => player.setMix(Number($('mix').value)));
player.mix = 0.5;
$('tone').addEventListener('change', () => player.setTone($('tone').value));
const toggle = (btn, on) => { btn.classList.toggle('on', on); btn.setAttribute('aria-pressed', on); };
$('metroBtn').addEventListener('click', () => { player.metronome = !player.metronome; toggle($('metroBtn'), player.metronome); });
let follow = true;
$('followBtn').addEventListener('click', () => { follow = !follow; toggle($('followBtn'), follow); });

let loopA = null;
$('loopA').addEventListener('click', () => {
  loopA = player.time;
  player.loop = null;
  toggle($('loopA'), true); toggle($('loopB'), false);
  tabView.setLoop(null);
});
$('loopB').addEventListener('click', () => {
  if (loopA == null || player.time <= loopA + 0.2) return;
  player.loop = { a: loopA, b: player.time };
  toggle($('loopB'), true);
  tabView.setLoop(player.loop);
  player.seek(loopA);
});
$('loopClear').addEventListener('click', () => {
  loopA = null; player.loop = null;
  toggle($('loopA'), false); toggle($('loopB'), false);
  tabView.setLoop(null);
});

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && state.result && !['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) {
    e.preventDefault();
    togglePlay();
  }
});

let lastShown = '';
function tick(force = false) {
  if (state.result && (player.playing || force)) {
    const t = player.time;
    const active = tabView.setCursor(t, follow && player.playing) || [];
    const sig = active.map((n) => n.string + ':' + n.fret).join(',');
    if (sig !== lastShown || force) { fretboard.show(active); lastShown = sig; }
    $('timeLabel').textContent = `${fmt(t)} / ${fmt(state.result.duration)}`;
  }
  syncPlayButton();
}
(function loop() { tick(); requestAnimationFrame(loop); })();

// ---------- export ----------

function asciiTab() {
  const r = state.result;
  const name = $('songTitle').textContent;
  const tuning = TUNINGS[$('tuning').value].label;
  const head = `${name}\n${MODES[r.mode].label} · ${r.key.name} · ${r.bpm} BPM · ${tuning}${r.capo ? ` · capo ${r.capo}` : ''}\nGenerated with TabForge`;
  return toAscii(r.notes, r.strings, { chords: r.chords, title: head });
}
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const baseName = () => $('songTitle').textContent.replace(/[^\w\- ]+/g, '').trim().slice(0, 60) || 'tab';
$('copyAscii').addEventListener('click', async () => {
  await navigator.clipboard.writeText(asciiTab());
  $('copyAscii').textContent = 'Copied ✓';
  setTimeout(() => { $('copyAscii').textContent = 'Copy tab as text'; }, 1500);
});
$('dlAscii').addEventListener('click', () => download(new Blob([asciiTab()], { type: 'text/plain' }), baseName() + '.txt'));
$('dlMidi').addEventListener('click', () => {
  const r = state.result;
  const program = { clean: 27, drive: 29, acoustic: 25 }[$('tone').value];
  // write the quantized grid at a steady tempo so bars line up in notation software
  const beat = 60 / r.bpm;
  const grid = r.notes.map((n) => ({ midi: n.midi, vel: n.vel, start: (n.step / 4) * beat, dur: (n.durSteps / 4) * beat }));
  download(new Blob([toMidi(grid, { bpm: r.bpm, program })], { type: 'audio/midi' }), baseName() + '.mid');
});
