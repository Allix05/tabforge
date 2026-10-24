// Client for the optional local helper (helper/tabforge_helper.py). Browsers
// can't download from YouTube/Spotify themselves, so links and stem separation
// go through a small free server the user runs on their own machine.

const LOCAL = 'http://127.0.0.1:8765';
const BASE = location.port === '8765' ? '' : LOCAL;

export async function helperStatus() {
  try {
    const r = await fetch(BASE + '/api/health', { signal: AbortSignal.timeout(1500) });
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

async function created(r) {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `Helper error ${r.status}`);
  return body;
}

export async function jobFromUrl(url, separate) {
  return created(await fetch(`${BASE}/api/jobs?separate=${separate ? 1 : 0}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }),
  }));
}

export async function jobFromFile(file, separate) {
  return created(await fetch(`${BASE}/api/jobs?separate=${separate ? 1 : 0}&name=${encodeURIComponent(file.name)}`, {
    method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file,
  }));
}

export async function waitForJob(id, onUpdate) {
  for (;;) {
    const job = await (await fetch(`${BASE}/api/jobs/${id}`)).json();
    onUpdate(job);
    if (job.status === 'done') return job;
    if (job.status === 'error') throw new Error(job.error || 'Helper job failed');
    await new Promise((r) => setTimeout(r, 700));
  }
}

export async function fetchStem(id, name) {
  const r = await fetch(`${BASE}/api/jobs/${id}/audio/${name}`);
  if (!r.ok) throw new Error(`Could not load the ${name} track`);
  return r.blob();
}
