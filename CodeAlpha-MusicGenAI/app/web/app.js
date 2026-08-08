/* ────────────────────────────────────────────────────────────────────────
   MusicGenAI Dashboard — app.js
   ────────────────────────────────────────────────────────────────────── */

"use strict";

/* ── State ───────────────────────────────────────────────────────────── */
const state = {
  mode: "mixed",
  selectedJobId: null,
  jobs: [],
  playerFiles: [],
  playerIndex: 0,
  playerIsPlaying: false,
  toneSynth: null,
  toneScheduledIds: [],
};

/* ── DOM refs ─────────────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);

const statusPanel     = $("statusPanel");
const jobsPanel       = $("jobsPanel");
const logsPanel       = $("logsPanel");
const activeJobLabel  = $("activeJobLabel");
const runHint         = $("runHint");
const jobCount        = $("jobCount");
const serverStatus    = $("serverStatus");
const serverLabel     = $("serverLabel");
const playerFiles     = $("playerFiles");
const playerPlay      = $("playerPlay");
const playerPrev      = $("playerPrev");
const playerNext      = $("playerNext");
const nowPlayingName  = $("nowPlayingName");
const waveform        = $("waveform");
const playerMeta      = $("playerMeta");

/* ── Accent picker ────────────────────────────────────────────────────── */
const ACCENTS = {
  orange: {
    accent: "var(--accent-orange)",
    hover: "var(--accent-orange-hover)",
  },
  cyan: {
    accent: "var(--accent-cyan)",
    hover: "var(--accent-cyan-hover)",
  },
};

function applyAccent(name) {
  const tone = ACCENTS[name] || ACCENTS.orange;
  const root = document.documentElement.style;
  root.setProperty("--accent", tone.accent);
  root.setProperty("--accent-hover", tone.hover);
}

const swatchRow = $("swatchRow");
if (swatchRow) {
  swatchRow.addEventListener("click", (e) => {
    const btn = e.target.closest(".swatch");
    if (!btn) return;
    swatchRow.querySelectorAll(".swatch").forEach((s) => s.classList.remove("swatch--active"));
    btn.classList.add("swatch--active");
    applyAccent(btn.dataset.accent);
  });
}

/* ── Mode pills ────────────────────────────────────────────────────────── */
document.getElementById("modePills").addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-pill");
  if (!btn) return;
  document.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("mode-pill--active"));
  btn.classList.add("mode-pill--active");
  state.mode = btn.dataset.mode;
  refreshStatus();
});

/* ── Utilities ─────────────────────────────────────────────────────────── */
function parseNum(value, fn) {
  if (value === "" || value == null) return null;
  const n = fn(value);
  return Number.isNaN(n) ? null : n;
}

function sanitizeOutputFilename(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const base = trimmed.split(/[\\/]+/).filter(Boolean).pop() || "";
  return base || null;
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

function setHint(text, type = "") {
  runHint.textContent = text;
  runHint.className = "run-hint" + (type ? " " + type : "");
}

/* ── Grid canvas (ambient) ─────────────────────────────────────────────── */
function drawGrid() {
  const canvas = $("gridCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  const step = 40;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#1e2a38";
  ctx.lineWidth = .5;
  for (let x = 0; x <= W; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y <= H; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
}
window.addEventListener("resize", drawGrid);

/* ── Server health ─────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    await fetchJson("/api/health");
    serverStatus.className = "pulse-dot ok";
    serverLabel.textContent = "Server online";
  } catch {
    serverStatus.className = "pulse-dot err";
    serverLabel.textContent = "Server offline";
  }
}

/* ── Status panel ──────────────────────────────────────────────────────── */
async function refreshStatus() {
  const btn = $("refreshStatusBtn");
  btn.classList.add("spinning");
  try {
    const st = await fetchJson(`/api/status?mode=${encodeURIComponent(state.mode)}`);
    statusPanel.innerHTML = `
      <div class="status-row"><dt>MIDI files</dt><dd>${st.midi_count}</dd></div>
      <div class="status-row"><dt>Preprocessed</dt><dd class="${st.preprocessed ? "ok" : "err"}">${st.preprocessed ? "Yes" : "No"}</dd></div>
      <div class="status-row"><dt>Trained</dt><dd class="${st.trained ? "ok" : "err"}">${st.trained ? "Yes" : "No"}</dd></div>
      <div class="status-row"><dt>Checkpoint</dt><dd class="mono" title="${st.checkpoint}">${basename(st.checkpoint)}</dd></div>
    `;
  } catch (err) {
    statusPanel.innerHTML = `<div class="status-row"><dt>Error</dt><dd class="err">${err.message}</dd></div>`;
  } finally {
    btn.classList.remove("spinning");
  }
}

function basename(p) {
  if (!p) return "—";
  return p.replace(/\\/g, "/").split("/").pop() || p;
}

/* ── Jobs ──────────────────────────────────────────────────────────────── */
function renderJobs() {
  jobCount.textContent = state.jobs.length;
  if (state.jobs.length === 0) {
    jobsPanel.innerHTML = "<p class='empty-hint'>No jobs yet.</p>";
    return;
  }
  jobsPanel.innerHTML = state.jobs.map(job => {
    const active = job.id === state.selectedJobId ? "active" : "";
    const time = job.created_at.replace("T", " ").replace("Z", "");
    return `
      <div class="job-item ${active}" data-id="${job.id}">
        <div class="job-item__info">
          <div class="job-item__action">${job.action}</div>
          <div class="job-item__meta">${job.mode} · ${time}</div>
        </div>
        <span class="badge ${job.status}">${job.status}</span>
      </div>
    `;
  }).join("");

  jobsPanel.querySelectorAll(".job-item").forEach(el => {
    el.addEventListener("click", () => {
      state.selectedJobId = el.dataset.id;
      renderJobs();
      refreshLogs();
    });
  });
}

async function refreshJobs() {
  try {
    state.jobs = await fetchJson("/api/jobs?limit=30");
    if (!state.selectedJobId && state.jobs.length > 0) {
      state.selectedJobId = state.jobs[0].id;
    }
    renderJobs();
  } catch (err) {
    setHint(`Failed to fetch jobs: ${err.message}`, "error");
  }
}

async function refreshLogs() {
  if (!state.selectedJobId) {
    activeJobLabel.textContent = "—";
    logsPanel.textContent = "Select a job to view logs.";
    return;
  }
  try {
    const [job, logs] = await Promise.all([
      fetchJson(`/api/jobs/${state.selectedJobId}`),
      fetchJson(`/api/jobs/${state.selectedJobId}/logs?tail=500`),
    ]);
    activeJobLabel.textContent = `${job.id} · ${job.action} · ${job.status}`;
    logsPanel.textContent = logs.lines.join("");
    logsPanel.scrollTop = logsPanel.scrollHeight;
  } catch (err) {
    logsPanel.textContent = `Error loading logs: ${err.message}`;
  }
}

/* ── Collect options ────────────────────────────────────────────────────── */
function collectTrainOptions() {
  return {
    resume:     $("trainResume").checked,
    epochs:     parseNum($("trainEpochs").value, Number.parseInt),
    batch_size: parseNum($("trainBatch").value,  Number.parseInt),
    lr:         parseNum($("trainLr").value,      Number.parseFloat),
  };
}

function collectGenerateOptions() {
  return {
    num_tokens:   parseNum($("genTokens").value,  Number.parseInt),
    temperature:  parseNum($("genTemp").value,    Number.parseFloat),
    top_k:        parseNum($("genTopK").value,    Number.parseInt),
    top_p:        parseNum($("genTopP").value,    Number.parseFloat),
    bpm:          parseNum($("genBpm").value,     Number.parseInt),
    seed_length:  parseNum($("genSeedLen").value, Number.parseInt),
    seed_file:    $("genSeedFile").value.trim() || null,
    output_filename: sanitizeOutputFilename($("genOutputFilename").value),
  };
}

const outputFilenameInput = $("genOutputFilename");
if (outputFilenameInput) {
  outputFilenameInput.addEventListener("input", () => {
    const sanitized = sanitizeOutputFilename(outputFilenameInput.value) || "";
    if (sanitized !== outputFilenameInput.value) {
      outputFilenameInput.value = sanitized;
    }
  });
}

/* ── Create Job ────────────────────────────────────────────────────────── */
async function createJob(action) {
  setHint(`Starting ${action}…`, "running");
  const body = {
    mode:     state.mode,
    action,
    train:    collectTrainOptions(),
    generate: collectGenerateOptions(),
  };
  try {
    const created = await fetchJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setHint(`Job started: ${created.id} (${created.action})`, "running");
    state.selectedJobId = created.id;
    await refreshJobs();
    await refreshLogs();
  } catch (err) {
    setHint(`Could not start job: ${err.message}`, "error");
  }
}

document.querySelectorAll("button[data-action]").forEach(btn => {
  btn.addEventListener("click", () => createJob(btn.dataset.action));
});

$("refreshStatusBtn").addEventListener("click", refreshStatus);

/* ── MIDI Player ────────────────────────────────────────────────────────── */
function outputFileUrl(file) {
  return `/api/outputs/${encodeURIComponent(basename(file))}`;
}

function setDownloadFallback(file, message) {
  if (!file) {
    playerMeta.textContent = message || "No file selected.";
    return;
  }
  playerMeta.innerHTML = `${message || "Download fallback:"} <a href="${outputFileUrl(file)}" download style="color:var(--accent);text-decoration:none;">⬇ ${basename(file)}</a>`;
}

function clearToneSchedule() {
  if (!window.Tone) return;
  state.toneScheduledIds.forEach((id) => window.Tone.Transport.clear(id));
  state.toneScheduledIds = [];
  window.Tone.Transport.cancel();
}

async function ensureToneReady() {
  if (!window.Tone || !window.Midi) {
    throw new Error("Tone.js or @tonejs/midi not loaded");
  }
  await window.Tone.start();
  if (!state.toneSynth) {
    state.toneSynth = new window.Tone.PolySynth(window.Tone.Synth, {
      oscillator: { type: "triangle" },
      envelope: { attack: 0.01, decay: 0.2, sustain: 0.25, release: 0.6 },
    }).toDestination();
  }
}

async function playSelectedMidi() {
  const file = state.playerFiles[state.playerIndex];
  if (!file) return;

  await ensureToneReady();
  clearToneSchedule();
  window.Tone.Transport.stop();
  window.Tone.Transport.position = 0;

  const res = await fetch(outputFileUrl(file));
  if (!res.ok) {
    throw new Error(`Could not load MIDI (${res.status})`);
  }

  const arrayBuffer = await res.arrayBuffer();
  const midi = new window.Midi(arrayBuffer);
  const firstTempo = midi.header?.tempos?.[0]?.bpm;
  if (firstTempo && Number.isFinite(firstTempo)) {
    window.Tone.Transport.bpm.value = firstTempo;
  }

  let hasNotes = false;
  let endTime = 0;

  midi.tracks.forEach((track) => {
    track.notes.forEach((note) => {
      hasNotes = true;
      const start = Math.max(note.time, 0);
      const duration = Math.max(note.duration, 0.03);
      const velocity = Number.isFinite(note.velocity) ? note.velocity : 0.8;

      const id = window.Tone.Transport.schedule((time) => {
        state.toneSynth.triggerAttackRelease(note.name, duration, time, velocity);
      }, start);
      state.toneScheduledIds.push(id);
      endTime = Math.max(endTime, start + duration);
    });
  });

  if (!hasNotes) {
    throw new Error("MIDI has no playable notes");
  }

  state.toneScheduledIds.push(
    window.Tone.Transport.schedule(() => {
      stopPlayer();
    }, endTime + 0.05),
  );

  window.Tone.Transport.start("+0.02");
  startPlayer();
  setDownloadFallback(file, "Download fallback:");
}

async function refreshPlayerFiles() {
  try {
    const data = await fetchJson("/api/outputs");
    state.playerFiles = data.files || [];
    if (state.playerIndex >= state.playerFiles.length) {
      state.playerIndex = Math.max(0, state.playerFiles.length - 1);
    }
    renderPlayerFiles();
  } catch {
    playerMeta.textContent = "Could not load output MIDI files.";
  }
}

function renderPlayerFiles() {
  if (state.playerFiles.length === 0) {
    playerFiles.innerHTML = "<p class='empty-hint'>No generated MIDI files found.</p>";
    return;
  }
  playerFiles.innerHTML = state.playerFiles.map((f, i) => `
    <button class="player-file-chip ${i === state.playerIndex ? "active" : ""}" data-index="${i}" title="${f}">
      ♩ ${basename(f)}
    </button>
  `).join("");

  playerFiles.querySelectorAll(".player-file-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      state.playerIndex = parseInt(chip.dataset.index);
      state.playerIsPlaying = false;
      stopPlayer();
      renderPlayerFiles();
      updateNowPlaying();
    });
  });

  updateNowPlaying();
}

function updateNowPlaying() {
  const file = state.playerFiles[state.playerIndex];
  nowPlayingName.textContent = file ? basename(file) : "No file selected";

  if (file) {
    setDownloadFallback(file, "Download fallback:");
  } else {
    playerMeta.textContent = "No file selected.";
  }
}

function stopPlayer() {
  state.playerIsPlaying = false;
  if (window.Tone) {
    window.Tone.Transport.stop();
    window.Tone.Transport.position = 0;
  }
  playerPlay.innerHTML = "&#9654;";
  waveform.classList.remove("playing");
}

function startPlayer() {
  state.playerIsPlaying = true;
  playerPlay.innerHTML = "&#9646;&#9646;";
  waveform.classList.add("playing");
}

playerPlay.addEventListener("click", async () => {
  if (state.playerFiles.length === 0) return;
  if (state.playerIsPlaying) {
    stopPlayer();
  } else {
    try {
      await playSelectedMidi();
    } catch (err) {
      const file = state.playerFiles[state.playerIndex];
      const message = err?.message || "Playback unavailable";
      stopPlayer();
      setDownloadFallback(file, `${message}. Download fallback:`);
    }
  }
});

playerPrev.addEventListener("click", async () => {
  if (state.playerFiles.length === 0) return;
  const wasPlaying = state.playerIsPlaying;
  stopPlayer();
  state.playerIndex = (state.playerIndex - 1 + state.playerFiles.length) % state.playerFiles.length;
  renderPlayerFiles();
  if (wasPlaying) {
    try {
      await playSelectedMidi();
    } catch (err) {
      const file = state.playerFiles[state.playerIndex];
      setDownloadFallback(file, `${err?.message || "Playback unavailable"}. Download fallback:`);
    }
  }
});

playerNext.addEventListener("click", async () => {
  if (state.playerFiles.length === 0) return;
  const wasPlaying = state.playerIsPlaying;
  stopPlayer();
  state.playerIndex = (state.playerIndex + 1) % state.playerFiles.length;
  renderPlayerFiles();
  if (wasPlaying) {
    try {
      await playSelectedMidi();
    } catch (err) {
      const file = state.playerFiles[state.playerIndex];
      setDownloadFallback(file, `${err?.message || "Playback unavailable"}. Download fallback:`);
    }
  }
});

/* ── Bootstrap ──────────────────────────────────────────────────────────── */
async function bootstrap() {
  drawGrid();
  applyAccent("orange");

  await checkHealth();
  await refreshStatus();
  await refreshJobs();
  await refreshLogs();
  await refreshPlayerFiles();

  setInterval(checkHealth,    10_000);
  setInterval(refreshStatus,   8_000);
  setInterval(refreshJobs,     3_500);
  setInterval(refreshLogs,     2_000);
  setInterval(refreshPlayerFiles, 15_000);
}

bootstrap();