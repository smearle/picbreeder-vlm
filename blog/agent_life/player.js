/* Life-of-an-agent player. Reads ./manifest.json (override with ?manifest=URL).

   Layout:
     LEFT  - the candidate grid (top) and the CPPN morph (centered below, half size)
     RIGHT - "VLM output": every step's reasoning as a transcript line; karaoke-style
             highlighting runs *inside* the line for whichever audio is playing in it.

   Playback model:
     - Default speed = 1.5x.
     - Each step is capped at STEP_CAP_MS (5s wall-clock) of dwell. When the cap
       fires, the new step starts and the outgoing audio becomes a half-volume
       "trailer" that keeps playing in its own transcript line.
     - If a second audio starts while a trailer is still playing, every existing
       trailer halves *again* (volume + highlight alpha) -- so an older trailer
       fades 1.0 -> 0.5 -> 0.25 -> ... while it lingers behind newer narrations.
*/

const qs = new URLSearchParams(location.search);
const MANIFEST_URL = qs.get("manifest") || "manifest.json";
const DEFAULT_SPEED = 1.5;
const TRAILER_LEVEL = 0.5;
const DIM_FACTOR    = 0.5;  // every new narration multiplies all trailers' level by this
// (No per-step cap: each generation's rationale plays in full before advancing.
// Manual Next / scrub still jumps immediately.)

const el = (id) => document.getElementById(id);
const els = {};
["title", "hero", "morph", "graph", "graph_static", "grid", "gridbox", "emptygrid", "gridcap",
 "gridlabel", "morphpanel", "transcript",
 "playpause", "prev", "next", "speed", "mute", "scrub", "progress",
 "audiohint"].forEach((k) => (els[k] = el(k)));

const S = {
  M: null, steps: [], N: 0,
  idx: 0,
  playing: false,
  speed: DEFAULT_SPEED,
  muted: false,
  primary: null,             // active track {audio, stepIdx, spans, wordIdx, level, tline}
  trailers: [],              // older tracks still playing (each fades by DIM_FACTOR each new narration)
};

init();

async function init() {
  // Always fetch a fresh manifest -- assets are cheap; a stale manifest hides
  // newly-added paths (e.g., new branching imagery) until the user wipes cache.
  const M = await (await fetch(MANIFEST_URL, { cache: "no-store" })).json();
  S.M = M;
  S.steps = buildSteps(M);
  S.N = S.steps.length;

  els.title.innerHTML = `Life of an Agent&mdash;<span class="ttl-accent">${escapeHtml(M.title || "")}</span>`;
  els.scrub.max = String(S.N - 1);
  els.speed.value = String(DEFAULT_SPEED);
  lockGridAspect();
  buildTranscript();
  showStep(0, false);

  // Autoplay on load. If the browser blocks audio (no user gesture yet),
  // visuals keep animating and we surface a small "click to enable narration"
  // hint that, when clicked, retroactively starts the current step's audio.
  play();
  els.audiohint.addEventListener("click", enableAudio);
  els.playpause.addEventListener("click", () => (S.playing ? pause() : play()));
  els.prev.addEventListener("click", () => jump(S.idx - 1));
  els.next.addEventListener("click", () => jump(S.idx + 1));
  els.scrub.addEventListener("input", () => jump(parseInt(els.scrub.value, 10)));
  els.speed.addEventListener("change", () => setSpeed(parseFloat(els.speed.value)));
  els.mute.addEventListener("click", toggleMute);
  els.morph.addEventListener("ended", () => onMorphEnd());
  document.addEventListener("keydown", onKey);
}

function buildSteps(M) {
  const steps = [];
  const wrap = (s, narrs) => ({ ...s, narrations: narrs || [] });
  if (M.branching) {
    const b = M.branching;
    const narrs = b.narrations || (b.audio ? [{
      voice: "announcer", audio: b.audio, audio_dur: b.audio_dur,
      chunks: b.chunks || [], text: b.rationale || ""
    }] : []);
    steps.push(wrap({
      kind: "branching",
      label: "Branching",
      rationale: b.rationale || "",
      grid: b.grid || null,
      empty: !!b.archive_empty,
      empty_text: b.empty_text || "",
      morph_out: null,
      hero: b.hero || null,
      graph_image: b.graph_image || null,
    }, narrs));
  }
  M.generations.forEach((g) => {
    const narrs = g.narrations || (g.audio ? [{
      voice: "agent", audio: g.audio, audio_dur: g.audio_dur,
      chunks: g.chunks || [], text: g.rationale || ""
    }] : []);
    steps.push(wrap({
      kind: "gen",
      label: `Gen ${g.generation}`,
      rationale: g.rationale || "",
      grid: g.grid,
      hero: g.hero,
      graph_image: g.graph_image || null,
      selected: g.selected,
      published: g.published,
      morph_out: g.morph_out,
      graph_out: g.graph_out || null,
      genNum: g.generation,
    }, narrs));
  });
  return steps;
}

/* ---------------- transcript (now also carries the speakable text) ---------------- */
function buildTranscript() {
  els.transcript.innerHTML = "";
  S.steps.forEach((st, i) => {
    const line = document.createElement("div");
    line.className = "tline" + (st.kind === "branching" ? " branching" : "");
    line.dataset.i = i;

    const num = document.createElement("div");
    num.className = "tnum";
    num.textContent = st.label;

    const body = document.createElement("div");
    body.className = "tbody";
    renderNarrationsInto(body, st.narrations, st.rationale);

    // small meta line under the prose (picks / published / archive-empty)
    const meta = document.createElement("div");
    meta.className = "picks";
    if (st.kind === "branching") {
      meta.innerHTML = st.empty ? "archive empty &middot; fresh start" : "archive sample";
    } else {
      const picks = st.selected.length ? `picks [${st.selected.join(", ")}]` : "no pick";
      const pub = st.published
        ? ` &middot; <span class="pub">published “${escapeHtml(st.published.title)}”</span>`
        : "";
      meta.innerHTML = picks + pub;
    }
    body.appendChild(meta);

    line.appendChild(num);
    line.appendChild(body);
    line.addEventListener("click", () => jump(i));
    els.transcript.appendChild(line);
  });
}

function renderNarrationsInto(target, narrations, fallbackText) {
  if (!narrations || !narrations.length) {
    target.textContent = fallbackText || "";
    return;
  }
  const frag = document.createDocumentFragment();
  narrations.forEach((narr, narrIdx) => {
    const wrap = document.createElement("span");
    wrap.className = "narr narr-" + (narr.voice || "agent");
    wrap.dataset.n = String(narrIdx);
    const chunks = narr.chunks || [];
    if (!chunks.length && narr.text) {
      wrap.textContent = narr.text;
    } else {
      for (const c of chunks) {
        if (c.t != null) {
          const span = document.createElement("span");
          span.className = "w";
          span.dataset.t = c.t;
          span.dataset.d = c.dur;
          span.dataset.n = String(narrIdx);
          span.textContent = c.text;
          wrap.appendChild(span);
        } else {
          wrap.appendChild(document.createTextNode(c.text));
        }
      }
    }
    frag.appendChild(wrap);
  });
  target.appendChild(frag);
}

/* ---------------- per-audio "track" (highlights words in its own tline) ---------------- */
function makeTrack(stepIdx, narrIdx, level) {
  const st = S.steps[stepIdx];
  const narr = st.narrations[narrIdx];
  const audio = new Audio(narr.audio);
  audio.muted = S.muted;
  audio.playbackRate = S.speed;
  audio.volume = level;
  const tline = els.transcript.children[stepIdx];
  const spans = tline ? tline.querySelectorAll(`.w[data-n="${narrIdx}"]`) : [];
  const track = { audio, stepIdx, narrIdx, spans, wordIdx: -1, level, tline };
  setLevel(track, level);
  audio.addEventListener("timeupdate", () => updateTrackHighlight(track));
  return track;
}

function setLevel(track, level) {
  track.level = level;
  track.audio.volume = level;
  if (track.tline) track.tline.style.setProperty("--alpha", String(level));
}

function updateTrackHighlight(track) {
  const spans = track.spans;
  if (!spans || !spans.length) return;
  const t = track.audio.currentTime;
  let i = Math.max(0, track.wordIdx);
  while (i > 0 && parseFloat(spans[i].dataset.t) > t) i--;
  while (i + 1 < spans.length && parseFloat(spans[i + 1].dataset.t) <= t) i++;
  if (parseFloat(spans[0].dataset.t) > t) i = -1;
  if (i !== track.wordIdx) {
    if (track.wordIdx >= 0 && spans[track.wordIdx]) spans[track.wordIdx].classList.remove("active");
    if (i >= 0 && spans[i]) spans[i].classList.add("active");
    track.wordIdx = i;
  }
}

function clearTrackHighlight(track) {
  if (track.spans && track.wordIdx >= 0 && track.spans[track.wordIdx]) {
    track.spans[track.wordIdx].classList.remove("active");
  }
  track.wordIdx = -1;
  if (track.tline) track.tline.style.removeProperty("--alpha");
}

/* ---------------- core navigation ---------------- */
function showStep(i, autoplay) {
  S.idx = Math.max(0, Math.min(S.N - 1, i));
  const st = S.steps[S.idx];

  els.gridlabel.textContent =
    st.kind === "branching"
      ? "Branching step—the archive shown to the agent"
      : "The candidate grid—its picks outlined in red";

  if (st.grid) {
    els.grid.src = st.grid;
    els.grid.style.display = "";
    els.emptygrid.style.display = "none";
  } else if (st.kind === "branching" && st.empty) {
    els.grid.style.display = "none";
    els.emptygrid.style.display = "";
    els.emptygrid.innerHTML = st.empty_text
      ? `<div><b>Archive empty.</b><br>${escapeHtml(st.empty_text)}</div>`
      : `<div><b>Archive empty.</b><br>This agent is the first user—it starts from a fresh random population.</div>`;
  } else {
    els.grid.style.display = "none";
    els.emptygrid.style.display = "";
    els.emptygrid.textContent = "—";
  }

  // Stop any morph carried over from the previous step. If this step has a
  // morph_out, narrate() will (re-)start it slowed to span the step interval.
  hideMorph();
  // The "static" layer of each stagebox: hero img on the right; graph_static
  // img on the left. During morph segments these are hidden under the video.
  if (st.kind === "branching") {
    els.morphpanel.classList.remove("dim");
    if (st.hero)   els.hero.src = st.hero;          else els.hero.removeAttribute("src");
    if (st.graph_image) els.graph_static.src = st.graph_image;
    else                els.graph_static.removeAttribute("src");
  } else {
    els.morphpanel.classList.remove("dim");
    els.hero.src = st.hero;
    if (st.graph_image) els.graph_static.src = st.graph_image;
  }

  if (st.kind === "branching") {
    els.gridcap.innerHTML = st.empty
      ? `<b>Branching step</b> · archive empty, defaulting to a fresh population`
      : `<b>Branching step</b> · the archive sample shown to the agent`;
  } else {
    const pub = st.published
      ? ` <span class="pub">— published “${escapeHtml(st.published.title)}”</span>`
      : "";
    els.gridcap.innerHTML = `<b>Generation ${st.genNum} / ${maxGenNum()}</b> · selected ${
      st.selected.length ? "[" + st.selected.join(", ") + "]" : "nothing"
    }${pub}`;
  }

  [...els.transcript.children].forEach((c, j) => {
    c.classList.toggle("active", j === S.idx);
    c.classList.toggle("done", j < S.idx);
  });
  const active = els.transcript.children[S.idx];
  if (active) scrollActiveIntoView(active);

  els.scrub.value = String(S.idx);
  els.progress.textContent = stepProgressLabel();

  if (autoplay && S.playing) narrate();
}

function narrate() {
  // Step transition: cascade dim trailers, demote outgoing primary to a 0.5 trailer.
  for (const tr of S.trailers) setLevel(tr, tr.level * DIM_FACTOR);
  if (S.primary) {
    const old = S.primary;
    S.primary = null;
    if (old.audio.ended) {
      clearTrackHighlight(old);
    } else {
      setLevel(old, TRAILER_LEVEL);
      S.trailers.push(old);
      old.audio.addEventListener("ended", () => {
        S.trailers = S.trailers.filter(t => t !== old);
        clearTrackHighlight(old);
      }, { once: true });
    }
  }
  // Start narration 0 of this step. Chained narrations follow naturally on
  // each clip's "ended" event; when the last narration in the step ends, we
  // advance to the next step.
  startNarration(S.idx, 0);
  // Play the step's CPPN morph (if any) concurrently with narration, slowed
  // to span the *full narration runtime* so the image is in motion for the
  // whole step rather than flashing as a brief transition. Hold steps (no
  // morph_out) keep their static hero.
  startStepMorph(S.idx);
}

function stepWallTarget(stepIdx) {
  /* Predicted wall-clock time spent on this step = total narration audio
     length / current playback speed. No cap: each rationale plays in full. */
  const st = S.steps[stepIdx];
  let total = 0;
  for (const n of st.narrations || []) total += n.audio_dur || 0;
  return (total || 1) / S.speed;
}

function startStepMorph(stepIdx) {
  const st = S.steps[stepIdx];
  const m = st.morph_out;
  if (!m || !m.dur) {
    // Hold step: no lineage advance this generation. Hero stays static.
    if (st.graph_out) startGraphSegment(st.graph_out, 1.0); else hideGraph();
    return;
  }
  const target = stepWallTarget(stepIdx);
  // playbackRate is clamped so we don't crawl below 0.0625x (browser floor) or
  // blast past 4x on tiny step targets.
  const rate = Math.max(0.0625, Math.min(4.0, m.dur / target));
  els.hero.classList.add("hide");
  els.morph.src = m.path;
  els.morph.classList.add("show");
  els.morph.muted = true;
  els.morph.playbackRate = rate;
  els.morph.currentTime = 0;
  els.morph.play().catch(() => {});

  // Graph plays in lockstep at the same rate (same frame count / fps).
  if (st.graph_out) startGraphSegment(st.graph_out, rate);
  else hideGraph();
}

function startGraphSegment(g, rate) {
  els.graph_static.classList.add("hide");
  els.graph.src = g.path;
  els.graph.classList.add("show");
  els.graph.muted = true;
  els.graph.playbackRate = rate;
  els.graph.currentTime = 0;
  els.graph.play().catch(() => {});
}

function hideGraph() {
  els.graph.pause();
  els.graph.classList.remove("show");
  els.graph_static.classList.remove("hide");
}

function startNarration(stepIdx, narrIdx) {
  const st = S.steps[stepIdx];
  const narr = st.narrations && st.narrations[narrIdx];
  if (!narr || !narr.audio) {
    // skip past empty narrations within this step
    if (narr && st.narrations && narrIdx + 1 < st.narrations.length) {
      startNarration(stepIdx, narrIdx + 1);
    }
    return;
  }
  const track = makeTrack(stepIdx, narrIdx, 1.0);
  track.audio.addEventListener("ended", () => {
    if (S.primary !== track) return;
    // chain to the next narration in the same step
    if (narrIdx + 1 < st.narrations.length) {
      clearTrackHighlight(track);
      S.primary = null;
      startNarration(stepIdx, narrIdx + 1);
      return;
    }
    // all narrations done in this step
    if (!S.playing) return;
    if (stepIdx >= S.N - 1) { S.playing = false; els.playpause.textContent = "↺"; return; }
    advance();
  });
  tryAudioPlay(track.audio);
  S.primary = track;
}

/* Wraps audio.play() so we can surface the "click to enable narration" hint
   when the browser blocks autoplay-with-sound. */
function tryAudioPlay(a) {
  const p = a.play();
  if (p && typeof p.catch === "function") {
    p.catch(() => {
      if (a.error || a.muted) return;
      els.audiohint.hidden = false;
    });
  }
}

function enableAudio() {
  els.audiohint.hidden = true;
  // Retroactively resume / start the current step's primary. The user click
  // is a fresh gesture, so this should succeed.
  if (S.primary) {
    S.primary.audio.muted = S.muted;
    S.primary.audio.play().catch(() => {});
  }
  S.trailers.forEach(t => t.audio.play().catch(() => {}));
  if (!S.playing) { S.playing = true; els.playpause.textContent = "❚❚"; }
}

function advance() {
  if (S.idx >= S.N - 1) {
    S.playing = false;
    els.playpause.textContent = "↺";
    return;
  }
  S.idx += 1;
  showStep(S.idx, true);
}

function onMorphEnd() {
  // Morph reached its end frame; the last frame is approximately the new hero,
  // so just hold it (don't flash back to the still). If this is a hold step
  // there's no morph showing in the first place; this handler is a no-op then.
  // We keep the video visible at its end frame.
}

/* ---------------- transport ---------------- */
function play() {
  S.playing = true;
  els.playpause.textContent = "❚❚";
  if (S.primary) S.primary.audio.play().catch(() => {});
  S.trailers.forEach(t => t.audio.play().catch(() => {}));
  if (els.morph.classList.contains("show")) els.morph.play().catch(() => {});
  if (els.graph.classList.contains("show")) els.graph.play().catch(() => {});
  if (!S.primary) narrate();
}

function pause() {
  S.playing = false;
  els.playpause.textContent = "▶";
  if (S.primary) S.primary.audio.pause();
  S.trailers.forEach(t => t.audio.pause());
  els.morph.pause();
  els.graph.pause();
}

function jump(i) {
  i = Math.max(0, Math.min(S.N - 1, i));
  const wasPlaying = S.playing;
  if (S.primary) {
    S.primary.audio.pause();
    clearTrackHighlight(S.primary);
    S.primary = null;
  }
  for (const t of S.trailers) { t.audio.pause(); clearTrackHighlight(t); }
  S.trailers = [];
  hideMorph();
  showStep(i, false);
  if (wasPlaying) { S.playing = true; narrate(); }
}

function setSpeed(v) {
  S.speed = v;
  if (S.primary) S.primary.audio.playbackRate = v;
  S.trailers.forEach(t => t.audio.playbackRate = v);
  els.morph.playbackRate = v;
  els.graph.playbackRate = v;
}

function toggleMute() {
  S.muted = !S.muted;
  if (S.primary) S.primary.audio.muted = S.muted;
  S.trailers.forEach(t => t.audio.muted = S.muted);
  els.mute.textContent = S.muted ? "🔇" : "🔊";
}

function onKey(e) {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); S.playing ? pause() : play(); }
  else if (e.code === "ArrowRight") jump(S.idx + 1);
  else if (e.code === "ArrowLeft") jump(S.idx - 1);
}

/* ---------------- helpers ---------------- */
/* Scroll ONLY the transcript container so the active line is centered.
   scrollIntoView() also scrolls every scrollable ancestor (including the page
   window), which makes the whole blog page jump on each step transition. */
function scrollActiveIntoView(child) {
  const c = els.transcript;
  if (!c || !child) return;
  const cRect = c.getBoundingClientRect();
  const chRect = child.getBoundingClientRect();
  const delta = (chRect.top - cRect.top) - (c.clientHeight - chRect.height) / 2;
  const max = c.scrollHeight - c.clientHeight;
  const target = Math.max(0, Math.min(max, c.scrollTop + delta));
  if (Math.abs(target - c.scrollTop) < 1) return;
  c.scrollTo({ top: target, behavior: "smooth" });
}

/* Lock the grid box to the candidate-grid aspect ratio so the panel keeps a
   constant height across steps. The branching step's archive strip has a very
   different aspect ratio; object-fit:contain letterboxes (pads) it inside the
   fixed box instead of resizing the whole panel. */
function lockGridAspect() {
  if (!els.gridbox) return;
  const gen = S.steps.find((s) => s.kind === "gen" && s.grid);
  if (!gen) return;
  const probe = new Image();
  probe.onload = () => {
    if (probe.naturalWidth && probe.naturalHeight) {
      els.gridbox.style.setProperty("--grid-ar", probe.naturalWidth + " / " + probe.naturalHeight);
    }
  };
  probe.src = gen.grid;
}

function hideMorph() {
  els.morph.classList.remove("show");
  els.graph.classList.remove("show");
  els.morph.pause();
  els.graph.pause();
  els.hero.classList.remove("hide");
  els.graph_static.classList.remove("hide");
  // Note: we do NOT removeAttribute("src") — the next startStepMorph swaps src
  // and plays, which is smoother than thrashing the element back to bare.
}

function maxGenNum() {
  return Math.max(...S.steps.filter(s => s.kind === "gen").map(s => s.genNum));
}
function stepProgressLabel() {
  const st = S.steps[S.idx];
  if (st.kind === "branching") return `Branching · 0 / ${maxGenNum()}`;
  return `Gen ${st.genNum} / ${maxGenNum()}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
