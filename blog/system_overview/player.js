/* Schematic system-overview player (curated, parallel evolution + evaluation).

   Reads ./curated_manifest.json (override with ?manifest=URL): a HAND-PICKED
   handful of "epochs" from one run, each an archive checkpoint + TWO independent
   draws (evolution + evaluation) + one evolution session + one rating session.

   The Sampling node is a DECK of stacked draw-cards, at the static figure's
   thumbnail scale. Per epoch:

     archive checkpoint  →  the EVOLUTION draw (front card: 5 shelves — top-rated,
       best-new, most-branched, latest, random) glides in from the Archive; the
       EVALUATION draw (2 shelves — least-rated, random) waits on the card behind.
     →  the evolver picks a branching parent from the front card and rides it into
       the generation grid.
     →  the deck SHUFFLES: the evaluation draw slides up to the front while a new
       card fades in behind it.
     →  IN PARALLEL: Evolution runs its 20-gen mutation/selection loop and
       publishes, while the rater streams the (now front) evaluation draw into VLM
       rating (1–5 stars) and the stars ride back to the Archive.
     →  the Archive grows: the publication rides the publish edge in, and as it lands a
       batch of images flips onto PART of the top layer — the other recent publications
       (gold-ringed) plus the older entries resurfacing from under the pile. They stay
       there, and the rest of the layer turns over to the next checkpoint. This is the
       ONLY time images arrive: the next epoch's draws simply lift half their images off
       the recent publications and half off the resurfaced ones  →  next epoch.

   Cross-edges follow system_overview.tex (orange feed/publish, green rate, dashed
   ×20 loop, black pipes). */

const qs = new URLSearchParams(location.search);
const MANIFEST_URL = qs.get("manifest") || "curated_manifest.json";

const DWELL = { sampleHold: 300, shuffle: 480, grow: 320 };

// Evolution-track beat durations (ms @ 1×). Named so the SAME numbers drive both
// the animation AND estimateEvoMs()/estimateGenLoopMs(), off which the evaluation
// stream paces itself so the two parallel branches finish together.
const EVO = {
  branchPre: 110, branchLiftAnim: 360, branchHold: 780,
  branchTravelAnim: 620, branchTravelWait: 560,
  spawnBase: 260, spawnStagger: 7, spawnTail: 40,
  selectPre: 120, convergeAnim: 240, convergeWait: 220, selectPost: 50, genPost: 50,
  pubPick: 640, pubFadeIn: 220, pubGrowWait: 300, pubHold: 800,
  pubLaunchAnim: 900, pubLaunchWait: 900, pubRideWait: 700,
};
function spawnMs() { const n = (S.M.grid_rows || 3) * (S.M.grid_cols || 5); return EVO.spawnBase + n * EVO.spawnStagger + EVO.spawnTail; }
function branchMs() { return EVO.branchPre + EVO.branchHold + EVO.branchTravelWait + spawnMs(); }
function estimateGenLoopMs(a) {
  const gens = a.generations || [], nG = gens.length, sp = spawnMs();
  let t = 0;
  gens.forEach((g, gi) => {
    if (g.published) t += EVO.pubPick + EVO.pubFadeIn + EVO.pubGrowWait + EVO.pubHold + EVO.pubLaunchWait + EVO.pubRideWait;
    else { t += EVO.selectPre + EVO.convergeWait + EVO.selectPost; if (gi < nG - 1) t += sp + EVO.genPost; }
  });
  return t;
}

const el = (id) => document.getElementById(id);
const els = {};
["fig", "arrows", "archive-img", "arch-back", "arch-front", "arch-fly", "fly-pub", "samp-deck",
 "p-archive", "p-sampling", "p-evolution", "p-evaluation",
 "sel", "mut", "rat", "gen-grid", "gen-wrap", "sel-overlay", "rated",
 "pub-clone", "pub-title", "pub-rider", "samp-fly", "branch-clone", "branch-title", "evo-fly",
 "rate-fly", "rate-stack", "rate-stars",
 "playpause", "speed", "scrub", "progress"].forEach((k) => (els[k] = el(k)));

const S = { M: null, agents: [], epoch: 0, playing: false, speed: 1,
            started: false, runToken: 0, edges: new Set(), phase: "",
            deck: { front: null, back: null },
            fresh: [],        // the recent publications sitting on the Archive's top layer
            deliverP: null,   // resolves once the whole publication batch has landed
            epochsRun: 0 };   // monotonic count of epochs entered (a render hook, see below)

// Offscreen hook for tools/render_system_overview_gif.py, which drives the show under CDP virtual
// time and needs to know when an epoch turns over so it can cut the GIF on a seamless loop.
window.__sysov = { get epoch() { return S.epoch; }, get epochsRun() { return S.epochsRun; },
                   get nEpochs() { return S.agents.length; }, play: () => play(), pause: () => pause(),
                   setSpeed: (v) => setSpeed(v) };

let pauseWaiters = [];

init();

async function init() {
  const M = await (await fetch(MANIFEST_URL, { cache: "no-store" })).json();
  S.M = M; S.agents = M.agents || [];
  if (els.scrub) els.scrub.max = String(Math.max(0, S.agents.length - 1));
  lockGridAspect();
  buildArrows();
  window.addEventListener("resize", () => { layoutArrows(); layoutDeck(); relayoutFresh(); });
  [120, 450, 1100].forEach((ms) => setTimeout(() => { layoutArrows(); layoutDeck(); }, ms));
  els["archive-img"].addEventListener("load", layoutArrows);

  renderStatic(0);
  // The visible control bar was removed; keyboard transport (space / arrows) still works.
  if (els.playpause) els.playpause.addEventListener("click", () => (S.playing ? pause() : play()));
  if (els.speed) els.speed.addEventListener("change", () => setSpeed(parseFloat(els.speed.value)));
  if (els.scrub) els.scrub.addEventListener("input", () => jumpTo(parseInt(els.scrub.value, 10)));
  document.addEventListener("keydown", onKey);
  play();
}

/* ---------------- transport ---------------- */
const alive = (tok) => tok === S.runToken;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms / S.speed));
function gate(tok) {
  if (!alive(tok) || S.playing) return Promise.resolve();
  return new Promise((res) => pauseWaiters.push(res));
}
function play() {
  S.playing = true; if (els.playpause) els.playpause.textContent = "❚❚";
  const w = pauseWaiters; pauseWaiters = []; w.forEach((r) => r());
  if (!S.started) { S.started = true; runShow(S.epoch); }
}
function pause() { S.playing = false; if (els.playpause) els.playpause.textContent = "▶"; }
function setSpeed(v) { S.speed = v; }
function jumpTo(ei) {
  S.runToken++; pauseWaiters = [];
  S.epoch = Math.max(0, Math.min(S.agents.length - 1, ei));
  resetAll(); renderStatic(S.epoch);
  S.playing = true; if (els.playpause) els.playpause.textContent = "❚❚"; S.started = true;
  runShow(S.epoch);
}
function onKey(e) {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); S.playing ? pause() : play(); }
  else if (e.code === "ArrowRight") jumpTo(S.epoch + 1);
  else if (e.code === "ArrowLeft") jumpTo(S.epoch - 1);
}

/* ---------------- the show: a loop over curated epochs ---------------- */
async function runShow(startEpoch) {
  const tok = ++S.runToken;
  let ei = ((startEpoch % S.agents.length) + S.agents.length) % S.agents.length;
  while (alive(tok)) {
    S.epoch = ei; if (els.scrub) els.scrub.value = String(ei);
    await gate(tok); if (!alive(tok)) return;
    const next = S.agents[(ei + 1) % S.agents.length];
    await runEpoch(S.agents[ei], next, tok);
    if (!alive(tok)) return;
    ei = (ei + 1) % S.agents.length;
  }
}

async function runEpoch(a, next, tok) {
  S.epochsRun++;
  // this epoch's checkpoint is already standing — the previous epoch grew into it. Only a
  // scrub/jump lands here with the wrong sheet up, and that one snaps.
  setArchive(a);
  setRated(a.evaluation);
  resetOverlays();
  // build the deck: front = this epoch's EVOLUTION draw (filled by the glide below),
  // back = this epoch's EVALUATION draw, waiting.
  buildDeck(a);
  await imgReady(els["archive-img"]); if (!alive(tok)) return;
  ensureArchiveBacking();
  layoutArrows(); layoutDeck();

  // 1) the evolution draw glides Archive → the front card
  clearActive();
  els["p-archive"].classList.add("active");
  sampGlow(true);
  S.edges.clear(); addEdges(["sample"]);
  setPhase(a, "sampling the archive");
  // wait for the front card's OWN strip image before flying tiles in: with height:auto it
  // renders zero-height until loaded, which would collapse every row onto the node's centre
  // line (a thin horizontal streak) instead of the grid.
  await imgReady(S.deck.front._img); if (!alive(tok)) return;
  await glideCard(S.deck.front, a.branching, a, tok); if (!alive(tok)) return;
  // the flying stopped -> Archive back to grey and the sample arrow back to faded
  els["p-archive"].classList.remove("active"); removeEdges(["sample"]);
  await sleep(DWELL.sampleHold); await gate(tok); if (!alive(tok)) return;

  // 2) the evolver picks its branching parent from the front card → rides the orange
  //    feed arrow into the grid
  setPhase(a, "branching a parent");
  els["p-evolution"].classList.add("active"); els.sel.classList.add("hot");
  addEdges(["feed", "pipe1", "pipe2", "loop"]);
  await branchPick(a, tok); if (!alive(tok)) return;
  // the feed arrow STAYS lit through breeding — the breeder keeps attending to this sample —
  // and its origin follows the evo card as it shuffles to the back (see layoutArrows/shuffleDeck).
  // the branching individual has been chosen -> the rest of the evolution sample clears out
  const evoImg = S.deck.front._img;
  evoImg.style.transition = `opacity ${0.42 / S.speed}s ease`; evoImg.style.opacity = "0";
  // branch done + the evo draw has emptied -> Sampling node goes quiet while evolution breeds,
  // until the evaluation draw glides in below.
  sampGlow(false);

  // 3) breeding starts NOW — the moment the branched individual arrives — and runs CONCURRENTLY
  //    with the evaluation setup + rating below (it is paced to finish a little before the rater).
  setPhase(a, "evolving + evaluating");
  const evoP = evoGenLoop(a, next, tok);

  // 3b) SWAP the two cards + deal the evaluation draw in from the Archive (evo breeds meanwhile,
  //     so DON'T clear its active state / pipe edges — only add the sample edge for the glide).
  await shuffleDeck(a, tok); if (!alive(tok)) return;
  els["p-archive"].classList.add("active"); sampGlow(true);
  addEdges(["sample"]);
  await imgReady(S.deck.front._img); if (!alive(tok)) return;
  // persist=true: the eval thumbnails land and STAY in the card (no snap to a composite image)
  const landedEval = await glideCard(S.deck.front, S.deck.front._spec, a, tok, true); if (!alive(tok)) return;
  // the flying stopped -> Archive back to grey and the sample arrow back to faded
  els["p-archive"].classList.remove("active"); removeEdges(["sample"]);
  await sleep(DWELL.sampleHold); await gate(tok); if (!alive(tok)) return;

  // 4) the rater streams the eval draw; evolution is already breeding and finishes a bit earlier
  await evalStream(a, tok, landedEval); if (!alive(tok)) return;
  sampGlow(false);   // rating draw emptied -> Sampling node goes quiet
  await evoP; if (!alive(tok)) return;

  // 5) the archive grows into the next checkpoint
  setPhase(a, "publication saved · archive grows");
  els["p-archive"].classList.add("active"); els["p-archive"].classList.add("grow");
  await growArchive(next, a, tok); if (!alive(tok)) return;
  await sleep(DWELL.grow); await gate(tok);
  els["p-archive"].classList.remove("grow");
  clearActive(); S.edges.clear(); applyEdges();
  clearDeck();
}

/* ================= the SAMPLING deck ================= */
// spec for a card: {image, row_labels, rows, cols, fill}
function evoSpec(a) {
  const b = a.branching || {};
  return { image: b.image, row_labels: b.row_labels || [], rows: b.rows || 5,
           cols: b.samples_per_row || 5, fill: b.fill || [], parent_row: b.parent_row,
           parent_col: b.parent_col, parent_title: b.parent_title, kind: "evo" };
}
function evalSpec(a) {
  const r = a.rating || {};
  return { image: r.image, row_labels: r.row_labels || ["least-rated", "random"],
           rows: r.rows || 2, cols: r.cols || 5, fill: r.fill || [], kind: "eval" };
}
function makeCard(spec) {
  const card = document.createElement("div");
  card.className = "samp-card k-" + spec.kind;
  card._spec = spec;
  const ul = document.createElement("ul"); ul.className = "row-labels";
  (spec.row_labels || []).forEach((lbl) => {
    const li = document.createElement("li");
    li.innerHTML = String(lbl).replace(/[<>]/g, "").replace(/\\+/g, "<br>");
    ul.appendChild(li);
  });
  const sf = document.createElement("div"); sf.className = "strip-frames";
  const img = document.createElement("img"); img.className = "contain"; img.alt = spec.kind + " sample";
  if (spec.image) img.src = relAsset(spec.image);
  sf.appendChild(img);
  card.appendChild(ul); card.appendChild(sf);
  card._img = img; card._labels = ul;
  img.addEventListener("load", layoutDeck);
  return card;
}
function setCardPos(card, pos) { if (card) { card.dataset.pos = pos; card.className = card.className.replace(/\s*pos-\w+/g, "") + " pos-" + pos; } }
function buildDeck(a) {
  els["samp-deck"].innerHTML = "";
  const front = makeCard(evoSpec(a));   // evolution draw (glide will fill it)
  const back = makeCard(evalSpec(a));   // evaluation draw, waiting behind
  els["samp-deck"].appendChild(back); els["samp-deck"].appendChild(front);
  setCardPos(back, "back"); setCardPos(front, "front");
  // BOTH draws start empty; each is dealt in from the Archive by its own glide (the evo
  // one now at the front, the eval one after the swap brings it to the front).
  front._img.style.opacity = "0";
  back._img.style.opacity = "0";
  S.deck = { front, back };
  layoutDeck();
}
function clearDeck() { els["samp-deck"].innerHTML = ""; S.deck = { front: null, back: null }; }
function layoutDeck() {
  els["samp-deck"] && els["samp-deck"].querySelectorAll(".samp-card").forEach((card) => {
    const img = card._img, lbl = card._labels;
    if (img && img.clientHeight) lbl.style.height = img.clientHeight + "px";
  });
}

// swap: the spent evolution draw (front) recedes to the back while the evaluation draw
// (back) comes forward — ONE consistent front↔back animation, exactly two cards, no third
// sheet. The eval card arrives empty at the front and is then dealt in from the Archive.
async function shuffleDeck(a, tok) {
  const evo = S.deck.front, evl = S.deck.back;
  setCardPos(evo, "back");
  setCardPos(evl, "front");
  S.deck = { front: evl, back: evo };
  // re-route the feed arrow every frame while the evo card slides to the back, so the arrow
  // tracks the card instead of snapping to its new position.
  await trackArrowsFor(DWELL.shuffle, tok); if (!alive(tok)) { return; }
  layoutDeck();
}
// rAF-loop layoutArrows for `ms` (playback-scaled) so SVG edges follow a CSS card transition.
function trackArrowsFor(ms, tok) {
  return new Promise((resolve) => {
    const t0 = performance.now(), dur = ms / S.speed;
    (function step() {
      layoutArrows();
      if (!alive(tok) || performance.now() - t0 >= dur) return resolve();
      requestAnimationFrame(step);
    })();
  });
}

/* ================= EVOLUTION branch — pick parent from the front (evo) card ================= */
async function branchPick(a, tok) {
  const card = S.deck.front; if (!card) return;
  const spec = card._spec;
  await imgReady(card._img); if (!alive(tok)) return;
  layoutDeck(); layoutArrows();
  const sp = S.speed;
  const rows = spec.rows, cols = spec.cols, pr = spec.parent_row, pc = spec.parent_col;
  const img = card._img, clone = els["branch-clone"], title = els["branch-title"];
  const si = rect(img), gw = rect(els["gen-wrap"]);
  els["gen-grid"].style.opacity = "0";

  if (pr == null || pc == null || !si.w) {
    els["gen-grid"].src = relAsset((a.generations[0] || {}).grid);
    await spawnGridCandidates((a.generations[0] || {}).grid, () => alive(tok));
    return;
  }
  const cw = si.w / cols, ch = si.h / rows;
  const pcx = si.l + (pc + 0.5) * cw, pcy = si.t + (pr + 0.5) * ch;
  const TR = (x, y) => `translate(${x}px,${y}px) translate(-50%,-50%)`;

  // Cancelling a fill:forwards animation on a display:none element leaves its end state to paint
  // on the frame the element is shown again — the previous epoch's parent would flash at the grid
  // centre for a frame or two. So: kill the leftovers, place the clone while it is still hidden,
  // and only then unhide it.
  clone.getAnimations().forEach((an) => an.cancel());
  clone.style.backgroundImage = `url("${img.src}")`;
  clone.style.backgroundSize = `${cols * 100}% ${rows * 100}%`;
  clone.style.backgroundPosition = `${cols > 1 ? (pc / (cols - 1)) * 100 : 0}% ${rows > 1 ? (pr / (rows - 1)) * 100 : 0}%`;
  clone.style.width = cw + "px"; clone.style.height = ch + "px";
  clone.style.transform = TR(pcx, pcy);
  clone.style.display = "block"; clone.style.opacity = "1";
  void clone.offsetWidth;

  await sleep(EVO.branchPre); if (!alive(tok)) return;
  const sSize = Math.min(si.w, si.h) * 0.7;
  const scx = si.l + si.w / 2, scy = si.t + si.h * 0.5;
  clone.animate([
    { transform: TR(pcx, pcy), width: cw + "px", height: ch + "px" },
    { transform: TR(scx, scy), width: sSize + "px", height: sSize + "px" },
  ], { duration: EVO.branchLiftAnim / sp, easing: "ease", fill: "forwards" });
  title.textContent = spec.parent_title || "";
  if (title.textContent) {
    title.style.left = scx + "px"; title.style.top = (scy + sSize / 2 + 0.012 * gw.height) + "px";
    title.style.display = "block"; title.style.opacity = "0"; void title.offsetWidth;
    title.style.transition = `opacity ${0.3 / sp}s ease`; title.style.opacity = "1";
  }

  await sleep(EVO.branchHold); if (!alive(tok)) return;
  if (title.textContent) { title.style.transition = `opacity ${0.25 / sp}s ease`; title.style.opacity = "0"; }
  layoutArrows();
  const cell = cellFig(0, gw), gcx = gw.l + gw.w / 2, gcy = gw.t + gw.h / 2;
  const pts = [{ x: scx, y: scy }];
  const feedEl = document.getElementById("e-feed");
  if (feedEl) { const L = feedEl.getTotalLength(); for (let i = 1; i <= 20; i++) { const p = feedEl.getPointAtLength((L * i) / 20); pts.push({ x: p.x, y: p.y }); } }
  const selR = rect(els.sel), mutR = rect(els.mut);
  pts.push({ x: selR.cx, y: selR.cy }, { x: mutR.cx, y: mutR.cy }, { x: gcx, y: gcy });
  // shrink to the destination grid-cell (thumbnail) size right as it leaves the card, then
  // ride the feed edge at that small size the rest of the way — travelling images match the
  // size they land at, not the big showcase size.
  const kf = pts.map((p, i) => {
    const f = i / (pts.length - 1);
    const s = Math.min(1, f / 0.12);
    return { transform: TR(p.x, p.y), width: (sSize + (cell.w - sSize) * s) + "px",
             height: (sSize + (cell.h - sSize) * s) + "px", offset: f };
  });
  clone.animate(kf, { duration: EVO.branchTravelAnim / sp, easing: "ease-in-out", fill: "forwards" });

  await sleep(EVO.branchTravelWait); if (!alive(tok)) return;
  clone.style.transition = `opacity ${0.26 / sp}s ease`; clone.style.opacity = "0";
  els["gen-grid"].src = relAsset((a.generations[0] || {}).grid);
  await spawnGridCandidates((a.generations[0] || {}).grid, () => alive(tok));
  if (!alive(tok)) return;
  clone.style.display = "none";
  clone.getAnimations().forEach((an) => an.cancel());   // don't leave a filled end state behind
}

/* ================= EVOLUTION gen-loop (runs parallel with the rater) ================= */
async function evoGenLoop(a, next, tok) {
  els["p-evolution"].classList.add("active");
  // the feed arrow stays lit for the whole breeding run (the breeder keeps drawing on the
  // sample), alongside the internal selection→mutation→grid pipe/loop arrows.
  addEdges(["feed", "pipe1", "pipe2", "loop"]);
  const gens = a.generations || [];
  const nGen = a.n_generations || 20;
  setLoopCount(nGen);
  for (let gi = 0; gi < gens.length; gi++) {
    const g = gens[gi];
    await gate(tok); if (!alive(tok)) return;
    els.sel.classList.add("hot"); els.mut.classList.remove("hot");
    if (g.published) { await runPublish(a, next, g, tok); break; }
    renderSelection(g.selected);
    await sleep(EVO.selectPre); if (!alive(tok)) return;
    await convergeSelected(g, tok); if (!alive(tok)) return;
    await sleep(EVO.selectPost);
    const nextGen = gens[gi + 1];
    if (nextGen) {
      els.mut.classList.add("hot"); els.sel.classList.remove("hot");
      pulseLoop();
      els["gen-grid"].src = relAsset(nextGen.grid);
      await spawnGridCandidates(nextGen.grid, () => alive(tok)); if (!alive(tok)) return;
      await sleep(EVO.genPost);
    }
  }
  els.sel.classList.remove("hot"); els.mut.classList.remove("hot");
  removeEdges(["feed", "pipe1", "pipe2", "loop"]);
  els["p-evolution"].classList.remove("active");   // breeding done -> node greys while the rater finishes
}

async function convergeSelected(g, tok) {
  const sp = S.speed;
  const gw = rect(els["gen-wrap"]);
  const rows = S.M.grid_rows || 3, cols = S.M.grid_cols || 5;
  const cx = gw.l + gw.w / 2, cy = gw.t + gw.h / 2;
  const sel = g.selected || [];
  renderSelection(null);
  const layer = els["evo-fly"]; layer.innerHTML = "";
  sel.forEach((idx) => {
    const cf = cellFig(idx, gw);
    const c = gridCropEl(g.grid, idx, rows, cols, true);
    c.style.width = cf.w + "px"; c.style.height = cf.h + "px";
    c.style.transform = `translate(${cf.cx}px,${cf.cy}px) translate(-50%,-50%)`;
    layer.appendChild(c);
    c.animate([
      { transform: `translate(${cf.cx}px,${cf.cy}px) translate(-50%,-50%)` },
      { transform: `translate(${cx}px,${cy}px) translate(-50%,-50%)` },
    ], { duration: EVO.convergeAnim / sp, easing: "cubic-bezier(.5,0,.3,1)", fill: "forwards" });
  });
  els["gen-grid"].style.transition = `opacity ${0.4 / sp}s ease`;
  els["gen-grid"].style.opacity = "0";
  await sleep(EVO.convergeWait);
}

async function runPublish(a, next, g, tok) {
  const sp = S.speed;
  const pubImg = relAsset((a.publication && a.publication.image) || g.grid);
  const cellIdx = (g.published && g.published.index != null) ? g.published.index
                  : (g.selected && g.selected[0]) || 0;
  const gw = els["gen-wrap"].getBoundingClientRect();
  const rc = cellRect(cellIdx);
  const x0 = rc.left / 100 * gw.width, y0 = rc.top / 100 * gw.height;
  const w0 = rc.w / 100 * gw.width, h0 = rc.h / 100 * gw.height;
  const clone = els["pub-clone"], title = els["pub-title"], rider = els["pub-rider"];
  clone.src = pubImg; rider.src = pubImg;
  title.textContent = (g.published && g.published.title) || (a.publication && a.publication.title) || "";
  els["evo-fly"].innerHTML = "";

  // 1) the last generation's grid stands, and the breeder picks its favourite out of it: the
  //    winning cell is outlined in place, exactly as a selection is on every other generation.
  els["gen-grid"].style.transition = "none";
  els["gen-grid"].style.opacity = "1";
  renderSelection([cellIdx]);
  await sleep(EVO.pubPick); if (!alive(tok)) return;

  // 2) the winner lifts off its cell — the clone takes over from the outlined crop at exactly its
  //    place and size — and the losing candidates (and the outline) fade out beneath it.
  clone.style.display = "block"; clone.style.transition = "none"; clone.style.opacity = "1";
  clone.style.left = x0 + "px"; clone.style.top = y0 + "px";
  clone.style.width = w0 + "px"; clone.style.height = h0 + "px";
  void clone.offsetWidth;
  renderSelection(null);
  els["gen-grid"].style.transition = `opacity ${0.3 / sp}s ease`;
  els["gen-grid"].style.opacity = "0";   // the losing candidates fade fully out; only the winner stays

  await sleep(EVO.pubFadeIn); if (!alive(tok)) return;
  const size = 0.62 * gw.height;
  const x1 = (gw.width - size) / 2, y1 = (gw.height - size) / 2 - 0.06 * gw.height;
  clone.style.transition = `left ${0.45 / sp}s ease, top ${0.45 / sp}s ease, width ${0.45 / sp}s ease, height ${0.45 / sp}s ease`;
  clone.style.left = x1 + "px"; clone.style.top = y1 + "px";
  clone.style.width = size + "px"; clone.style.height = size + "px";

  await sleep(EVO.pubGrowWait); if (!alive(tok)) return;
  title.style.top = (y1 + size + 0.02 * gw.height) + "px";
  title.style.display = "block"; title.style.opacity = "0"; void title.offsetWidth;
  title.style.transition = `opacity ${0.35 / sp}s ease`; title.style.opacity = "1";

  await sleep(EVO.pubHold); if (!alive(tok)) return;
  els["p-archive"].classList.add("active");
  await launchPublication(tok); if (!alive(tok)) return;
  addEdges(["publish"]);
  S.deliverP = deliverPublications(a, next, tok);
  await sleep(EVO.pubRideWait); if (!alive(tok)) return;
  // the publish edge stays lit until the LAST of the batch lands (see deliverPublications)
}

/* The showcased publication doesn't dissolve and reappear on the arrow — it IS the image that
   leaves. The rider takes over from the clone in the same frame, at exactly its place and size,
   then shrinks to Archive-cell size as it rides the dashed ×20 back-edge out of the generation
   grid and through the VLM-selection box, ending on the mouth of the publish edge, where
   deliverPublications picks it up without a seam. */
async function launchPublication(tok) {
  const sp = S.speed;
  const clone = els["pub-clone"], title = els["pub-title"], rider = els["pub-rider"];
  layoutArrows();
  const cr = rect(clone), gm = archGeom();
  const loopEl = document.getElementById("e-loop"), pubEl = document.getElementById("e-publish");
  const at = (p) => `translate(${p.x}px,${p.y}px) translate(-50%,-50%)`;

  const pts = [{ x: cr.cx, y: cr.cy }];
  if (loopEl) { const L = loopEl.getTotalLength(); for (let i = 1; i <= 18; i++) pts.push(loopEl.getPointAtLength((L * i) / 18)); }
  if (pubEl) pts.push(pubEl.getPointAtLength(0));

  rider.getAnimations().forEach((an) => an.cancel());
  rider.style.display = "block"; rider.style.opacity = "1";
  rider.style.width = cr.w + "px"; rider.style.height = cr.h + "px";
  rider.style.transform = at(pts[0]);
  clone.style.transition = "none"; clone.style.display = "none"; clone.style.opacity = "0";
  title.style.transition = `opacity ${0.24 / sp}s ease`; title.style.opacity = "0";
  els["gen-grid"].style.transition = "none"; els["gen-grid"].style.opacity = "0";
  if (pts.length < 2) { rider.style.width = gm.cw + "px"; rider.style.height = gm.ch + "px"; return; }

  pulseLoop();
  // shrink over the first third of the flight, so it is already thumbnail-sized by the time it
  // threads the selection box rather than deflating on the doorstep
  const kf = pts.map((p, i) => {
    const f = i / (pts.length - 1), s = Math.min(1, f / 0.34);
    return { transform: at(p), width: (cr.w + (gm.cw - cr.w) * s) + "px",
             height: (cr.h + (gm.ch - cr.h) * s) + "px", offset: f };
  });
  rider.animate(kf, { duration: EVO.pubLaunchAnim / sp, easing: "ease-in-out", fill: "forwards" });

  await sleep(EVO.pubLaunchWait); if (!alive(tok)) return;
  title.style.display = "none";
  // park the rider on the publish mouth for real, rather than leaving a forwards fill to hold it
  // there over an inline transform still pointing at the generation grid
  rider.getAnimations().forEach((an) => an.cancel());
  rider.style.width = gm.cw + "px"; rider.style.height = gm.ch + "px";
  rider.style.transform = at(pts[pts.length - 1]);
}

/* The featured publication — the one we watched being bred — is the ONLY image that travels the
   publish edge. Everything else the top layer gains flips into the cell it keeps as the featured
   one touches down: the other recent publications (gold-ringed; they came from Evolution nodes we
   never see, so they have no journey to show) and the older entries resurfacing from under the
   pile. This is the ONE arrival — the next epoch's draws lift images off these cells and nothing
   flips in again. The layer beneath them turns over to the next checkpoint DURING this same wave,
   so the Archive changes exactly once: when the wave settles it is already showing what the next
   epoch will sample from. (growArchive afterwards only waits for it.) */
function deliverPublications(a, next, tok) {
  layoutArrows();
  const pathEl = document.getElementById("e-publish");
  if (!pathEl) return Promise.resolve();
  const sp = S.speed;
  const L = pathEl.getTotalLength();
  const N = 28, pts = [];
  for (let i = 0; i <= N; i++) { const p = pathEl.getPointAtLength((L * i) / N); pts.push(p); }
  const g = archGeom();
  const archN = Math.min((next && next.archive_n) || g.n, g.n);
  const pubCell = pubCellIndex(a, archN);
  const end = pts[pts.length - 1];
  const at = (p) => `translate(${p.x}px,${p.y}px) translate(-50%,-50%)`;
  const nextSrc = next && next.archive_image && relAsset(next.archive_image);
  const A = nextSrc ? arrivalCells(archN, pubCell) : { pubs: [pubCell], surf: [] };
  // publications first, then the resurfacing entries behind them
  const ks = [...A.pubs, ...A.surf];
  const isPub = (i) => i < A.pubs.length;

  // the previous batch of arrivals has been sampled from all epoch; it is superseded now
  const layer = els["arch-fly"];
  layer.innerHTML = ""; S.fresh = [];

  // the layer BENEATH the arrivals turns over to the next checkpoint under cover of the same
  // wave: the old sheet (#arch-front, below #arch-fly) fades out while the cells flip in, so the
  // whole Archive settles into its next state in one motion instead of turning over a second
  // time on the way to the next sample.
  const prevSrc = els["archive-img"].getAttribute("src");
  const turned = (nextSrc && nextSrc !== prevSrc && els["archive-img"].offsetWidth)
    ? turnOverTopLayer(prevSrc, nextSrc, tok,
        { delay: DELIVER.lead, duration: ks.length * DELIVER.flipGap + DELIVER.flip })
    : Promise.resolve();

  // every arrival gets a persistent overlay cell, held invisible until it is due
  const cells = ks.map((k, i) => {
    const cell = document.createElement("div");
    cell.className = "arch-fly-cell" + (isPub(i) ? " fresh" : "");
    if (nextSrc) archCrop(cell, nextSrc, k, g);
    cell.style.width = g.cw + "px"; cell.style.height = g.ch + "px";
    cell.style.opacity = "0";
    layer.appendChild(cell);
    S.fresh.push({ k, el: cell, pub: isPub(i) });
    return cell;
  });

  // 1) the featured publication rides the edge (at the Archive cell size it shrank to on its way
  //    out of Evolution, and lands at) and settles on its cell. The ride picks the rider up exactly
  //    where launchPublication parked it, so drop that animation's forwards fill first.
  const rider = els["pub-rider"];
  rider.getAnimations().forEach((an) => an.cancel());
  rider.style.display = "block"; rider.style.opacity = "1";
  rider.style.width = g.cw + "px"; rider.style.height = g.ch + "px";
  const featIdx = ks.indexOf(pubCell), featDst = g.at(pubCell);
  const ridden = rider.animate(pts.map((p, j) => ({ transform: at(p), offset: j / N })),
    { duration: DELIVER.ride / sp, easing: "ease-in-out", fill: "forwards" })
    .finished.then(() => {
      if (!alive(tok)) return;
      return rider.animate([{ transform: at(end) }, { transform: at(featDst) }],
        { duration: DELIVER.land / sp, easing: "ease-in", fill: "forwards" }).finished;
    }).then(() => {
      if (!alive(tok)) return;
      placeFresh(cells[featIdx], featDst, g);   // the overlay takes over from the rider
      fadeRider();
      removeEdges(["publish"]);
    }).catch(() => {});

  // 2) the rest flip in around it, first ones landing just before the rider does
  const flips = ks.map((k, i) => {
    if (k === pubCell) return Promise.resolve();
    const dst = g.at(k);
    const T = (rot) => `translate(${dst.x}px,${dst.y}px) translate(-50%,-50%) perspective(220px) rotateY(${rot}deg)`;
    const order = i < featIdx ? i : i - 1;    // the featured cell takes no slot in the wave
    return cells[i].animate([
      { transform: T(-72), opacity: 0, offset: 0 },
      { transform: T(0), opacity: 1, offset: 1 },
    ], { duration: DELIVER.flip / sp, delay: (DELIVER.lead + order * DELIVER.flipGap) / sp,
         easing: "cubic-bezier(.3,.8,.3,1)", fill: "both" })
      .finished.then(() => { if (alive(tok)) placeFresh(cells[i], dst, g); }).catch(() => {});
  });
  return Promise.all([ridden, turned, ...flips]);
}
// the featured publication has been absorbed into the top layer -> let the rider go
function fadeRider() {
  els["pub-rider"].animate([{ opacity: 1 }, { opacity: 0 }],
    { duration: 300 / S.speed, easing: "ease", fill: "forwards" });
}

/* ================= EVALUATION stream — from the front (eval) card ================= */
function rateStar(seedA, seedB) { return ((seedA * 37 + seedB * 101 + 3) % 5) + 1; }
function bgCrop(elm, src, r, c, rows, cols) {
  elm.style.backgroundImage = `url("${src}")`;
  elm.style.backgroundRepeat = "no-repeat";
  elm.style.backgroundSize = `${cols * 100}% ${rows * 100}%`;
  elm.style.backgroundPosition =
    `${cols > 1 ? (c / (cols - 1)) * 100 : 0}% ${rows > 1 ? (r / (rows - 1)) * 100 : 0}%`;
}
// The rater flips through the evaluation sample ONE image at a time, showing each
// centred in the Evaluation node with its title + star rating underneath, then
// replacing it with the next. One persistent frame whose content cycles (no flying
// thumbnails). Paced to finish together with the parallel evolution gen-loop.
async function evalStream(a, tok, landed) {
  const sp = S.speed;
  const card = S.deck.front;                // after the shuffle, the eval draw is on front
  if (!card || card._spec.kind !== "eval") return;
  landed = landed || [];
  const spec = card._spec;
  const cellsMeta = (a.rating && a.rating.cells) || [];
  els["p-evaluation"].classList.add("active");
  addEdges(["ratefeed", "pipe3"]);     // sample→rating lit; the rating→Archive arrow waits for the batch to finish
  els.rat.classList.add("hot");
  els.rated.style.visibility = "hidden";       // the reveal replaces the static examples grid
  const rows = spec.rows, cols = spec.cols, fill = spec.fill || [];
  const src = relAsset(spec.image);
  await imgReady(card._img); if (!alive(tok)) return;
  layoutArrows();

  const cells = [];
  for (let r = 0; r < rows; r++) { const f = (fill[r] != null ? fill[r] : cols); for (let c = 0; c < f; c++) cells.push({ r, c }); }
  const n = cells.length || 1;

  // reveal zone = the Evaluation node's examples region; rated cards PILE UP in the middle
  // (a growing stack, newest on top) — nothing fades in or out between images.
  const rg = rect(els.rated);
  const host = els["rate-stack"]; host.innerHTML = "";
  const cx = rg.l + rg.w / 2, cy = rg.t + rg.h * 0.44;
  const size = Math.min(rg.w * 0.42, rg.h * 0.46);
  const step = size * 0.02;   // tight diagonal offset between successive cards in the pile

  // each sample vanishes from the eval DRAW as it is rated by fading out its REAL landed
  // thumbnail (glideCard left them in place), keyed by cell so order-independent.
  const cellOf = {};
  landed.forEach((L) => { cellOf[L.r + "," + L.c] = L.el; });

  // Evolution began breeding ~HEADSTART ms earlier (during the swap + eval glide) and should
  // finish a little BEFORE this rater does — so the eval node is the last one busy and never
  // sits empty before the next epoch. Pace the rating window off the evo duration minus that
  // head start and the closing star-flight (STAR_MS), plus a small endGap so eval lands last.
  // STAR_MS tracks sendRatingStars()'s real cost (fat-star ride + burst + tail)
  const evalFlush = 300, HEADSTART = 1550, STAR_MS = STAR_RIDE + STAR_BURST + 400, endGap = 450;
  const windowMs = Math.max(1300, estimateGenLoopMs(a) - HEADSTART - STAR_MS - evalFlush + endGap);
  const hold = Math.max(205, windowMs / n);    // per-image dwell (paced to the evo window)

  let prevMeta = null;   // the .er-meta of the card that a newer one is about to bury
  for (let i = 0; i < n; i++) {
    if (!alive(tok)) return;
    await gate(tok);
    const { r, c } = cells[i];
    const meta = cellsMeta[i] || {};
    // remove this sample from the eval draw for real: fade out its landed thumbnail
    const cellEl = cellOf[r + "," + c];
    if (cellEl) cellEl.animate([{ opacity: 1 }, { opacity: 0 }], { duration: Math.min(300, hold * 0.9) / sp, easing: "ease", fill: "forwards" });
    // the card below loses its title + stars the moment a new one lands on top
    if (prevMeta) { prevMeta.remove(); prevMeta = null; }
    // build a fresh rated card and SNAP it onto the pile (no entry animation). The pile runs
    // top-left (i=0) -> bottom-right (i=n-1), centred so first and last are equidistant from cx,cy.
    const frame = document.createElement("div"); frame.className = "eval-reveal";
    const off = (i - (n - 1) / 2) * step;
    frame.style.left = (cx + off) + "px"; frame.style.top = (cy + off) + "px";
    frame.style.zIndex = String(7 + i);
    const im = document.createElement("div"); im.className = "er-img";
    im.style.width = size + "px"; im.style.height = size + "px";
    bgCrop(im, src, r, c, rows, cols);
    const mt = document.createElement("div"); mt.className = "er-meta";
    const ti = document.createElement("div"); ti.className = "er-title"; ti.textContent = meta.title || "";
    const st = document.createElement("div"); st.className = "er-stars rt-stars";
    st.innerHTML = stars(meta.rating != null ? meta.rating : rateStar(S.epoch + 1, i));
    mt.appendChild(ti); mt.appendChild(st);
    frame.appendChild(im); frame.appendChild(mt);
    host.appendChild(frame);
    prevMeta = mt;
    await sleep(hold);
  }
  if (!alive(tok)) return;
  // the batch is complete -> NOW the Evaluation→Archive arrow lights, the LAST card drops its
  // title + stars (like every card before it), and in that same instant a single fat star
  // swells up at the foot of the node and rides the edge to burst over the Archive. The pile of
  // rated cards is what the star carries off: it clears as the star sets out, so the node is
  // empty behind it rather than holding a stale stack all the way to the Archive.
  addEdges(["rate"]);
  if (prevMeta) { prevMeta.remove(); prevMeta = null; }
  host.animate([{ opacity: 1 }, { opacity: 0 }],
    { duration: Math.min(320, STAR_RIDE * 0.32) / sp, easing: "ease", fill: "forwards" });
  await sendRatingStars(a, tok); if (!alive(tok)) return;
  host.innerHTML = ""; host.getAnimations().forEach((an) => an.cancel());
  // the draw is spent -> every landed thumbnail has faded out; clear the fly layer so nothing
  // lingers over the (now grey) Sampling card.
  els["samp-fly"].innerHTML = "";
  els.rated.style.visibility = "";
  els.rat.classList.remove("hot");
  removeEdges(["ratefeed", "rate", "pipe3"]);
}

// batch-complete flourish: ONE fat star swells into being at the FOOT of the Evaluation node,
// where the rate edge starts (the instant the last card's title + stars are dropped), rides
// that edge, and on arrival BURSTS into baby stars that scatter across the Archive
// (deterministic spread) and wink out.
const STAR_RIDE = 1000, STAR_BURST = 760;
async function sendRatingStars(a, tok) {
  const layer = els["rate-stars"]; if (!layer) return;
  layer.innerHTML = "";
  const sp = S.speed;
  const pathEl = document.getElementById("e-rate");
  const ai = rect(els["archive-img"]);
  const pts = [];
  if (pathEl) { const L = pathEl.getTotalLength(), N = 22;
    for (let i = 0; i <= N; i++) { const p = pathEl.getPointAtLength((L * i) / N); pts.push({ x: p.x, y: p.y }); } }
  const ev = rect(els["p-evaluation"]);
  const start = pts.length ? pts[0] : { x: ev.cx, y: ev.b };
  const end = pts.length ? pts[pts.length - 1] : { x: ai.cx, y: ai.cy };
  const TR = (x, y, s) => `translate(${x}px,${y}px) translate(-50%,-50%) scale(${s})`;

  // 1) the single fat star: swells into being at the edge source, then rides the edge
  const big = document.createElement("div"); big.className = "fly-star big"; big.textContent = "★";
  layer.appendChild(big);
  const kf = [
    { transform: TR(start.x, start.y, 0.45), opacity: 0, offset: 0 },
    { transform: TR(start.x, start.y, 1), opacity: 1, offset: 0.12 },
  ];
  pts.forEach((p, i) => { const f = pts.length > 1 ? i / (pts.length - 1) : 0;
    kf.push({ transform: TR(p.x, p.y, 1), opacity: 1, offset: 0.12 + 0.88 * f }); });
  big.animate(kf, { duration: STAR_RIDE / sp, easing: "ease-in-out", fill: "both" });
  await sleep(STAR_RIDE + 40);
  if (!alive(tok)) { layer.innerHTML = ""; return; }

  // 2) it bursts: baby stars fan out to scattered spots across the Archive
  big.remove();
  const K = 12;
  const H = (k) => { const x = Math.sin((k + 1) * 12.9898 + (S.epoch + 1) * 3.71) * 43758.5453; return x - Math.floor(x); };
  for (let k = 0; k < K; k++) {
    const s = document.createElement("div"); s.className = "fly-star baby"; s.textContent = "★";
    layer.appendChild(s);
    const tx = ai.l + (0.1 + 0.8 * H(k)) * ai.w, ty = ai.t + (0.1 + 0.8 * H(k + 50)) * ai.h;
    // bow the fan out slightly so the babies spray rather than run straight lines
    const mx = (end.x + tx) / 2 + (H(k + 7) - 0.5) * 0.35 * ai.w;
    const my = (end.y + ty) / 2 + (H(k + 19) - 0.5) * 0.35 * ai.h;
    s.animate([
      { transform: TR(end.x, end.y, 1.15), opacity: 1, offset: 0 },
      { transform: TR(mx, my, 0.95), opacity: 1, offset: 0.55 },
      { transform: TR(tx, ty, 0.4), opacity: 0, offset: 1 },
    ], { duration: STAR_BURST / sp, delay: (k * 18) / sp, easing: "cubic-bezier(.2,.7,.3,1)", fill: "both" });
  }
  await sleep(STAR_BURST + K * 18 + 120);
  if (alive(tok)) layer.innerHTML = "";
}

/* Where the j-th image of a draw is lifted from. Every draw comes off the batch that arrived with
   the last publication (S.fresh): even j off a recent publication, odd j off an entry that
   resurfaced from under the pile. Both are already sitting on the top layer, so a draw never
   makes anything appear in the Archive — it only takes. Before the first publication there is no
   batch, so the opening draw sweeps the sheet instead. */
function drawSource(j, nF, archN, g) {
  const live = (S.fresh || []).filter((f) => f.el && f.el.isConnected);
  const want = live.filter((f) => f.pub === (j % 2 === 0));
  const pool = want.length ? want : live;
  if (!pool.length) return g.at(Math.min(archN - 1, Math.floor(((j + 0.5) / nF) * archN)));
  return g.at(pool[Math.floor(hash01(j, S.epoch + 2) * pool.length)].k);
}

/* ---------------- shared sample glide (Archive → a card's strip) ----------------
   persist=true (the evaluation draw): the flown cells STAY put as the landed thumbnails —
   no swap to the standalone composite strip image (which would "snap") — and the array of
   {el, r, c} is returned so the rater can remove each real cell as it rates it. */
function glideCard(card, spec, a, tok, persist) {
  return new Promise((resolve) => {
    if (!card || !spec || !spec.image) { if (card) { if (!persist) card._img.style.opacity = "1"; card.classList.add("dealt"); } resolve(persist ? [] : null); return; }
    const sp = S.speed;
    const img = card._img, layer = els["samp-fly"];
    const rows = spec.rows || (spec.row_labels || []).length || 5;
    const cols = spec.samples_per_row || spec.cols || 5;
    const si = rect(img), ai = rect(els["archive-img"]);
    if (!si.w || !si.h || !ai.w) { if (!persist) img.style.opacity = "1"; card.classList.add("dealt"); resolve(persist ? [] : null); return; }
    const cw = si.w / cols, ch = si.h / rows;
    const g = archGeom();
    const acw = g.cw, ach = g.ch;
    const archN = Math.min(a.archive_n || g.n, g.n);
    const fill = spec.fill || [];
    const toFly = [];
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
      if (c < (fill[r] != null ? fill[r] : cols)) toFly.push([r, c]);
    const nF = toFly.length || 1;
    img.style.transition = "none"; img.style.opacity = "0";
    layer.innerHTML = "";
    const landed = [];
    toFly.forEach(([r, c], j) => {
      const f = document.createElement("div");
      f.className = "samp-fly-cell";
      f.style.backgroundImage = `url("${img.src}")`;
      f.style.backgroundSize = `${cols * 100}% ${rows * 100}%`;
      f.style.backgroundPosition = `${cols > 1 ? (c / (cols - 1)) * 100 : 0}% ${rows > 1 ? (r / (rows - 1)) * 100 : 0}%`;
      const src = drawSource(j, nF, archN, g);
      const dx = si.l + (c + 0.5) * cw, dy = si.t + (r + 0.5) * ch;
      const TR = (x, y) => `translate(${x}px,${y}px) translate(-50%,-50%)`;
      f.style.transform = TR(src.x, src.y);
      f.style.width = acw + "px"; f.style.height = ach + "px"; f.style.opacity = "0";
      layer.appendChild(f);
      landed.push({ el: f, r, c });
      f.animate([
        { transform: TR(src.x, src.y), width: acw + "px", height: ach + "px", opacity: 0 },
        { transform: TR(src.x, src.y), width: acw + "px", height: ach + "px", opacity: 1, offset: .18 },
        { transform: TR(dx, dy), width: cw + "px", height: ch + "px", opacity: 1 },
      ], { duration: 560 / sp, delay: (j * 14) / sp, easing: "ease-in-out", fill: "both" });
    });
    setTimeout(() => {
      if (!alive(tok)) { if (!persist) layer.innerHTML = ""; resolve(persist ? landed : null); return; }
      card.classList.add("dealt");   // thumbnails are in place -> the shelf labels may fade in
      if (persist) { resolve(landed); return; }   // leave cells in place; composite stays hidden
      img.style.transition = `opacity ${0.2 / sp}s ease`; img.style.opacity = "1";
      layer.innerHTML = "";
      resolve(null);
    }, (560 + nF * 14) / sp + 40);
  });
}

/* children appear full-size at the grid centre, then move out to their cells */
async function spawnGridCandidates(gridSrc, live) {
  const sp = S.speed;
  const gw = rect(els["gen-wrap"]);
  const rows = S.M.grid_rows || 3, cols = S.M.grid_cols || 5;
  const cx = gw.l + gw.w / 2, cy = gw.t + gw.h / 2;
  const grid = els["gen-grid"], layer = els["evo-fly"];
  grid.style.opacity = "0";
  const n = rows * cols;
  for (let i = 0; i < n; i++) {
    const cf = cellFig(i, gw);
    const c = gridCropEl(gridSrc, i, rows, cols);
    c.style.width = cf.w + "px"; c.style.height = cf.h + "px"; c.style.zIndex = "1";
    c.style.transform = `translate(${cx}px,${cy}px) translate(-50%,-50%)`; c.style.opacity = "0";
    layer.appendChild(c);
    c.animate([
      { transform: `translate(${cx}px,${cy}px) translate(-50%,-50%)`, opacity: 0 },
      { transform: `translate(${cx}px,${cy}px) translate(-50%,-50%)`, opacity: 1, offset: .16 },
      { transform: `translate(${cf.cx}px,${cf.cy}px) translate(-50%,-50%)`, opacity: 1 },
    ], { duration: EVO.spawnBase / sp, delay: (i * EVO.spawnStagger) / sp, easing: "cubic-bezier(.4,0,.3,1)", fill: "both" });
  }
  await sleep(EVO.spawnBase + n * EVO.spawnStagger + EVO.spawnTail);
  if (live && !live()) return;
  grid.style.opacity = "1"; layer.innerHTML = "";
}

/* ================= the ARCHIVE pile + its growth =================
   Each epoch's archive_image is a fresh RANDOM 50-cell sample of the archive as it stood at
   that agent's branching time (see render_archive_grid in build_system_fig_assets.py). Between
   the curated epochs ~170 other agents publish, so the archive grows by ~22% — yet all 50
   visible cells change, because the sample is redrawn from the whole (much larger) archive.

   The animation splits that into the two things it actually is:

     GROWTH (when the publication lands): one batch of images flips onto the top layer, covering
       maybe half its cells, never all of it, and STAYS there. Two kinds, and the next draws
       take half their images from each: the other recent PUBLICATIONS (gold-ringed) and the
       older entries RESURFACING from under the pile, which the redrawn sample now shows.
       Underneath them the rest of the top layer quietly turns over.
     SAMPLING (next epoch): the draws just lift images straight off that batch — half off the
       recent publications, half off the resurfaced entries. Nothing new arrives at sample time.

   Sheets are opaque and offset by only ARCH_D px, so the pile reads as a tight solid stack. */
const ARCH_D = 2, ARCH_MAX = 4;
const GROW = { fade: 460, pubFrac: 0.30, surfFrac: 0.26 };
// the featured publication's ride down the publish edge, and the wave of flips it arrives with:
// `lead` is tuned so the first cells land just before the rider touches down at ride+land.
const DELIVER = { ride: 880, land: 440, lead: 1000, flip: 320, flipGap: 22 };

// deterministic pseudo-random in [0,1) — Math.random() would reshuffle on every replay
function hash01(i, salt) { const x = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453; return x - Math.floor(x); }

/* Geometry of the front sheet's cell grid, in figure coordinates. */
function archGeom() {
  const img = els["archive-img"], ai = rect(img);
  const cols = S.M.archive_side || 10, rows = S.M.archive_rows || cols;
  const cw = ai.w / cols, ch = ai.h / rows;
  return { ai, cols, rows, cw, ch, n: cols * rows,
           at: (k) => ({ x: ai.l + ((k % cols) + 0.5) * cw, y: ai.t + (Math.floor(k / cols) + 0.5) * ch }) };
}
function archCrop(elm, src, k, g) {
  elm.style.backgroundImage = `url("${src}")`;
  elm.style.backgroundRepeat = "no-repeat";
  elm.style.backgroundSize = `${g.cols * 100}% ${g.rows * 100}%`;
  elm.style.backgroundPosition =
    `${g.cols > 1 ? ((k % g.cols) / (g.cols - 1)) * 100 : 0}% ${g.rows > 1 ? (Math.floor(k / g.cols) / (g.rows - 1)) * 100 : 0}%`;
}
// The cells that take an arriving image, split by what kind it is: `pubs` are the recent
// publications (the featured one's cell always among them), `surf` are older entries resurfacing
// from under the pile. Both are exact fractions of the layer so the two together never cover all
// of it; ranking by hash rather than thresholding it keeps the counts steady across epochs.
function arrivalCells(archN, pubCell) {
  const nP = Math.max(1, Math.round(archN * GROW.pubFrac));
  const nS = Math.max(1, Math.round(archN * GROW.surfFrac));
  const rest = [];
  for (let k = 0; k < archN; k++) if (k !== pubCell) rest.push(k);
  rest.sort((p, q) => hash01(p, S.epoch + 1) - hash01(q, S.epoch + 1));
  return { pubs: [pubCell, ...rest.slice(0, nP - 1)], surf: rest.slice(nP - 1, nP - 1 + nS) };
}

// The node only shows a 50-image sample of the archive. Stand the rest of it up as a tight
// stack of sheets behind the front one (other real checkpoints of the same archive), so the
// Archive reads as a deep pile — like the Evaluation node's — rather than a lone grid.
function ensureArchiveBacking() {
  const back = els["arch-back"], img = els["archive-img"];
  if (!back || back.children.length || !img.offsetWidth) return;
  const L = img.offsetLeft, T = img.offsetTop, W = img.offsetWidth, H = img.offsetHeight;
  const cur = img.getAttribute("src");
  const others = (S.agents || []).map((x) => x.archive_image && relAsset(x.archive_image))
    .filter((s) => s && s !== cur);
  others.slice(0, ARCH_MAX).forEach((src, i) => {
    const d = i + 1;
    const s = document.createElement("img");
    s.className = "arch-sheet";
    s.style.cssText = `left:${L}px;top:${T}px;width:${W}px;height:${H}px;`;
    s.src = src; s.dataset.depth = String(d);
    back.appendChild(s); styleBacking(s, d);
  });
}

function setArchive(a) {
  if (!a.archive_image) return;
  const img = els["archive-img"], next = relAsset(a.archive_image);
  img.style.opacity = "1";
  if (img.getAttribute("src") === next) return;         // already standing -> nothing to do
  clearArchiveLayers(); img.src = next;
}

/* The archive grows — and by the time we get here it already has. The arriving images flew in with
   the publication and the layer they landed on turned over beneath them in the same wave (see
   deliverPublications); all this does is wait for that to settle before the next epoch samples. */
async function growArchive(next, a, tok) {
  if (!next || !next.archive_image) return;
  await (S.deliverP || Promise.resolve()); if (!alive(tok)) return;
  // a scrub/jump can land here with the batch never delivered — snap the sheet over in that case
  const img = els["archive-img"], nextSrc = relAsset(next.archive_image);
  if (img.getAttribute("src") !== nextSrc) img.src = nextSrc;
}

function pubCellIndex(a, archN) {
  const c = a.publication && a.publication.archive_cell;
  return (c && c.index != null) ? Math.min(c.index, archN - 1) : archN - 1;
}
function placeFresh(cell, d, g) {
  cell.style.opacity = "1";
  cell.style.width = g.cw + "px"; cell.style.height = g.ch + "px";
  cell.style.transform = `translate(${d.x}px,${d.y}px) translate(-50%,-50%)`;
}
// the recent arrivals are anchored in figure coordinates, so a resize has to re-place them
function relayoutFresh() {
  if (!S.fresh || !S.fresh.length || !els["archive-img"].offsetWidth) return;
  const g = archGeom();
  S.fresh.forEach(({ k, el }) => { if (el.isConnected) placeFresh(el, g.at(k), g); });
}

// The arrivals are crops of the incoming sheet at their own cells (and sit above #arch-front), so
// the new checkpoint slides in beneath them without a flicker — they just keep their gold ring.
function turnOverTopLayer(prevSrc, nextSrc, tok, opts) {
  const img = els["archive-img"], front = els["arch-front"];
  if (!front || !img.offsetWidth) { img.src = nextSrc; return Promise.resolve(); }
  const { delay = 0, duration = GROW.fade } = opts || {};
  const L = img.offsetLeft, T = img.offsetTop, W = img.offsetWidth, H = img.offsetHeight;
  const sheet = document.createElement("img");        // the outgoing top layer, held in place
  sheet.className = "arch-sheet";
  sheet.style.cssText = `left:${L}px;top:${T}px;width:${W}px;height:${H}px;`;
  sheet.src = prevSrc;
  front.appendChild(sheet);
  img.src = nextSrc;
  return img.decode().catch(() => {}).then(() => {
    if (!alive(tok)) { sheet.remove(); return; }
    pushArchiveBacking(prevSrc, L, T, W, H);           // buried before the fade, so no 2px pop
    return sheet.animate([{ opacity: 1 }, { opacity: 0 }],
      { duration: duration / S.speed, delay: delay / S.speed, easing: "ease", fill: "both" }).finished
      .then(() => sheet.remove()).catch(() => sheet.remove());
  });
}

function pushArchiveBacking(src, L, T, W, H) {
  const back = els["arch-back"]; if (!back) return;
  Array.from(back.children).forEach((s) => {           // everything already there sinks a step
    const d = (parseInt(s.dataset.depth, 10) || 0) + 1;
    if (d > ARCH_MAX) { s.remove(); return; }
    s.dataset.depth = String(d); styleBacking(s, d);
  });
  const s = document.createElement("img");
  s.className = "arch-sheet";
  s.style.cssText = `left:${L}px;top:${T}px;width:${W}px;height:${H}px;`;
  s.src = src; s.dataset.depth = "1";
  back.appendChild(s); styleBacking(s, 1);
}
// solid, not translucent: a buried sheet shows only the ARCH_D-px L of itself poking out
// past the sheet in front, which is what makes the pile read as paper rather than ghosts.
function styleBacking(s, d) {
  s.style.transform = `translate(${-ARCH_D * d}px,${-ARCH_D * d}px)`;
  s.style.zIndex = String(10 - d);                     // deeper sheets sit further back
}
function clearArchiveLayers() {
  ["arch-back", "arch-front", "arch-fly"].forEach((k) => { if (els[k]) els[k].innerHTML = ""; });
  S.fresh = []; S.deliverP = null;
}
function setRated(ev) {
  els.rated.innerHTML = "";
  ((ev && ev.cells) || []).forEach((c) => {
    const cell = document.createElement("div"); cell.className = "rated-cell";
    const img = document.createElement("img"); if (c.image) img.src = relAsset(c.image); cell.appendChild(img);
    if (c.title) { const t = document.createElement("div"); t.className = "rt-title"; t.textContent = c.title; cell.appendChild(t); }
    if (c.rating != null) { const s = document.createElement("div"); s.className = "rt-stars"; s.innerHTML = stars(c.rating); cell.appendChild(s); }
    els.rated.appendChild(cell);
  });
}
function stars(n) {
  n = Math.max(0, Math.min(5, Math.round(n))); let s = "";
  for (let i = 0; i < 5; i++) s += i < n ? '<span class="on">★</span>' : '<span class="off">★</span>';
  return s;
}
function renderStatic(ei) {
  const a = S.agents[ei]; if (!a) return;
  setArchive(a); setRated(a.evaluation);   // static frame: no growth animation
  buildDeck(a);
  // the static first frame shows the front draw already dealt, so its labels are visible
  if (S.deck.front) { S.deck.front._img.style.opacity = "1"; S.deck.front.classList.add("dealt"); }
  // the generation grid stays empty (hidden) until Evolution actually spawns candidates —
  // no static placeholder image in the node before it animates.
  els["gen-grid"].style.opacity = "0";
  clearActive(); S.edges.clear(); applyEdges();
  if (els.progress) els.progress.textContent = `epoch ${ei + 1}/${S.agents.length} · ${a.title || ""}`;
}
function setPhase(a, phase) {
  S.phase = phase;
  if (els.progress) els.progress.textContent = `epoch ${S.epoch + 1}/${S.agents.length} · ${a.title || ""} · ${phase}`;
}

/* ---------------- selection outlines ---------------- */
function cellRect(i) {
  const R = S.M.grid_rows || 3, Cc = S.M.grid_cols || 5;
  const t = S.M.grid_thumb || 128, m = S.M.grid_margin || 10;
  const W = Cc * t + (Cc + 1) * m, H = R * t + (R + 1) * m;
  const r = Math.floor(i / Cc), c = i % Cc;
  return { left: (m + c * (t + m)) / W * 100, top: (m + r * (t + m)) / H * 100,
           w: t / W * 100, h: t / H * 100 };
}
function renderSelection(sel) {
  const ov = els["sel-overlay"]; ov.innerHTML = "";
  if (!sel) return;
  sel.forEach((i, k) => {
    const d = document.createElement("div"); d.className = "sel-cell";
    const r = cellRect(i);
    d.style.left = r.left + "%"; d.style.top = r.top + "%";
    d.style.width = r.w + "%"; d.style.height = r.h + "%";
    d.style.animationDelay = (k * 60) + "ms";
    ov.appendChild(d);
  });
}
function cellFig(idx, gw) {
  const rc = cellRect(idx);
  return { cx: gw.l + ((rc.left + rc.w / 2) / 100) * gw.w, cy: gw.t + ((rc.top + rc.h / 2) / 100) * gw.h,
           w: (rc.w / 100) * gw.w, h: (rc.h / 100) * gw.h };
}
function gridCropEl(src, idx, rows, cols, selected) {
  const e = document.createElement("div");
  e.className = "evo-cell" + (selected ? " sel" : "");
  e.style.backgroundImage = `url("${src}")`;
  e.style.backgroundSize = `${cols * 100}% ${rows * 100}%`;
  const r = Math.floor(idx / cols), c = idx % cols;
  e.style.backgroundPosition = `${cols > 1 ? (c / (cols - 1)) * 100 : 0}% ${rows > 1 ? (r / (rows - 1)) * 100 : 0}%`;
  return e;
}

/* ---------------- reset ---------------- */
function resetOverlays() {
  // NB: #arch-fly is NOT cleared here — the recent publications on the Archive's top layer
  // deliberately outlive the epoch that put them there (the next draws sample off them).
  ["samp-fly", "evo-fly", "rate-fly", "rate-stack", "rate-stars"].forEach((k) => {
    const e = els[k]; if (e) { e.getAnimations && e.getAnimations().forEach((an) => an.cancel()); e.innerHTML = ""; }
  });
  ["pub-clone", "pub-title", "pub-rider", "branch-clone", "branch-title"].forEach((k) => {
    const e = els[k]; if (!e) return;
    e.getAnimations && e.getAnimations().forEach((an) => an.cancel());
    e.style.transition = "none"; e.style.opacity = "0"; e.style.display = "none";
  });
  renderSelection(null);
  setLoopCount(((S.agents[S.epoch] || {}).n_generations) || 20);
  const g = els["gen-grid"];
  g.getAnimations && g.getAnimations().forEach((an) => an.cancel());
  g.style.transition = ""; g.style.opacity = "0";   // empty until Evolution spawns
}
function resetAll() { resetOverlays(); clearActive(); clearDeck(); clearArchiveLayers(); }

/* ---------------- arrows (SVG, routed like system_overview.tex) ---------------- */
const EDGES = ["sample", "feed", "publish", "ratefeed", "rate", "pipe1", "pipe2", "pipe3", "loop"];
const EDGE_COLOR = {
  sample: "feed", feed: "feed", publish: "feed", ratefeed: "feed",
  rate: "rate", pipe1: "pipe", pipe2: "pipe", pipe3: "pipe", loop: "loop",
};
function buildArrows() {
  const svg = els.arrows;
  svg.innerHTML = `<defs>${["feed", "rate", "pipe", "loop"].map((c) =>
    `<marker id="ah-${c}" markerWidth="7" markerHeight="7" refX="5.4" refY="3" orient="auto">
       <path d="M0,0 L6,3 L0,6 Z" fill="${markerColor(c)}"></path></marker>`).join("")}</defs>`;
  for (const id of EDGES) {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("id", "e-" + id);
    p.setAttribute("fill", "none");
    p.setAttribute("stroke", cssVar(EDGE_COLOR[id]));
    p.setAttribute("stroke-width", id.startsWith("pipe") || id === "loop" ? "1.5" : "1.9");
    p.setAttribute("marker-end", `url(#ah-${EDGE_COLOR[id]})`);
    p.setAttribute("opacity", id === "feed" ? "0" : "0.42");
    p.style.transition = "opacity .38s ease";   // smooth fade in/out (esp. the feed arrow)
    if (id === "loop") p.style.strokeDasharray = "4 3";
    svg.appendChild(p);
  }
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("id", "loop-label"); t.setAttribute("font-style", "italic");
  t.setAttribute("font-size", "12"); t.setAttribute("fill", cssVar("loop"));
  t.setAttribute("font-family", "var(--cm)"); t.textContent = "×20";
  els.arrows.appendChild(t);
  layoutArrows();
}
function markerColor(c) { return { feed: "#bf6000", rate: "#0f8a8a", pipe: "#404040", loop: "#00008c" }[c]; }
function cssVar(c) { return { feed: "var(--feed)", rate: "var(--rate)", pipe: "var(--pipe)", loop: "var(--vlm-draw)" }[c]; }
function rect(elm) {
  const f = els.fig.getBoundingClientRect(); const r = elm.getBoundingClientRect();
  return { l: r.left - f.left, r: r.right - f.left, t: r.top - f.top, b: r.bottom - f.top,
           cx: r.left - f.left + r.width / 2, cy: r.top - f.top + r.height / 2, w: r.width, h: r.height };
}
function setPath(id, d) { const p = document.getElementById("e-" + id); if (p) p.setAttribute("d", d); }
function reportHeight() {
  if (window.parent === window) return;
  const h = Math.ceil(document.getElementById("app").getBoundingClientRect().height);
  try { window.parent.postMessage({ sysov: "height", h }, "*"); } catch (e) {}
}
window.addEventListener("resize", reportHeight);
[200, 600, 1200].forEach((ms) => setTimeout(reportHeight, ms));
document.addEventListener("click", (e) => {
  if (window.parent === window) return;
  if (e.target.closest(".controls")) return;
  try { window.parent.postMessage({ sysov: "collapse" }, "*"); } catch (err) {}
});

function layoutArrows() {
  reportHeight();
  if (!els.arrows.children.length) return;
  const A = rect(els["p-archive"]), Sm = rect(els["p-sampling"]),
        Ev = rect(els["p-evolution"]), El = rect(els["p-evaluation"]),
        sel = rect(els.sel), mut = rect(els.mut), gw = rect(els["gen-wrap"]),
        rat = rect(els.rat), rated = rect(els.rated);
  const midX = (A.r + Sm.l) / 2;
  const gutter = Math.max(8, Ev.l - 42);
  // 1) sample: Archive.east → Sampling.west
  setPath("sample", `M${A.r},${A.cy} L${Sm.l - 2},${A.cy}`);
  // 2) feed (evolver's draw): originates at the EVO draw card's west edge (so it follows the
  //    card front↔back as the deck shuffles) → left to the gap → down → into the VLM breeder.
  const evoCard = [S.deck.front, S.deck.back].find((c) => c && c._spec && c._spec.kind === "evo");
  const fc = evoCard ? rect(evoCard) : Sm;
  // leave from LOW on the card (not its centre) so this run never collides with the
  // Archive→Sampling edge, which sits at the archive's mid-height.
  const feedX = Math.min(fc.l, midX), feedY = fc.t + fc.h * 0.78;
  setPath("feed", `M${fc.l},${feedY} L${feedX},${feedY} L${feedX},${sel.t - 16} L${sel.cx},${sel.t - 16} L${sel.cx},${sel.t - 2}`);
  // 3) publish: VLM-selection.west → left gutter → up → Archive.west (below mid)
  setPath("publish", `M${sel.l},${sel.cy} L${gutter},${sel.cy} L${gutter},${A.cy + 14} L${A.l - 2},${A.cy + 14}`);
  // 4) ratefeed: Sampling.south → VLM rating top
  setPath("ratefeed", `M${rat.cx},${Sm.b} L${rat.cx},${rat.t - 2}`);
  // 5) rate: Evaluation.south → down → left gutter → up → Archive.west (above mid)
  setPath("rate", `M${El.cx},${El.b} L${El.cx},${El.b + 16} L${gutter - 16},${El.b + 16} L${gutter - 16},${A.cy - 14} L${A.l - 2},${A.cy - 14}`);
  // pipes inside Evolution / Evaluation
  setPath("pipe1", `M${sel.cx},${sel.b} L${mut.cx},${mut.t - 1}`);
  setPath("pipe2", `M${mut.cx},${mut.b} L${gw.cx},${gw.t - 1}`);
  setPath("pipe3", `M${rat.cx},${rat.b} L${rated.cx},${rated.t - 1}`);
  // ×20 back-edge — the bow apex and the "×20" label must both stay INSIDE the node's
  // right edge (Ev.r), so cap the bow to leave a margin and clamp the label's right edge.
  const bow = Math.max(20, (Ev.r - gw.r) - 14);
  setPath("loop",
    `M${gw.r},${gw.cy} C${gw.r + bow},${gw.cy} ${gw.r + bow},${sel.cy} ${gw.r},${sel.cy} L${sel.r + 2},${sel.cy}`);
  const lbl = document.getElementById("loop-label");
  if (lbl) { lbl.setAttribute("x", Math.min(gw.r + bow * 0.72 + 3, Ev.r - 24)); lbl.setAttribute("y", (gw.cy + sel.cy) / 2 + 3); }
}

/* ---------------- edge state (additive) ---------------- */
function addEdges(ids) { ids.forEach((id) => S.edges.add(id)); applyEdges(); }
function removeEdges(ids) { ids.forEach((id) => S.edges.delete(id)); applyEdges(); }
function applyEdges() {
  for (const id of EDGES) {
    const p = document.getElementById("e-" + id); if (!p) continue;
    const on = S.edges.has(id);
    // the feed arrow into Evolution fully fades away when idle (it only means something
    // while a parent is branching); the other edges keep a faint circuit-diagram backdrop.
    const rest = id === "feed" ? "0" : "0.42";
    p.setAttribute("opacity", on ? "1" : rest);
    p.style.animation = on ? "flow .6s linear infinite" : "none";
  }
}
function pulseLoop() {
  const p = document.getElementById("e-loop");
  if (p) { p.setAttribute("opacity", "1"); p.style.animation = "flow .5s linear infinite"; }
}
/* The ×20 label is a caption on the back-edge: it states how many generations the loop runs for
   this epoch, and stays fixed there for the whole run. */
function setLoopCount(n) {
  const lbl = document.getElementById("loop-label");
  if (lbl) lbl.textContent = "×" + Math.max(0, Math.round(n));
}
function clearActive() {
  ["p-archive", "p-sampling", "p-evolution", "p-evaluation"].forEach((k) => { els[k].classList.remove("active"); els[k].classList.remove("grow"); });
  els.sel.classList.remove("hot"); els.mut.classList.remove("hot"); els.rat.classList.remove("hot");
  sampGlow(false);
}
// green-glow the Sampling node's FRONT draw card (the one holding the sampled images),
// never the outer panel or the waiting back card. Always targets the current front.
function sampGlow(on) {
  const d = S.deck || {};
  if (d.back) d.back.classList.remove("samp-active");
  if (d.front) d.front.classList.toggle("samp-active", !!on);
}

/* ---------------- helpers ---------------- */
function relAsset(p) {
  const base = MANIFEST_URL.includes("/") ? MANIFEST_URL.slice(0, MANIFEST_URL.lastIndexOf("/") + 1) : "";
  const url = /^https?:|^\//.test(p) ? p : base + p;
  // cache-bust: the archive/sample PNGs are regenerated under the SAME filenames, so
  // append the manifest's asset_v (bump it in curated_manifest.json when assets change)
  // to force browsers/CDN to fetch fresh copies instead of a stale layout.
  const v = S.M && S.M.asset_v;
  return v ? url + (url.includes("?") ? "&" : "?") + "v=" + v : url;
}
const imgReady = (img) => (img && img.complete && img.naturalWidth)
  ? Promise.resolve()
  : new Promise((res) => { if (!img) return res();
      img.addEventListener("load", res, { once: true });
      img.addEventListener("error", res, { once: true }); });
function lockGridAspect() {
  const a = S.agents.find((x) => x.generations && x.generations[0]);
  if (!a) return;
  const probe = new Image();
  probe.onload = () => { if (probe.naturalWidth) els["gen-wrap"].style.setProperty("--grid-ar", probe.naturalWidth + " / " + probe.naturalHeight); layoutArrows(); };
  probe.src = relAsset(a.generations[0].grid);
}

const _st = document.createElement("style");
_st.textContent = "@keyframes flow { to { stroke-dashoffset: -7; } } #e-loop { stroke-dasharray:4 3; }";
document.head.appendChild(_st);
