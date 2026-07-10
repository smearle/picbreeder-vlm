(function () {
  if (typeof newPopulation !== 'function') return;     // needs breed/cppn.js
  var grid = document.getElementById('cxbGrid'), statusEl = document.getElementById('cxbStatus');
  var evolveBtn = document.getElementById('cxbEvolve'), undoBtn = document.getElementById('cxbUndo'),
      resetBtn = document.getElementById('cxbReset'), newParentBtn = document.getElementById('cxbNewParent'),
      publishBtn = document.getElementById('cxbPublish'), pubRow = document.getElementById('cxbPublished');
  if (!grid) return;
  // The candidate grid is CSS auto-fill (column count depends on the card's width),
  // so a fixed POP almost always leaves a ragged last row (one empty cell on a
  // laptop). TARGET is the rough tile count we aim for; POP is recomputed from the
  // live column count so the population always fills whole rows. Lineages don't mind
  // a generation-to-generation size change, so we just re-fit on resize.
  var TARGET = 15, RES = 128, STRENGTH = 0.5;
  var POP = TARGET;
  var pop, generation = 0, selected = new Set(), cells = [], hist = [];

  // Number of columns the grid currently resolves to. auto-fill keeps empty tracks,
  // so getComputedStyle reports one length per column even before cells are added.
  function gridCols() {
    var tpl = (getComputedStyle(grid).gridTemplateColumns || '').trim();
    if (!tpl || tpl === 'none') return 1;
    return Math.max(1, tpl.split(/\s+/).length);
  }
  // Round POP to a whole number of rows nearest TARGET tiles. Returns whether it changed.
  function recomputePop() {
    var cols = gridCols();
    var rows = Math.max(1, Math.round(TARGET / cols));
    var next = cols * rows;
    if (next === POP) return false;
    POP = next; return true;
  }
  // Seed genome: an ANCESTOR of the apple explored in the two CPPN figures above —
  // generation 140 of the apple's own Picbreeder lineage (series 5736, final = gen 320),
  // a green stem-and-leaf apple that predates the red fruit. Generation 0 is its mutated
  // offspring rather than random noise, so the breeder picks up the apple's lineage just
  // before it ripened and re-discovers the apple (and elsewhere) from there. Built by
  // tools/build_apple_ancestor_seed.py. null until it loads (and on failure) — reset()
  // then falls back to a fresh random scaffold population.
  var seedGenome = null;
  var SEED_URL = 'breed/data/apple_ancestor.json';

  // Transient hint / publish-result line above the grid (no "Generation N" caption).
  // Empty by default — the shortcuts are spelled out in the figure caption — and its
  // height is reserved in CSS. Messages may carry a link (Published! …), hence innerHTML.
  function setStatus(msg, neutral) {
    statusEl.innerHTML = msg || '';
    statusEl.classList.toggle('hint', !!msg && !neutral);   // red 'hint' for warnings; neutral grey otherwise
  }
  function buildGrid() {
    grid.innerHTML = ''; cells.length = 0;
    for (var i = 0; i < POP; i++) {
      (function (idx) {
        var cell = document.createElement('div'); cell.className = 'bcell';
        var cv = document.createElement('canvas'); cv.width = cv.height = RES;
        cell.appendChild(cv); grid.appendChild(cell);
        cell.addEventListener('click', function () { toggleSelect(idx); });
        cells.push({ cell: cell, ctx: cv.getContext('2d') });
      })(i);
    }
  }
  function toggleSelect(i) {
    if (selected.has(i)) { selected.delete(i); cells[i].cell.classList.remove('sel'); }
    else { selected.add(i); cells[i].cell.classList.add('sel'); }
    // The Space shortcut is advertised permanently in the figure caption, so no
    // transient status nudge is needed the first time a tile is selected.
    setStatus();
  }
  function clearSelection() { selected.clear(); cells.forEach(function (c) { c.cell.classList.remove('sel'); }); }

  var token = 0;
  function renderAll() {                  // time-sliced paint so a big grid never blocks the page
    var my = ++token, i = 0;
    (function step() {
      if (my !== token) return;
      var t0 = performance.now();
      while (i < POP && performance.now() - t0 < 12) {
        cells[i].ctx.putImageData(renderGenome(pop.genomes[i], RES, true), 0, 0); i++;
      }
      if (i < POP) requestAnimationFrame(step);
    })();
  }
  function pushHist() { hist.push({ genomes: pop.genomes.map(genomeToJSON), ctr: pop.ctr, gen: generation }); undoBtn.disabled = hist.length <= 1; }

  function evolve() {                      // mutation + crossover from the selected tiles (NEAT reproduce)
    if (!selected.size) { setStatus('Pick at least one tile (click it), then Evolve.'); return; }
    var parents = []; selected.forEach(function (i) { parents.push(pop.genomes[i]); });
    pop.genomes = reproduce(parents, pop, POP, STRENGTH);
    generation++; clearSelection(); pushHist(); renderAll(); setStatus();
  }
  function undo() {                        // step back one generation
    if (hist.length <= 1) return;
    hist.pop(); var e = hist[hist.length - 1];
    pop = new Population(); pop.ctr = e.ctr; pop.genomes = e.genomes.map(genomeFromJSON);
    generation = e.gen; clearSelection(); renderAll(); setStatus(); undoBtn.disabled = hist.length <= 1;
  }
  function freshPop() {                    // gen-0: mutated offspring of the apple, else random
    if (!seedGenome) return newPopulation(POP);
    var p = new Population();
    // advance the node-key counter past the apple's highest key so add-node mutations
    // mint fresh keys instead of colliding with the seed's existing nodes.
    p.ctr = 3; seedGenome.nodes.forEach(function (_n, k) { if (k + 1 > p.ctr) p.ctr = k + 1; });
    p.genomes = [];
    for (var i = 0; i < POP; i++) { var c = cloneGenome(seedGenome); mutate(c, p, STRENGTH); p.genomes.push(c); }
    return p;
  }
  function reset() {                       // back to generation 0
    pop = freshPop(); generation = 0; clearSelection(); hist = []; pushHist(); renderAll(); setStatus();
  }

  // ---- New parent: branch from a random human-archive publication -------------
  // The human Picbreeder archive ships its genomes alongside the rest of its data on
  // HF (site/human/genomes.json.gz, {id: genomeJSON} — the same shape genomeFromJSON
  // consumes everywhere else). Fetch it once, pick a random publication, make it the
  // gen-0 seed, and reset() so the grid fills with its mutated offspring to evolve on.
  function archiveBase() {
    var o = new URLSearchParams(location.search).get('archiveBase');
    if (o && !/^(https?:)?\/\//.test(o) && o !== '.') o = 'http://' + o;
    return (o || 'https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive/resolve/main').replace(/\/$/, '');
  }
  // One shared in-flight promise so the idle preload and a New-parent click never
  // double-fetch; cleared on failure so a later click can retry (e.g. once the HF
  // dataset goes public). The .gz is a few hundred KB, so warming it ahead of the
  // first click (see preload at the bottom) makes New parent feel instant.
  var humanGenomesP = null;
  function fetchHumanGenomes() {
    if (!humanGenomesP) {
      humanGenomesP = fetch(archiveBase() + '/site/human/genomes.json.gz').then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        var stream = r.body.pipeThrough(new DecompressionStream('gzip'));
        return new Response(stream).text();
      }).then(function (t) { return JSON.parse(t); })
        .catch(function (e) { humanGenomesP = null; throw e; });
    }
    return humanGenomesP;
  }
  // Spinner overlay over the grid while a parent loads; the cells grey out beneath it.
  // buildGrid() wipes the grid (and the spinner with it), so re-attach on demand.
  var spinner = null;
  function setLoading(on) {
    grid.classList.toggle('loading', on);
    if (on) {
      if (!spinner || spinner.parentNode !== grid) {
        spinner = document.createElement('div'); spinner.className = 'cxb-spin'; grid.appendChild(spinner);
      }
      spinner.style.display = '';
    } else if (spinner) {
      spinner.style.display = 'none';
    }
  }
  function newParent() {
    setStatus(''); setLoading(true);
    fetchHumanGenomes().then(function (gm) {
      var ids = Object.keys(gm);
      if (!ids.length) { setLoading(false); setStatus('The human archive is empty.'); return; }
      var id = ids[Math.floor(Math.random() * ids.length)];
      seedGenome = genomeFromJSON(gm[id]);
      reset();                              // re-seeds generation 0 from the new parent
      setLoading(false);
    }).catch(function (e) {
      setLoading(false);
      setStatus('Couldn’t load the human archive (' + e.message + ').');
    });
  }

  // ---- Published row: your session's publications, above the console -----------
  // Each Publish drops a half-scale thumbnail here (newest on the right). Clicking a
  // thumbnail makes that genome the parent and re-seeds the candidate grid from it.
  function addPublished(genome, png, title) {
    if (!pubRow) return;
    pubRow.hidden = false;
    var g = cloneGenome(genome);          // snapshot so later evolution can't mutate it
    var item = document.createElement('div');
    item.className = 'cxb-pub-item';
    item.title = (title ? '“' + title + '”—' : '') + 'click to breed from this';
    var img = document.createElement('img');
    img.src = png; img.alt = title || 'Published image';
    item.appendChild(img);
    item.addEventListener('click', function () { useAsParent(g, item); });
    pubRow.appendChild(item);
    pubRow.scrollLeft = pubRow.scrollWidth;  // keep the newest in view on the right
  }
  function useAsParent(genome, item) {
    seedGenome = genome; reset();
    for (var k = 0; k < pubRow.children.length; k++) pubRow.children[k].classList.remove('active');
    if (item) item.classList.add('active');
  }

  // ---- Publish the selected tile (local, this-session row only) ----------------
  // Sharing to an online gallery isn't available, so Publish just drops the selected
  // tile into your row above the console — no prompts, no network. Click a collected
  // thumbnail to breed a fresh candidate grid from it.
  function publish() {
    if (selected.size !== 1) { setStatus('Select exactly one tile as your representative, then Publish.'); return; }
    var i = selected.values().next().value, HI = 512;
    var genomeObj = pop.genomes[i];
    var c = document.createElement('canvas'); c.width = c.height = HI;
    c.getContext('2d').putImageData(renderGenome(genomeObj, HI, true), 0, 0);
    var png = c.toDataURL('image/png');
    addPublished(genomeObj, png, '');       // collect into your row (this session only)
    setStatus('Added to your images above—click one to breed from it.', true);
  }

  evolveBtn.addEventListener('click', evolve);
  undoBtn.addEventListener('click', undo);
  resetBtn.addEventListener('click', reset);
  newParentBtn.addEventListener('click', newParent);
  publishBtn.addEventListener('click', publish);

  // Space/Enter = evolve selected · z or u = undo · r = reset · p = new parent. Scoped to keydowns
  // inside the (focusable) card so we never hijack page scrolling/typing elsewhere;
  // clicking a tile focuses the card, and we leave space/enter to natively activate
  // a focused Undo/Reset/Evolve button.
  var cardEl = document.getElementById('cppnxBreed');
  cardEl.addEventListener('keydown', function (e) {
    if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
    var onBtn = /^(BUTTON|INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName);
    var k = e.key;
    if (k === 'Enter' || k === ' ' || k === 'Spacebar') {
      if (onBtn) return;                 // let space/enter activate a focused button
      e.preventDefault(); evolve();
    } else if (k === 'z' || k === 'Z' || k === 'u' || k === 'U') {
      e.preventDefault(); undo();
    } else if (k === 'r' || k === 'R') {
      e.preventDefault(); reset();
    } else if (k === 'p' || k === 'P') {
      e.preventDefault(); newParent();
    }
  });

  // Build the (empty) grid and show the spinner while the tiny apple-ancestor seed
  // loads — then seed generation 0 from it. We deliberately DON'T paint a random
  // generation first: the seed file is local and quick, so waiting avoids a flash of
  // random CPPNs being replaced by apple variants. On failure, fall back to random.
  recomputePop(); buildGrid(); setLoading(true);

  // Keep the grid rectangular across viewport changes: when the column count shifts,
  // re-fit the current generation to the new POP (trim, or clone-and-mutate extras),
  // rebuild the cells, and restart history at the resized generation so Undo stays
  // consistent. Debounced so a drag-resize doesn't thrash.
  function applySelClasses() { selected.forEach(function (i) { if (cells[i]) cells[i].cell.classList.add('sel'); }); }
  var resizeT = null;
  function onResize() {
    if (!recomputePop() || !pop) return;
    var g = pop.genomes;
    if (g.length > POP) g.length = POP;
    else while (g.length < POP) { var c = cloneGenome(g[Math.floor(Math.random() * g.length)] || g[0]); mutate(c, pop, STRENGTH); g.push(c); }
    selected.forEach(function (i) { if (i >= POP) selected.delete(i); });
    hist = []; pushHist();
    buildGrid(); applySelClasses(); renderAll();
  }
  window.addEventListener('resize', function () { clearTimeout(resizeT); resizeT = setTimeout(onResize, 150); });

  fetch(SEED_URL).then(function (r) { return r.json(); }).then(function (d) {
    if (d && d.nodes) seedGenome = genomeFromJSON(d);
    reset(); setLoading(false);
  }).catch(function () { reset(); setLoading(false); /* random fallback */ });

  // Warm the human archive once the page's visible assets are in, so the first
  // New-parent click is instant instead of waiting on the gzipped fetch.
  function schedulePreload() {
    if (window.requestIdleCallback) requestIdleCallback(function () { fetchHumanGenomes().catch(function () {}); }, { timeout: 5000 });
    else setTimeout(function () { fetchHumanGenomes().catch(function () {}); }, 2500);
  }
  if (document.readyState === 'complete') schedulePreload();
  else window.addEventListener('load', schedulePreload);
})();
