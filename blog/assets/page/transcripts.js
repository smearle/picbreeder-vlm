/* Transcript newsreel viewer.
 *
 * Parallel columns (one per agent) of what each VLM agent SAW (its branching
 * archive sample, then each generation's candidate grid) and SAID (rationale +
 * the cells it picked), branching -> publication.
 *
 * A run can hold thousands of agents, so the wall streams as a vertical masonry:
 * N columns wide (N = however many fit the pane), agents packed top-to-bottom and
 * appended in batches as the user scrolls DOWN — no pages to step through — each
 * agent lazy-loading its atlas when it nears view. Search / filter / sort narrow
 * the set and rebuild the wall from the top.
 *
 * Data (tools/build_transcript_data.py):
 *   <base>/manifest.json          -> { runs:[{run,label,arc,model,seed,hf?,disabled?}] }
 *   <base>/<run>/index.json       -> { run,arc,model,seed,n_agents,total_agents,agents:[{id,title,n_gen,published}] }
 *   <base>/<run>/<agent>/transcript.json + atlas.webp
 * Local (bundled) runs resolve under ../assets/transcripts/; runs flagged hf:true
 * resolve from the HF dataset (archiveBase proxy honoured for the private repo).
 */
(function () {
  'use strict';

  var qs = new URLSearchParams(location.search);
  var ARCHIVE_BASE = (qs.get('archiveBase') || '').replace(/\/$/, '');
  var HF_BASE = (ARCHIVE_BASE ||
    'https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive/resolve/main');
  var LOCAL_BASE = '../assets/transcripts/';   // viewer is /transcripts/index.html
  var COLW = 220;                              // px width a grid block scales to
  var COLSTEP = 260;                           // px a column occupies — sets how many columns fit the pane
  var LOAD_MARGIN = 800;                       // append the next batch this many px before the bottom

  var wall, sel, metaEl, searchEl, pubEl, sortEl, pager;
  var STATE = { run: null, hf: false, all: [], view: [], rendered: 0, total: 0, cols: 1, colEls: [] };
  var colObserver = null;

  function el(t, c, h) { var e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; }
  function fetchJSON(u) { return fetch(u).then(function (r) { if (!r.ok) throw new Error(r.status + ' ' + u); return r.json(); }); }
  function escapeHTML(s) { var d = el('div'); d.textContent = s == null ? '' : s; return d.innerHTML; }

  function runBase(run, hf) { return (hf ? HF_BASE + '/transcripts/' : LOCAL_BASE) + run + '/'; }

  // ---- atlas-block geometry: cell i at (margin + col*(cell+margin), ...) ----
  function cellRect(b, i) {
    var c = b.cell, m = b.margin, col = i % b.cols, row = Math.floor(i / b.cols);
    return { x: m + col * (c + m), y: m + row * (c + m), w: c, h: c };
  }
  function drawBlock(atlas, block, picks, dispW) {
    var scale = dispW / block.w;
    var cv = el('canvas');
    cv.width = Math.round(block.w * scale);
    cv.height = Math.round(block.h * scale);
    var ctx = cv.getContext('2d');
    ctx.drawImage(atlas, block.x, block.y, block.w, block.h, 0, 0, cv.width, cv.height);
    if (picks && picks.length) {
      ctx.lineWidth = Math.max(2, 3 * scale); ctx.strokeStyle = '#FF6C00';
      picks.forEach(function (i) {
        var r = cellRect(block, i);
        ctx.strokeRect(r.x * scale + ctx.lineWidth / 2, r.y * scale + ctx.lineWidth / 2,
                       r.w * scale - ctx.lineWidth, r.h * scale - ctx.lineWidth);
      });
    }
    return cv;
  }
  function picksLabel(a) { return (a && a.length) ? 'picked ' + a.map(function (i) { return '#' + i; }).join(', ') : ''; }

  // ---- timeline cards ----
  function renderStep(atlas, step) {
    var card = el('div', 'tx-card');
    if (step.kind === 'branching') {
      card.appendChild(el('div', 'tx-step', 'Branching &middot; archive sample'));
      if (step.block) {
        var wrap = el('div', 'tx-grid-wrap');
        wrap.appendChild(drawBlock(atlas, step.block, step.selected_indices || [], COLW));
        wrap.addEventListener('click', function () {
          openLightbox(atlas, step.block, step.selected_indices || [], 'Branching',
                       'Archive sample shown before generation 0', step.rationale); });
        card.appendChild(wrap);
      } else {
        card.appendChild(el('div', 'tx-empty', step.empty_text ||
          'The archive was empty—this agent started from a fresh random population.'));
      }
      if (step.rationale) card.appendChild(saidEl(step.rationale, null));
      return card;
    }
    if (step.kind === 'publication') {
      var pub = el('div', 'tx-pub');
      pub.appendChild(el('div', 'tx-pub lab', 'Published &middot; gen ' + step.generation));
      pub.appendChild(el('div', 'ptitle', escapeHTML(step.title || '')));
      if (step.reason) pub.appendChild(el('div', 'preason', escapeHTML(step.reason)));
      card.appendChild(pub);
      return card;
    }
    card.appendChild(el('div', 'tx-step', 'Generation ' + step.generation));
    var w = el('div', 'tx-grid-wrap');
    w.appendChild(drawBlock(atlas, step.block, step.selected || [], COLW));
    w.addEventListener('click', function () {
      openLightbox(atlas, step.block, step.selected || [], 'Generation ' + step.generation,
                   picksLabel(step.selected), step.rationale); });
    card.appendChild(w);
    card.appendChild(saidEl(step.rationale, step.selected));
    return card;
  }
  function saidEl(text, picks) {
    var s = el('div', 'tx-said');
    if (picks && picks.length) s.appendChild(el('span', 'picks', picksLabel(picks) + '—'));
    s.appendChild(document.createTextNode(text || '(no rationale recorded)'));
    return s;
  }

  // ---- lightbox (click anywhere to zoom back out) ----
  var lb, lbInner;
  function openLightbox(atlas, block, picks, title, sub, rationale) {
    lbInner.innerHTML = '';
    lbInner.appendChild(drawBlock(atlas, block, picks, Math.min(880, block.w * 2.4)));
    var side = el('div', 'tx-lb-side');
    side.appendChild(el('div', 'aid', escapeHTML(sub || '')));
    side.appendChild(el('div', 'ttl', escapeHTML(title)));
    if (picks && picks.length) side.appendChild(el('div', 'picks', picksLabel(picks)));
    side.appendChild(el('div', 'said', escapeHTML(rationale || '(no rationale recorded)')));
    lbInner.appendChild(side);
    lb.classList.add('on');
  }
  function closeLightbox() { lb.classList.remove('on'); lbInner.innerHTML = ''; }

  // ---- lazy-load one column's atlas + render its timeline ----
  function loadColumn(col) {
    var body = col.querySelector('.tx-col-body');
    var agent = JSON.parse(col.dataset.agent);
    body.innerHTML = '<div class="tx-loading">loading…</div>';
    var base = runBase(STATE.run, STATE.hf) + agent.id + '/';
    fetchJSON(base + 'transcript.json').then(function (tx) {
      var atlas = new Image(); atlas.decoding = 'async';
      atlas.onload = function () {
        body.innerHTML = '';
        tx.timeline.forEach(function (s) { body.appendChild(renderStep(atlas, s)); });
      };
      atlas.onerror = function () { body.innerHTML = '<div class="tx-empty">atlas failed to load</div>'; };
      atlas.src = base + tx.atlas.file;
    }).catch(function (e) { body.innerHTML = '<div class="tx-empty">' + escapeHTML(String(e.message || e)) + '</div>'; });
  }

  // ---- vertical masonry: N columns wide, agents packed top-to-bottom, appended
  //      in batches as the user scrolls down (no pages) ----
  function colCount() { return Math.max(1, Math.floor((wall.clientWidth || 1000) / COLSTEP)); }
  function batchSize() { return STATE.cols * 2; }   // ~two rows of agents per batch

  function makeCard(a) {
    var card = el('div', 'tx-col');
    card.dataset.agent = JSON.stringify(a);
    var head = el('div', 'tx-col-head');
    head.appendChild(el('div', 'aid', escapeHTML(a.id.replace('agent_', 'agent ')) +
      (a.n_gen != null ? ' · ' + a.n_gen + ' gens' : '')));
    head.appendChild(el('div', 'title', escapeHTML(a.title || a.id)));
    card.appendChild(head);
    card.appendChild(el('div', 'tx-col-body', '<div class="tx-loading">scroll to load…</div>'));
    return card;
  }
  function appendBatch() {
    var end = Math.min(STATE.view.length, STATE.rendered + batchSize());
    for (var i = STATE.rendered; i < end; i++) {
      var card = makeCard(STATE.view[i]);
      STATE.colEls[i % STATE.cols].appendChild(card);   // round-robin keeps sort order left→right
      colObserver.observe(card);
    }
    STATE.rendered = end;
    updateStatus();
  }
  // Top up whenever the scroll position comes within LOAD_MARGIN of the bottom.
  function pump() {
    if (!wall.clientWidth || !wall.clientHeight || !STATE.colEls.length) return;
    var guard = 0;
    while (STATE.rendered < STATE.view.length &&
           wall.scrollTop + wall.clientHeight > wall.scrollHeight - LOAD_MARGIN) {
      appendBatch();
      if (++guard > 500) break;   // safety against a zero-growth layout
    }
  }
  var pumpQueued = false;
  function schedulePump() {
    if (pumpQueued) return;
    pumpQueued = true;
    requestAnimationFrame(function () { pumpQueued = false; pump(); });
  }
  // Resizing can change how many columns fit; a changed count needs a full re-lay-out.
  function onResize() {
    if (STATE.view.length && colCount() !== STATE.cols) renderView();
    else schedulePump();
  }

  // rebuild the wall from the top for the current view (after search/filter/sort/run/resize)
  function renderView() {
    if (colObserver) colObserver.disconnect();
    wall.innerHTML = '';
    wall.scrollLeft = 0; wall.scrollTop = 0;
    STATE.rendered = 0; STATE.colEls = [];
    if (!STATE.view.length) {
      wall.innerHTML = '<div class="tx-empty" style="margin:24px">No agents match.</div>';
      updateStatus(); return;
    }
    STATE.cols = colCount();
    for (var c = 0; c < STATE.cols; c++) {
      var colEl = el('div', 'tx-masonry-col');
      wall.appendChild(colEl);
      STATE.colEls.push(colEl);
    }
    colObserver = new IntersectionObserver(function (ents) {
      ents.forEach(function (en) {
        if (en.isIntersecting && !en.target.dataset.loaded) {
          en.target.dataset.loaded = '1'; loadColumn(en.target); colObserver.unobserve(en.target);
        }
      });
    }, { root: wall, rootMargin: '600px 0px', threshold: 0.01 });
    pump();   // fill the first screenful(s)
  }

  function updateStatus() {
    pager.innerHTML = '';
    if (!STATE.view.length) { pager.appendChild(el('span', 'pg-info', 'no agents match')); return; }
    pager.appendChild(el('span', 'pg-info', 'showing ' + STATE.rendered + ' of ' + STATE.view.length +
      ' agents' + (STATE.view.length !== STATE.total ? ' (of ' + STATE.total + ' built)' : '')));
    if (STATE.rendered < STATE.view.length) pager.appendChild(el('span', 'pg-more', 'scroll for more ↓'));
  }

  // ---- search / filter / sort ----
  function applyView() {
    var q = (searchEl.value || '').trim().toLowerCase();
    var pubOnly = pubEl.checked;
    var v = STATE.all.filter(function (a) {
      if (pubOnly && !a.published) return false;
      if (q && (a.title || a.id).toLowerCase().indexOf(q) < 0 && a.id.indexOf(q) < 0) return false;
      return true;
    });
    var key = sortEl.value;
    if (key === 'gens') v.sort(function (a, b) { return (b.n_gen || 0) - (a.n_gen || 0); });
    else if (key === 'title') v.sort(function (a, b) { return (a.title || '').localeCompare(b.title || ''); });
    else v.sort(function (a, b) { return a.id.localeCompare(b.id); }); // chronological (agent id)
    STATE.view = v; renderView();
  }

  // ---- load a run ----
  function buildRun(run, hf) {
    STATE.run = run; STATE.hf = !!hf;
    wall.innerHTML = '<div class="tx-loading" style="margin:24px">loading run…</div>';
    fetchJSON(runBase(run, hf) + 'index.json').then(function (idx) {
      STATE.all = idx.agents || [];
      STATE.total = STATE.all.length;
      var built = STATE.all.length, totRun = idx.total_agents || built;
      metaEl.textContent = built + ' transcripts' +
        (totRun > built ? ' (sampled from ' + totRun + ' agents)' : '') +
        ' · ' + (idx.model || '') + (idx.seed != null ? ' · seed ' + idx.seed : '');
      applyView();
    }).catch(function (e) {
      wall.innerHTML = '<div class="tx-empty" style="margin:24px">Could not load “' + escapeHTML(run) +
        '”: ' + escapeHTML(String(e.message || e)) + '</div>';
      pager.innerHTML = '';
    });
  }

  // ---- run selector ----
  function initRuns() {
    fetchJSON(LOCAL_BASE + 'manifest.json').then(function (m) {
      var runs = m.runs || [];
      sel.innerHTML = '';
      runs.forEach(function (r) {
        var o = el('option'); o.value = r.run; o.textContent = r.label || r.run;
        if (r.disabled) o.disabled = true; o.dataset.hf = r.hf ? '1' : '';
        sel.appendChild(o);
      });
      // ?run= pins the viewer to one run (the archive-viewer "Transcripts" tab uses
      // this, hiding the selector via body.embed); otherwise default to the first run.
      var want = qs.get('run');
      var pinned = want && runs.find(function (r) { return r.run === want && !r.disabled; });
      if (want && !pinned) {
        // Pinned run isn't in the committed manifest — it's a wall built & pushed to HF
        // after the manifest was baked (the gallery only offers the tab once it has
        // probed a live wall). Load it straight from HF. ?hf=0 forces a local lookup.
        var hf = qs.get('hf') !== '0';
        var o = el('option'); o.value = want; o.textContent = want; o.dataset.hf = hf ? '1' : '';
        sel.appendChild(o); sel.value = want; buildRun(want, hf);
        return;
      }
      var target = pinned || runs.find(function (r) { return !r.disabled; });
      if (target) { sel.value = target.run; buildRun(target.run, target.hf); }
    }).catch(function () { sel.innerHTML = '<option>no transcript data found</option>'; });
  }

  document.addEventListener('DOMContentLoaded', function () {
    // Embedded in the archive viewer's Transcripts tab: hide the run selector (the
    // host pins the run via ?run=) so this reads as one run's wall, not a sub-app.
    if (qs.has('embed') && qs.get('run')) document.body.classList.add('embed');
    wall = document.getElementById('tx-wall');
    sel = document.getElementById('tx-run');
    metaEl = document.getElementById('tx-meta');
    searchEl = document.getElementById('tx-search');
    pubEl = document.getElementById('tx-pub-only');
    sortEl = document.getElementById('tx-sort');
    pager = document.getElementById('tx-pager');
    lb = document.getElementById('tx-lb');
    lbInner = document.getElementById('tx-lb-inner');

    // streaming wall: scrolling right (or resizing wider) tops up columns
    wall.addEventListener('scroll', schedulePump, { passive: true });
    window.addEventListener('resize', onResize);

    sel.addEventListener('change', function () {
      var o = sel.options[sel.selectedIndex]; buildRun(sel.value, o && o.dataset.hf === '1'); });
    var t;
    searchEl.addEventListener('input', function () { clearTimeout(t); t = setTimeout(applyView, 180); });
    pubEl.addEventListener('change', applyView);
    sortEl.addEventListener('change', applyView);
    lb.addEventListener('click', closeLightbox);   // click anywhere zooms out
    document.getElementById('tx-lb-close').addEventListener('click', closeLightbox);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeLightbox(); });
    initRuns();
  });
})();
