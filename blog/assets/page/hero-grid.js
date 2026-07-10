(function(){
  var GRID = document.getElementById('hero-grid');
  if(!GRID) return;

  /* ============================ TUNABLES ============================ */
  var MORPH_FPS          = 20;    // internal unit of the frame timeline: frames-per-second used to convert a segment's wall-clock duration into frame span (buildGpuLineage) and back into travel speed (tick). It CANCELS OUT -- traversal takes sum(segment durations) seconds whatever its value. To change morph speed, use EDGES_PER_SEC / MIN_SEG_SEC below.
  var HOLD_MS            = 400;   // on first hover of a STILL cell, hold its published title this long before rewinding, so the reader can read the name of the image they pointed at
  var KEYFRAME_HOLD_MS   = 350;   // as a morph passes through an actual published image in the lineage (a keyframe from the morph JSON's pubs/titles), land exactly on it and pause this long, so each named ancestor reads as a held still before the morph flows on
  // "Flux" = how agitated the grid is (0..1). It RISES while the hero is off-screen
  // and DECAYS while the reader is watching it; the chance a given image starts
  // morphing scales with flux. So: scroll away for a while -> come back to a grid in
  // greater flux; keep watching -> it settles down to FLUX_MIN, a low hum that never
  // fully stops (each idle image keeps a small flux-scaled chance to start a morph).
  var FLUX_START         = 0.12;  // agitation right after page load
  var FLUX_MIN           = 0.2;  // <<< LOW HUM: floor flux never decays below, so morphing never fully stops while watching
  var FLUX_RISE_PER_SEC  = 0.03;  // build-up/sec while OFF-screen  (~33s from calm to full)
  var FLUX_DECAY_PER_SEC = 0.12;  // settling/sec while ON-screen   (~8s from full to calm)
  var FLUX_MAX           = 1.0;
  var MAX_CONCURRENT     = 14;    // images morphing at once at full flux
  var SPAWN_PER_TICK     = 4;     // most new morphs started per check (ramps the "burst")
  var SPAWN_CHECK_MS     = 700;   // how often we consider starting new morphs
  var CELL_PX            = 112;   // <<< IMAGE SIZE: target cell px. Columns are chosen to keep cells near this at any width; the partial last row is centered, so its raggedness reads as intentional.
  /* ================================================================= */

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var cells = [], active = new Set();
  var titleZ = 0;   // monotonic stacking counter so the most recently hovered cell's title wins overlaps
  var flux = FLUX_START, inView = true;

  // Double-click a cell -> open the inline archive viewer at that image's source run
  // with the loupe parked on it. provByFig maps hero id -> {run, arc, model, seed, index, ...}.
  var provByFig = {};
  fetch('assets/hero_sprites/provenance.json').then(function(r){return r.json();})
    .then(function(p){ provByFig = p || {}; })
    .catch(function(){ /* provenance absent -> double-click is a no-op */ });

  fetch('assets/hero_sprites/manifest.json').then(function(r){return r.json();}).then(setup)
    .catch(function(e){ console.error('hero manifest load failed', e); });

  function watching(){ return inView && !document.hidden; }   // reader can actually see it

  // Cells render their lineage live on the GPU from genome JSON (PB.HeroMorph):
  // resolution-independent, ~0.37 MB total. WebGL is required -- without it the grid
  // stays a static thumbnail mosaic (cells never become `loaded`, so nothing morphs).
  // Morph pacing is by PUBLICATION SEGMENT, in wall-clock seconds (so it's
  // frame-rate-independent: a faster display just renders more in-between frames -->
  // smoother, never faster). Each published->published transition lasts
  // max(MIN_SEG_SEC, edgesInSegment / EDGES_PER_SEC) seconds, i.e. we travel between
  // publications at a capped speed and dwell a minimum beat on short hops. Busier
  // lineages (more publications) therefore take proportionally longer, rather than
  // every cell racing tip-to-root in a fixed time regardless of how much happens
  // along the way.
  var EDGES_PER_SEC = 8;                   // <<< SPEED CAP between publications (lower = slower morphs)
  var MIN_SEG_SEC   = 0.7;                 // floor: a short hop still reads as a beat, not a jump
  var GPU_STEPS_INIT = 100;                // placeholder maxFrame until a cell's genomes load
  var morphEng = null;                     // resolved in setup(), after cppn.js/hero-morph.js have loaded
  var MORPH_BASE = 'assets/hero_morph/';
  function fetchGz(url){
    return fetch(url).then(function(r){
      if(!r.ok) throw new Error(url + ' ' + r.status);
      return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json();
    });
  }

  var setupTries = 0;
  function setup(manifest){
    // hero-morph.js / cppn.js load lower in the page, so on a fast (cached) manifest
    // fetch this can fire first; wait a few ticks for PB.HeroMorph before deciding.
    if(!(window.PB && PB.HeroMorph) && setupTries++ < 40){
      return setTimeout(function(){ setup(manifest); }, 25);
    }
    if(window.PB && PB.HeroMorph){
      var eng = PB.HeroMorph.create();
      if(eng && eng.ok) morphEng = eng;
    }
    if(!morphEng) console.warn('hero: WebGL unavailable -- static thumbnail grid, no morphs');

    // Two-tier load. Tier 1 is a SINGLE local atlas image (thumbs_atlas.jpg, ~150 KB)
    // holding every cell's final frame; one request paints the whole grid, so the
    // banner never looks unloaded. Tier 2 is the per-lineage genome chain, fetched
    // per cell and compiled on the GPU; once a cell's chain lands it starts morphing
    // and draw() replaces the thumb with the live CPPN render of the same tip image.
    var morphReady = 0;
    var startThreshold = Math.max(1, Math.round(manifest.length * 0.5));

    manifest.forEach(function(m){
      var cv = document.createElement('canvas');
      cv.width = m.fw; cv.height = m.fw;
      var wrap = document.createElement('div');
      wrap.className = 'hero-cell';
      wrap.appendChild(cv);
      var titleHost = document.createElement('div');
      titleHost.className = 'hero-title-host';
      var titleA = document.createElement('div');
      var titleB = document.createElement('div');
      titleA.className = 'hero-title-under';
      titleB.className = 'hero-title-under';
      titleA.style.opacity = '0';
      titleB.style.opacity = '0';
      titleHost.appendChild(titleA);
      titleHost.appendChild(titleB);
      wrap.appendChild(titleHost);
      // wrap is placed into a row by relayout(), not appended to GRID directly
      var cell = { m:m, cv:cv, ctx:cv.getContext('2d'),
                   maxFrame:GPU_STEPS_INIT, frame:GPU_STEPS_INIT, target:GPU_STEPS_INIT, phase:'rest',
                   loaded:false, drawn:-1, hover:false,
                   wrap:wrap, titleA:titleA, titleB:titleB,
                   lineage:null, kfIdx:-1, gpu:null };

      // Tier 1 (thumbnails) is painted in bulk from the atlas after this loop.
      // Tier 2: fetch this lineage's genome chain and compile it -- then it's ready
      // to morph. Keyframes come from data.pubs (published-genome canon indices)
      // zipped with data.titles, so each title dwells exactly on its published image.
      if(morphEng){
        fetchGz(MORPH_BASE + m.fig + '.json.gz').then(function(data){
          cell.gpu = morphEng.addCell(m.fig, data);
          cell.pubs = data.pubs || [];
          cell.pubTitles = data.titles || [];
          buildGpuLineage(cell);
          cell.loaded = true;
          draw(cell, true);
          morphReady++;
          if(morphReady === startThreshold) startSpawn();
        }).catch(function(e){ console.warn('hero-morph load failed', m.fig, e); });
      }

      cv.addEventListener('mouseenter', function(){
        cell.wrap.classList.add('show-title');     // reveal title immediately
        cell.wrap.style.zIndex = ++titleZ;         // bring this title in front of any overlapping neighbor
        if(!cell.loaded) return;
        cell.hover = true;
        if(cell.phase === 'rest'){
          // First hover of a STILL cell: hold its published title a beat so the
          // reader can read the name of the image they pointed at, then rewind.
          cell.holdTimer = setTimeout(function(){
            cell.holdTimer = null;
            if(cell.hover){ cell.phase = 'hover'; cell.target = 0; active.add(cell); }
          }, HOLD_MS);
        } else {
          cell.phase = 'hover'; cell.target = 0; active.add(cell);   // already moving: rewind now
        }
      });
      cv.addEventListener('dblclick', function(){
        var prov = provByFig[cell.m.fig] || provByFig[cell.m.id];
        if(prov && typeof window.PBVLMOpenHeroArc === 'function') window.PBVLMOpenHeroArc(prov);
      });
      // Right-click a cell -> navigate to the breeding console with this
      // individual seeded as the parent. The archive id is img_{index+1:06d}
      // (the hero fig id is a separate global numbering, so it can't be reused);
      // play.html's ?run=&id= path fetches that genome and startFromFounder()s it.
      cv.addEventListener('contextmenu', function(e){
        var prov = provByFig[cell.m.fig] || provByFig[cell.m.id];
        if(!prov || prov.run == null || prov.index == null) return;  // no provenance -> normal menu
        e.preventDefault();
        var id = 'img_' + String(prov.index + 1).padStart(6, '0');
        window.location.href = 'breed/play.html?run=' + encodeURIComponent(prov.run) +
                               '&id=' + encodeURIComponent(id);
      });
      cv.addEventListener('mouseleave', function(){
        cell.hover = false;
        if(cell.holdTimer){ clearTimeout(cell.holdTimer); cell.holdTimer = null; }  // left during the hold: never morphed
        // If mid hover-rewind, turn around and play forward to the tip, keeping
        // the title visible the whole way (the class is cleared in tick when that
        // morph finishes). Otherwise there's no return morph, so hide now.
        if(cell.phase === 'hover'){ cell.phase = 'return'; cell.target = cell.maxFrame; active.add(cell); }
        else { cell.wrap.classList.remove('show-title'); }
      });
      cells.push(cell);
    });

    // Tier 1: paint every cell from one atlas image (cells are in manifest order,
    // so cell i is at atlas slot i). ATLAS_* must match tools/build_hero_thumbs.py.
    var ATLAS_FILE = 'assets/hero_sprites/thumbs_atlas.jpg', ATLAS_COLS = 8, ATLAS_CELL = 128;
    function paintFromAtlas(img){
      cells.forEach(function(c, i){
        if(c.loaded) return;                   // live GPU render already took over
        c.ctx.drawImage(img, (i % ATLAS_COLS) * ATLAS_CELL, Math.floor(i / ATLAS_COLS) * ATLAS_CELL,
                        ATLAS_CELL, ATLAS_CELL, 0, 0, c.cv.width, c.cv.height);
      });
    }
    var atlas = new Image();
    atlas.decoding = 'async';
    atlas.onload = function(){ paintFromAtlas(atlas); };
    atlas.onerror = function(){    // fall back to per-cell thumbs if the atlas is missing
      cells.forEach(function(c){
        var t = new Image(); t.decoding = 'async';
        t.onload = function(){ if(!c.loaded){ c.ctx.drawImage(t, 0, 0, c.cv.width, c.cv.height); } };
        t.src = c.m.thumb;
      });
    };
    atlas.src = ATLAS_FILE;

    if('IntersectionObserver' in window){
      new IntersectionObserver(function(es){ inView = es[0].isIntersecting; }, {threshold:0.12}).observe(GRID);
    }
    // Pick the column count that keeps cells nearest CELL_PX while mildly
    // preferring fuller last rows, then lay cells out as a stack of rows.
    var N = cells.length, GAP = 4;
    var rowObjs = [];   // {el, track, offset, min, pitch, shiftable, timer}

    function colsFor(avail){
      var ideal = (avail + GAP) / (CELL_PX + GAP);
      var lo = Math.max(1, Math.floor(ideal) - 2), hi = Math.ceil(ideal) + 2;
      var best = Math.max(1, Math.round(ideal)), bestScore = Infinity;
      for(var c = lo; c <= hi; c++){
        var cw = (avail - (c - 1) * GAP) / c;
        if(cw <= 0) continue;
        var missing = Math.ceil(N / c) * c - N;            // empty slots in last row
        var score = Math.abs(Math.log(cw / CELL_PX)) + 0.12 * (missing / c);
        if(score < bestScore){ bestScore = score; best = c; }
      }
      return best;
    }

    function clearRows(){
      rowObjs.forEach(function(r){
        if(r.timer) clearTimeout(r.timer);
        if(r.snap) clearTimeout(r.snap);
        if(r.raf) cancelAnimationFrame(r.raf);
      });
      rowObjs = [];
      while(GRID.firstChild) GRID.removeChild(GRID.firstChild);
    }

    // A shiftable row is an infinite ring: it holds T > cols cells, shows `cols`
    // of them through the clip window, and the rest sit off the ends. Scrolling
    // never hits a wall — when a cell passes fully off one edge we move that one
    // DOM node to the other end and compensate the transform by one cell-pitch,
    // so the same images loop forever in either direction (O(1) per step, no
    // cloning). r.cells mirrors the live DOM order; r.offset is translateX(px).
    function setX(r){ r.track.style.transform = 'translateX(' + r.offset + 'px)'; }
    function recycleFrontToEnd(r){            // first cell exited left -> send it right
      r.track.appendChild(r.cells[0].wrap); r.cells.push(r.cells.shift()); r.offset += r.pitch;
    }
    function recycleEndToFront(r){            // pull the last cell around to the left
      r.track.insertBefore(r.cells[r.cells.length - 1].wrap, r.track.firstChild);
      r.cells.unshift(r.cells.pop()); r.offset -= r.pitch;
    }
    function ringNormalize(r){                // keep offset within one pitch of 0 by recycling
      var g = 0, n = r.cells.length;
      while(r.offset <= -r.pitch + 0.5 && g++ < n) recycleFrontToEnd(r);
      while(r.offset > 0.5 && g++ < 2 * n) recycleEndToFront(r);
    }
    function settle(r){                       // recycle + reposition with no animation
      r.track.style.transition = 'none';
      ringNormalize(r); setX(r);
      r.track.offsetHeight;                   // flush so the restore won't animate the snap
      r.track.style.transition = '';
    }

    function shiftRow(r, d){                   // d=+1 reveal a cell from the LEFT, d=-1 from the RIGHT
      if(r.raf || r.el.classList.contains('dragging')) return;   // not while flinging/dragging
      if(d > 0){                               // park a cell just off the left, invisibly, to reveal
        r.track.style.transition = 'none';
        recycleEndToFront(r); setX(r); r.track.offsetHeight; r.track.style.transition = '';
      }
      r.offset += d * r.pitch;
      setX(r);                                 // animates via the CSS transition
      clearTimeout(r.snap);
      r.snap = setTimeout(function(){ settle(r); }, 600);
    }

    function scheduleShift(r){
      if(!r.shiftable || reduce) return;
      r.timer = setTimeout(function(){
        // don't shift a row out from under the cursor while a cell in it is hovered
        var hovered = r.cells.some(function(c){ return c.hover; });
        if(!document.hidden && !hovered) shiftRow(r, Math.random() < 0.5 ? 1 : -1);
        scheduleShift(r);
      }, 8000 + Math.random() * 14000);        // avg ~15 s, random direction (loops both ways)
    }

    function snapTo(r){                        // ease to the nearest cell boundary, then settle
      r.track.style.transition = '';
      r.offset = Math.round(r.offset / r.pitch) * r.pitch; setX(r);
      clearTimeout(r.snap); r.snap = setTimeout(function(){ settle(r); }, 600);
    }
    function cancelFling(r){ if(r.raf){ cancelAnimationFrame(r.raf); r.raf = 0; } }
    function fling(r, v0){                      // v0 px/ms; glide with friction, then snap to a cell
      cancelFling(r);
      // Nonlinear launch: power>1 makes soft flicks travel much less while letting
      // hard, intentional flicks travel more (the gap widens around 1 px/ms).
      var sp = Math.min(3.5, Math.abs(v0));
      var v = (v0 < 0 ? -1 : 1) * 0.5 * Math.pow(sp, 1.9);
      r.track.style.transition = 'none';
      var last = 0;
      function step(ts){
        if(!last) last = ts;
        var dt = Math.min(40, ts - last); last = ts;
        r.offset += v * dt;
        v *= Math.pow(0.9965, dt);              // gentler friction -> longer cool-down
        ringNormalize(r); setX(r);
        if(Math.abs(v) > 0.02){ r.raf = requestAnimationFrame(step); }
        else { r.raf = 0; snapTo(r); }          // glide spent -> rest on a cell border
      }
      r.raf = requestAnimationFrame(step);
    }

    function makeDraggable(r){
      var track = r.track, startX = 0, lastX = 0, lastT = 0, vel = 0, dragging = false, down = false;
      track.addEventListener('pointerdown', function(e){
        down = true; dragging = false; startX = lastX = e.clientX; lastT = e.timeStamp; vel = 0;
        cancelFling(r); clearTimeout(r.snap);
      });
      track.addEventListener('pointermove', function(e){
        if(!down) return;
        if(!dragging){
          if(Math.abs(e.clientX - startX) < 5) return;   // under threshold: leave hover/click intact
          dragging = true; r.el.classList.add('dragging');
          try { track.setPointerCapture(e.pointerId); } catch(_){}
        }
        e.preventDefault();
        var dx = e.clientX - lastX, dt = e.timeStamp - lastT;
        r.offset += dx;                                    // incremental, so ring recycling survives
        if(dt > 0) vel = 0.7 * (dx / dt) + 0.3 * vel;      // smoothed px/ms for the flick
        lastX = e.clientX; lastT = e.timeStamp;
        ringNormalize(r); setX(r);                         // .dragging disables the transition
      });
      function end(){
        down = false;
        if(dragging){
          dragging = false; r.el.classList.remove('dragging');
          // Scale momentum by how far they actually swept, so a quick SHORT flick
          // (little drag, high release speed) throws gently, while a long sweep throws
          // far. reach ramps 0->1 over ~250 px (~2 cells).
          var reach = Math.pow(Math.min(1, Math.abs(lastX - startX) / 250), 1.6);  // curved: short sweeps gentler
          if(Math.abs(vel) > 0.12) fling(r, vel * reach);  // released with speed -> momentum
          else snapTo(r);                                  // gentle release -> snap to grid
        }
      }
      track.addEventListener('pointerup', end);
      track.addEventListener('pointercancel', end);
      track.addEventListener('lostpointercapture', end);
    }

    function relayout(){
      var avail = GRID.clientWidth || window.innerWidth;
      var cols = colsFor(avail);
      var cw = Math.floor((avail - (cols - 1) * GAP) / cols);
      var pitch = cw + GAP;
      cells.forEach(function(o){ o.cv.style.width = cw + 'px'; o.cv.style.height = cw + 'px'; });

      // For EVERY row to loop, each needs >= 1 off-screen buffer cell, so we make
      // floor(N/(cols+1)) rows of `cols` visible cells and spread all N images so
      // each row holds cols+(1..2); the extras cycle through hidden as it loops.
      // Too few images for two such rows -> plain centered (non-scrolling) grid.
      var loopRows = Math.floor(N / (cols + 1));
      var scrolling = loopRows >= 2;
      var nRows = scrolling ? loopRows : Math.ceil(N / cols);

      // Narrow viewports take few columns, which would otherwise stack the images
      // into a tower tall enough to shove the article title off the bottom of the
      // screen. Cap the rows at what fits above the fold (leaving TITLE_RESERVE for
      // the title + byline) and let the surplus images ride the rows' rings, where
      // they cycle into view instead of demanding a row of their own.
      var ROW_PX = cw + 28;                                     // cell + .hero-row padding-bottom
      var TITLE_RESERVE = 190;
      var vh = window.innerHeight || 800;
      var maxRows = Math.max(1, Math.floor((vh - TITLE_RESERVE) / ROW_PX));
      if(nRows > maxRows){
        nRows = maxRows;
        scrolling = N > nRows * cols;
      }

      clearRows();

      var sizes = [];
      if(scrolling){
        var base = Math.floor(N / nRows), extraRows = N - base * nRows;   // base >= cols+1
        for(var i = 0; i < nRows; i++) sizes.push(base + (i < extraRows ? 1 : 0));
      } else {
        for(var i = 0; i < nRows; i++) sizes.push(Math.min(cols, N - i * cols));
      }

      var winW = cols * cw + (cols - 1) * GAP, idx = 0;
      sizes.forEach(function(size){
        var el = document.createElement('div'); el.className = 'hero-row';
        var track = document.createElement('div'); track.className = 'hero-row-track';
        var ordered = cells.slice(idx, idx + size); idx += size;
        ordered.forEach(function(c){ track.appendChild(c.wrap); });
        el.appendChild(track); GRID.appendChild(el);
        var extra = size - cols;
        var r = { el:el, track:track, cells:ordered.slice(), pitch:pitch, offset:0,
                  shiftable:extra > 0, timer:null, snap:null, raf:0 };
        if(extra > 0){
          el.classList.add('shiftable');
          el.style.width = winW + 'px';                // clip window = exactly `cols` cells
          var leftBuf = Math.floor(extra / 2);         // start with buffers split off both ends
          r.offset = -leftBuf * pitch;
          r.track.style.transition = 'none'; setX(r); r.track.offsetHeight; r.track.style.transition = '';
          makeDraggable(r); scheduleShift(r);
        }
        rowObjs.push(r);
      });
    }
    // GPU keyframes ARE the published genomes. Anchors in edge space are the random
    // root (0) then each publication (pubs ends at nEdges = the tip). Each segment
    // between anchors is given a wall-clock duration -- max(MIN_SEG_SEC, edges/EDGES_PER_SEC)
    // -- and that duration is converted to a slice of the frame timeline (segF[]). The
    // frame->frac map (gpuFrac) then interpolates edge position linearly within each
    // slice, so we travel between publications at a capped speed and a title dwells on
    // its EXACT published image (frac lands on pubs[k]/nEdges). maxFrame grows with the
    // number of publications, so busy lineages take longer instead of racing.
    function buildGpuLineage(cell){
      if(!cell.gpu) return;                      // genomes not loaded yet
      var pubs = cell.pubs || [], titles = cell.pubTitles || [];
      var nE = cell.gpu.nEdges;
      if(!nE || !pubs.length){ cell.lineage = null; return; }
      var anchors = [0];                         // random seed root...
      for(var i = 0; i < pubs.length; i++) anchors.push(pubs[i]);   // ...then each publication (last = tip)
      var segF = [0], lin = [{ f: 0, title: '' }];                  // untitled root, never dwelt on
      for(var s = 0; s < anchors.length - 1; s++){
        var span = anchors[s + 1] - anchors[s];                    // edges in this segment
        var dur  = Math.max(MIN_SEG_SEC, span / EDGES_PER_SEC);    // seconds (capped speed, floored beat)
        segF.push(segF[s] + dur * MORPH_FPS);                      // frames = seconds * frames-per-second
        lin.push({ f: segF[s + 1], title: titles[s] || '' });      // title dwells at the pub ending this segment
      }
      cell.segF = segF; cell.segA = anchors; cell.nE = nE;
      cell.maxFrame = segF[segF.length - 1];
      cell.lineage = lin;
      cell.kfIdx = -1;
      if(cell.phase === 'rest'){ cell.frame = cell.target = cell.maxFrame; }   // sit on the tip
      else {                                     // clamp an in-flight morph into the new timeline
        if(cell.frame  > cell.maxFrame) cell.frame  = cell.maxFrame;
        if(cell.target > cell.maxFrame) cell.target = cell.maxFrame;
      }
      updateTitle(cell, cell.frame);
    }
    relayout();
    var rzt; window.addEventListener('resize', function(){ clearTimeout(rzt); rzt = setTimeout(relayout, 120); });
    requestAnimationFrame(tick);
    if(!reduce){
      // flux integrator on a timer, so agitation builds even while scrolled away / tab hidden
      var lastFlux = Date.now();
      setInterval(function(){
        var t = Date.now(), dt = (t - lastFlux) / 1000; lastFlux = t;
        flux += (watching() ? -FLUX_DECAY_PER_SEC : FLUX_RISE_PER_SEC) * dt;
        flux = Math.max(FLUX_MIN, Math.min(FLUX_MAX, flux));
      }, 400);
      // Hard ceiling: even on a slow connection, kick off the morph loop after
      // 8s so the banner doesn't stay perfectly still. The genome-load callback
      // will usually beat this timer once `startThreshold` cells are ready.
      setTimeout(startSpawn, 8000);
    }
  }

  var spawnStarted = false;
  function startSpawn(){
    if(spawnStarted || reduce || !morphEng) return;   // no engine -> nothing can morph
    spawnStarted = true;
    setInterval(spawn, SPAWN_CHECK_MS);
  }

  function spawn(){
    if(!watching()) return;                                   // only morph when visible
    var running = 0;
    cells.forEach(function(c){ if(c.phase==='toRoot'||c.phase==='toCurrent') running++; });
    var slots = Math.min(SPAWN_PER_TICK, Math.round(flux*MAX_CONCURRENT) - running);
    for(var i=0; i<slots; i++){
      if(Math.random() > flux) continue;                      // likelihood scales with flux
      var idle = cells.filter(function(c){ return c.loaded && c.phase==='rest' && !c.hover; });
      if(!idle.length) break;
      var cell = idle[Math.floor(Math.random()*idle.length)];
      cell.phase = 'toRoot'; cell.target = 0; active.add(cell);
    }
  }

  // Sharper-than-smoothstep S-curve (Inigo Quilez "gain"): keeps zero slope at
  // the keyframes (each name dwells at full opacity at its publication frame)
  // but steepens the midpoint so we spend less time in the half-faded,
  // illegible two-title overlap. Higher XFADE_K = snappier title swap.
  var XFADE_K = 3.5;
  function xfade(t){
    var x = t < 0.5 ? t : 1 - t;
    var a = 0.5 * Math.pow(2 * x, XFADE_K);
    return t < 0.5 ? a : 1 - a;
  }
  // Next lineage keyframe in the direction of travel that is an ACTUAL published
  // image (non-empty title). The random root (f=0, no title) is skipped so the
  // morph never pauses on the noise seed; the tip is handled by phase rest, and
  // cell.target is excluded by the caller so we never dwell where we're stopping.
  function nextKeyframe(cell, from, dir){
    var lin = cell.lineage;
    if(!lin) return null;
    var best = null;
    for(var i = 0; i < lin.length; i++){
      if(!lin[i].title) continue;                  // unnamed root -> not a published image
      var f = lin[i].f;
      if(dir > 0 && f > from + 1e-3 && (best === null || f < best)) best = f;
      if(dir < 0 && f < from - 1e-3 && (best === null || f > best)) best = f;
    }
    return best;
  }
  function updateTitle(cell, fr){
    var lin = cell.lineage;
    if(!lin) return;
    // bracket: largest i with lin[i].f <= fr
    var idx = 0;
    for(var i = 0; i < lin.length - 1; i++){
      if(fr >= lin[i].f) idx = i; else break;
    }
    var a = lin[idx], b = lin[idx + 1] || a;
    var span = b.f - a.f;
    var t = span > 0 ? (fr - a.f) / span : 0;
    if(t < 0) t = 0; else if(t > 1) t = 1;
    var s = xfade(t);
    if(idx !== cell.kfIdx){
      cell.titleA.textContent = a.title;
      cell.titleB.textContent = b.title;
      cell.kfIdx = idx;
    }
    cell.titleA.style.opacity = (1 - s).toFixed(3);
    cell.titleB.style.opacity = s.toFixed(3);
  }
  // Map a GPU cell's frame position onto its edge-space fraction via the per-segment
  // timeline (segF frame boundaries -> segA edge anchors). Linear within each segment.
  function gpuFrac(cell, fr){
    var F = cell.segF, A = cell.segA;
    if(!F) return cell.maxFrame > 0 ? fr / cell.maxFrame : 1;     // pre-timeline fallback
    if(fr <= 0) return 0;
    if(fr >= cell.maxFrame) return 1;
    var s = 0;
    for(var i = 0; i < F.length - 1; i++){ if(fr >= F[i]) s = i; else break; }
    var seg = F[s + 1] - F[s];
    var t = seg > 0 ? (fr - F[s]) / seg : 0;
    return (A[s] + t * (A[s + 1] - A[s])) / cell.nE;
  }
  function draw(cell, force){
    if(!cell.loaded) return;                  // only true once the GPU chain compiled
    var fr = cell.frame, max = cell.maxFrame;
    if(fr < 0) fr = 0; else if(fr > max) fr = max;
    if(!force && fr === cell.drawn) return;   // nothing moved since last paint
    cell.drawn = fr;
    if(cell.lineage){
      try { updateTitle(cell, fr); }
      catch(e){ console.warn('updateTitle failed', e); cell.lineage = null; }
    }
    // Evaluate the CPPN live at this fraction of the lineage. Continuous by
    // construction (weights/activations lerp on the GPU), so no cross-fade needed.
    // gpuFrac walks the per-segment timeline so publication-to-publication speed is
    // capped and each title lands on its exact published image.
    var frac = gpuFrac(cell, fr);
    var res = cell.cv.width;
    if(morphEng.render(cell.gpu, frac, res)){
      cell.ctx.drawImage(morphEng.canvas, 0, 0, res, res, 0, 0, cell.cv.width, cell.cv.height);
    }
  }

  var last = performance.now();
  function tick(now){
    // dt is wall-clock elapsed, so morph speed is frame-rate-independent -- a faster
    // display just renders more in-between frames (smoother), never a faster morph. The
    // clamp only guards against a huge jump after the tab was backgrounded; keep it
    // generous (0.1s) so speed stays accurate down to ~10fps rather than dragging.
    var dt = Math.min(0.1, (now-last)/1000); last = now;
    var step = MORPH_FPS * dt;
    active.forEach(function(cell){
      if(cell.dwellEnd){                       // currently paused on a published image
        if(now < cell.dwellEnd){ draw(cell); return; }
        cell.dwellEnd = 0;                      // beat's up -> let the morph flow on
      }
      if(cell.frame !== cell.target){
        var dir = cell.target > cell.frame ? 1 : -1;
        var nf = dir > 0 ? Math.min(cell.target, cell.frame+step)
                         : Math.max(cell.target, cell.frame-step);
        var kf = nextKeyframe(cell, cell.frame, dir);
        if(kf !== null && kf !== cell.target && ((dir > 0 && nf >= kf) || (dir < 0 && nf <= kf))){
          nf = kf;                              // land exactly on the published image...
          cell.dwellEnd = now + KEYFRAME_HOLD_MS;  // ...and hold a beat there
        }
        cell.frame = nf;
      }
      draw(cell);
      if(cell.frame === cell.target){
        if(cell.phase === 'toRoot'){ cell.phase = 'toCurrent'; cell.target = cell.maxFrame; }
        else if(cell.phase === 'return'){ cell.phase = 'rest'; active.delete(cell); cell.wrap.classList.remove('show-title'); }
        else if(cell.phase === 'toCurrent'){ cell.phase = 'rest'; active.delete(cell); }
        else if(cell.phase === 'hover'){ active.delete(cell); }   // frozen at root until mouseleave
        else { active.delete(cell); }
      }
    });
    requestAnimationFrame(tick);
  }
})();
