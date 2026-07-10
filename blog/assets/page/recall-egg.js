/* Semantic Recall panel: the THINGS leaderboard reel.
 *
 * The panel is a canvas you drag-scroll through the WHOLE THINGS vocabulary --
 * all 1,823 nouns, each paired with the single closest archive image to it
 * (across every VLM run + the human Picbreeder archive), sorted from best match
 * to worst. Dragging has momentum, like the rows of the hero banner. There is no
 * static/frozen mode: the reel goes live on its own and stays live.
 *
 * Data (assets/recall_things/, built by tools/build_recall_easter_egg.py):
 *   manifest.json  {cell, cols, count, atlas, items:[{noun, sim, cond}, ...]}
 *   things_atlas.jpg  one atlas, items in order; slot i = (i%cols, i//cols)
 *
 * The atlas (~4 MB) is fetched when the panel first scrolls into view rather than
 * at page load, so it never weighs on the initial paint. grid_recall.png shows
 * until the atlas decodes -- its first screenful is pixel-identical, so the swap
 * is invisible -- and remains as the fallback if the atlas fails to load.
 *
 * The panel sits inside the metric figure's horizontal swap track (index.html),
 * so the two gestures share a surface and are separated by AXIS: a vertical drag
 * scrolls this reel, a horizontal one slides the figure. Whichever axis the
 * pointer commits to first wins the gesture, and each side marks the DOM so the
 * other stands down for the rest of it.
 */
(function(){
  var root = document.getElementById('recall-egg');
  if(!root) return;
  var gridImg = root.querySelector('.recall-egg-grid');
  if(!gridImg) return;
  var BASE = root.getAttribute('data-base') || 'assets/recall_things/';
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var state = null;     // populated by activate(); null until the atlas is on its way

  // -------------------------------------------------------------- activation
  var activating = false;
  function activate(){
    if(state || activating) return;
    activating = true;

    var canvas = document.createElement('canvas');
    canvas.className = 'recall-egg-canvas';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label',
      'All 1,823 THINGS concept nouns, each paired with the closest archive image; drag to scroll.');
    root.appendChild(canvas);

    var ctx = canvas.getContext('2d');
    var s = {
      canvas: canvas, ctx: ctx,
      atlas: null, items: null, atlasCols: 0, atlasCell: 0, count: 0,
      offset: 0, maxOffset: 0, contentH: 0, ready: false,
      W: 0, H: 0, C: 0, TR: 0, thumb: 0, raf: 0
    };
    state = s;

    sizeCanvas(s);

    // Keep the static grid_recall.png visible until the atlas is decoded, THEN
    // swap to the canvas — whose first screenful is pixel-identical — so there's
    // no loading flash.
    fetch(BASE + 'manifest.json').then(function(r){ return r.json(); }).then(function(m){
      s.items = m.items; s.count = m.count;
      s.atlasCols = m.cols; s.atlasCell = m.cell;
      var atlas = new Image();
      atlas.decoding = 'async';
      atlas.onload = function(){
        s.atlas = atlas; s.ready = true;
        sizeCanvas(s); layout(s); draw(s);
        root.classList.add('recall-egg--live');   // reveal canvas / hide grid now
      };
      atlas.onerror = abort;                      // leave the static grid in place
      atlas.src = BASE + (m.atlas || 'things_atlas.jpg');
    }).catch(abort);
  }

  // Only reachable if the atlas or manifest fails: fall back to the static PNG.
  function abort(){
    if(!state) return;
    var s = state;
    if(s.raf) cancelAnimationFrame(s.raf);
    root.classList.remove('recall-egg--live');
    if(s.canvas.parentNode) s.canvas.parentNode.removeChild(s.canvas);
    state = null;
  }

  // Go live the moment the panel is first revealed. It lives inside the swap
  // frame's overflow:hidden, so it stays unintersected — and unfetched — until
  // the figure is on screen AND swiped to.
  if(window.IntersectionObserver){
    var io = new IntersectionObserver(function(entries){
      for(var i = 0; i < entries.length; i++){
        if(entries[i].isIntersecting){ io.disconnect(); activate(); return; }
      }
    }, { rootMargin: '200px' });
    io.observe(root);
  } else {
    activate();
  }

  // -------------------------------------------------------------- layout/size
  function sizeCanvas(s){
    var W = gridImg.clientWidth, H = gridImg.clientHeight || Math.round(W * 1.48);
    var dpr = window.devicePixelRatio || 1;
    s.W = W; s.H = H;
    s.canvas.width = Math.round(W * dpr);
    s.canvas.height = Math.round(H * dpr);
    s.canvas.style.width = W + 'px';
    s.canvas.style.height = H + 'px';
    s.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if(s.ready) layout(s);
  }

  // Geometry copied from build_global_recall_grid (build_metric_fig_assets.py),
  // as fractions of the panel width, so the FIRST screenful (offset 0) is pixel-
  // faithful to the static grid_recall.png: a 3x3 of thumbnails with a noun + up
  // to two attribution lines under each. fig_w=4.6", M=.06 gutter, G=.07 gap,
  // cell=1.4467", label band lb=.74"; the figure was saved at 919x1363.
  var SIDE = 3;
  var MF = 0.06 / 4.6, GF = 0.07 / 4.6, CELLF = ((4.6 - 0.12 - 0.14) / 3) / 4.6,
      LBF = 0.74 / 4.6;
  // text sizes (pt -> px @200dpi -> displayed) and offsets below the cell (as
  // fractions of figure height), matching the static fig.text placement: the
  // noun, then the run's attribution (model + knob) on ONE line right under it.
  var NOUN_PT = 9.6, COND_PT = 7.4, PT2IMG = 200 / 72 / 919;   // *W -> displayed px
  var NOUN_DY = 0.006, COND_DY = 0.032;                        // *figH

  function layout(s){
    var W = s.W;
    s.C = SIDE;
    s.m = MF * W; s.g = GF * W; s.thumb = CELLF * W; s.lb = LBF * W;
    s.rowPitch = s.thumb + s.lb + s.g;
    s.nounFs = NOUN_PT * PT2IMG * W;
    s.condFs = COND_PT * PT2IMG * W;
    s.nounDy = NOUN_DY * s.H;
    s.condDy = COND_DY * s.H;
    // Row-major grid scrolled VERTICALLY; clamped to top/bottom (no wrap).
    var TR = Math.ceil(s.count / SIDE);
    s.TR = TR;
    s.contentH = 2 * s.m + TR * (s.thumb + s.lb) + (TR - 1) * s.g;
    s.maxOffset = Math.max(0, s.contentH - s.H);
    s.offset = clamp(s, s.offset);
  }

  function clamp(s, y){ return Math.max(0, Math.min(s.maxOffset, y)); }

  // ---------------------------------------------------------------- rendering
  function clearBG(s){
    s.ctx.fillStyle = '#fff';
    s.ctx.fillRect(0, 0, s.W, s.H);
  }

  function draw(s){
    if(!s.ready) return;
    var ctx = s.ctx;
    clearBG(s);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var pitchX = s.thumb + s.g;
    var firstRow = Math.max(0, Math.floor((s.offset - s.m) / s.rowPitch));
    var lastRow = Math.min(s.TR - 1, Math.ceil((s.offset + s.H - s.m) / s.rowPitch));
    for(var row = firstRow; row <= lastRow; row++){
      var y = s.m + row * s.rowPitch - s.offset;
      for(var col = 0; col < s.C; col++){
        var idx = row * s.C + col;
        if(idx >= s.count) continue;
        drawCell(s, idx, s.m + col * pitchX, y);
      }
    }
  }

  function drawCell(s, idx, x, y){
    var ctx = s.ctx, item = s.items[idx];
    var ac = idx % s.atlasCols, ar = Math.floor(idx / s.atlasCols);
    ctx.drawImage(s.atlas, ac * s.atlasCell, ar * s.atlasCell, s.atlasCell, s.atlasCell,
                  x, y, s.thumb, s.thumb);
    var cx = x + s.thumb / 2, cellBottom = y + s.thumb, maxW = s.thumb - 2;
    // noun (purple italic bold), then the run's attribution (model · knob) on ONE
    // grey line right under it — same content + colours as the static grid_recall.png.
    ctx.fillStyle = '#3a1a70';
    ctx.font = 'italic 700 ' + s.nounFs.toFixed(1) + 'px "Helvetica Neue",Arial,sans-serif';
    fitText(ctx, item.noun, cx, cellBottom + s.nounDy, maxW);
    ctx.fillStyle = '#666';
    ctx.font = s.condFs.toFixed(1) + 'px "Helvetica Neue",Arial,sans-serif';
    fitText(ctx, item.cond || '', cx, cellBottom + s.condDy, maxW);
  }

  // shrink the noun until it fits the cell (THINGS has a few long compound nouns)
  function fitText(ctx, txt, cx, y, maxW){
    var base = ctx.font;
    var m = ctx.measureText(txt);
    if(m.width > maxW){
      var px = parseFloat(base.match(/(\d+(?:\.\d+)?)px/)[1]);
      var scaled = Math.max(8, px * maxW / m.width);
      ctx.font = base.replace(/(\d+(?:\.\d+)?)px/, scaled.toFixed(1) + 'px');
    }
    ctx.fillText(txt, cx, y);
    ctx.font = base;
  }

  // ------------------------------------------------------- drag + momentum
  // Same feel as the hero banner rows: incremental drag, smoothed release velocity
  // scaled by sweep length, then a friction glide. Operates on the live `state`.
  var down = false, dragging = false, startX = 0, startY = 0, lastY = 0, lastT = 0, vel = 0;

  function cancelFling(s){ if(s && s.raf){ cancelAnimationFrame(s.raf); s.raf = 0; } }
  function fling(s, v0){
    cancelFling(s);
    var sp = Math.min(3.5, Math.abs(v0));
    var v = (v0 < 0 ? -1 : 1) * 0.5 * Math.pow(sp, 1.9);
    var last = 0;
    function step(ts){
      if(!last){ last = ts; s.raf = requestAnimationFrame(step); return; }  // seed clock; dt>0 next frame
      var dt = Math.min(40, ts - last); last = ts;
      var next = clamp(s, s.offset + v * dt);
      if(dt > 0 && next === s.offset){ s.raf = 0; draw(s); return; }   // hit top/bottom -> stop
      s.offset = next;
      v *= Math.pow(0.9965, dt);
      draw(s);
      if(Math.abs(v) > 0.02){ s.raf = requestAnimationFrame(step); }
      else { s.raf = 0; }
    }
    s.raf = requestAnimationFrame(step);
  }

  // The figure's swap track marks itself while IT owns the gesture; stand down.
  function swapHasGesture(){
    var t = root.closest('.metric-track');
    return !!(t && t.classList.contains('dragging'));
  }

  root.addEventListener('pointerdown', function(e){
    if(e.button) return;
    down = true; dragging = false;
    startX = e.clientX; startY = lastY = e.clientY; lastT = e.timeStamp; vel = 0;
    cancelFling(state);
  });
  root.addEventListener('pointermove', function(e){
    if(!down || !state) return;
    var s = state;
    if(!dragging){
      if(swapHasGesture()){ down = false; return; }   // the figure claimed this drag
      var dy = e.clientY - startY, dx = e.clientX - startX;
      if(Math.abs(dy) < 5) return;                    // tap, not a drag
      if(Math.abs(dx) > Math.abs(dy)){ down = false; return; }  // horizontal: it's the figure's
      dragging = true;
      root.setAttribute('data-vdrag', '');            // tell the swap track to stand down
      s.canvas.classList.add('dragging');
      try { s.canvas.setPointerCapture(e.pointerId); } catch(_){}
    }
    e.preventDefault();
    var dy2 = e.clientY - lastY, dt = e.timeStamp - lastT;
    s.offset = clamp(s, s.offset - dy2);           // content follows the finger
    if(dt > 0) vel = 0.7 * (-dy2 / dt) + 0.3 * vel; // smoothed d(offset)/dt
    lastY = e.clientY; lastT = e.timeStamp;
    if(s.ready) draw(s);
  });
  function dragEnd(){
    if(!down) return;
    down = false;
    var s = state;
    if(!s){ dragging = false; return; }
    if(dragging){
      dragging = false;
      root.removeAttribute('data-vdrag');
      s.canvas.classList.remove('dragging');
      var reach = Math.pow(Math.min(1, Math.abs(lastY - startY) / 250), 1.6);
      if(!reduce && Math.abs(vel) > 0.12) fling(s, vel * reach);
    }
    dragging = false;
  }
  root.addEventListener('pointerup', dragEnd);
  root.addEventListener('pointercancel', dragEnd);
  root.addEventListener('lostpointercapture', dragEnd);

  // Wheel/trackpad scroll drives the reel, but passes through once the reel is
  // clamped and the wheel keeps pushing that way — so the page never gets stuck
  // scrolling past the panel.
  root.addEventListener('wheel', function(e){
    var s = state;
    if(!s || !s.ready) return;
    var atEnd = (e.deltaY < 0 && s.offset <= 0) || (e.deltaY > 0 && s.offset >= s.maxOffset);
    if(atEnd) return;                              // let the page have it
    e.preventDefault();
    cancelFling(s);
    var unit = e.deltaMode === 1 ? 16 : (e.deltaMode === 2 ? s.H : 1);
    s.offset = clamp(s, s.offset + e.deltaY * unit);
    draw(s);
  }, { passive: false });

  // re-size the live canvas with the figure
  var rzt;
  window.addEventListener('resize', function(){
    if(!state) return;
    clearTimeout(rzt);
    rzt = setTimeout(function(){ if(state){ sizeCanvas(state); draw(state); } }, 120);
  });
})();
