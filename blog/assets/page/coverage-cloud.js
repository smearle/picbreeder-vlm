/* Visual / Semantic Coverage cloud hover-bloom.
 *
 * Each coverage panel (grid_visual.png / grid_semantic.png) is a static UMAP
 * scatter of ~24k pooled archive items. This turns the dots into pictures under
 * the cursor: mouse over the cloud and the nearest sampled individuals snap in
 * — nearest = biggest — from a lazily-loaded thumbnail atlas, so nothing weighs
 * on page load and the static figure is untouched until you actually hover it.
 *
 * The bloom is a spotlight, not paint: only the individuals near the cursor
 * right now are drawn, and the canvas clears the moment the cursor leaves.
 * Opacity and size lerp toward their targets over a few frames, and opacity is
 * pinned to 1 wherever the cursor is actually inside a thumbnail — so whatever
 * you point at is solid, and its neighbours ghost off with distance.
 *
 * On the Semantic panel — a picture of caption space — the centre individual (the
 * one under the cursor) also shows the caption a VLM wrote for it, on a translucent
 * white plate. The Visual panel carries no captions and shows none.
 *
 * Data (assets/coverage_clouds/<space>/, built by tools/build_coverage_clouds.py):
 *   manifest.json  {space,w,h,cell,cols,count,atlas,pts:[[nx,ny],...],caps?:[str,...]}
 *                  pts: normalized [0,1] positions in the PNG frame (y down);
 *                  atlas slot i -> (col=i%cols, row=i//cols).
 *                  caps (semantic only): the VLM caption of point i, "" if unknown.
 *   atlas.jpg      cell x cell thumbnails, atlas order.
 *
 * A transparent canvas is overlaid exactly on the panel <img> (which keeps its
 * dots and alt text). Only fine-hover pointers activate it, so touch devices
 * never fetch the atlas.
 */
(function () {
  var fineHover = !window.matchMedia || window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  if (!fineHover) return;                       // no cursor -> leave the static figure

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var panels = document.querySelectorAll('.cov-cloud');
  for (var i = 0; i < panels.length; i++) setup(panels[i]);

  function setup(root) {
    var img = root.querySelector('img');
    if (!img) return;
    var base = root.getAttribute('data-base');
    if (!base) return;

    var s = {
      root: root, img: img, base: base,
      canvas: null, ctx: null, atlas: null, pts: null, caps: null, capIdx: -1,
      cols: 0, cell: 0, count: 0,
      W: 0, H: 0,                 // CSS px size of the panel
      mx: -1, my: -1, over: false,
      alpha: null, szv: null,     // Float32Arrays, per-point eased opacity / size 0..1
      live: [],                   // indices with alpha > 0, i.e. worth drawing
      raf: 0, loading: false, ready: false
    };

    // Lazily build everything on first hover; a bare mouseenter just kicks the fetch.
    root.addEventListener('pointerenter', function () { s.over = true; ensure(s); });
    root.addEventListener('pointermove', function (e) { s.over = true; onMove(s, e); });
    root.addEventListener('pointerleave', function () { s.over = false; s.mx = s.my = -1; if (s.ready) tick(s); });
    window.addEventListener('resize', function () { if (s.ready) { size(s); tick(s); } }, { passive: true });
  }

  // ------------------------------------------------------------------ activation
  function ensure(s) {
    if (s.ready || s.loading) return;
    s.loading = true;
    fetch(s.base + 'manifest.json').then(function (r) { return r.json(); }).then(function (m) {
      s.pts = m.pts; s.count = m.count; s.cols = m.cols; s.cell = m.cell;
      s.caps = m.caps || null;                  // semantic cloud only: VLM caption per point
      s.alpha = new Float32Array(m.count);
      s.szv = new Float32Array(m.count);
      var atlas = new Image();
      atlas.decoding = 'async';
      atlas.onload = function () {
        s.atlas = atlas;
        buildCanvas(s);
        s.ready = true;
        size(s);
        tick(s);
      };
      atlas.onerror = function () { s.loading = false; };  // leave the static figure
      atlas.src = s.base + (m.atlas || 'atlas.jpg');
    }).catch(function () { s.loading = false; });
  }

  function buildCanvas(s) {
    var c = document.createElement('canvas');
    c.className = 'cov-cloud-canvas';
    s.root.appendChild(c);
    s.canvas = c;
    s.ctx = c.getContext('2d');
  }

  function size(s) {
    var r = s.img.getBoundingClientRect();
    s.W = r.width; s.H = r.height;
    var dpr = window.devicePixelRatio || 1;
    s.canvas.width = Math.max(1, Math.round(s.W * dpr));
    s.canvas.height = Math.max(1, Math.round(s.H * dpr));
    s.canvas.style.width = s.W + 'px';
    s.canvas.style.height = s.H + 'px';
    s.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function onMove(s, e) {
    var r = s.img.getBoundingClientRect();
    s.mx = e.clientX - r.left;   // track even pre-load, so it blooms the instant
    s.my = e.clientY - r.top;    // the atlas is ready without needing another move
    if (!s.ready) { ensure(s); return; }
    tick(s);
  }

  // ------------------------------------------------------------------ painting
  // Reveal radius, thumbnail sizes and the nearest-N cap all scale with panel
  // width so the bloom feels identical at any column width.
  function tick(s) {
    if (s.raf) return;                          // coalesce moves to one paint per frame
    s.raf = requestAnimationFrame(function () { s.raf = 0; frame(s); });
  }

  // Opacity vs. distance. Solid while the cursor is inside the thumbnail's own
  // footprint, then falling off to nothing at the edge of `reach` — so the image
  // you are pointing at reads at full strength and its neighbours ghost away.
  // The exponent < 1 keeps the near ring substantial rather than washing it out.
  function opacityAt(d, szPx, reach) {
    var solid = szPx * 0.5;
    if (d <= solid) return 1;
    var u = (d - solid) / (reach - solid);
    return Math.pow(1 - Math.min(1, u), 0.7);
  }

  function frame(s) {
    var W = s.W, H = s.H, pts = s.pts;
    var reach = 0.42 * W;                        // reveal radius
    var reach2 = reach * reach;
    var maxS = 0.155 * W;                        // (kept in sync with thumbSize())
    var sep = maxS * 0.62;                       // min gap between bloomed thumbs
    var CAP = 11;                                // how many thumbs bloom at once

    // Within `reach`, greedily accept points nearest-first but only if they clear
    // `sep` from every already-accepted thumb, so the bloom spreads into a legible
    // cluster (nearest = biggest) instead of piling up under the cursor. The
    // evenly-spaced FPS sample is dense, so unfiltered nearest-N would all land in
    // the same spot.
    var hot = [];
    if (s.over && s.mx >= 0) {
      var cand = [];
      for (var i = 0; i < pts.length; i++) {
        var dx = pts[i][0] * W - s.mx, dy = pts[i][1] * H - s.my;
        var d2 = dx * dx + dy * dy;
        if (d2 <= reach2) cand.push([i, d2, pts[i][0] * W, pts[i][1] * H]);
      }
      cand.sort(function (a, b) { return a[1] - b[1]; });
      var acc = [], sep2 = sep * sep;
      for (var k = 0; k < cand.length && acc.length < CAP; k++) {
        var px = cand[k][2], py = cand[k][3], ok = true;
        for (var a2 = 0; a2 < acc.length; a2++) {
          var ax = acc[a2][0] - px, ay = acc[a2][1] - py;
          if (ax * ax + ay * ay < sep2) { ok = false; break; }
        }
        if (!ok) continue;
        acc.push([px, py]);
        var d = Math.sqrt(cand[k][1]);
        var ts = Math.pow(Math.max(0, 1 - d / reach), 0.55);   // 1 at cursor -> 0 at edge
        hot.push([cand[k][0], ts, opacityAt(d, thumbSize(s, ts), reach)]);
      }
    }

    // Ease opacity and size toward those targets; anything the cursor has left
    // behind targets 0 and fades out. `live` carries the still-visible points
    // between frames so we never rescan all ~24k.
    var targ = {}, moving = false;
    for (var n = 0; n < hot.length; n++) targ[hot[n][0]] = hot[n];
    var pending = s.live.slice();
    for (var n = 0; n < hot.length; n++) if (!s.alpha[hot[n][0]]) pending.push(hot[n][0]);

    var ease = reduce ? 1 : 0.24;               // per-frame approach to target
    var live = [];
    for (var p = 0; p < pending.length; p++) {
      var i2 = pending[p], h = targ[i2];
      var ta = h ? h[2] : 0, tz = h ? h[1] : s.szv[i2];   // fading out keeps its size
      var a = s.alpha[i2] + (ta - s.alpha[i2]) * ease;
      var z = s.szv[i2] + (tz - s.szv[i2]) * ease;
      if (Math.abs(ta - a) > 0.004 || Math.abs(tz - z) > 0.004) moving = true;
      else { a = ta; z = tz; }
      s.alpha[i2] = a; s.szv[i2] = z;
      if (a > 0.004) live.push(i2);
    }
    s.live = live;

    // The nearest thumb — the biggest, the one you're pointing at — is the "centre"
    // individual. Remember it while fading out so its caption fades with it rather
    // than vanishing the instant the cursor leaves.
    if (hot.length) s.capIdx = hot[0][0];

    var ctx = s.ctx;
    ctx.clearRect(0, 0, W, H);
    live.sort(function (a, b) { return s.szv[a] - s.szv[b]; });   // biggest last -> topmost
    for (var n = 0; n < live.length; n++) drawThumb(s, ctx, live[n]);
    drawCaption(s, ctx);
    if (moving) tick(s);                        // keep easing until settled
  }

  // ------------------------------------------------------------------- caption
  // Semantic Coverage is a picture of CAPTION space, so the caption is the thing the
  // panel is actually about: show the centre individual's, on a translucent white
  // plate under its thumbnail. The visual cloud has no captions and gets none.
  function wrap(ctx, text, maxW) {
    var words = text.split(/\s+/), lines = [], line = '';
    for (var i = 0; i < words.length; i++) {
      var t = line ? line + ' ' + words[i] : words[i];
      if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = words[i]; }
      else line = t;
    }
    if (line) lines.push(line);
    if (lines.length > 3) {                    // keep the plate small; elide the rest
      lines = lines.slice(0, 3);
      lines[2] = lines[2].replace(/[.,]$/, '') + '…';
    }
    return lines;
  }

  function drawCaption(s, ctx) {
    if (!s.caps || s.capIdx < 0) return;
    var i = s.capIdx, a = s.alpha[i];
    if (a <= 0.01) return;
    var txt = s.caps[i];
    if (!txt) return;                          // the handful of points with no caption

    var W = s.W, H = s.H;
    var fs = Math.max(9, 0.034 * W);           // ~the scale of the panel's own callouts
    ctx.font = 'italic ' + fs.toFixed(1) + 'px "Helvetica Neue",Arial,sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    var maxTextW = Math.min(0.62 * W, 220);
    var lines = wrap(ctx, txt, maxTextW);
    var lh = fs * 1.22, padX = 0.6 * fs, padY = 0.42 * fs;
    var boxW = 0, k;
    for (k = 0; k < lines.length; k++) boxW = Math.max(boxW, ctx.measureText(lines[k]).width);
    boxW += 2 * padX;
    var boxH = lines.length * lh + 2 * padY;

    // Under the thumb by default; flip above when that would run off the panel.
    var sz = thumbSize(s, s.szv[i]);
    var cx = s.pts[i][0] * W, cy = s.pts[i][1] * H;
    var top = cy + sz / 2 + 0.42 * fs;
    if (top + boxH > H - 2) top = cy - sz / 2 - 0.42 * fs - boxH;
    top = Math.max(2, Math.min(H - boxH - 2, top));
    var left = Math.max(2, Math.min(W - boxW - 2, cx - boxW / 2));

    ctx.globalAlpha = a;
    ctx.fillStyle = 'rgba(255,255,255,0.86)';
    ctx.strokeStyle = 'rgba(0,0,0,0.10)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(left, top, boxW, boxH, 3);
    else ctx.rect(left, top, boxW, boxH);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#444';
    for (k = 0; k < lines.length; k++) {
      ctx.fillText(lines[k], left + boxW / 2, top + padY + k * lh);
    }
    ctx.globalAlpha = 1;
  }

  function thumbSize(s, a) {
    var W = s.W;
    return 0.06 * W + (0.155 * W - 0.06 * W) * a;
  }

  function drawThumb(s, ctx, i) {
    var sz = thumbSize(s, s.szv[i]);
    var cx = s.pts[i][0] * s.W, cy = s.pts[i][1] * s.H;
    var sx = (i % s.cols) * s.cell, sy = Math.floor(i / s.cols) * s.cell;
    ctx.globalAlpha = s.alpha[i];
    ctx.drawImage(s.atlas, sx, sy, s.cell, s.cell, cx - sz / 2, cy - sz / 2, sz, sz);
    ctx.globalAlpha = 1;
  }
})();
