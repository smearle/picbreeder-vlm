(function () {
  var DATA_URL = 'breed/data/cppn_explainer.json';
  var store = null;

  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
  // Weight & node-value colormaps live in breed/weight-cmap.js (the PB_CMAP single
  // source of truth — flip its ACTIVE toggle to revert/swap schemes). Tiny grey
  // fallback if it's somehow absent. weightColor draws connection weights (with the
  // canvas figure's magnitude-fade alpha); valFill draws node-activation magnitude.
  var CMAP = window.PB_CMAP || {
    link: function (w, a) { return 'rgba(136,136,136,' + (a == null ? 1 : a) + ')'; },
    node: function () { return '#33384a'; },
  };
  function weightColor(w, alpha) { return CMAP.link(w, alpha); }

  fetch(DATA_URL).then(function (r) { return r.json(); }).then(function (d) {
    store = d.genomes || {};
    initDNA();
    initAnim();
  }).catch(function (e) { /* leave figures static on failure */ });

  // ------------------------------------------------ (B) DNA inspector
  // The editor IS the shared PB.DnaEditor widget (breed/dna-editor.js) — the same code that
  // powers breed/dna.html and breed/play.html — mounted here and driven by the Human/AI
  // bundle. The blog used to carry a near-verbatim fork of that editor, which kept drifting
  // out of sync (zoom-invariant pie menus, double-click-to-add, drag-node-to-link all went
  // missing); reusing the widget keeps the graph/menu/edit behaviour one-sourced. The
  // .cppnx-dna CSS above re-skins it for the blog; opts.minGraphH (canvas floor) and
  // opts.playUrl (Breed hand-off) cover the two breeder-site-specific details.
  function initDNA() {
    function metaOf(rec) { return { title: rec.title, model: rec.model || '', agent: rec.agent || '' }; }
    // Fixed to the human-evolved butterfly — no genome picker (the editor shows this one DNA).
    var rec0 = store['h_butterfly'] || store[Object.keys(store)[0]];
    if (!rec0) return;
    PB.DnaEditor(document.getElementById('cppnxDNAeditor'), {
      genome: rec0.g, color: rec0.color !== false, meta: metaOf(rec0),
      instructions: false, minGraphH: 440, playUrl: 'breed/play.html?dna=1',
      lockView: true,          // editing only — no zoom toolbar/wheel
      allowPan: true,          // …but keep click-and-drag panning to reach an overflowing graph
      actions: ['reset'],      // the sole bottom button
      // Fill-the-canvas layout (homepage only): big activation tiles, tight grid spacing,
      // fat draggable links, and a fit that fills the canvas width while auto-sizing its
      // height — so the graph fills the figure instead of sitting tiny and letterboxed.
      tileR: 46, colW: 120, rowV: 96, padX: 50, padY: 20,
      linkW: 11, linkWDisabled: 4,
      fitWidth: true, fitMargin: 0.98, fitMaxScale: 4
    });
  }

  // ------------------------------------------------------------- (A) animation
  function initAnim() {
    var gridC = document.getElementById('cppnxGrid'), gx = gridC.getContext('2d');
    var netC = document.getElementById('cppnxNet'), nx = netC.getContext('2d');
    var outC = document.getElementById('cppnxOut'), ox = outC.getContext('2d');
    var pixC = document.getElementById('cppnxPix'), px = pixC.getContext('2d');
    var hsvC = document.getElementById('cppnxHsv'), hx = hsvC.getContext('2d');
    // Logical drawing sizes (captured from the canvas attributes BEFORE we scale
    // them). All the geometry below works in these units; we blow the backing
    // store + on-screen box up by SCALE and draw through a matching transform, so
    // the whole pipeline renders crisply at 1.5× without touching any constant.
    var GS = gridC.width, OS = outC.width;
    var NETW = netC.width, NETH = netC.height, HVS = hsvC.width;
    var PIXW = pixC.width, PIXH = pixC.height;
    var SCALE = 1;
    [gridC, netC, outC, pixC, hsvC].forEach(function (c) {
      c.width = Math.round(c.width * SCALE); c.height = Math.round(c.height * SCALE);
    });
    [gx, nx, ox, px, hx].forEach(function (c) { c.setTransform(SCALE, 0, 0, SCALE, 0, 0); });
    var R = 44;                       // coarse render res for the demo
    // stay on the apple: the figure paints in ONE genome and never cycles away once filled
    var SHOW = ['h_apple'].filter(function (k) { return store[k]; });
    if (!SHOW.length) SHOW = Object.keys(store).slice(0, 1);

    var small = document.createElement('canvas'); small.width = small.height = R;
    var sctx = small.getContext('2d');

    // per-cell coordinates (constant): drive the forward pass and hover hit-test
    var CX = new Float32Array(R * R), CY = new Float32Array(R * R), CR = new Float32Array(R * R);
    for (var _c = 0; _c < R * R; _c++) {
      var _x = ((_c % R) * 2 - (R - 1)) / R, _y = (Math.floor(_c / R) * 2 - (R - 1)) / R;
      CX[_c] = _x; CY[_c] = _y; CR[_c] = Math.hypot(_x, _y);
    }

    var gi = 0, full = null, color = true, selCell = 0;
    // the output image is revealed by hovering: `build` accumulates the cells the
    // user has "coloured", `painted`/`paintedCount` track coverage, prev* lets us
    // join consecutive hover positions so fast moves don't leave gaps
    var build = sctx.createImageData(R, R), painted = new Uint8Array(R * R), paintedCount = 0;
    var prevCxp = -1, prevCyp = -1, brushRad = 1;   // current brush radius (grows on hold)
    // The probe is a ball with inertia: the cursor sets a target in cell-space and
    // the ball springs toward it, so it lags, overshoots and coasts to a stop
    // instead of teleporting. Position/velocity live in cell units, and are NOT
    // bounded by the grid — the ball is free to swing anywhere on the page, the
    // canvas is just a window onto it. Off-grid it paints nothing, but the network
    // still evaluates there (the CPPN extrapolates happily past the unit patch) so
    // the r-circle it's impaled on keeps swelling and shrinking as it roams.
    // A weak spring under light friction: the ball is slow off the mark, sails past
    // the cursor, and swings back through it a couple of times before it settles.
    var probeX = 0, probeY = 0, velX = 0, velY = 0, tgtX = 0, tgtY = 0, raf = 0;
    // false until the cursor has been seen at all; until then the ball has nowhere
    // to be but the seed cell.
    var aimed = false;
    var SPRING = 0.012, DAMP = 0.95;
    // Raw, the spring's pull grows without bound with the target's distance, so a
    // cursor a page away hurls the ball across the grid at tens of pixels a frame
    // (~47px/f for a cursor 600px off, ~79px/f from the corner of a big window).
    // Two knobs tame it, without flattening the bounce over the patch:
    //   MAXACC  a ceiling on how hard the ball is pulled, so a distant cursor tugs
    //           no harder than one just off the far corner of the grid.
    //   speedCap a ceiling on how fast it may ever GO, which tightens the further it
    //           strays from the centre of the patch: brisk over the grid, wading
    //           through the rim. This is the "gravity". It is no leash — the ball
    //           still chases the cursor ALL the way, wherever it is on the page.
    //           Past FAR0 (70 cells, far outside the canvas, where the ball is
    //           invisible and only the CPPN's saturated tail is on show) the ceiling
    //           lifts again, gently (FLY), so crossing a wide window takes ~3s
    //           rather than ~40s. It has slowed back to VRIM well before it
    //           re-enters view, so there is no wall at the canvas edge.
    // Tuned by measuring the OLD spring: it peaked at 4.9px/f chasing an on-grid
    // cursor, and these caps sit just under that — the same swing, a little calmer.
    var CC = (R - 1) / 2, PATCH = R / 2 * Math.SQRT2, SOFT = R / 2;
    var MAXACC = 0.55, MAXVEL = 1.9, VRIM = 0.6, FAR0 = 70, FLY = 0.022;
    function speedCap(d) {
      return VRIM + (MAXVEL - VRIM) / (1 + Math.max(0, d - PATCH) / SOFT)
                  + FLY * Math.max(0, d - FAR0);
    }
    var REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    // The figure waits behind a play glyph until the first click on the grid. Nothing
    // moves, nothing paints. Reduced-motion users have no animation to arm, so they
    // skip the gate entirely and get the live grid straight away.
    var started = REDUCED;

    // The actual network we are sampling: topology + per-pixel forward pass.
    // `net` is rebuilt for each genome; `drawNet(v)` colours every node by its
    // activation value `v[node]` at the pixel currently being painted, so the
    // graph stays in lock-step with the sweep instead of pulsing decoratively.
    var net = null;

    function nodeLabel(k) {
      if (k === -1) return 'x'; if (k === -2) return 'y';
      if (k === -3) return 'r'; if (k === -4) return 'b';
      if (k === 0) return 'h'; if (k === 1) return 's'; if (k === 2) return 'v';
      return null;
    }
    // Lay the real genome out left-to-right by feed-forward depth (inputs in
    // column 0, outputs in the last column) and record the compiled eval order
    // + input/output transforms so we can re-run the exact render-time pass.
    function buildLayout(g) {
      var evals = buildNet(g);
      var inF = g.inAct ? g.inAct.map(function (n) { return ACT[n] || ACT.identity; }) : null;
      var outF = g.outAct ? g.outAct.map(function (n) { return ACT[n] || ACT.identity; }) : null;
      var conns = [];
      g.conns.forEach(function (c) { if (c.enabled) conns.push([c.i, c.o]); });
      var layers = feedForwardLayers(INPUT_KEYS, OUTPUT_KEYS, conns).map(function (s) { return Array.from(s); });
      var lastCol = layers.length;            // outputs occupy the rightmost column
      var colOf = Object.create(null);
      INPUT_KEYS.forEach(function (k) { colOf[k] = 0; });
      layers.forEach(function (layer, i) { layer.forEach(function (n) { colOf[n] = i + 1; }); });
      OUTPUT_KEYS.forEach(function (k) { colOf[k] = lastCol; });
      var buckets = {};
      Object.keys(colOf).forEach(function (kk) { var k = +kk; (buckets[colOf[k]] = buckets[colOf[k]] || []).push(k); });
      Object.keys(buckets).forEach(function (c) { buckets[c].sort(function (a, b) { return a - b; }); });
      var W = NETW, H = NETH, padX = 24;
      var nodes = Object.create(null);
      Object.keys(buckets).forEach(function (cs) {
        var c = +cs, arr = buckets[c], n = arr.length;
        var x = lastCol > 0 ? padX + (W - 2 * padX) * (c / lastCol) : W / 2;
        arr.forEach(function (k, i) {
          var node = g.nodes.get(k) || {};
          var kind = INPUT_KEYS.indexOf(k) >= 0 ? 'input'
            : (OUTPUT_KEYS.indexOf(k) >= 0 ? 'output' : (node.affinity === 'color' ? 'color' : 'grey'));
          nodes[k] = { x: x, y: H * (i + 1) / (n + 1), kind: kind, label: nodeLabel(k) };
        });
      });
      var edges = [];
      g.conns.forEach(function (c) {
        if (!c.enabled) return;
        var a = nodes[c.i], b = nodes[c.o];
        if (a && b) edges.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, w: c.weight });
      });
      net = { evals: evals, inF: inF, outF: outF, nodes: nodes, edges: edges };
    }
    // Forward pass for one pixel, returning every node's value (mirrors the
    // inner loop of renderGenome so the values match the painted colour exactly).
    function forwardAt(sx, sy, d) {
      var inF = net.inF, outF = net.outF, evals = net.evals;
      var ix = sx, iy = sy, id = d, ib = 1.0;
      if (inF) { ix = inF[0](sx); iy = inF[1](sy); id = inF[2](d); ib = inF[3](1.0); }
      var v = Object.create(null);
      v[-1] = ix; v[-2] = iy; v[-3] = id; v[-4] = ib; v[0] = 0; v[1] = 0; v[2] = 0;
      for (var e = 0; e < evals.length; e++) {
        var ev = evals[e], s = 0, L = ev.links;
        for (var k = 0; k < L.length; k++) s += v[L[k][0]] * L[k][1];
        v[ev.node] = ev.act(s);
      }
      if (outF) { v[0] = outF[0](v[0]); v[1] = outF[1](v[1]); v[2] = outF[2](v[2]); }
      return v;
    }

    function loadGenome() {
      var rec = store[SHOW[gi % SHOW.length]];
      color = rec.color !== false;
      try {
        var g = genomeFromJSON(rec.g);
        full = renderGenome(g, R, color);
        buildLayout(g);
      } catch (e) { full = null; net = null; }
      if (!full) return;
      // start with a BLANK output image (just white) — the user reveals it by
      // hovering the grid. Reduced-motion users get the finished image up front.
      var d = build.data;
      for (var i = 0; i < d.length; i += 4) { d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = 255; }
      painted.fill(0); paintedCount = 0; prevCxp = -1; prevCyp = -1;
      selCell = Math.floor(Math.random() * R * R);
      if (!aimed) { tgtX = selCell % R; tgtY = Math.floor(selCell / R); }
      probeX = tgtX; probeY = tgtY;    // wherever the cursor is, even off the patch
      velX = velY = 0; launched = false; pinned = false;
      if (REDUCED) { for (var c = 0; c < R * R; c++) paintCell(c); }
      else { paintCell(selCell); }          // only the initial random cell is shown
      flushOut();
      renderProbe();
    }
    function paintCell(cell) {
      if (painted[cell]) return;
      painted[cell] = 1; paintedCount++;
      var i = cell * 4;
      build.data[i] = full.data[i]; build.data[i + 1] = full.data[i + 1];
      build.data[i + 2] = full.data[i + 2]; build.data[i + 3] = 255;
    }
    function paintBrush(cxp, cyp, rad) {     // round brush of radius `rad` (matches the ball)
      var lim = (rad + 0.5) * (rad + 0.5);
      for (var dy = -rad; dy <= rad; dy++) for (var dx = -rad; dx <= rad; dx++) {
        if (dx * dx + dy * dy > lim) continue;
        var x = cxp + dx, y = cyp + dy;
        if (x >= 0 && x < R && y >= 0 && y < R) paintCell(y * R + x);
      }
    }
    function flushOut() {
      sctx.putImageData(build, 0, 0);
      ox.imageSmoothingEnabled = false; ox.drawImage(small, 0, 0, R, R, 0, 0, OS, OS);
    }

    // node border = kind (categorical, below); fill = activation value, via the shared
    // PB_CMAP.node() ramp so it stays in the same family as the weight ramp.
    var BORDER = { input: '#3a59bc', output: '#FF6C00', color: '#8190c6', grey: '#9a9a9a' };
    function valFill(t) { return CMAP.node(t); }
    function drawNet(v) {
      nx.clearRect(0, 0, NETW, NETH);
      if (!net) return;
      net.edges.forEach(function (e) {                       // blue (+) / red (−) by weight, thicker + bolder = stronger
        var mag = Math.min(1, Math.abs(e.w) / 3);
        nx.lineWidth = 0.4 + 1.8 * mag;
        nx.strokeStyle = weightColor(e.w, 0.3 + 0.55 * mag);
        nx.beginPath(); nx.moveTo(e.x1, e.y1); nx.lineTo(e.x2, e.y2); nx.stroke();
      });
      nx.font = '9px ui-monospace, Menlo, monospace'; nx.textBaseline = 'middle';
      Object.keys(net.nodes).forEach(function (kk) {
        var k = +kk, nd = net.nodes[k], val = v ? v[k] : 0;
        var r = (nd.kind === 'input' || nd.kind === 'output') ? 6 : 5;
        nx.beginPath(); nx.arc(nd.x, nd.y, r, 0, 6.2832);
        nx.fillStyle = valFill(val == null ? 0 : val); nx.fill();
        nx.lineWidth = 1.6; nx.strokeStyle = BORDER[nd.kind] || '#9a9a9a'; nx.stroke();
        if (nd.label) {                                      // x,y,r,b on the left; h,s,v on the right
          nx.fillStyle = '#555';
          if (nd.kind === 'input') { nx.textAlign = 'right'; nx.fillText(nd.label, nd.x - r - 3, nd.y); }
          else { nx.textAlign = 'left'; nx.fillText(nd.label, nd.x + r + 3, nd.y); }
        }
      });
    }

    // ---- HSV-space locator: a hue/saturation wheel (angle = hue, radius =
    // saturation) plus a value bar; a marker shows where THIS pixel's output
    // colour sits. The wheel (at full value) is rendered once, then re-used. ----
    var DR = 46, DCx = 64, DCy = 50;
    var hsvDisc = (function () {
      var d = document.createElement('canvas'); d.width = d.height = 2 * DR;
      var dc = d.getContext('2d'), im = dc.createImageData(2 * DR, 2 * DR), data = im.data;
      for (var yy = 0; yy < 2 * DR; yy++) {
        for (var xx = 0; xx < 2 * DR; xx++) {
          var dx = xx - DR + 0.5, dy = yy - DR + 0.5, rr = Math.hypot(dx, dy), oi = (yy * 2 * DR + xx) * 4;
          if (rr <= DR) {
            var ang = Math.atan2(-dy, dx), hue = ((ang / (2 * Math.PI)) % 1 + 1) % 1;
            var c = hsv2rgb(hue, rr / DR, 1);
            data[oi] = c[0]; data[oi + 1] = c[1]; data[oi + 2] = c[2]; data[oi + 3] = 255;
          } else data[oi + 3] = 0;
        }
      }
      dc.putImageData(im, 0, 0); return d;
    })();
    // white-haloed single-char tag (matches the grid's x/y/r gauge labels)
    function hsvTag(ch, x, y) {
      hx.font = '9px ui-monospace, Menlo, monospace'; hx.textAlign = 'center'; hx.textBaseline = 'middle';
      hx.lineWidth = 2.5; hx.strokeStyle = '#fff'; hx.strokeText(ch, x, y);
      hx.fillStyle = '#333'; hx.fillText(ch, x, y);
    }
    function drawHSV(v) {
      hx.clearRect(0, 0, HVS, HVS); hx.fillStyle = '#fff'; hx.fillRect(0, 0, HVS, HVS);
      // hue/sat wheel — always at full value (value does NOT dim it)
      hx.drawImage(hsvDisc, DCx - DR, DCy - DR);
      hx.lineWidth = 1; hx.strokeStyle = 'rgba(0,0,0,.15)';
      hx.beginPath(); hx.arc(DCx, DCy, DR, 0, 6.2832); hx.stroke();
      // value bar — horizontal, lying under the wheel
      var bx0 = DCx - DR, bx1 = DCx + DR, by = DCy + DR + 12, bh = 11;
      var grad = hx.createLinearGradient(bx0, 0, bx1, 0);
      grad.addColorStop(0, '#000'); grad.addColorStop(1, '#fff');
      hx.fillStyle = grad; hx.fillRect(bx0, by, bx1 - bx0, bh);
      hx.lineWidth = 1; hx.strokeStyle = 'rgba(0,0,0,.25)'; hx.strokeRect(bx0, by, bx1 - bx0, bh);
      if (v) {
        var hue = pmod(v[0] + 1, 1), sat = color ? clamp(v[1], 0, 1) : 0, val = clamp(Math.abs(v[2]), 0, 1);
        // marker at (hue angle, saturation radius) on the wheel
        var ang = hue * 2 * Math.PI, mx = DCx + Math.cos(ang) * sat * DR, my = DCy - Math.sin(ang) * sat * DR;
        // --- make the placement legible: hue sets the ANGLE, saturation the RADIUS ---
        // saturation = distance from the hub: a white-cased spoke from centre to marker
        hx.lineCap = 'round';
        hx.lineWidth = 4; hx.strokeStyle = 'rgba(255,255,255,.85)';
        hx.beginPath(); hx.moveTo(DCx, DCy); hx.lineTo(mx, my); hx.stroke();
        hx.lineWidth = 2; hx.strokeStyle = 'rgba(20,20,20,.65)';
        hx.beginPath(); hx.moveTo(DCx, DCy); hx.lineTo(mx, my); hx.stroke();
        hx.lineCap = 'butt';
        // hue = angle swept from the 0° (red) reference: dashed baseline + a short arc to the marker
        hx.setLineDash([2, 2]); hx.lineWidth = 1; hx.strokeStyle = 'rgba(0,0,0,.28)';
        hx.beginPath(); hx.moveTo(DCx, DCy); hx.lineTo(DCx + DR, DCy); hx.stroke();
        hx.setLineDash([]);
        var arcR = 15;
        hx.lineWidth = 2; hx.strokeStyle = 'rgba(20,20,20,.65)';
        hx.beginPath(); hx.arc(DCx, DCy, arcR, 0, -ang, true); hx.stroke();
        // 's' beside the spoke (radius), 'h' inside the swept arc (angle)
        hsvTag('s', DCx + Math.cos(ang) * sat * DR * 0.5 - Math.sin(ang) * 8,
                    DCy - Math.sin(ang) * sat * DR * 0.5 - Math.cos(ang) * 8);
        hsvTag('h', DCx + Math.cos(ang * 0.5) * (arcR + 8),
                    DCy - Math.sin(ang * 0.5) * (arcR + 8));
        hx.beginPath(); hx.arc(mx, my, 4.5, 0, 6.2832);
        hx.lineWidth = 2.5; hx.strokeStyle = '#fff'; hx.stroke();
        hx.lineWidth = 1.4; hx.strokeStyle = '#222'; hx.stroke();
        // value marker on the horizontal bar
        var vx = bx0 + val * (bx1 - bx0);
        hx.beginPath(); hx.moveTo(vx, by - 3); hx.lineTo(vx, by + bh + 3);
        hx.lineWidth = 2; hx.strokeStyle = '#FF6C00'; hx.stroke();
      }
      hx.font = '8px ui-monospace, Menlo, monospace'; hx.fillStyle = '#888';
      hx.textAlign = 'left'; hx.textBaseline = 'alphabetic'; hx.fillText('value', bx0, by + bh + 9);
    }

    // colored bar with a faint casing so pale (near-zero) values still read
    function gridBar(x1, y1, x2, y2, val, w) {
      gx.lineCap = 'round';
      gx.lineWidth = w + 2; gx.strokeStyle = 'rgba(120,120,120,.35)';
      gx.beginPath(); gx.moveTo(x1, y1); gx.lineTo(x2, y2); gx.stroke();
      gx.lineWidth = w; gx.strokeStyle = valFill(val == null ? 0 : val);
      gx.beginPath(); gx.moveTo(x1, y1); gx.lineTo(x2, y2); gx.stroke();
      gx.lineCap = 'butt';
    }
    // faint full-extent track that a fixed gauge bar fills against
    function gridTrack(x1, y1, x2, y2) {
      gx.lineCap = 'round'; gx.lineWidth = 7; gx.strokeStyle = 'rgba(0,0,0,.05)';
      gx.beginPath(); gx.moveTo(x1, y1); gx.lineTo(x2, y2); gx.stroke(); gx.lineCap = 'butt';
    }
    function barLabel(x, y, ch) {
      gx.font = '8px ui-monospace, Menlo, monospace'; gx.textAlign = 'center'; gx.textBaseline = 'middle';
      gx.fillStyle = '#fff'; gx.lineWidth = 2.5; gx.strokeStyle = '#fff';
      gx.strokeText(ch, x, y); gx.fillStyle = '#555'; gx.fillText(ch, x, y);
    }
    function drawGrid(v) {
      gx.clearRect(0, 0, GS, GS); gx.fillStyle = '#fff'; gx.fillRect(0, 0, GS, GS);
      // The canvas is bigger than the sampled patch so the r-circle (radius up to
      // U·√2 at the corners) always fits; the faint lattice covers ONLY the unit
      // patch [-1,1]^2 we actually sample.
      var C = GS / 2, U = 46, cs = 2 * U / R;
      function PX(s) { return C + s * U; }
      gx.strokeStyle = '#eee'; gx.lineWidth = 1;
      for (var k = 0; k <= R; k += 4) {
        var t = C + (-1 + 2 * k / R) * U;
        gx.beginPath(); gx.moveTo(C - U, t); gx.lineTo(C + U, t); gx.stroke();
        gx.beginPath(); gx.moveTo(t, C - U); gx.lineTo(t, C + U); gx.stroke();
      }
      // axes through the centre (within the patch)
      gx.strokeStyle = '#dcdcdc';
      gx.beginPath(); gx.moveTo(C - U, C); gx.lineTo(C + U, C); gx.stroke();
      gx.beginPath(); gx.moveTo(C, C - U); gx.lineTo(C, C + U); gx.stroke();

      // Before the first click the grid is inert: a bare lattice under a play glyph,
      // no ball, no gauges, no r-circle (which would otherwise sit right under it).
      // Clicking starts the probe for good — there is no way back to this state.
      if (!started) {
        gx.save();
        gx.shadowColor = 'rgba(0,0,0,.28)'; gx.shadowBlur = 7; gx.shadowOffsetY = 1.5;
        gx.beginPath(); gx.arc(C, C, 17, 0, 6.2832);
        gx.fillStyle = '#fff'; gx.fill();       // white disc lifts the glyph off the lattice
        gx.restore();
        gx.lineWidth = 1; gx.strokeStyle = 'rgba(0,0,0,.10)'; gx.stroke();
        var pw = 13, ph = 15;                    // triangle, optically centred (not bbox-centred)
        gx.beginPath();
        gx.moveTo(C - pw * 0.42, C - ph / 2); gx.lineTo(C - pw * 0.42, C + ph / 2);
        gx.lineTo(C + pw * 0.58, C); gx.closePath();
        gx.fillStyle = '#000'; gx.fill();
        return;
      }

      // mark already-visited cells in grey (mirrors what's been coloured in the output)
      gx.fillStyle = 'rgba(110,110,110,.32)';
      for (var pc = 0; pc < R * R; pc++) {
        if (!painted[pc]) continue;
        gx.fillRect(PX(CX[pc]) - cs / 2, PX(CY[pc]) - cs / 2, cs + 0.6, cs + 0.6);
      }

      // the ball sits at its exact (fractional) position, so its drift reads smoothly
      // even though the sampled values below come from the cell it currently covers
      var dotx = PX((probeX * 2 - (R - 1)) / R), doty = PX((probeY * 2 - (R - 1)) / R);
      // coordinate bars, each coloured by the value its INPUT NODE takes on in
      // the network panel (same valFill): x is a fixed gauge along the bottom,
      // y a fixed gauge along the left, r the circle of that radius about centre.
      if (v) {
        var yBot = GS - 5, xLeft = 5;
        // faint guides projecting the pixel onto each gauge
        gx.setLineDash([2, 2]); gx.lineWidth = 1; gx.strokeStyle = 'rgba(0,0,0,.13)';
        gx.beginPath(); gx.moveTo(dotx, doty); gx.lineTo(dotx, yBot); gx.stroke();
        gx.beginPath(); gx.moveTo(dotx, doty); gx.lineTo(xLeft, doty); gx.stroke();
        gx.setLineDash([]);
        var rad = Math.hypot(dotx - C, doty - C);
        if (rad > 1) {
          gx.beginPath(); gx.arc(C, C, rad, 0, 6.2832);
          gx.lineWidth = 5; gx.strokeStyle = 'rgba(120,120,120,.30)'; gx.stroke();
          gx.lineWidth = 3; gx.strokeStyle = valFill(v[-3]); gx.stroke();
        }
        // the gauges only span the unit patch, so a ball that has wandered off the
        // grid pins its bar at the end of the track rather than running off-canvas
        gridTrack(C - U, yBot, C + U, yBot);      // x — fixed gauge along the bottom
        gridBar(C, yBot, clamp(dotx, C - U, C + U), yBot, v[-1], 5);
        gridTrack(xLeft, C - U, xLeft, C + U);    // y — fixed gauge along the left
        gridBar(xLeft, C, xLeft, clamp(doty, C - U, C + U), v[-2], 5);
        barLabel(C - U - 6, yBot, 'x');           // x label to the left of its axis
        barLabel(xLeft, C - U - 6, 'y');
        barLabel(C, C, 'r');                      // r label at the centre of the grid
      }
      // the ball: a black ring enclosing the cells the brush covers (so the bigger
      // held-brush is visible), with a solid hub; distinct from the orange accents.
      // Pinned, it wears a dashed collar so it reads as parked rather than stalled.
      var hr = Math.max(cs * (brushRad + 0.5), 3.5);
      gx.beginPath(); gx.arc(dotx, doty, hr, 0, 6.2832);
      gx.fillStyle = 'rgba(0,0,0,.14)'; gx.fill();
      gx.lineWidth = 1.4; gx.strokeStyle = '#000'; gx.stroke();
      gx.beginPath(); gx.arc(dotx, doty, 2.4, 0, 6.2832); gx.fillStyle = '#000'; gx.fill();
      if (pinned) {
        gx.setLineDash([2, 2]); gx.lineWidth = 1.4; gx.strokeStyle = '#FF6C00';
        gx.beginPath(); gx.arc(dotx, doty, hr + 3, 0, 6.2832); gx.stroke();
        gx.setLineDash([]);
      }
    }

    // Inspect wherever the ball is — a continuous (x, y), on the grid or well off it.
    // The forward pass and the swatch colour come straight from that coordinate (the
    // same maths renderGenome uses per pixel), so the grid bars, network nodes and
    // swatch agree everywhere instead of only at sampled cell centres.
    function renderProbe() {
      if (!full) return;
      var sx = (probeX * 2 - (R - 1)) / R, sy = (probeY * 2 - (R - 1)) / R;
      var vv = net ? forwardAt(sx, sy, Math.hypot(sx, sy) * Math.SQRT2) : null;
      drawGrid(vv);
      drawNet(vv);
      drawHSV(vv);
      var rgb = [255, 255, 255];
      if (vv) {
        rgb = color ? hsv2rgb(pmod(vv[0] + 1, 1), clamp(vv[1], 0, 1), clamp(Math.abs(vv[2]), 0, 1))
                    : (function (g) { return [g, g, g]; })(Math.round(clamp(Math.abs(vv[2]), 0, 1) * 255));
      }
      px.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
      px.fillRect(0, 0, PIXW, PIXH);
    }

    // The cursor sets the ball's target — anywhere on the page, not just over the
    // grid — through the same unit-patch geometry drawGrid uses. No clamping: a
    // cursor parked three figures away is simply a target three figures away.
    // The viewport coords are kept so the target can be re-derived against a fresh
    // canvas rect whenever the figure moves under a stationary cursor (see reaim).
    var curCX = 0, curCY = 0;
    function aimAt(e) { curCX = e.clientX; curCY = e.clientY; aimed = true; reaim(); }
    function reaim() {
      var rect = gridC.getBoundingClientRect();
      var mx = (curCX - rect.left) * (GS / rect.width);
      var my = (curCY - rect.top) * (GS / rect.height);
      var C = GS / 2, U = 46;
      tgtX = ((mx - C) / U * R + (R - 1)) / 2;
      tgtY = ((my - C) / U * R + (R - 1)) / 2;
    }
    var down = false, SMALL = 1, BIG = 4;    // press/hold → a bigger brush
    // `pinned`: a click on the grid parks the ball there until the next click.
    // pdX/pdY/pdMoved tell a click apart from a drag (see release()).
    var pinned = false, pdX = 0, pdY = 0, pdMoved = false;
    // colour the brush along the path from the last position (no gaps), then repaint
    // every panel. Off the grid there is nothing to colour, so we only move the ball
    // — and drop the stroke, so coming back on doesn't chord across the patch.
    function paintTo(fx, fy, rad) {
      probeX = fx; probeY = fy; brushRad = rad;   // so the grid ball shows the brush extent
      var cxp = Math.round(fx), cyp = Math.round(fy);
      if (cxp >= 0 && cxp < R && cyp >= 0 && cyp < R) {
        selCell = cyp * R + cxp;
        if (prevCxp >= 0) {
          var steps = Math.max(1, Math.max(Math.abs(cxp - prevCxp), Math.abs(cyp - prevCyp)));
          for (var s = 1; s <= steps; s++) {
            paintBrush(Math.round(prevCxp + (cxp - prevCxp) * s / steps),
                       Math.round(prevCyp + (cyp - prevCyp) * s / steps), rad);
          }
        } else paintBrush(cxp, cyp, rad);
        prevCxp = cxp; prevCyp = cyp;
        flushOut();
      } else { prevCxp = -1; prevCyp = -1; }
      renderProbe();
      // once the whole grid is coloured in, the apple simply stays revealed — we do
      // NOT advance to another genome (the figure is fixed on the apple by design).
    }
    // One spring step per frame. The loop parks itself once the ball has caught up
    // with the cursor and stopped, and any pointer move kicks it back awake.
    function step() {
      raf = 0;
      if (pinned) return;
      var dx = tgtX - probeX, dy = tgtY - probeY;
      var ax = dx * SPRING, ay = dy * SPRING;
      var am = Math.hypot(ax, ay);                      // cap the pull toward the cursor
      if (am > MAXACC) { ax = ax / am * MAXACC; ay = ay / am * MAXACC; }
      velX = (velX + ax) * DAMP; velY = (velY + ay) * DAMP;
      // …and cap the speed by where the ball IS: brisk over the patch, slow far out
      var cap = speedCap(Math.hypot(probeX - CC, probeY - CC)), sp = Math.hypot(velX, velY);
      if (sp > cap) { velX = velX / sp * cap; velY = velY / sp * cap; sp = cap; }
      var settled = sp < 0.02 && Math.hypot(ax, ay) < 0.0008;
      var nx2 = probeX + velX, ny2 = probeY + velY;
      if (settled) {
        if (Math.hypot(dx, dy) < 0.05) { nx2 = tgtX; ny2 = tgtY; }
        velX = velY = 0;
      }
      paintTo(nx2, ny2, down ? BIG : SMALL);
      if (!settled) raf = requestAnimationFrame(step);
    }
    function kick() { if (!raf && !REDUCED && inView && !pinned && started) raf = requestAnimationFrame(step); }

    // Scrolling the figure into view flicks the ball off with a random velocity, so
    // it arrives already drifting — the spring is far too weak to make a standing
    // start look like anything but a stall. The ball starts under the cursor, off
    // the patch as often as not, and orbits it; only if the cursor has never been
    // seen does it start on the seed cell instead. Re-armed by each reveal, so "r"
    // re-flicks it too.
    var launched = false, inView = false;
    function launch() {
      if (launched || REDUCED || !full || !started) return;
      launched = true;
      if (aimed) { probeX = tgtX; probeY = tgtY; }
      var ang = Math.random() * 2 * Math.PI, sp = 0.5 + Math.random() * 0.4;
      velX = Math.cos(ang) * sp; velY = Math.sin(ang) * sp;
      kick();
    }
    if (window.IntersectionObserver) {
      // stays observing: off screen, inView gates the rAF loop off entirely
      new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          inView = en.isIntersecting;
          if (inView) { launch(); kick(); }
        });
      }, { threshold: 0.5 }).observe(gridC);
    } else { inView = true; launch(); }
    gridC.style.cursor = started ? 'crosshair' : 'pointer';
    gridC.style.touchAction = 'none';        // let press-drag paint instead of scrolling
    // The cursor is tracked page-wide, not just over the canvas: the ball chases it
    // out of the figure and back in again, painting only the stretch it drags across
    // the grid. Only while the figure is on screen (kick() no-ops otherwise).
    window.addEventListener('pointermove', function (e) {
      if (down && Math.hypot(e.clientX - pdX, e.clientY - pdY) > 4) pdMoved = true;
      aimAt(e);
      if (pinned) return;
      if (REDUCED) paintTo(tgtX, tgtY, down ? BIG : SMALL); else kick();
    });
    gridC.addEventListener('pointerdown', function (e) {
      down = true; pdX = e.clientX; pdY = e.clientY; pdMoved = false;
      aimAt(e);
      if (!started) {                       // the play glyph: first click arms the probe
        started = true; pdMoved = true;     // …and does NOT also count as a pin-toggle click
        gridC.style.cursor = 'crosshair';
        launch();
        return;
      }
      if (pinned) { renderProbe(); return; }
      if (REDUCED) paintTo(tgtX, tgtY, BIG); else kick();
    });
    // A press-and-drag on the grid is the fat brush; a plain click (pointer barely
    // moved) pins the ball where it stands, and the next click cuts it loose again
    // from that same spot. Pinned, it ignores the cursor entirely.
    function release(e) {
      var wasDown = down; down = false;
      if (wasDown && !pdMoved && e && e.type === 'pointerup' && e.target === gridC) {
        pinned = !pinned;
        if (pinned) {
          velX = velY = 0;
          if (raf) { cancelAnimationFrame(raf); raf = 0; }
        }
      }
      if (pinned) { brushRad = SMALL; renderProbe(); return; }
      kick();                                       // redraw the ball at the smaller radius
      if (REDUCED) paintTo(probeX, probeY, SMALL);
    }
    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);

    // Scrolling (or resizing) slides the grid under a cursor that never moved, so no
    // pointermove fires and the target would be left behind at its old cell. Re-derive
    // it from the stored viewport coords: the ball chases the figure as it rises up
    // the page, and a cursor left over the grid keeps painting the whole way.
    function drift() {
      if (!aimed) return;
      reaim();
      if (!REDUCED) kick();
    }
    window.addEventListener('scroll', drift, { passive: true });
    window.addEventListener('resize', drift);

    // "r" wipes the revealed image back to blank (one random seed cell), so the
    // figure can be re-explored from scratch without reloading the page. Ignore
    // it while typing in a field or when a modifier is held.
    window.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (t && t.isContentEditable)) return;
      if (e.key === 'r' || e.key === 'R') { loadGenome(); if (inView) launch(); }
    });

    loadGenome();
  }
})();
