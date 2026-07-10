/* ===================================================================
 * PB.DnaEditor — the "Analyze Image DNA" editor as a reusable widget.
 *
 * This is the editor that powers dna.html, factored out so the DNA
 * panel on play.html can reuse it verbatim instead of duplicating the
 * graph/render/edit machinery. Both pages share dna-editor.css.
 *
 *   const ed = PB.DnaEditor(containerEl, {
 *     genome,                 // a cppn.js Genome  OR  genome JSON
 *     color: true,            // initial color-mode for the preview
 *     meta: { title, model, agent },   // labelling + "Breed" hand-off
 *     actions: ['breed','download','png','reset'],   // toolbar buttons
 *     instructions: true,     // show the help list under the toolbar
 *     onChange(genome) { ... }// fired after EVERY edit (and reset/load)
 *   });
 *   ed.setGenome(g, {color, title});  ed.getGenome();  ed.reset();
 *
 * The editor mutates the genome object it is given in place (set
 * activation / weight / add / delete all edit it directly), so a host
 * that passes a live genome sees edits immediately; onChange also fires
 * so the host can re-render and so it can follow the reset (which swaps
 * in a fresh object rebuilt from the original snapshot).
 *
 * Depends on cppn.js globals (Genome, genomeFromJSON, genomeToJSON,
 * renderGenome, buildNet, ACT, ACT_OPTIONS, INPUT_KEYS, OUTPUT_KEYS,
 * WEIGHT_MIN/MAX, clamp, choice, randWeight, createsCycle) and on PB.
 * =================================================================== */
(function () {
  // Weight/value colormap — single source of truth in breed/weight-cmap.js
  // (load it before this script). Tiny grey fallback if it's somehow absent.
  const WCMAP = window.PB_CMAP || { link: function () { return '#888'; }, pos: '#888', neg: '#888' };
  // GPU CPPN renderer (breed/cppn-gl.js). Optional: every page that mounts the
  // editor loads it, but if it (or WebGL) is missing we transparently fall back
  // to the cppn.js CPU path. Re-checked per call so a lost context degrades live.
  // NB read window.PB explicitly: on the breeder site pbsite.js declares a *lexical*
  // `const PB` that shadows window.PB, but cppn-gl.js attaches CppnGL to window.PB.
  const GL = () => { const E = window.PB && window.PB.CppnGL; return (E && E.available()) ? E : null; };
  const ACT_ABBR = { identity: '=', sin: 'sin', cos: 'cos', gaussian: 'gau', sigmoid: 'sig' };
  const BTN_ACT = [['=', 'identity'], ['sin', 'sin'], ['cos', 'cos'], ['gau', 'gaussian'], ['sig', 'sigmoid']];
  const IN_NAME = { '-1': 'x', '-2': 'y', '-3': 'r', '-4': 'bias' };
  const OUT_NAME = { '0': 'hue', '1': 'sat', '2': 'value' };
  const THUMB_RES = 32, SEL_RES = 200, EP_RES = 150;

  const INSTRUCTIONS_HTML =
    '<h3>Instructions</h3>' +
    '<ul class="instr">' +
    '<li><b>Click a node</b> to select it—its output renders on the right and a ring of circular buttons appears:' +
      '<ul>' +
      '<li><code>=</code> identity, <code>sin</code>, <code>cos</code>, <code>gau</code>ssian, <code>sig</code>moid—set the node’s activation function.</li>' +
      '<li><code>Lnk</code>—start a new link from this node, then click the target node.</li>' +
      '<li><code>Edt</code>—open a panel to edit the node’s incoming weights.</li>' +
      '<li><code>Del</code>—delete the node and every link to or from it.</li>' +
      '</ul></li>' +
    '<li><b>Click a link</b> for its ring: <code>Nod</code> (insert a node mid-link), <code>Del</code>, <code>±</code> (type a weight).' +
      ' Or just <b>drag a link up/down</b> to slide its weight value (most weights lie between −3 and 3).</li>' +
    '<li><b>Drag a node</b> to move it—its links follow. <b>Drag the small ⊕ handle</b> on the node’s edge onto' +
      ' another node to wire up a new link (a click on the handle does the same in two steps).</li>' +
    '<li><b>Double-click</b> empty space to add a new node (or double-click a link to split it with a node).</li>' +
    '<li><b>Drag</b> empty space to pan; <b>drag the zoom dial</b> on the right (or pinch the canvas) to zoom; <b>Fit</b> re-centers everything.</li>' +
    '</ul>';

  function template(showInstr, lockView, allowPan) {
    return '' +
      '<div class="linkmode-banner" id="linkbanner">Click the node you want to link <b id="linkfromlbl"></b> &rarr; … &nbsp;(<a href="#" id="linkcancel">cancel</a>)</div>' +
      '<div class="dna-wrap">' +
        '<div class="graphbox">' +
          // When the host locks the view (no zoom), the Fit + zoom toolbar is dropped
          // entirely and the hint drops its zoom verb — but keeps "drag to pan" if the
          // host opted panning back in (allowPan).
          (lockView ?
            '<div class="graphtools">' +
              '<span class="hint">' + (allowPan ? 'drag to pan · ' : '') + 'double-click to add a node · drag a node to move it · drag its ⊕ handle onto another node to link · drag a link up/down to change its weight</span>' +
            '</div>'
          :
            '<div class="graphtools">' +
              '<button class="pbbtn" id="g-fit" title="Fit the whole network in view">Fit</button>' +
              '<span class="zoomctl" title="Zoom">' +
                '<span class="zlbl">&minus;</span>' +
                '<input type="range" id="zoom" class="zoomrange" min="0" max="1" step="0.001" value="0.5" aria-label="Zoom">' +
                '<span class="zlbl">+</span>' +
              '</span>' +
              '<span class="hint">drag to pan · double-click to add a node · drag a node to move it · drag its ⊕ handle onto another node to link · drag a link up/down to change its weight</span>' +
            '</div>'
          ) +
          '<div class="graphframe">' +
            '<svg id="svg" xmlns="http://www.w3.org/2000/svg"><g id="vp"></g><g id="ovl"></g></svg>' +
            '<div class="wtip" id="wtip"></div>' +
            // vertical "thermometer" scale shown while a link's weight is being dragged:
            // a track from max(top) to min(bottom), a zero baseline, a fill from zero to the
            // current value, and a knob the user can watch slide into the min/max stops.
            '<div class="wscale" id="wscale">' +
              '<span class="wscale-lbl" id="wscale-top"></span>' +
              '<div class="wscale-track">' +
                '<div class="wscale-zero" id="wscale-zero"></div>' +
                '<div class="wscale-fill" id="wscale-fill"></div>' +
                '<div class="wscale-knob" id="wscale-knob"></div>' +
              '</div>' +
              '<span class="wscale-lbl" id="wscale-bot"></span>' +
            '</div>' +
          '</div>' +
          '<p class="legend">' +
            'Border color = node kind: <span class="dot" style="background:#fff;border:2px solid #555;"></span>input' +
            ' <span class="dot" style="background:#fff;border:2px solid #8a8a8a;"></span>gray' +
            ' <span class="dot" style="background:#fff;border:2px solid #2d68d4;"></span>color' +
            ' <span class="dot" style="background:#fff;border:2px solid #cc8a1a;"></span>output (H/S/V).' +
            ' Links: <span class="sw" style="border-top-color:' + WCMAP.pos + ';"></span>positive' +
            ' <span class="sw" style="border-top-color:' + WCMAP.neg + ';"></span>negative (saturation = strength)' +
            ' <span class="sw" style="border-top-color:#bbb;border-top-style:dashed;"></span>disabled.' +
          '</p>' +
        '</div>' +
        '<div class="rpanel">' +
          '<div class="card"><h4>Final image</h4><div class="body">' +
            '<canvas id="pheno" width="256" height="256"></canvas>' +
            '<div class="row"><label><input type="checkbox" id="colorchk" checked> Color</label></div>' +
            '<div class="meta" id="phenometa"></div>' +
          '</div></div>' +
          '<div class="card"><h4 id="sel-head">Selected node</h4><div class="body" id="sel-body">' +
            '<div class="placeholder">Click a node in the graph to render its output here.</div>' +
          '</div></div>' +
        '</div>' +
      '</div>' +
      '<div class="dna-status note" id="dna-status" style="display:none;"></div>' +
      '<div class="editpanel" id="editpanel"></div>' +
      '<div class="dna-actions" id="actions"></div>' +
      (showInstr ? INSTRUCTIONS_HTML : '');
  }

  PB.DnaEditor = function (root, opts) {
    opts = opts || {};
    const lockView = !!opts.lockView;   // host wants editing only — no zoom toolbar/wheel
    // Drag-to-pan is normally coupled to lockView, but a locked host (the blog homepage)
    // can opt panning back in on its own: keeps the graph reachable when it overflows the
    // fill-width frame, without bringing back the zoom UI.
    const allowPan = !lockView || !!opts.allowPan;
    // Per-instance layout knobs. Defaults reproduce the breed-site editor exactly; a host
    // (e.g. the blog homepage) can pass larger tiles / tighter spacing / thicker links and a
    // fill-the-width fit so the graph fills its canvas instead of sitting tiny and letterboxed.
    const R = opts.tileR || 21;                                   // node tile half-size
    const COLW = opts.colW || 80, ROWV = opts.rowV || 104;        // grid cell spacing
    const PADX = opts.padX != null ? opts.padX : 50, PADY = opts.padY != null ? opts.padY : 50;
    const LINKW = opts.linkW || 6, LINKW_OFF = opts.linkWDisabled || 2.5;   // link stroke widths
    const FIT_MARGIN = opts.fitMargin || 0.92, FIT_MAXSCALE = opts.fitMaxScale || 2.4;
    const FIT_WIDTH = !!opts.fitWidth;   // scale to fill width + auto-size the frame height
    root.innerHTML = template(opts.instructions !== false, lockView, allowPan);

    const qs = (id) => root.querySelector('#' + id);
    const svg = qs('svg');
    const vp = qs('vp');
    const ovl = qs('ovl');
    const pheno = qs('pheno');
    const pctx = pheno.getContext('2d');
    const banner = qs('linkbanner');
    const colorChk = qs('colorchk');
    const statusEl = qs('dna-status');

    let genome = null, originalJSON = null;
    let ctr = 3, colorMode = true;
    let curMeta = opts.meta || {};
    let linkFrom = null;       // node key while choosing a link target
    let selNode = null, selConn = null;   // current selection (node key, or "i,o")
    let selConnPt = null;                 // where on the edge the user clicked (content coords) → menu anchor
    let weightDragging = false;           // a link's weight is being live-dragged (sidebar repaints in place)
    let view = { x: 0, y: 0, s: 1 };
    let ndrag = null;                     // handle→node link drag in progress
    let mdrag = null;                     // node reposition drag in progress
    let linkDragLine = false;             // overlay is showing the rubber-band link line

    function setStatus(m) {
      if (opts.setStatus) { opts.setStatus(m); return; }
      statusEl.innerHTML = m || '';
      statusEl.style.display = m ? '' : 'none';
    }
    function notifyChange() { if (opts.onChange) opts.onChange(genome); }

    // ---------- per-pixel CPPN evaluation, capturing EVERY node's pattern ----------
    // Mirrors renderGenome's forward pass but stores each node's activation across
    // the whole grid, so we can draw the "intermediate pattern" at each node (Fig. 5).
    function computeNodeFields(g, res) {
      const evals = buildNet(g);
      const inF = g.inAct ? g.inAct.map((n) => ACT[n] || ACT.identity) : null;
      const fields = new Map();
      for (const k of g.nodes.keys()) fields.set(k, new Float32Array(res * res));
      const f1 = fields.get(-1), f2 = fields.get(-2), f3 = fields.get(-3), f4 = fields.get(-4);
      let idx = 0;
      for (let yy = 0; yy < res; yy++) {
        const sy = ((yy * 2) - (res - 1)) / res;
        for (let xx = 0; xx < res; xx++) {
          const sx = ((xx * 2) - (res - 1)) / res;
          const dd = Math.hypot(sx, sy) * Math.SQRT2;
          let ix = sx, iy = sy, id = dd, ib = 1.0;
          if (inF) { ix = inF[0](sx); iy = inF[1](sy); id = inF[2](dd); ib = inF[3](1.0); }
          const v = Object.create(null);
          v[-1] = ix; v[-2] = iy; v[-3] = id; v[-4] = ib; v[0] = 0; v[1] = 0; v[2] = 0;
          if (f1) f1[idx] = ix; if (f2) f2[idx] = iy; if (f3) f3[idx] = id; if (f4) f4[idx] = ib;
          for (let e = 0; e < evals.length; e++) {
            const ev = evals[e]; let s = 0; const L = ev.links;
            for (let k = 0; k < L.length; k++) s += v[L[k][0]] * L[k][1];
            const val = ev.act(s); v[ev.node] = val;
            const f = fields.get(ev.node); if (f) f[idx] = val;
          }
          idx++;
        }
      }
      return fields;
    }
    // Grayscale render of a node's intermediate pattern, contrast-stretched to the
    // node's own [min,max] so its structure is legible no matter the value range
    // (deep nodes often vary only slightly; the stretch is what makes Fig. 5 readable).
    function fieldToImageData(field, res) {
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < field.length; i++) { const v = field[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      const rng = mx - mn;
      const img = new ImageData(res, res), d = img.data;
      for (let i = 0; i < field.length; i++) {
        const t = rng > 1e-6 ? (field[i] - mn) / rng : 0.5;
        const g = (t * 255 + 0.5) | 0;
        d[i * 4] = g; d[i * 4 + 1] = g; d[i * 4 + 2] = g; d[i * 4 + 3] = 255;
      }
      return img;
    }
    const _tc = document.createElement('canvas'); _tc.width = _tc.height = THUMB_RES;
    const _tctx = _tc.getContext('2d');

    // The three OUTPUT nodes carry h/s/v, and a grayscale, per-node contrast-stretch
    // is a poor (misleading) view of them: hue is cyclic, so a smooth ramp reads as
    // flat grey even though `fract(o0+1)` wraps it into vivid colour, and value
    // saturates through `clamp(|o2|,0,1)`. So we render the output tiles through the
    // SAME per-channel transform the final image uses (renderGenome / the GL pheno
    // shader), isolating one channel each so they visibly map onto the final picture:
    //   hue  → full-sat/full-value hue field   sat → white→colour ramp   value → black→white
    // (transform_outputs is applied first, exactly as the final render does.)
    const OUT_SAT_HUE = 0.6;                 // reference hue for the saturation tile's ramp
    function outFieldImageData(field, res, ch) {
      const oa = genome.outAct, of = oa ? (ACT[oa[ch]] || ACT.identity) : null;
      const img = new ImageData(res, res), d = img.data;
      for (let i = 0; i < field.length; i++) {
        let o = field[i]; if (of) o = of(o);
        let r, g, b;
        if (ch === 0) { const c = hsv2rgb(pmod(o + 1, 1), 1, 1); r = c[0]; g = c[1]; b = c[2]; }
        else if (ch === 1) { const c = hsv2rgb(OUT_SAT_HUE, clamp(o, 0, 1), 1); r = c[0]; g = c[1]; b = c[2]; }
        else { r = g = b = (clamp(Math.abs(o), 0, 1) * 255 + 0.5) | 0; }
        d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
      }
      return img;
    }
    // One node's field as ImageData: the h/s/v output nodes go through the colour
    // mapping above; every other node stays the grayscale contrast-stretch. Shared
    // by the graph tiles, the click-to-enlarge view, and the edge endpoint canvases
    // so a node looks the same everywhere.
    function nodeImageData(k, field, res) {
      return (colorMode && OUTPUT_KEYS.includes(k)) ? outFieldImageData(field, res, OUTPUT_KEYS.indexOf(k)) : fieldToImageData(field, res);
    }
    function thumbURL(k, field) { _tctx.putImageData(nodeImageData(k, field, THUMB_RES), 0, 0); return _tc.toDataURL(); }

    // One node's intermediate pattern as a Float32Array (top-row-first, the same
    // layout fieldToImageData expects). Uses the GPU shader when available — this
    // is the live weight-scrub hot path — and falls back to the CPU sweep.
    function nodeField(k, res) {
      const g = GL();
      if (g) { const f = g.renderField(genome, k, res); if (f) return f; }
      return computeNodeFields(genome, res).get(k);
    }

    // ---------- final phenotype ----------
    function rerender() {
      genome._net = null;
      const g = GL();
      let ok = false;
      if (g) { try { ok = g.renderPheno(genome, colorMode, pctx, 256); } catch (e) { ok = false; } }
      if (!ok) { try { pctx.putImageData(renderGenome(genome, 256, colorMode), 0, 0); } catch (e) { pctx.clearRect(0, 0, 256, 256); } }
      const nN = genome.nodes.size, nC = [...genome.conns.values()].filter((c) => c.enabled).length;
      qs('phenometa').innerHTML = '<b>' + nN + '</b> nodes · <b>' + nC + '</b> active links';
      notifyChange();
    }

    // ---------- layered layout with edge routing (à la graphviz's `dot`) ----------
    // The same renderer (graphviz dot) draws the Python debug diagrams. We mirror its
    // pipeline: (1) longest-path depth → rows; (2) insert virtual "dummy" nodes along
    // every edge that spans more than one row, so long edges route THROUGH the
    // intermediate rows in thin channels instead of cutting straight across the graph;
    // (3) barycenter + transpose ordering over the augmented graph to minimise
    // crossings; (4) bottom-to-top coordinates (inputs at the bottom, outputs on top).
    function computeDepth() {
      // Longest-path layering, but cycle-safe. CPPNs can contain recurrent loops
      // (e.g. a 50→49 back-edge); naive relaxation diverges along a cycle and flings
      // its nodes to absurd depths — the "long neck". Drop DFS back-edges first so the
      // ranking runs over a DAG and stays bounded. (The renderer's feedForwardLayers
      // already ignores such loops, so this just keeps the drawing in sync.)
      const enabled = [...genome.conns.values()].filter((c) => c.enabled);
      const adj = new Map();
      enabled.forEach((c) => { if (!adj.has(c.i)) adj.set(c.i, []); adj.get(c.i).push(c.o); });
      const color = new Map(), back = new Set();   // 1 = on DFS stack, 2 = done
      const dfs = (u) => {
        color.set(u, 1);
        for (const v of (adj.get(u) || [])) {
          const cs = color.get(v) || 0;
          if (cs === 1) back.add(u + ',' + v);       // edge into an ancestor → back-edge
          else if (cs === 0) dfs(v);
        }
        color.set(u, 2);
      };
      INPUT_KEYS.forEach((k) => { if (genome.nodes.has(k) && !color.has(k)) dfs(k); });
      [...genome.nodes.keys()].forEach((k) => { if (!color.has(k)) dfs(k); });   // orphan subgraphs / pure cycles
      const fwd = enabled.filter((c) => !back.has(c.i + ',' + c.o));
      const depth = new Map();
      for (const k of genome.nodes.keys()) depth.set(k, genome.nodes.get(k).input ? 0 : 1);
      for (let it = 0; it < genome.nodes.size + 2; it++) {
        let changed = false;
        for (const c of fwd) { const nd = Math.max(depth.get(c.o) || 1, (depth.get(c.i) || 0) + 1); if (nd !== depth.get(c.o)) { depth.set(c.o, nd); changed = true; } }
        if (!changed) break;
      }
      let maxd = 0; depth.forEach((v) => { if (v > maxd) maxd = v; });
      OUTPUT_KEYS.forEach((k) => { if (genome.nodes.has(k)) depth.set(k, Math.max(maxd, depth.get(k) || 1)); });
      return depth;
    }

    // Order augmented layers (string ids; dummies included) to reduce edge crossings.
    // Rows 0 (inputs) and last (outputs) are pinned to their canonical order.
    function orderAug(layers, up, down) {
      if (layers.length <= 2) return;
      const idx = new Map();
      layers.forEach((col) => col.forEach((id, i) => idx.set(id, i)));
      const last = layers.length - 1;
      const pairCross = (a, b, nbr) => {
        const A = nbr.get(a), B = nbr.get(b); if (!A.length || !B.length) return 0;
        let c = 0; for (const x of A) { const px = idx.get(x); for (const y of B) if (idx.get(y) < px) c++; } return c;
      };
      const tie = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
      for (let s = 0; s < 12; s++) {
        const downward = s % 2 === 0;            // alternate ordering by lower / upper neighbours
        const seq = []; for (let d = 1; d < last; d++) seq.push(d);
        if (!downward) seq.reverse();
        for (const d of seq) {
          const nbr = downward ? up : down, col = layers[d];
          const bary = new Map();
          col.forEach((id, i) => { const ns = nbr.get(id); let b = i; if (ns.length) { let sum = 0; for (const n of ns) sum += idx.get(n); b = sum / ns.length; } bary.set(id, b); });
          col.sort((a, b) => (bary.get(a) - bary.get(b)) || tie(a, b));
          col.forEach((id, i) => idx.set(id, i));
        }
        let improved = true, guard = 0;       // transpose: greedy adjacent swaps
        while (improved && guard++ < 4) {
          improved = false;
          for (let d = 1; d < last; d++) {
            const col = layers[d];
            for (let i = 0; i < col.length - 1; i++) {
              const u = col[i], v = col[i + 1];
              const cur = pairCross(u, v, up) + pairCross(u, v, down);
              const swp = pairCross(v, u, up) + pairCross(v, u, down);
              if (swp < cur) { col[i] = v; col[i + 1] = u; idx.set(v, i); idx.set(u, i + 1); improved = true; }
            }
          }
        }
      }
    }

    const pos = new Map();        // real node key -> {x,y}
    // Node keys the user has placed by hand — either a free node dropped by double-click or
    // any node dragged to a new spot. They opt out of the layered layout (which keeps running
    // for everyone else) and their links are drawn straight rather than through stale
    // dummy-node waypoints.
    const manualPos = new Map();  // node key -> {x,y}
    let routes = new Map();       // "i,o" -> [intermediate waypoints {x,y}], i→o order
    let contentW = 100, contentH = 100;
    function layout() {
      // Free (unwired) hidden nodes are kept OUT of the layered layout entirely so they don't
      // distort the rows, and are placed at the spot the user double-clicked (manualPos);
      // the moment a node gains any connection it rejoins the automatic layout below.
      const isIO = (k) => genome.nodes.get(k).input || OUTPUT_KEYS.includes(k);
      const hasConn = new Set();
      for (const c of genome.conns.values()) { hasConn.add(c.i); hasConn.add(c.o); }
      const free = new Set();
      for (const k of genome.nodes.keys()) if (!isIO(k) && !hasConn.has(k)) free.add(k);
      const depth = computeDepth();
      let maxd = 0; depth.forEach((v) => { if (v > maxd) maxd = v; });
      // augmented layered lists: real ids 'n<key>', dummy ids 'd<n>'. For every enabled
      // edge depth(o) > depth(i) holds (longest-path relaxation), so chains only ascend.
      const layers = []; for (let i = 0; i <= maxd; i++) layers.push([]);
      [...genome.nodes.keys()].sort((a, b) => a - b).forEach((k) => { if (!free.has(k)) layers[depth.get(k)].push('n' + k); });
      const chains = new Map();   // "i,o" -> [dummyId per crossed row, i→o]
      let dctr = 0;
      for (const c of genome.conns.values()) {
        if (!c.enabled) continue;
        const di = depth.get(c.i), dj = depth.get(c.o);
        if (di == null || dj == null || dj - di <= 1) continue;
        const chain = [];
        for (let L = di + 1; L < dj; L++) { const id = 'd' + (dctr++); layers[L].push(id); chain.push(id); }
        chains.set(c.i + ',' + c.o, chain);
      }
      // augmented adjacency (segments between consecutive rows)
      const up = new Map(), down = new Map();
      layers.forEach((col) => col.forEach((id) => { up.set(id, []); down.set(id, []); }));
      const seg = (loId, hiId) => { down.get(loId).push(hiId); up.get(hiId).push(loId); };
      for (const c of genome.conns.values()) {
        if (!c.enabled) continue;
        const di = depth.get(c.i), dj = depth.get(c.o); if (di == null || dj == null) continue;
        const chain = chains.get(c.i + ',' + c.o);
        if (!chain) { seg('n' + c.i, 'n' + c.o); continue; }
        let prev = 'n' + c.i;
        for (const id of chain) { seg(prev, id); prev = id; }
        seg(prev, 'n' + c.o);
      }
      orderAug(layers, up, down);
      // bottom-to-top coordinates; dummy columns are narrow so real nodes stay close.
      // COLW / ROWV / PADX / PADY come from the per-instance layout knobs (set near the top).
      const nLev = layers.length;
      // measure each row's width counting dummies as a fraction of a real slot
      const DUMW = 0.34;
      const rowWidth = (col) => col.reduce((s, id) => s + (id[0] === 'd' ? DUMW : 1), 0) - 1;
      const maxRow = Math.max(1, ...layers.map(rowWidth));
      contentW = PADX * 2 + Math.max(1, maxRow) * COLW;
      contentH = PADY * 2 + Math.max(1, nLev - 1) * ROWV;
      const xy = new Map();
      layers.forEach((col, d) => {
        const y = PADY + (nLev - 1 - d) * ROWV;
        const rw = rowWidth(col) * COLW, x0 = (contentW - rw) / 2;
        let cx = x0;
        col.forEach((id, i) => {
          if (i > 0) cx += ((col[i - 1][0] === 'd' ? DUMW : 1) / 2 + (id[0] === 'd' ? DUMW : 1) / 2) * COLW;
          xy.set(id, { x: cx, y: y });
        });
      });
      // ---- straighten: pull each node toward its neighbours' mean x so chains of
      // single-parent / single-child nodes collapse into vertical lines, and long edges
      // (routed through their dummy waypoints) run straight up instead of bowing out.
      // Per row we find the feasible x-assignment closest (L2) to those targets while
      // preserving the row's left→right order and minimum slot gaps — isotonic regression
      // (pool-adjacent-violators) after subtracting the cumulative minimum gaps.
      const nbrsOf = (id) => (up.get(id) || []).concat(down.get(id) || []);
      const halfSlot = (id) => (id[0] === 'd' ? DUMW : 1) * COLW / 2;
      const minGap = (a, b) => halfSlot(a) + halfSlot(b);
      const isotonicL2 = (v) => {                 // nearest non-decreasing fit (PAVA, equal weights)
        const bv = [], bl = [];
        for (let i = 0; i < v.length; i++) {
          bv.push(v[i]); bl.push(1);
          while (bv.length > 1 && bv[bv.length - 2] > bv[bv.length - 1]) {
            const v2 = bv.pop(), l2 = bl.pop(), v1 = bv.pop(), l1 = bl.pop();
            bv.push((v1 * l1 + v2 * l2) / (l1 + l2)); bl.push(l1 + l2);
          }
        }
        const out = [];
        for (let b = 0; b < bv.length; b++) for (let j = 0; j < bl[b]; j++) out.push(bv[b]);
        return out;
      };
      const straightenRow = (col) => {
        if (!col.length) return;
        const pref = [0];
        for (let i = 1; i < col.length; i++) pref[i] = pref[i - 1] + minGap(col[i - 1], col[i]);
        const v = col.map((id, i) => {            // target x = mean neighbour x (else keep current)
          const ns = nbrsOf(id);
          let d = xy.get(id).x;
          if (ns.length) { let s = 0; for (const n of ns) s += xy.get(n).x; d = s / ns.length; }
          return d - pref[i];
        });
        const yh = isotonicL2(v);
        col.forEach((id, i) => { xy.get(id).x = yh[i] + pref[i]; });
      };
      for (let pass = 0; pass < 16; pass++) {      // Gauss-Seidel sweeps, alternating direction
        const order = layers.map((_, d) => d);
        if (pass % 2) order.reverse();
        for (const d of order) straightenRow(layers[d]);
      }
      // re-fit the content box to the straightened coordinates
      let minx = Infinity, maxx = -Infinity;
      xy.forEach((p) => { if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x; });
      const shiftX = PADX - minx;
      xy.forEach((p) => { p.x += shiftX; });
      contentW = (maxx - minx) + PADX * 2;

      pos.clear();
      for (const k of genome.nodes.keys()) {
        const m = manualPos.get(k);
        if (m) { pos.set(k, { x: m.x, y: m.y }); continue; }  // hand-placed: dropped / dragged there
        if (free.has(k)) { pos.set(k, { x: contentW / 2, y: contentH / 2 }); continue; }
        const p = xy.get('n' + k); if (p) pos.set(k, p);
      }
      routes = new Map();
      for (const [key, chain] of chains) routes.set(key, chain.map((id) => xy.get(id)));
    }

    // ---------- draw the graph (into the #vp pan/zoom group) ----------
    // Diverging weight ramp lives in breed/weight-cmap.js (the WCMAP single source
    // of truth — flip its ACTIVE toggle to revert/swap schemes). Disabled links use
    // '#bbb' below, independent of the scheme.
    function weightColor(w) { return WCMAP.link(w); }
    // A link's polyline, routed through its dummy-node waypoints — except when either
    // endpoint has been hand-placed, in which case the layered route no longer describes
    // where the edge runs and a straight segment is the honest drawing.
    function connPoints(c) {
      const a = pos.get(c.i), b = pos.get(c.o); if (!a || !b) return '';
      const routed = c.enabled && !manualPos.has(c.i) && !manualPos.has(c.o);
      const wp = (routed && routes.get(c.i + ',' + c.o)) || [];
      return [a, ...wp, b].map((p) => p.x.toFixed(1) + ',' + p.y.toFixed(1)).join(' ');
    }

    // ---------- the link handle ----------
    // Every node carries one small ⊕ handle, revealed on hover/selection, that you drag onto
    // another node to wire a link (dragging the node body moves the node instead). It sits on
    // the tile's border, on the ray toward the centre of mass of the other nodes — i.e. facing
    // the graph, roughly the way a new link would travel. Facing it outward instead would park
    // the handles of the boundary nodes (every input, every output) beyond the frame's clip.
    const HR = Math.max(7.5, R * 0.45);
    function handlePt(k) {
      const p = pos.get(k);
      let cx = 0, cy = 0, n = 0;
      for (const [j, q] of pos) if (j !== k) { cx += q.x; cy += q.y; n++; }
      let dx = 0, dy = -1;                       // lone node: point up, the way links flow
      if (n) {
        dx = cx / n - p.x; dy = cy / n - p.y;
        const L = Math.hypot(dx, dy);
        if (L < 1e-3) { dx = 0; dy = -1; } else { dx /= L; dy /= L; }
      }
      const t = R / Math.max(Math.abs(dx), Math.abs(dy));   // ray ∩ the square tile's border
      return { x: p.x + dx * t, y: p.y + dy * t };
    }
    function handleSVG(k) {
      const h = handlePt(k);
      const x = h.x.toFixed(1), y = h.y.toFixed(1);
      return `<g class="linkhandle" data-handle="${k}">` +
        `<circle class="lh-hit" cx="${x}" cy="${y}" r="${(HR * 2).toFixed(1)}"></circle>` +
        `<circle class="lh-dot" cx="${x}" cy="${y}" r="${HR.toFixed(1)}"></circle>` +
        `<text class="lh-plus" x="${x}" y="${y}" font-size="${(HR * 1.9).toFixed(1)}">+</text>` +
        `<title>drag onto another node to link</title></g>`;
    }

    // Move one node to (x,y) by patching the SVG in place: the full draw() re-renders every
    // node's pattern tile (a CPPN sweep + toDataURL each), far too heavy for a drag frame.
    // Only the moved node's geometry and the links touching it can have changed.
    function moveNodeTo(k, x, y) {
      pos.set(k, { x: x, y: y });
      const g = vp.querySelector('[data-node="' + k + '"]');
      if (g) {
        const x0 = x - R, y0 = y - R;
        g.querySelectorAll('rect, image').forEach((el) => { el.setAttribute('x', x0); el.setAttribute('y', y0); });
        const cap = g.querySelector('text.cap');
        if (cap) { cap.setAttribute('x', x); cap.setAttribute('y', y + R + 12); }
        const hg = g.querySelector('.linkhandle');
        if (hg) hg.outerHTML = handleSVG(k);
      }
      for (const [key, c] of genome.conns) {
        if (c.i !== k && c.o !== k) continue;
        const grp = vp.querySelector('[data-conn="' + key + '"]');
        if (grp) { const pts = connPoints(c); grp.querySelectorAll('polyline').forEach((pl) => pl.setAttribute('points', pts)); }
      }
      if (selNode === k) drawOverlay();       // the ring follows its node
    }

    function draw() {
      // A weight-only drag never moves a node (layout reads topology + activations,
      // not weights), so reuse the existing positions and skip the whole graphviz
      // layout pass each frame — only the link colours and tile patterns change.
      if (!weightDragging || !pos.size) layout();

      const fields = computeNodeFields(genome, THUMB_RES);
      let h = '';

      // connections, routed through their dummy waypoints as polylines (fat transparent
      // hit polyline for hover/scroll/drag). Disabled links stay straight + dashed.
      for (const c of genome.conns.values()) {
        if (!pos.has(c.i) || !pos.has(c.o)) continue;
        const pts = connPoints(c);
        const col = c.enabled ? weightColor(c.weight) : '#bbb';
        // Fat constant stroke: weight magnitude is shown by the colour ramp's saturation
        // (red↔blue), so width stays flat — but kept thick so links are an easy click+drag
        // target (drag a link up/down to change its weight).
        const w = c.enabled ? LINKW : LINKW_OFF;
        const dash = c.enabled ? '' : ' stroke-dasharray="4 3"';
        h += `<g class="connwrap" data-conn="${c.i},${c.o}">` +
          `<polyline class="graph-conn" points="${pts}" fill="none" stroke="${col}" stroke-width="${w}"${dash}></polyline>` +
          `<polyline class="graph-hit" points="${pts}" fill="none"><title>weight ${c.weight.toFixed(2)}${c.enabled ? '' : ' (disabled)'} — drag to change</title></polyline>` +
          `</g>`;
      }
      // nodes: a SQUARE tile showing the node's own intermediate pattern (Fig. 5),
      // framed by a border whose color marks the node kind.
      for (const [k, n] of genome.nodes) {
        const p = pos.get(k); if (!p) continue;
        let frame = 'frame-color';
        if (n.input) frame = 'frame-input'; else if (OUTPUT_KEYS.includes(k)) frame = 'frame-output'; else if (n.affinity === 'grey') frame = 'frame-grey';
        const url = thumbURL(k, fields.get(k));
        const cap = n.input ? IN_NAME[k] : (OUTPUT_KEYS.includes(k) ? OUT_NAME[k] : (ACT_ABBR[n.activation] || '?'));
        const sel = (k === selNode) ? ' node-sel' : (k === linkFrom ? ' node-lit' : '');
        const x0 = p.x - R, y0 = p.y - R, sz = R * 2;
        h += `<g class="graph-node${sel}" data-node="${k}">` +
          `<rect class="tilebg" x="${x0}" y="${y0}" width="${sz}" height="${sz}"></rect>` +
          `<image href="${url}" x="${x0}" y="${y0}" width="${sz}" height="${sz}" preserveAspectRatio="none"></image>` +
          `<rect class="frame ${frame}" x="${x0}" y="${y0}" width="${sz}" height="${sz}"></rect>` +
          `<rect class="hit" x="${x0}" y="${y0}" width="${sz}" height="${sz}"></rect>` +
          `<text class="cap" x="${p.x}" y="${p.y + R + 12}">${cap}</text>` +
          handleSVG(k) +
          `</g>`;
      }
      vp.innerHTML = h;
      drawOverlay();
      applyView();
    }

    // ---------- radial control buttons ----------
    // The ring lives in the (untransformed) #ovl overlay, NOT inside the pan/zoom group,
    // so its circles + labels keep a constant on-screen size and stay legible at any
    // canvas zoom. Anchors are projected from content coords to screen coords by hand.
    function ctrlButton(cx, cy, label, ctrl, target, extra) {
      return `<g class="ctrlbtn ${extra || ''}" data-ctrl="${ctrl}" data-target="${target}">` +
        `<circle cx="${cx}" cy="${cy}" r="11"></circle><text x="${cx}" y="${cy}">${label}</text></g>`;
    }
    // ring radius: clears the node tile when zoomed in, but never collapses below a
    // legible, non-overlapping minimum when zoomed out (buttons are a fixed 11px).
    function ring(cx, cy, btns) {
      const RAD = Math.max(40, R * view.s + 18), n = btns.length;
      return btns.map((b, i) => {
        const ang = -Math.PI / 2 + i * (2 * Math.PI / n);
        return ctrlButton(cx + RAD * Math.cos(ang), cy + RAD * Math.sin(ang), b.l, b.c, b.t, b.x);
      }).join('');
    }
    const toScreen = (p) => ({ x: view.x + p.x * view.s, y: view.y + p.y * view.s });
    const toContent = (clientX, clientY) => {
      const r = svg.getBoundingClientRect();
      return { x: (clientX - r.left - view.x) / view.s, y: (clientY - r.top - view.y) / view.s };
    };
    // Which node's tile covers this client point, if any. Used to resolve a link drag's drop:
    // releasing anywhere over a node connects to it, so the target needs no handle of its own
    // (and hit-testing the DOM would fail on whatever child element happened to be on top).
    function nodeAt(clientX, clientY) {
      const p = toContent(clientX, clientY);
      for (const [k, q] of pos) if (Math.abs(p.x - q.x) <= R && Math.abs(p.y - q.y) <= R) return k;
      return null;
    }
    function nodeRingBtns(k) {
      const n = genome.nodes.get(k), btns = [];
      if (!n.input) BTN_ACT.forEach(([lbl, act]) => btns.push({ l: lbl, c: 'act:' + act, t: k, x: n.activation === act ? 'cur' : '' }));
      btns.push({ l: 'Lnk', c: 'lnk', t: k });
      if (!n.input) btns.push({ l: 'Edt', c: 'edt', t: k });
      if (!n.input && !OUTPUT_KEYS.includes(k)) btns.push({ l: 'Del', c: 'del', t: k, x: 'danger' });
      return btns;
    }
    // (Re)draw the active selection's ring into the overlay. Called from draw() and on
    // every pan/zoom (applyView) so the ring tracks its anchor without changing size.
    function drawOverlay() {
      if (linkDragLine) return;                         // a node→node link drag owns the overlay
      let h = '';
      if (selNode != null) {
        const p = pos.get(selNode), n = genome.nodes.get(selNode);
        if (p && n) { const s = toScreen(p); h = ring(s.x, s.y, nodeRingBtns(selNode)); }
      } else if (selConn != null && !weightDragging) {   // hide the edge ring mid-drag; it returns on release
        const c = genome.conns.get(selConn), a = c && pos.get(c.i), b = c && pos.get(c.o);
        if (a && b) {
          let mx, my;
          if (selConnPt) { mx = selConnPt.x; my = selConnPt.y; }
          else { const wp = routes.get(selConn); const m = (wp && wp.length) ? wp[wp.length >> 1] : { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; mx = m.x; my = m.y; }
          const s = toScreen({ x: mx, y: my });
          const btns = [{ l: 'Nod', c: 'nod', t: selConn }, { l: 'Del', c: 'delc', t: selConn, x: 'danger' }, { l: '±', c: 'pm', t: selConn }];
          h = ring(s.x, s.y, btns);
        }
      }
      ovl.innerHTML = h;
    }

    // ---------- mutation primitives ----------
    function bumpCtr() { let mx = 2; genome.nodes.forEach((_, k) => { if (k > mx) mx = k; }); ctr = Math.max(ctr, mx + 1); return ctr; }
    function refresh() {
      rerender(); draw();
      if (selNode != null) renderSelected(selNode);
      else if (selConn != null) {
        // During a live weight-drag, update the weight readout AND repaint the endpoint
        // patterns in place (the target node's output changes with the weight) — but skip
        // the innerHTML rebuild, so the sidebar tracks the drag as live as the graph does.
        if (weightDragging) {
          const wb = qs('sel-conn-w'), c = genome.conns.get(selConn); if (wb && c) wb.textContent = c.weight.toFixed(3);
          paintConnEndpoints(selConn);
        }
        else renderSelectedConn(selConn);
      }
    }
    function setActivation(k, act) { const n = genome.nodes.get(k); if (n && !n.input) { n.activation = act; refresh(); } }
    function deleteNode(k) {
      if (genome.nodes.get(k).input || OUTPUT_KEYS.includes(k)) return;
      genome.nodes.delete(k); manualPos.delete(k);
      for (const ck of [...genome.conns.keys()]) { const c = genome.conns.get(ck); if (c.i === k || c.o === k) genome.conns.delete(ck); }
      if (selNode === k) { selNode = null; clearSelected(); }
      rerender(); draw();
    }
    function deleteConn(key) { genome.conns.delete(key); if (selConn === key) { selConn = null; clearSelected(); } refresh(); }
    function setWeight(key, w) { const c = genome.conns.get(key); if (c) { c.weight = clamp(w, WEIGHT_MIN, WEIGHT_MAX); refresh(); } }
    function addLink(i, o) {
      if (i === o || genome.hasConn(i, o) || INPUT_KEYS.includes(o)) return false;
      const enabled = [...genome.conns.values()].filter((c) => c.enabled).map((c) => [c.i, c.o]);
      if (createsCycle(enabled, [i, o])) { setStatus('That link would create a cycle—not allowed in a feed-forward CPPN.'); return false; }
      genome.addConn(i, o, randWeight()); setStatus(''); refresh(); return true;
    }
    function addNode(at) {                 // free-standing hidden node (double-click empty canvas); wire it up by dragging
      const nk = bumpCtr(); ctr = nk + 1;
      genome.nodes.set(nk, { activation: choice(ACT_OPTIONS), affinity: 'color' });
      if (at) manualPos.set(nk, { x: at.x, y: at.y });   // drop it under the cursor (content coords)
      refresh(); selectNode(nk);
      setStatus('Added a node—drag the ⊕ handle on its edge onto another node to wire up its links.');
      return nk;
    }
    function insertNodeOnLink(key) {     // classic NEAT add-node: disable i→o, route i→new→o preserving function
      const c = genome.conns.get(key); if (!c) return;
      const nk = bumpCtr(); ctr = nk + 1;
      const aff = OUTPUT_KEYS.includes(c.i) ? genome.nodes.get(c.o).affinity : choice([genome.nodes.get(c.i).affinity, genome.nodes.get(c.o).affinity]);
      genome.nodes.set(nk, { activation: choice(ACT_OPTIONS), affinity: aff });
      c.enabled = false; genome.addConn(c.i, nk, 1.0); genome.addConn(nk, c.o, c.weight);
      selConn = null; refresh();
    }

    // ---------- selection + right render panel ----------
    function clearSelected() {
      qs('sel-head').textContent = 'Selected node';
      qs('sel-body').innerHTML = '<div class="placeholder">Click a node in the graph to render its output here.</div>';
      alignGraphHeight();
    }
    function renderSelected(k) {
      const n = genome.nodes.get(k); if (!n) return;
      const name = n.input ? IN_NAME[k] : (OUT_NAME[k] || 'node #' + k);
      qs('sel-head').textContent = 'Node: ' + name;
      const nin = [...genome.conns.values()].filter((c) => c.o === k && c.enabled).length;
      const nout = [...genome.conns.values()].filter((c) => c.i === k && c.enabled).length;
      const act = n.input ? '— (input)' : n.activation;
      qs('sel-body').innerHTML =
        '<canvas id="selcanvas" width="' + SEL_RES + '" height="' + SEL_RES + '"></canvas>' +
        '<div class="meta"><b>' + name + '</b><br>activation: <b>' + act + '</b><br>' +
        nin + ' incoming · ' + nout + ' outgoing link' + (nout === 1 ? '' : 's') + '</div>';
      const f = nodeField(k, SEL_RES);
      if (f) root.querySelector('#selcanvas').getContext('2d').putImageData(nodeImageData(k, f, SEL_RES), 0, 0);
      alignGraphHeight();
    }
    const nodeLabel = (k) => { const n = genome.nodes.get(k); return n && n.input ? IN_NAME[k] : (OUT_NAME[k] || 'node #' + k); };
    // Edge selected → show the two endpoints' activation patterns (source feeding IN,
    // target coming OUT) with the link's weight between them, so the panel reads as
    // "this pattern, scaled by this weight, contributes to that pattern".
    // Repaint just the two endpoint canvases for the selected edge (no innerHTML rebuild),
    // so a live weight-drag can refresh the patterns in place every frame.
    function paintConnEndpoints(key) {
      const c = genome.conns.get(key); if (!c) return;
      [c.i, c.o].forEach((k) => {
        const cv = root.querySelector('#sel-body canvas[data-ep="' + k + '"]');
        if (!cv) return;
        const f = nodeField(k, EP_RES);
        if (f) cv.getContext('2d').putImageData(nodeImageData(k, f, EP_RES), 0, 0);
      });
    }
    function renderSelectedConn(key, skipPaint) {
      const c = genome.conns.get(key); if (!c) { clearSelected(); return; }
      qs('sel-head').textContent = 'Edge: ' + nodeLabel(c.i) + ' → ' + nodeLabel(c.o);
      // No per-endpoint caption — the pattern thumbnails (and the weight arrow between
      // them, source at the bottom feeding the target at the top) speak for themselves.
      const endpt = (k) =>
        '<div class="row" style="margin-top:0;">' +
        '<canvas data-ep="' + k + '" width="' + EP_RES + '" height="' + EP_RES + '" style="width:150px;height:150px;"></canvas></div>';
      qs('sel-body').innerHTML =
        endpt(c.o) +
        '<div class="meta" style="text-align:center;margin:6px 0;">↑ weight <b id="sel-conn-w">' + c.weight.toFixed(3) + '</b>' + (c.enabled ? '' : ' · disabled') + '</div>' +
        endpt(c.i);
      // skipPaint = layout only (used to measure the tallest pane height); the per-node
      // field computation is the expensive part, so skip it when we just need the size.
      if (!skipPaint) paintConnEndpoints(key);
      alignGraphHeight();
    }
    function selectNode(k) { selNode = k; selConn = null; selConnPt = null; draw(); renderSelected(k); }
    function selectConn(key, pt) { selConn = key; selNode = null; selConnPt = pt || null; renderSelectedConn(key); draw(); }
    function deselect() { selNode = null; selConn = null; selConnPt = null; clearSelected(); draw(); }

    // ---------- link-target mode ----------
    function startLink(k) { linkFrom = k; selNode = null; banner.classList.add('on'); qs('linkfromlbl').textContent = (genome.nodes.get(k).input ? IN_NAME[k] : ('#' + k)); draw(); }
    function cancelLink() { if (linkFrom == null) return; linkFrom = null; banner.classList.remove('on'); draw(); }
    qs('linkcancel').onclick = (e) => { e.preventDefault(); cancelLink(); };

    // ---------- node "Edt" panel ----------
    function openEdit(k) {
      const n = genome.nodes.get(k); if (!n) return;
      const panel = qs('editpanel');
      const incoming = [...genome.conns.values()].filter((c) => c.o === k);
      const lbl = (j) => genome.nodes.get(j) && genome.nodes.get(j).input ? IN_NAME[j] : ('#' + j);
      let rows = incoming.map((c) =>
        `<tr><td>${lbl(c.i)} &rarr;</td>` +
        `<td><input type="number" step="0.1" value="${c.weight.toFixed(3)}" data-i="${c.i}"></td>` +
        `<td><label><input type="checkbox" data-en="${c.i}" ${c.enabled ? 'checked' : ''}> enabled</label></td>` +
        `<td><button class="pbbtn" data-del="${c.i}">delete</button></td></tr>`).join('');
      if (!rows) rows = '<tr><td colspan="4" class="note">No incoming links.</td></tr>';
      panel.innerHTML =
        `<h4>Editing node ${OUT_NAME[k] || ('#' + k)}</h4>` +
        `<p>Activation: <select id="ed-act">` + ACT_OPTIONS.map((a) => `<option value="${a}" ${n.activation === a ? 'selected' : ''}>${a}</option>`).join('') + `</select></p>` +
        `<table><thead><tr><th>source</th><th>weight</th><th></th><th></th></tr></thead><tbody>${rows}</tbody></table>` +
        `<p style="margin-top:8px;"><button class="pbbtn" id="ed-close">Done</button></p>`;
      panel.classList.add('on');
      panel.querySelector('#ed-act').onchange = (e) => { n.activation = e.target.value; refresh(); };
      panel.querySelectorAll('input[data-i]').forEach((inp) => inp.oninput = () => { const c = genome.conns.get(inp.dataset.i + ',' + k); if (c && !isNaN(parseFloat(inp.value))) { c.weight = clamp(parseFloat(inp.value), WEIGHT_MIN, WEIGHT_MAX); rerender(); if (selNode === k) draw(); } });
      panel.querySelectorAll('input[data-en]').forEach((inp) => inp.onchange = () => { const c = genome.conns.get(inp.dataset.en + ',' + k); if (c) { c.enabled = inp.checked; refresh(); } });
      panel.querySelectorAll('button[data-del]').forEach((b) => b.onclick = () => { deleteConn(parseInt(b.dataset.del, 10) + ',' + k); openEdit(k); });
      panel.querySelector('#ed-close').onclick = () => panel.classList.remove('on');
      panel.scrollIntoView({ block: 'nearest' });
    }

    // ---------- floating weight readout (follows the cursor during scroll/drag) ----------
    const wtip = qs('wtip');
    let wtipTimer = null;
    function showWtip(clientX, clientY, text, sticky) {
      const r = svg.getBoundingClientRect();
      wtip.textContent = text;
      wtip.style.left = (clientX - r.left) + 'px';
      wtip.style.top = (clientY - r.top) + 'px';
      wtip.classList.add('on');
      if (wtipTimer) { clearTimeout(wtipTimer); wtipTimer = null; }
      if (!sticky) wtipTimer = setTimeout(() => wtip.classList.remove('on'), 850);
    }
    const wlabel = (key) => { const c = genome.conns.get(key); return c ? 'weight ' + c.weight.toFixed(2) : ''; };

    // ---------- vertical weight scale (thermometer) shown while dragging a link ----------
    const wscale = qs('wscale'), wscaleKnob = qs('wscale-knob'), wscaleFill = qs('wscale-fill');
    const WS_TRACK_H = 132;                 // must match .wscale-track height in dna-editor.css
    const WS_SPAN = WEIGHT_MAX - WEIGHT_MIN;
    const wsY = (w) => (1 - clamp((w - WEIGHT_MIN) / WS_SPAN, 0, 1)) * WS_TRACK_H;   // value → px from track top
    let wscaleTimer = null;
    qs('wscale-top').textContent = (WEIGHT_MAX > 0 ? '+' : '') + WEIGHT_MAX;
    qs('wscale-bot').textContent = String(WEIGHT_MIN);
    qs('wscale-zero').style.top = wsY(0) + 'px';
    function showWscale(clientX, clientY, w) {
      const r = svg.getBoundingClientRect();
      // anchor just to the right of the pinned readout, vertically centred on the click point —
      // clamped so the ~160px-tall widget isn't clipped by the frame's overflow near an edge
      const HALF = WS_TRACK_H / 2 + 24;
      wscale.style.left = (clientX - r.left + 64) + 'px';   // clear the right edge of the pinned "weight XX" label
      wscale.style.top = clamp(clientY - r.top, HALF, r.height - HALF) + 'px';
      const knobY = wsY(w), zeroY = wsY(0);
      wscaleKnob.style.top = knobY + 'px';
      // fill from the zero baseline to the knob — positive up, negative down (scheme colours)
      wscaleFill.style.top = Math.min(knobY, zeroY) + 'px';
      wscaleFill.style.height = Math.abs(knobY - zeroY) + 'px';
      wscaleFill.style.background = w >= 0 ? WCMAP.pos : WCMAP.neg;
      wscale.classList.toggle('lim', w <= WEIGHT_MIN + 1e-6 || w >= WEIGHT_MAX - 1e-6);
      wscale.classList.add('on');
      if (wscaleTimer) { clearTimeout(wscaleTimer); wscaleTimer = null; }
    }
    function hideWscale(delay) {
      if (wscaleTimer) clearTimeout(wscaleTimer);
      wscaleTimer = setTimeout(() => wscale.classList.remove('on'), delay || 0);
    }

    // ---------- pan / zoom ----------
    function applyView() { vp.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.s})`); drawOverlay(); updateZoomUI(); }
    function fitView() {
      const w = svg.clientWidth || 600;
      // Fill-width mode (blog homepage): scale to the canvas width and grow the frame's
      // height to wrap the content exactly, so the graph fills the canvas with essentially
      // no letterboxing whatever the (fluid) column width happens to be.
      if (FIT_WIDTH) {
        const s = Math.max(0.05, Math.min(FIT_MAXSCALE, (w / contentW) * FIT_MARGIN));
        view.s = s;
        view.x = (w - contentW * s) / 2;          // tiny symmetric side pad from the margin
        // The drawn graph extends R above the top row's centres and R+caption below the
        // bottom row's; size the frame (and offset the view) to that true extent so nothing
        // clips — content y-centres span [PADY, contentH-PADY].
        const pad = 4, top = PADY - R, bot = contentH - PADY + R + 18;
        view.y = pad - top * s;
        if (graphframe) graphframe.style.height = Math.round((bot - top) * s + pad * 2) + 'px';
        applyView();
        return;
      }
      const hgt = svg.clientHeight || 560;
      const sFit = Math.min(w / contentW, hgt / contentH) * FIT_MARGIN;
      // Never shrink below a legible tile size — deep/tall networks overflow and pan
      // rather than collapsing into an unreadable strip.
      const s = Math.max(0.3, Math.min(FIT_MAXSCALE, sFit));
      view.s = s;
      view.x = (w - contentW * s) / 2;
      // fits vertically → centre; taller than the frame → anchor the top (outputs) so
      // the user starts at the output row and pans downward toward the inputs.
      view.y = (contentH * s <= hgt) ? (hgt - contentH * s) / 2 : 16;
      applyView();
    }
    function zoomAt(cx, cy, factor) {
      const ns = Math.max(0.15, Math.min(4, view.s * factor));
      const gx = (cx - view.x) / view.s, gy = (cy - view.y) / view.s;
      view.s = ns; view.x = cx - gx * ns; view.y = cy - gy * ns; applyView();
    }
    // Keep the graph canvas's bottom border lined up with the bottom of the right render
    // panel at its TALLEST — the "Edge" pane. That height is held constant whatever is (or
    // isn't) selected, so the canvas never jumps as the user clicks around. `primedH` is
    // measured once per load from the edge layout (see primeGraphHeight); the canvas never
    // shrinks below its 560px default, and on a wrapped/narrow layout it reverts to default.
    const graphframe = root.querySelector('.graphframe'), rpanel = root.querySelector('.rpanel');
    // Floor for the canvas height. Defaults to the breeder site's 560px; a host with a more
    // compact layout (the blog uses 440px) can lower it via opts.minGraphH. Must match the
    // CSS default height of .graphframe so the wrapped-layout revert lines up.
    const MIN_GRAPH_H = opts.minGraphH || 560;
    let primedH = 0;
    function panelBottomOffset() {           // panel height relative to the canvas top, or null if wrapped below
      const fr = graphframe.getBoundingClientRect(), rp = rpanel.getBoundingClientRect();
      return (rp.top > fr.top + 40) ? null : (rp.bottom - fr.top);
    }
    function alignGraphHeight() {
      if (FIT_WIDTH) return;            // fill-width hosts drive the frame height from content (fitView)
      if (!graphframe || !rpanel) return;
      const cur = panelBottomOffset();
      if (cur == null) { graphframe.style.height = ''; return; }   // wrapped → CSS default
      graphframe.style.height = Math.max(MIN_GRAPH_H, Math.round(Math.max(primedH, cur))) + 'px';
    }
    // Measure how tall the panel gets with an edge selected (the tallest pane) and lock the
    // canvas to it. Renders the first edge's layout invisibly (no pixel paint), measures,
    // then restores the placeholder — all synchronously, so nothing flashes on screen.
    function primeGraphHeight() {
      primedH = 0;
      const fc = genome && genome.conns && genome.conns.keys().next().value;
      if (fc) { renderSelectedConn(fc, true); const m = panelBottomOffset(); if (m != null) primedH = m; }
      clearSelected();
    }
    window.addEventListener('resize', () => { if (FIT_WIDTH) fitView(); else alignGraphHeight(); });
    // mousedown: on a link → drag up/down to slide its weight; on empty space → pan
    let drag = null, wdrag = null, moved = false;
    svg.addEventListener('mousedown', (e) => {
      if (e.target.closest('[data-ctrl]')) return;
      const hnd = e.target.closest('[data-handle]');
      if (hnd) {
        // Arm a possible handle→node link drag. Like the weight-drag, we don't commit until
        // the pointer actually moves, so a click-and-release stays a click (which enters the
        // two-step "now pick a target" link mode).
        ndrag = { from: parseInt(hnd.dataset.handle, 10), cx: e.clientX, cy: e.clientY, started: false };
        moved = false; e.preventDefault();
        return;
      }
      const nodeG = e.target.closest('[data-node]');
      if (nodeG) {
        // Arm a possible node reposition. Below the movement threshold this stays a click,
        // which selects the node.
        const k = parseInt(nodeG.dataset.node, 10), p = pos.get(k);
        if (p) { mdrag = { k: k, cx: e.clientX, cy: e.clientY, x0: p.x, y0: p.y, started: false }; }
        moved = false; e.preventDefault();
        return;
      }
      const link = e.target.closest('[data-conn]');
      if (link) {
        const key = link.dataset.conn, c = genome.conns.get(key);
        // Arm a possible weight-drag, but DON'T pointer-lock yet — locking on mousedown
        // would swallow the click, so a click-and-release must stay a click (which opens
        // the edge's radial menu). We only lock once the pointer actually moves (below).
        if (c) { wdrag = { key: key, w0: c.weight, acc: 0, cx: e.clientX, cy: e.clientY, lockReq: false }; moved = false; e.preventDefault(); }
        return;
      }
      if (!allowPan) return;            // panning disabled (a plain click still deselects)
      drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y }; moved = false; svg.classList.add('grabbing');
    });
    window.addEventListener('mousemove', (e) => {
      if (mdrag) {
        if (!mdrag.started && Math.abs(e.clientX - mdrag.cx) + Math.abs(e.clientY - mdrag.cy) > 3) {
          mdrag.started = true; moved = true; svg.classList.add('grabbing');
        }
        if (mdrag.started) moveNodeTo(mdrag.k, mdrag.x0 + (e.clientX - mdrag.cx) / view.s, mdrag.y0 + (e.clientY - mdrag.cy) / view.s);
        return;
      }
      if (ndrag) {
        if (!ndrag.started && Math.abs(e.clientX - ndrag.cx) + Math.abs(e.clientY - ndrag.cy) > 4) {
          ndrag.started = true; moved = true; linkDragLine = true;
          svg.classList.add('linking');       // no handles on the nodes we're dragging over
          const p = handlePt(ndrag.from); ndrag.sx = view.x + p.x * view.s; ndrag.sy = view.y + p.y * view.s;
        }
        if (ndrag.started) {
          const r = svg.getBoundingClientRect();
          ovl.innerHTML = `<line class="linkdrag" x1="${ndrag.sx}" y1="${ndrag.sy}" x2="${e.clientX - r.left}" y2="${e.clientY - r.top}"></line>`;
          // light up whichever node the pointer is over — that's the node a release would link to
          const over = nodeAt(e.clientX, e.clientY);
          const tgt = (over != null && over !== ndrag.from) ? over : null;
          if (tgt !== ndrag.over) {
            const cls = (k) => k != null && vp.querySelector('[data-node="' + k + '"]');
            const old = cls(ndrag.over); if (old) old.classList.remove('node-lit');
            const now = cls(tgt); if (now) now.classList.add('node-lit');
            ndrag.over = tgt;
          }
        }
        return;
      }
      if (wdrag) {
        if (document.pointerLockElement === svg) {
          wdrag.acc += e.movementY;                    // locked → accumulate relative movement
        } else {
          wdrag.acc = e.clientY - wdrag.cy;            // not locked yet → delta from the click point
          if (!wdrag.lockReq && Math.abs(wdrag.acc) > 3) {
            // crossed the drag threshold: pin the cursor (devtools/Blender-style scrubbing)
            wdrag.lockReq = true;
            try { const pr = svg.requestPointerLock && svg.requestPointerLock(); if (pr && pr.catch) pr.catch(() => {}); } catch (_) {}
          }
        }
        const dy = wdrag.acc;
        if (Math.abs(dy) > 2) {
          moved = true;
          if (!wdrag.sel) {                       // first real movement → select the link so its two
            wdrag.sel = true;                     // endpoints show in the sidebar, just like clicking it.
            selConn = wdrag.key; selNode = null; selConnPt = null;
            weightDragging = true;
            renderSelectedConn(selConn);          // paint endpoints once; weight number updates live below
          }
        }
        // drag UP (dy<0) raises the weight; ~0.02 per pixel for fine control
        setWeight(wdrag.key, wdrag.w0 - dy * 0.02);
        showWtip(wdrag.cx, wdrag.cy, wlabel(wdrag.key), true);   // readout pinned at the click point
        const wc = genome.conns.get(wdrag.key);
        if (wc) showWscale(wdrag.cx, wdrag.cy, wc.weight);       // thermometer tracks the value into its stops
        return;
      }
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      view.x = drag.vx + dx; view.y = drag.vy + dy; applyView();
    });
    window.addEventListener('mouseup', (e) => {
      if (mdrag) {
        if (mdrag.started) {
          // Commit the drop: the node opts out of the layered layout from here on, and a full
          // draw() re-routes its links (straight now) and re-fits the handle directions.
          const p = pos.get(mdrag.k);
          manualPos.set(mdrag.k, { x: p.x, y: p.y });
          svg.classList.remove('grabbing');
          draw();
        }
        mdrag = null;
        return;
      }
      if (ndrag) {
        if (ndrag.started) {
          linkDragLine = false;
          svg.classList.remove('linking');
          const to = nodeAt(e.clientX, e.clientY);          // released over a node → link to it
          let ok = false;
          if (to != null && to !== ndrag.from) ok = addLink(ndrag.from, to);
          if (!ok) draw();                 // addLink→refresh already redraws; otherwise drop the
        }                                  // rubber-band and the drop target's highlight
        ndrag = null;
        return;
      }
      if (wdrag) {
        if (document.pointerLockElement === svg && document.exitPointerLock) document.exitPointerLock();  // restores the cursor to the click point
        if (weightDragging) { weightDragging = false; if (selConn === wdrag.key) { renderSelectedConn(selConn); drawOverlay(); } }  // repaint endpoints at the settled weight + show the edge ring
        if (wtipTimer) clearTimeout(wtipTimer); wtipTimer = setTimeout(() => wtip.classList.remove('on'), 700);  // let the readout fade where it sits
        hideWscale(700);                          // fade the thermometer out alongside the readout
      }
      drag = null; wdrag = null; svg.classList.remove('grabbing');
    });
    // wheel always zooms toward the cursor (weight editing is drag-only, so scrolling
    // over a link no longer fights with zooming the canvas). Gentle + magnitude-aware:
    // normalise deltaMode to pixels, clamp trackpad spikes, ~3% per mouse notch.
    // Plain wheel / two-finger scroll is left alone so the page scrolls normally; only a
    // trackpad pinch (delivered as a wheel event with ctrlKey) zooms. The toolbar slider is
    // the explicit mouse-driven control.
    svg.addEventListener('wheel', (e) => {
      if (lockView || !e.ctrlKey) return;
      e.preventDefault();
      const r = svg.getBoundingClientRect();
      let dy = e.deltaY;
      if (e.deltaMode === 1) dy *= 16; else if (e.deltaMode === 2) dy *= 400;
      dy = Math.max(-50, Math.min(50, dy));
      zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-dy * 0.003));   // pinch sensitivity (was .0006 — much stronger now)
    }, { passive: false });

    if (qs('g-fit')) qs('g-fit').onclick = fitView;   // absent when the view is locked

    // ---- horizontal zoom slider (toolbar): value 0..1 maps log-scaled to ZMIN..ZMAX, canvas-centred ----
    const ZMIN = 0.2, ZMAX = 4, zrange = qs('zoom');
    let zdrag = false;
    const scaleToT = (s) => clamp((Math.log(s) - Math.log(ZMIN)) / (Math.log(ZMAX) - Math.log(ZMIN)), 0, 1);
    function updateZoomUI() { if (zrange && !zdrag) zrange.value = scaleToT(view.s); }
    function zoomToScale(s) { zoomAt(svg.clientWidth / 2, svg.clientHeight / 2, clamp(s, ZMIN, ZMAX) / view.s); }
    if (zrange) {
      zrange.addEventListener('input', () => { zdrag = true; zoomToScale(ZMIN * Math.pow(ZMAX / ZMIN, parseFloat(zrange.value))); zdrag = false; });
      updateZoomUI();
    }

    // ---------- click routing (suppressed right after a pan drag) ----------
    svg.addEventListener('click', (e) => {
      if (moved) { moved = false; return; }
      const ctrl = e.target.closest('[data-ctrl]');
      if (ctrl) {
        const act = ctrl.dataset.ctrl, t = ctrl.dataset.target;
        if (act.indexOf('act:') === 0) setActivation(parseInt(t, 10), act.slice(4));
        else if (act === 'lnk') startLink(parseInt(t, 10));
        else if (act === 'edt') openEdit(parseInt(t, 10));
        else if (act === 'del') deleteNode(parseInt(t, 10));
        else if (act === 'nod') insertNodeOnLink(t);
        else if (act === 'delc') deleteConn(t);
        else if (act === 'pm') { const c = genome.conns.get(t); const v = prompt('New weight for this link (most are between -3 and 3):', c.weight.toFixed(3)); if (v != null && v.trim() !== '' && !isNaN(parseFloat(v))) setWeight(t, parseFloat(v)); }
        return;
      }
      const hnd = e.target.closest('[data-handle]');
      if (hnd) {                       // clicked (not dragged) the handle → pick a target next
        const k = parseInt(hnd.dataset.handle, 10);
        if (linkFrom != null) { if (k !== linkFrom) addLink(linkFrom, k); cancelLink(); }
        else startLink(k);
        return;
      }
      const nodeG = e.target.closest('[data-node]');
      if (nodeG) {
        const k = parseInt(nodeG.dataset.node, 10);
        if (linkFrom != null) { if (k !== linkFrom) addLink(linkFrom, k); cancelLink(); return; }
        (k === selNode) ? deselect() : selectNode(k);
        return;
      }
      const link = e.target.closest('[data-conn]');
      if (link) {
        if (link.dataset.conn === selConn) deselect();
        else selectConn(link.dataset.conn, toContent(e.clientX, e.clientY));   // ring opens where clicked
        return;
      }
      if (linkFrom != null) { cancelLink(); return; }
      deselect();
    });
    // double-click: on a link → split it with a node; on empty canvas → add a free node
    svg.addEventListener('dblclick', (e) => {
      if (e.target.closest('[data-ctrl]')) return;
      e.preventDefault();
      const link = e.target.closest('[data-conn]');
      if (link) { insertNodeOnLink(link.dataset.conn); return; }
      if (e.target.closest('[data-node]')) return;
      addNode(toContent(e.clientX, e.clientY));
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { cancelLink(); deselect(); return; }
      // Backspace / Delete removes the current selection — but only when the editor owns a
      // selection and the user isn't typing in a field (weight editor, etc.), so we never
      // steal the key from a text input or trigger browser "back".
      if (e.key === 'Backspace' || e.key === 'Delete') {
        const t = e.target, tag = t && t.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
        if (selConn != null) { e.preventDefault(); deleteConn(selConn); }
        else if (selNode != null) { e.preventDefault(); deleteNode(selNode); }
      }
    });
    colorChk.addEventListener('change', () => { colorMode = colorChk.checked; rerender(); if (opts.onColor) opts.onColor(colorMode); });

    // ---------- toolbar ----------
    function blobDownload(blob, name) { const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
    const titleSlug = () => (curMeta.title || 'genome').replace(/\s+/g, '_');
    // The toolbar buttons a host can ask for, by key. Hosts pick a subset via opts.actions.
    const ACTION_DEFS = {
      breed: { html: '<button class="pbbtn" id="act-breed"><span class="ic">&#9654;</span> Breed from this DNA</button>',
        wire: () => qs('act-breed').onclick = () => {
          sessionStorage.setItem('pb_dna_genome', JSON.stringify(genomeToJSON(genome)));
          sessionStorage.setItem('pb_dna_meta', JSON.stringify({ title: curMeta.title, color: colorMode, model: curMeta.model, agent: curMeta.agent }));
          // play.html sits beside this widget on the breeder site, but a host embedding the
          // editor elsewhere (e.g. the blog, served from the site root) passes its own URL.
          location.href = opts.playUrl || 'play.html?dna=1';
        } },
      download: { html: '<button class="pbbtn" id="act-dl">Download DNA</button>',
        wire: () => qs('act-dl').onclick = () => blobDownload(new Blob([JSON.stringify(genomeToJSON(genome), null, 2)], { type: 'application/json' }), titleSlug() + '_edited.json') },
      png: { html: '<button class="pbbtn" id="act-png">Save PNG</button>',
        wire: () => qs('act-png').onclick = () => { const c = document.createElement('canvas'); c.width = 512; c.height = 512; c.getContext('2d').putImageData(renderGenome(genome, 512, colorMode), 0, 0); c.toBlob((b) => blobDownload(b, titleSlug() + '.png')); } },
      reset: { html: '<button class="pbbtn" id="act-reset">Reset to original</button>',
        wire: () => qs('act-reset').onclick = () => reset() },
    };
    const actionKeys = opts.actions || ['breed', 'download', 'png', 'reset'];
    function buildActions() {
      const el = qs('actions');
      el.innerHTML = actionKeys.map((k) => ACTION_DEFS[k] ? ACTION_DEFS[k].html : '').join(' ');
      actionKeys.forEach((k) => { if (ACTION_DEFS[k]) ACTION_DEFS[k].wire(); });
    }

    // ---------- public ops ----------
    function reset() {
      genome = genomeFromJSON(JSON.parse(originalJSON)); bumpCtr();
      manualPos.clear();
      qs('editpanel').classList.remove('on');
      linkFrom = null; banner.classList.remove('on'); selNode = null; selConn = null; clearSelected();
      rerender(); draw(); primeGraphHeight(); fitView(); setStatus('');
    }
    // Point the editor at a (new) genome. `g` may be a cppn.js Genome (edited in
    // place by the host) or genome JSON. `meta` may carry {color, title, model, agent}.
    function setGenome(g, meta) {
      genome = (g instanceof Genome) ? g : genomeFromJSON(g);
      // Stored genomes omit the four implicit inputs (x, y, r, bias) — their links
      // reference keys -1..-4 but carry no node objects. Re-add them so the graph can
      // draw and link from them (harmless to the renderer, which seeds inputs directly).
      INPUT_KEYS.forEach((k) => { if (!genome.nodes.has(k)) genome.nodes.set(k, { activation: null, affinity: 'grey', input: true }); });
      originalJSON = JSON.stringify(genomeToJSON(genome));
      manualPos.clear();
      if (meta) { curMeta = Object.assign({}, curMeta, meta); if (meta.color !== undefined) { colorMode = meta.color !== false; colorChk.checked = colorMode; } }
      bumpCtr();
      linkFrom = null; banner.classList.remove('on'); selNode = null; selConn = null; selConnPt = null;
      qs('editpanel').classList.remove('on'); clearSelected(); setStatus('');
      rerender(); draw(); primeGraphHeight(); fitView();
    }

    buildActions();
    clearSelected();
    if (opts.genome) setGenome(opts.genome, { color: opts.color, title: (opts.meta && opts.meta.title) });

    // Set color-mode from outside (e.g. host syncing a shared toggle). Does NOT fire
    // opts.onColor — only a user click on this checkbox does — so hosts can two-way bind
    // without an echo loop.
    function setColor(c) {
      colorMode = c !== false; colorChk.checked = colorMode;
      if (genome) rerender();
    }

    return {
      root: root,
      setGenome: setGenome,
      getGenome: () => genome,
      reset: reset,
      fit: fitView,
      redraw: () => { rerender(); draw(); },
      setColor: setColor,
      getColor: () => colorMode,
    };
  };
})();
