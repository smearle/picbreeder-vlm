/* ===========================================================================
 * morph-graph-inset.js — PB.MorphGraphInset
 *
 * A small canvas inset for the scrubbable phylogeny viewer (sfdp-morph-demo.html)
 * that draws the CPPN's *topology* morphing from parent to child, in the same
 * visual language as the blog's "Explore a CPPN" editor (breed/dna-editor.js):
 *
 *   - a graphviz-`dot`-style layered graph, inputs at the bottom and the H/S/V
 *     outputs at the top, long edges routed through thin dummy channels;
 *   - each node drawn as a square tile of its own intermediate pattern
 *     (contrast-stretched grayscale, à la Fig. 5), framed by a colour marking
 *     its kind (input / grey / color / output);
 *   - links coloured by their weight on the shared PB_CMAP diverging ramp.
 *
 * Unlike the editor (one static genome), this renders a *blend*: it lays the
 * superset graph (parent ∪ child) out ONCE per branch so nodes never jump, then
 * for the morph parameter t it
 *   - interpolates every connection's weight (matching the GPU image morph),
 *   - blends each node's activation output,        mix(actA(S), actB(S), t)
 *   - fades nodes/edges in (child-only) or out (parent-only) by opacity.
 * Node patterns are evaluated on the CPU at a small resolution every frame —
 * cheap because the whole superset is one forward pass per pixel.
 *
 * Depends on cppn.js globals (feedForwardLayers, INPUT_KEYS, OUTPUT_KEYS, ACT)
 * and weight-cmap.js (window.PB_CMAP). No interactivity, no DOM beyond the one
 * canvas it is handed.
 *
 *   const inset = PB.MorphGraphInset(canvasEl);
 *   inset.update(childId, rec, t);   // rec = {sup, nodePairs, connStats, nodeIn}
 *   inset.clear();  inset.resize();
 * ======================================================================== */
(function (root) {
  const PB = root.PB || (root.PB = {});
  const CMAP = root.PB_CMAP || { link: function () { return '#888'; } };

  const IN_NAME  = { '-1': 'x', '-2': 'y', '-3': 'r', '-4': 'bias' };
  const OUT_NAME = { '0': 'hue', '1': 'sat', '2': 'value' };
  const ACT_ABBR = { identity: '=', sin: 'sin', cos: 'cos', gaussian: 'gau', sigmoid: 'sig' };

  // Layout knobs (content units). Kept close to dna-editor's so the look matches,
  // but a touch tighter since the inset is small; fit() scales the whole thing down.
  const R = 21, COLW = 60, ROWV = 82, PADX = 34, PADY = 34;
  const LINKW = 4.5, LINKW_OFF = 2;
  const DUMW = 0.34;                                   // dummy column width (slot fraction)

  // ----- layered layout (port of dna-editor.js computeDepth/orderAug/layout) -----
  // Operates on a plain {nodes:Map, conns:Map} where every laid-out connection is
  // marked enabled. Returns {pos, routes, contentW, contentH}.
  function computeDepth(G) {
    const enabled = [...G.conns.values()].filter((c) => c.enabled);
    const adj = new Map();
    enabled.forEach((c) => { if (!adj.has(c.i)) adj.set(c.i, []); adj.get(c.i).push(c.o); });
    const color = new Map(), back = new Set();          // cycle-safe: drop DFS back-edges
    const dfs = (u) => {
      color.set(u, 1);
      for (const v of (adj.get(u) || [])) {
        const cs = color.get(v) || 0;
        if (cs === 1) back.add(u + ',' + v);
        else if (cs === 0) dfs(v);
      }
      color.set(u, 2);
    };
    INPUT_KEYS.forEach((k) => { if (G.nodes.has(k) && !color.has(k)) dfs(k); });
    [...G.nodes.keys()].forEach((k) => { if (!color.has(k)) dfs(k); });
    const fwd = enabled.filter((c) => !back.has(c.i + ',' + c.o));
    const depth = new Map();
    for (const k of G.nodes.keys()) depth.set(k, G.nodes.get(k).input ? 0 : 1);
    for (let it = 0; it < G.nodes.size + 2; it++) {
      let changed = false;
      for (const c of fwd) { const nd = Math.max(depth.get(c.o) || 1, (depth.get(c.i) || 0) + 1); if (nd !== depth.get(c.o)) { depth.set(c.o, nd); changed = true; } }
      if (!changed) break;
    }
    let maxd = 0; depth.forEach((v) => { if (v > maxd) maxd = v; });
    OUTPUT_KEYS.forEach((k) => { if (G.nodes.has(k)) depth.set(k, Math.max(maxd, depth.get(k) || 1)); });
    return depth;
  }

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
      const downward = s % 2 === 0;
      const seq = []; for (let d = 1; d < last; d++) seq.push(d);
      if (!downward) seq.reverse();
      for (const d of seq) {
        const nbr = downward ? up : down, col = layers[d];
        const bary = new Map();
        col.forEach((id, i) => { const ns = nbr.get(id); let b = i; if (ns.length) { let sum = 0; for (const n of ns) sum += idx.get(n); b = sum / ns.length; } bary.set(id, b); });
        col.sort((a, b) => (bary.get(a) - bary.get(b)) || tie(a, b));
        col.forEach((id, i) => idx.set(id, i));
      }
      let improved = true, guard = 0;
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

  function layout(G) {
    const depth = computeDepth(G);
    let maxd = 0; depth.forEach((v) => { if (v > maxd) maxd = v; });
    const layers = []; for (let i = 0; i <= maxd; i++) layers.push([]);
    [...G.nodes.keys()].sort((a, b) => a - b).forEach((k) => layers[depth.get(k)].push('n' + k));
    const chains = new Map();
    let dctr = 0;
    for (const c of G.conns.values()) {
      if (!c.enabled) continue;
      const di = depth.get(c.i), dj = depth.get(c.o);
      if (di == null || dj == null || dj - di <= 1) continue;
      const chain = [];
      for (let L = di + 1; L < dj; L++) { const id = 'd' + (dctr++); layers[L].push(id); chain.push(id); }
      chains.set(c.i + ',' + c.o, chain);
    }
    const up = new Map(), down = new Map();
    layers.forEach((col) => col.forEach((id) => { up.set(id, []); down.set(id, []); }));
    const seg = (loId, hiId) => { down.get(loId).push(hiId); up.get(hiId).push(loId); };
    for (const c of G.conns.values()) {
      if (!c.enabled) continue;
      const di = depth.get(c.i), dj = depth.get(c.o); if (di == null || dj == null) continue;
      const chain = chains.get(c.i + ',' + c.o);
      if (!chain) { seg('n' + c.i, 'n' + c.o); continue; }
      let prev = 'n' + c.i;
      for (const id of chain) { seg(prev, id); prev = id; }
      seg(prev, 'n' + c.o);
    }
    orderAug(layers, up, down);
    const nLev = layers.length;
    const rowWidth = (col) => col.reduce((s, id) => s + (id[0] === 'd' ? DUMW : 1), 0) - 1;
    const maxRow = Math.max(1, ...layers.map(rowWidth));
    let contentW = PADX * 2 + Math.max(1, maxRow) * COLW;
    const contentH = PADY * 2 + Math.max(1, nLev - 1) * ROWV;
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
    // straighten: pull each node toward its neighbours' mean x (isotonic per row)
    const nbrsOf = (id) => (up.get(id) || []).concat(down.get(id) || []);
    const halfSlot = (id) => (id[0] === 'd' ? DUMW : 1) * COLW / 2;
    const minGap = (a, b) => halfSlot(a) + halfSlot(b);
    const isotonicL2 = (v) => {
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
      const v = col.map((id, i) => {
        const ns = nbrsOf(id);
        let d = xy.get(id).x;
        if (ns.length) { let s = 0; for (const n of ns) s += xy.get(n).x; d = s / ns.length; }
        return d - pref[i];
      });
      const yh = isotonicL2(v);
      col.forEach((id, i) => { xy.get(id).x = yh[i] + pref[i]; });
    };
    for (let pass = 0; pass < 16; pass++) {
      const order = layers.map((_, d) => d);
      if (pass % 2) order.reverse();
      for (const d of order) straightenRow(layers[d]);
    }
    let minx = Infinity, maxx = -Infinity;
    xy.forEach((p) => { if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x; });
    const shiftX = PADX - minx;
    xy.forEach((p) => { p.x += shiftX; });
    contentW = (maxx - minx) + PADX * 2;

    const pos = new Map();
    for (const k of G.nodes.keys()) { const p = xy.get('n' + k); if (p) pos.set(k, p); }
    const routes = new Map();
    for (const [key, chain] of chains) routes.set(key, chain.map((id) => xy.get(id)));
    return { pos, routes, contentW, contentH };
  }

  // ----- contrast-stretched grayscale of one node's intermediate pattern -----
  function fieldToImageData(field, res, alpha) {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < field.length; i++) { const v = field[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    const rng = mx - mn;
    const img = new ImageData(res, res), d = img.data, a = Math.round((alpha == null ? 1 : alpha) * 255);
    for (let i = 0; i < field.length; i++) {
      const t = rng > 1e-6 ? (field[i] - mn) / rng : 0.5;
      const g = (t * 255 + 0.5) | 0;
      d[i * 4] = g; d[i * 4 + 1] = g; d[i * 4 + 2] = g; d[i * 4 + 3] = a;
    }
    return img;
  }

  PB.MorphGraphInset = function (canvas, opts) {
    opts = opts || {};
    const ctx = canvas.getContext('2d');
    const DPR = Math.min(2, root.devicePixelRatio || 1);

    let edgeId = null;            // current branch (child id) — re-layout only when it changes
    let LAY = null;               // {pos, routes, contentW, contentH}
    let conns = null;             // [{key, i, o, sw, ew, se, ee, enabledEither}]
    let nodeList = null;          // node keys in layout, with kind/labels/membership
    let evalSeq = null;           // [{node, links:[{i,key}], a0, a1}] topo order for the field pass
    let fields = null;            // Map node -> Float32Array(res*res)
    let res = 24;
    let view = { s: 1, x: 0, y: 0 };
    let tile = null, tctx = null; // reusable offscreen for putImageData→drawImage

    function ensureTile() {
      if (tile && tile.width === res) return;
      tile = document.createElement('canvas'); tile.width = tile.height = res; tctx = tile.getContext('2d');
    }

    function nodeKind(k, n) {
      if (n.input) return { frame: '#555', lw: 2 };
      if (OUTPUT_KEYS.includes(k)) return { frame: '#cc8a1a', lw: 3 };
      if (n.affinity === 'grey') return { frame: '#8a8a8a', lw: 2.5 };
      return { frame: '#2d68d4', lw: 2.5 };
    }
    function nodeCaption(k, n, pair) {
      if (n.input) return IN_NAME[k] || '';
      if (OUTPUT_KEYS.includes(k)) return OUT_NAME[k] || '';
      const act = pair ? pair[1] : n.activation;        // child activation
      return ACT_ABBR[act] || '?';
    }

    function setEdge(rec) {
      const sup = rec.sup, connStats = rec.connStats, nodePairs = rec.nodePairs, nodeIn = rec.nodeIn || {};
      // Layout genome: every connection present in EITHER parent or child is laid
      // out (enabled); a gene disabled on both ends doesn't shape the graph.
      const LGconns = new Map();
      conns = [];
      for (const key in connStats) {
        const s = connStats[key];
        const enabledEither = !!(s.se || s.ee);
        LGconns.set(key, { i: s.i, o: s.o, enabled: enabledEither });
        conns.push({ key: key, i: s.i, o: s.o, sw: s.sw, ew: s.ew, se: s.se, ee: s.ee, enabledEither: enabledEither });
      }
      const LG = { nodes: sup.nodes, conns: LGconns };
      LAY = layout(LG);

      // node count drives the field resolution (legibility vs. cost)
      const nN = sup.nodes.size;
      res = nN > 70 ? 16 : nN > 40 ? 20 : 26;
      ensureTile();

      nodeList = [];
      for (const [k, n] of sup.nodes) {
        nodeList.push({ k: k, n: n, kind: nodeKind(k, n),
          cap: nodeCaption(k, n, nodePairs[k]),
          mem: nodeIn[k] || 'both' });
      }

      // topo order for the per-pixel forward pass (union, enabled-either)
      const eConns = conns.filter((c) => c.enabledEither).map((c) => [c.i, c.o]);
      const layers = feedForwardLayers(INPUT_KEYS, OUTPUT_KEYS, eConns);
      const declared = new Set(INPUT_KEYS);
      evalSeq = [];
      for (const layer of layers) for (const node of layer) {
        const links = [];
        for (const c of conns) if (c.enabledEither && c.o === node && (c.i < 0 || declared.has(c.i))) links.push({ i: c.i, key: c.key });
        const pair = nodePairs[node] || ['identity', 'identity'];
        evalSeq.push({ node: node, links: links, a0: ACT[pair[0]] || ACT.identity, a1: ACT[pair[1]] || ACT.identity });
        declared.add(node);
      }
      fields = new Map();
      for (const k of sup.nodes.keys()) fields.set(k, new Float32Array(res * res));
      fit();
    }

    // weight at parameter t (mirrors the GPU image morph's glFrameUniforms)
    function interpW(c, t) {
      if (c.se === c.ee) return (1 - t) * c.sw + t * c.ew;
      if (c.se && !c.ee) return c.sw * (1 - t);
      return c.ew * t;
    }
    // node opacity from its parent/child membership
    function nodeAlpha(mem, t) { return mem === 'both' ? 1 : mem === 'child' ? t : 1 - t; }
    // connection opacity (same logic as the field weight fade)
    function connAlpha(c, t) { return (c.se === c.ee) ? 1 : (c.se && !c.ee) ? 1 - t : t; }

    // one forward pass per pixel → every node's blended intermediate pattern
    function computeFields(t) {
      const W = {};
      for (const c of conns) if (c.enabledEither) W[c.key] = interpW(c, t);
      const N = res, span = N;
      let idx = 0;
      const v = Object.create(null);
      for (let yy = 0; yy < N; yy++) {
        const sy = ((yy * 2) - (N - 1)) / span;
        for (let xx = 0; xx < N; xx++) {
          const sx = ((xx * 2) - (N - 1)) / span;
          const dd = Math.hypot(sx, sy) * Math.SQRT2;
          v[-1] = sx; v[-2] = sy; v[-3] = dd; v[-4] = 1.0;
          const f1 = fields.get(-1), f2 = fields.get(-2), f3 = fields.get(-3), f4 = fields.get(-4);
          if (f1) f1[idx] = sx; if (f2) f2[idx] = sy; if (f3) f3[idx] = dd; if (f4) f4[idx] = 1.0;
          for (let e = 0; e < evalSeq.length; e++) {
            const ev = evalSeq[e]; let s = 0; const L = ev.links;
            for (let j = 0; j < L.length; j++) s += v[L[j].i] * W[L[j].key];
            const val = (ev.a0 === ev.a1) ? ev.a0(s) : (ev.a0(s) * (1 - t) + ev.a1(s) * t);
            v[ev.node] = val;
            const f = fields.get(ev.node); if (f) f[idx] = val;
          }
          idx++;
        }
      }
    }

    function fit() {
      if (!LAY) return;
      const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300, m = 10;
      const s = Math.min((w - 2 * m) / LAY.contentW, (h - 2 * m) / LAY.contentH);
      view.s = s;
      view.x = (w - LAY.contentW * s) / 2;
      view.y = (h - LAY.contentH * s) / 2;
    }
    const tx = (x) => view.x + x * view.s;
    const ty = (y) => view.y + y * view.s;

    function draw(t) {
      const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!LAY) return;
      computeFields(t);

      // ---- links (routed polylines), coloured by interpolated weight ----
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      for (const c of conns) {
        const a = LAY.pos.get(c.i), b = LAY.pos.get(c.o); if (!a || !b) continue;
        const al = connAlpha(c, t); if (al < 0.03) continue;
        const wpath = c.enabledEither ? (LAY.routes.get(c.key) || []) : [];
        ctx.beginPath();
        ctx.moveTo(tx(a.x), ty(a.y));
        for (const p of wpath) ctx.lineTo(tx(p.x), ty(p.y));
        ctx.lineTo(tx(b.x), ty(b.y));
        if (c.enabledEither) {
          ctx.strokeStyle = CMAP.link(interpW(c, t), al);
          ctx.lineWidth = Math.max(0.8, LINKW * view.s);
          ctx.setLineDash([]);
        } else {
          ctx.strokeStyle = 'rgba(170,170,170,' + (0.6 * al).toFixed(3) + ')';
          ctx.lineWidth = Math.max(0.6, LINKW_OFF * view.s);
          ctx.setLineDash([4 * view.s, 3 * view.s]);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // ---- nodes: a tile of the intermediate pattern + a kind-coloured frame ----
      const sz = 2 * R * view.s;
      const capPx = Math.max(7, Math.min(12, 11 * view.s));
      ctx.font = '700 ' + capPx.toFixed(1) + 'px Arial, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const item of nodeList) {
        const p = LAY.pos.get(item.k); if (!p) continue;
        const al = nodeAlpha(item.mem, t); if (al < 0.03) continue;
        const cx = tx(p.x), cy = ty(p.y), x0 = cx - sz / 2, y0 = cy - sz / 2;
        const field = fields.get(item.k);
        ctx.globalAlpha = al;
        // tile background (so a faint field still reads as a square), then pattern
        ctx.fillStyle = '#fff'; ctx.fillRect(x0, y0, sz, sz);
        if (field) { tctx.putImageData(fieldToImageData(field, res), 0, 0); ctx.drawImage(tile, x0, y0, sz, sz); }
        ctx.strokeStyle = item.kind.frame; ctx.lineWidth = Math.max(1, item.kind.lw * view.s * 0.85);
        ctx.strokeRect(x0, y0, sz, sz);
        // caption below the tile
        if (item.cap) {
          ctx.globalAlpha = al;
          ctx.fillStyle = '#fbfcfe'; // halo
          const ty2 = cy + sz / 2 + capPx * 0.9;
          ctx.lineWidth = 3; ctx.strokeStyle = '#fbfcfecc';
          ctx.strokeText(item.cap, cx, ty2);
          ctx.fillStyle = '#333'; ctx.fillText(item.cap, cx, ty2);
        }
      }
      ctx.globalAlpha = 1;
    }

    function resize() {
      const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300;
      canvas.width = Math.round(w * DPR); canvas.height = Math.round(h * DPR);
      fit();
    }

    function update(id, rec, t) {
      if (id !== edgeId) { edgeId = id; setEdge(rec); }
      draw(t);
    }
    function clear() {
      edgeId = null; LAY = null; conns = null; nodeList = null; evalSeq = null; fields = null;
      const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.clearRect(0, 0, w, h);
    }

    resize();
    return { update: update, clear: clear, resize: resize };
  };
})(typeof window !== 'undefined' ? window : this);
