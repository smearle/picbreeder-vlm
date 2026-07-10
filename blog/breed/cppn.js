/* ===========================================================================
 * cppn.js — a faithful, in-browser port of the Picbreeder-VLM CPPN substrate.
 *
 * This mirrors the Python used in the paper (neat_components.py,
 * picbreeder_reproduction.py, picture2d/common.py) closely enough that the
 * images you breed here are drawn from the same distribution the VLMs (and the
 * original Picbreeder users) explored:
 *
 *   - Inputs:  x, y, r=hypot(x,y)*sqrt(2), bias=1            (picture2d _canvas_coords)
 *   - Outputs: [hue, saturation, brightness] in HSV          (restrict_hsb_channels)
 *   - Functions: sin, sigmoid(bipolar), gaussian(bipolar), cos, identity
 *   - Genome scaffold: grayscale inputs->hidden->ink, plus an InkToHSB color
 *     subnet (hue=sin, sat=sigmoid), exactly as seed_initial_population builds it
 *   - Mutation: weighted generators (mutate_weights 10, add_connection 6,
 *     add_node 4, mutate_activation 1) with grey/color node affinities
 *   - Reproduction: selected genomes carry over unchanged (elitism), the rest
 *     are clones/crossovers of selected parents, each mutated once
 * ======================================================================== */

// ---------- RNG / math helpers ----------
function uniform(a, b) { return a + (b - a) * Math.random(); }
function randint(n) { return Math.floor(Math.random() * n); }
function choice(arr) { return arr[randint(arr.length)]; }
function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
function pmod(a, n) { return ((a % n) + n) % n; }       // python-style modulo
function gauss(mean, sd) {                               // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- activation functions (match neat_components._*_unit) ----------
const ACT = {
  identity: (z) => z,
  sigmoid:  (z) => { z = clamp(z, -60, 60); return 2 / (1 + Math.exp(-z)) - 1; },
  gaussian: (z) => { z = clamp(z, -60, 60); return 2 * Math.exp(-z * z) - 1; },
  cos:      (z) => { z = clamp(z, -60, 60); return Math.cos(z); },
  sin:      (z) => { z = clamp(z, -60, 60); return Math.sin(z); },
};
const ACT_OPTIONS = ['sin', 'sigmoid', 'gaussian', 'cos', 'identity'];

// ---------- structural constants ----------
const INPUT_KEYS = [-1, -2, -3, -4];   // x, y, r, bias
const OUTPUT_KEYS = [0, 1, 2];         // hue, sat, brightness
const WEIGHT_MIN = -3, WEIGHT_MAX = 3;
function randWeight() { return uniform(-3, 3); }

// generator mix + rates (apply_picbreeder_config_defaults)
const GEN_WEIGHTS = [['mutate_weights', 10], ['add_connection', 6], ['add_node', 4], ['mutate_activation', 1]];
const ACT_RATE = 0.05, W_MUT_RATE = 0.20, POW_MIN = 0.01, POW_MAX = 2.0;
const COLOR_LINK = 0.50, GREY_LINK = 0.50, GREY_TO_COLOR = 0.10;
const CROSSOVER_RATE = 0.30;

// ---------- genome ----------
class Genome {
  constructor() {
    this.nodes = new Map();   // key -> {activation, affinity, input?}
    this.conns = new Map();   // "i,o" -> {i, o, weight, enabled}
    this.inAct = null;        // optional [4] input-activation names (transform_inputs)
    this.outAct = null;       // optional [3] output-activation names (transform_outputs)
    this._net = null;         // cached compiled network
  }
  addConn(i, o, w, enabled = true) {
    this.conns.set(i + ',' + o, { i, o, weight: w, enabled });
    this._net = null;
  }
  hasConn(i, o) { return this.conns.has(i + ',' + o); }
}

function cloneGenome(g) {
  const c = new Genome();
  for (const [k, n] of g.nodes) c.nodes.set(k, { activation: n.activation, affinity: n.affinity, input: n.input });
  for (const [k, cn] of g.conns) c.conns.set(k, { i: cn.i, o: cn.o, weight: cn.weight, enabled: cn.enabled });
  c.inAct = g.inAct ? g.inAct.slice() : null;
  c.outAct = g.outAct ? g.outAct.slice() : null;
  return c;
}

// Rebuild a Genome from the JSON produced by genomeToJSON / the Python exporter.
function genomeFromJSON(j) {
  const g = new Genome();
  for (const n of j.nodes) g.nodes.set(n.key, { activation: n.activation, affinity: n.affinity, input: !!n.input });
  for (const c of j.connections) g.addConn(c.from, c.to, c.weight, c.enabled !== false);
  g.inAct = j.inAct || null;
  g.outAct = j.outAct || null;
  return g;
}

// ---------- population (owns the global node-key counter) ----------
class Population {
  constructor() { this.ctr = 3; this.genomes = []; }  // node keys 0..2 are outputs
  newNodeKey() { return this.ctr++; }
}

// Build the initial CPPN scaffold (seed_initial_population + InkToHSB bootstrap).
function makeSeed(pop) {
  const g = new Genome();
  for (const k of INPUT_KEYS) g.nodes.set(k, { activation: null, affinity: 'grey', input: true });
  g.nodes.set(0, { activation: choice(ACT_OPTIONS), affinity: 'color' });
  g.nodes.set(1, { activation: choice(ACT_OPTIONS), affinity: 'color' });
  g.nodes.set(2, { activation: choice(ACT_OPTIONS), affinity: 'grey' });

  // grayscale subnet: inputs -> 1 grey hidden -> brightness(2)
  const gh = pop.newNodeKey();
  g.nodes.set(gh, { activation: choice(ACT_OPTIONS), affinity: 'grey' });
  for (const inp of INPUT_KEYS) g.addConn(inp, gh, randWeight());
  g.addConn(gh, 2, randWeight());

  // color subnet (InkToHSB): hue=sin, sat=sigmoid; ink drives color initially.
  g.nodes.get(0).activation = 'sin';
  g.nodes.get(1).activation = 'sigmoid';
  const ch = [pop.newNodeKey(), pop.newNodeKey()];
  for (const k of ch) g.nodes.set(k, { activation: choice(ACT_OPTIONS), affinity: 'color' });
  g.addConn(2, 0, randWeight());
  g.addConn(2, 1, randWeight());
  const sources = [...INPUT_KEYS, 2];
  for (const s of sources) for (const h of ch) g.addConn(s, h, randWeight());
  for (const h of ch) for (const t of [0, 1]) g.addConn(h, t, 0.0);  // dormant until mutated
  return g;
}

function newPopulation(size = 15) {
  const pop = new Population();
  for (let i = 0; i < size; i++) pop.genomes.push(makeSeed(pop));
  return pop;
}

// ---------- feed-forward compilation (port of neat.graphs) ----------
function requiredForOutput(inputs, outputs, conns) {
  const inSet = new Set(inputs);
  let required = new Set(outputs);
  let s = new Set(outputs);
  while (true) {
    const t = new Set();
    for (const [a, b] of conns) if (s.has(b) && !s.has(a)) t.add(a);
    if (!t.size) break;
    const layerNodes = new Set([...t].filter((x) => !inSet.has(x)));
    if (!layerNodes.size) break;
    for (const n of layerNodes) required.add(n);
    for (const n of t) s.add(n);
  }
  return required;
}

function feedForwardLayers(inputs, outputs, conns) {
  const required = requiredForOutput(inputs, outputs, conns);
  const layers = [];
  let s = new Set(inputs);
  while (true) {
    const c = new Set();
    for (const [a, b] of conns) if (s.has(a) && !s.has(b)) c.add(b);
    const t = new Set();
    for (const n of c) {
      if (!required.has(n)) continue;
      let allKnown = true;
      for (const [a, b] of conns) if (b === n && !s.has(a)) { allKnown = false; break; }
      if (allKnown) t.add(n);
    }
    if (!t.size) break;
    layers.push(t);
    for (const n of t) s.add(n);
  }
  return layers;
}

function createsCycle(conns, test) {
  const [i, o] = test;
  if (i === o) return true;
  const visited = new Set([o]);
  while (true) {
    let added = 0;
    for (const [a, b] of conns) {
      if (visited.has(a) && !visited.has(b)) {
        if (b === i) return true;
        visited.add(b); added++;
      }
    }
    if (added === 0) return false;
  }
}

function buildNet(g) {
  if (g._net) return g._net;
  const conns = [];
  for (const c of g.conns.values()) if (c.enabled) conns.push([c.i, c.o]);
  const layers = feedForwardLayers(INPUT_KEYS, OUTPUT_KEYS, conns);
  const evals = [];
  for (const layer of layers) {
    for (const node of layer) {
      const links = [];
      for (const c of g.conns.values()) if (c.enabled && c.o === node) links.push([c.i, c.weight]);
      evals.push({ node, act: ACT[g.nodes.get(node).activation] || ACT.identity, links });
    }
  }
  g._net = evals;
  return evals;
}

// colorsys.hsv_to_rgb + _to_byte
function hsv2rgb(h, s, v) {
  let r, gr, b;
  if (s === 0) { r = gr = b = v; }
  else {
    let i = Math.floor(h * 6); const f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    i = ((i % 6) + 6) % 6;
    if (i === 0) { r = v; gr = t; b = p; }
    else if (i === 1) { r = q; gr = v; b = p; }
    else if (i === 2) { r = p; gr = v; b = t; }
    else if (i === 3) { r = p; gr = q; b = v; }
    else if (i === 4) { r = t; gr = p; b = v; }
    else { r = v; gr = p; b = q; }
  }
  return [Math.floor(r * 255 + 0.5), Math.floor(gr * 255 + 0.5), Math.floor(b * 255 + 0.5)];
}

// Render a genome into an ImageData of size res x res.
function renderGenome(g, res, colorMode) {
  const evals = buildNet(g);
  // optional input/output activation transforms (transform_inputs / transform_outputs)
  const inF = g.inAct ? g.inAct.map((n) => ACT[n] || ACT.identity) : null;
  const outF = g.outAct ? g.outAct.map((n) => ACT[n] || ACT.identity) : null;
  const img = new ImageData(res, res);
  const data = img.data;
  let idx = 0;
  for (let yy = 0; yy < res; yy++) {
    const sy = ((yy * 2) - (res - 1)) / res;
    for (let xx = 0; xx < res; xx++) {
      const sx = ((xx * 2) - (res - 1)) / res;
      const d = Math.hypot(sx, sy) * Math.SQRT2;
      // forward pass (inputs x, y, r, bias=1 — keyed -1..-4)
      let ix = sx, iy = sy, id = d, ib = 1.0;
      if (inF) { ix = inF[0](sx); iy = inF[1](sy); id = inF[2](d); ib = inF[3](1.0); }
      const v = Object.create(null);
      v[-1] = ix; v[-2] = iy; v[-3] = id; v[-4] = ib; v[0] = 0; v[1] = 0; v[2] = 0;
      for (let e = 0; e < evals.length; e++) {
        const ev = evals[e]; let s = 0; const L = ev.links;
        for (let k = 0; k < L.length; k++) s += v[L[k][0]] * L[k][1];
        v[ev.node] = ev.act(s);
      }
      let o0 = v[0], o1 = v[1], o2 = v[2];
      if (outF) { o0 = outF[0](o0); o1 = outF[1](o1); o2 = outF[2](o2); }
      let r, gg, b;
      if (colorMode) {
        const h = pmod(o0 + 1, 1), s = clamp(o1, 0, 1), val = clamp(Math.abs(o2), 0, 1);
        const rgb = hsv2rgb(h, s, val); r = rgb[0]; gg = rgb[1]; b = rgb[2];
      } else {
        const val = Math.floor(clamp(Math.abs(o2), 0, 1) * 255 + 0.5); r = gg = b = val;
      }
      data[idx++] = r; data[idx++] = gg; data[idx++] = b; data[idx++] = 255;
    }
  }
  return img;
}

// ---------- mutation ----------
function selectGenerator() {
  const total = GEN_WEIGHTS.reduce((a, [, w]) => a + w, 0);
  let t = Math.random() * total;
  for (const [name, w] of GEN_WEIGHTS) { t -= w; if (t <= 0) return name; }
  return 'mutate_weights';
}

function mutateWeights(g, strength) {
  const sigma = POW_MIN + (POW_MAX - POW_MIN) * clamp(strength, 0, 1);
  if (sigma <= 0) return;
  for (const c of g.conns.values()) {
    if (!c.enabled) continue;
    if (Math.random() < W_MUT_RATE) c.weight = clamp(c.weight + gauss(0, sigma), WEIGHT_MIN, WEIGHT_MAX);
  }
}

function mutateActivation(g) {
  for (const [k, n] of g.nodes) {
    if (n.input) continue;
    if (Math.random() < ACT_RATE) n.activation = choice(ACT_OPTIONS);
  }
}

function mutateAddNode(g, pop) {
  const conns = [...g.conns.values()];
  if (!conns.length) return;
  const c = choice(conns);
  const i = c.i, o = c.o;
  const nk = pop.newNodeKey();
  let aff;
  if (OUTPUT_KEYS.includes(i)) aff = g.nodes.get(o).affinity;
  else aff = choice([g.nodes.get(i).affinity, g.nodes.get(o).affinity]);
  g.nodes.set(nk, { activation: choice(ACT_OPTIONS), affinity: aff });
  g.addConn(i, nk, randWeight());
  g.addConn(nk, o, randWeight());   // original conn left intact, matching the Python
}

function addConnBetween(g, srcAff, dstAff) {
  const sources = [], dests = [];
  for (const [k, n] of g.nodes) {
    if (n.affinity === srcAff && !OUTPUT_KEYS.includes(k)) sources.push(k);
    if (n.affinity === dstAff && !INPUT_KEYS.includes(k)) dests.push(k);
  }
  if (!sources.length || !dests.length) return false;
  const connKeys = [];
  for (const c of g.conns.values()) if (c.enabled) connKeys.push([c.i, c.o]);
  for (let t = 0; t < 64; t++) {
    const i = choice(sources), o = choice(dests);
    if (i === o) continue;
    if (g.hasConn(i, o)) continue;
    if (createsCycle(connKeys, [i, o])) continue;
    g.addConn(i, o, randWeight());
    return true;
  }
  return false;
}

function mutateAddConnection(g) {
  if (Math.random() < COLOR_LINK) addConnBetween(g, 'color', 'color');
  if (Math.random() < GREY_LINK) addConnBetween(g, 'grey', 'grey');
  if (Math.random() < GREY_TO_COLOR) addConnBetween(g, 'grey', 'color');
}

function mutateActArray(arr) {
  if (!arr) return;
  for (let i = 0; i < arr.length; i++) if (Math.random() < ACT_RATE) arr[i] = choice(ACT_OPTIONS);
}

function mutate(g, pop, strength) {
  const gen = selectGenerator();
  if (gen === 'add_node') mutateAddNode(g, pop);
  else if (gen === 'add_connection') mutateAddConnection(g);
  else if (gen === 'mutate_activation') mutateActivation(g);
  else mutateWeights(g, strength);
  mutateActArray(g.inAct);    // no-ops unless input/output activations are enabled on this genome
  mutateActArray(g.outAct);
  g._net = null;
}

// Picbreeder crossover (picbreeder_reproduction._picbreeder_crossover):
// clone parent2; graft parent1's missing nodes/conns, blend shared activations/weights.
function crossover(p1, p2) {
  const child = cloneGenome(p2);
  for (const [k, n] of p1.nodes) {
    if (!child.nodes.has(k)) child.nodes.set(k, { activation: n.activation, affinity: n.affinity, input: n.input });
    else if (Math.random() < 0.5) child.nodes.get(k).activation = n.activation;
  }
  for (const [k, c] of p1.conns) {
    if (!child.conns.has(k)) child.conns.set(k, { i: c.i, o: c.o, weight: c.weight, enabled: c.enabled });
    else { const rc = child.conns.get(k); rc.weight = (rc.weight + c.weight) / 2; }
  }
  child._net = null;
  return child;
}

// One generation of selective breeding (PicbreederReproduction.reproduce).
function reproduce(selected, pop, popSize, strength) {
  const out = [];
  for (const g of selected) out.push(cloneGenome(g));   // selected carry over unchanged
  const parents = selected;
  while (out.length < popSize) {
    const i = randint(parents.length);
    let child;
    if (parents.length > 1 && Math.random() < CROSSOVER_RATE) {
      let j = i; while (j === i) j = randint(parents.length);
      child = crossover(parents[i], parents[j]);
      mutate(child, pop, strength);
    } else {
      child = cloneGenome(parents[i]);
      mutate(child, pop, strength);
    }
    out.push(child);
  }
  return out;
}

// ---------- serialization (for the "DNA" download) ----------
function genomeToJSON(g) {
  const j = {
    format: 'picbreeder-vlm-cppn/1',
    inputs: ['x', 'y', 'r', 'bias'],
    outputs: ['hue', 'saturation', 'brightness'],
    nodes: [...g.nodes].map(([k, n]) => ({ key: k, activation: n.activation, affinity: n.affinity, input: !!n.input })),
    connections: [...g.conns.values()].map((c) => ({ from: c.i, to: c.o, weight: c.weight, enabled: c.enabled })),
  };
  if (g.inAct) j.inAct = g.inAct.slice();
  if (g.outAct) j.outAct = g.outAct.slice();
  return j;
}

// expose for Node-based tests / tooling
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Genome, Population, newPopulation, makeSeed, cloneGenome, genomeFromJSON,
    genomeToJSON, renderGenome, mutate, reproduce, crossover, buildNet };
}
