/* ===========================================================================
 * cppn-gl.js — PB.CppnGL: a shared, GPU-backed CPPN renderer.
 *
 * The phylogeny "morph" viewer (sfdp-morph-demo.html) discovered that the only
 * way to evaluate a CPPN per-pixel at interactive rates in the browser is to
 * compile the network to a GLSL fragment shader and let the GPU run all 256²
 * forward passes in parallel. The CPU port in cppn.js (renderGenome /
 * computeNodeFields) is fine for one-shot renders but melts under live weight
 * scrubbing in the DNA editor (a full per-pixel forward pass every mousemove).
 *
 * This module factors that GPU trick out of the morph demo into one reusable
 * engine for the STATIC, single-genome case (no morph blend):
 *
 *   PB.CppnGL.available()                         -> bool
 *   PB.CppnGL.renderPheno(genome, color, ctx, n)  -> bool   (final HSV image)
 *   PB.CppnGL.renderField(genome, nodeKey, res)   -> Float32Array | null
 *                                                   (one node's intermediate
 *                                                    pattern, top-row-first,
 *                                                    same layout the CPU uses)
 *
 * Both share ONE lazily-created offscreen WebGL context (so a page with many
 * DnaEditors doesn't blow past the browser's ~16-live-context cap) and an LRU
 * cache of compiled programs keyed by a topology+activation signature. During a
 * weight drag the signature is constant, so every frame is a cache hit: we just
 * re-upload the weight uniform and redraw. Structural edits (add/delete a node
 * or link, change an activation) bump the signature and trigger one recompile.
 *
 * Callers should treat every entry point as best-effort: a false / null return
 * (no WebGL, shader compile failure, lost context) means "fall back to the
 * cppn.js CPU path". The output is built to match that path pixel-for-pixel
 * (same coordinate convention, same exact HSV, same per-node contrast inputs).
 *
 * Depends on cppn.js globals: feedForwardLayers, INPUT_KEYS, OUTPUT_KEYS.
 * ======================================================================== */
(function (root) {
  const PB = root.PB || (root.PB = {});

  // GLSL fragments for each activation (mirror cppn.js ACT, incl. the ±60 clamp).
  const GLSL_ACT = {
    identity: (s) => `(${s})`,
    sin:      (s) => `sin(clamp(${s},-60.0,60.0))`,
    cos:      (s) => `cos(clamp(${s},-60.0,60.0))`,
    sigmoid:  (s) => `(2.0/(1.0+exp(-clamp(${s},-60.0,60.0)))-1.0)`,
    gaussian: (s) => { const z = `clamp(${s},-60.0,60.0)`; return `(2.0*exp(-(${z})*(${z}))-1.0)`; },
  };
  const glslAct = (n) => (GLSL_ACT[n] ? n : 'identity');

  // Exact port of cppn.js hsv2rgb (colorsys.hsv_to_rgb) so the live GPU preview
  // matches the CPU render and the saved PNG. (The morph viewer uses the cheaper
  // IQ approximation; here we want parity with renderGenome.)
  const HSV2RGB_GLSL = `
vec3 hsv2rgb(float h, float s, float v){
  if(s<=0.0) return vec3(v);
  float i6=floor(h*6.0); float f=h*6.0-i6;
  float p=v*(1.0-s), q=v*(1.0-s*f), t=v*(1.0-s*(1.0-f));
  float i=mod(i6,6.0);
  if(i<0.5) return vec3(v,t,p);
  else if(i<1.5) return vec3(q,v,p);
  else if(i<2.5) return vec3(p,v,t);
  else if(i<3.5) return vec3(p,q,v);
  else if(i<4.5) return vec3(t,p,v);
  return vec3(v,p,q);
}`;

  // Pack/unpack a scalar in [-FIELD_LIM, FIELD_LIM] into RG (16-bit) so a node's
  // intermediate value survives an 8-bit readback with ~0.001 precision — enough
  // for the contrast-stretch fieldToImageData applies. Activation outputs live in
  // [-1,1]; identity/output nodes can run wider, so the limit is generous.
  const FIELD_LIM = 32.0;
  const PACK_GLSL = `
vec4 packv(float v){
  float nv=clamp((v+${FIELD_LIM.toFixed(1)})/${(2*FIELD_LIM).toFixed(1)},0.0,1.0);
  float u=floor(nv*65535.0+0.5);
  float hi=floor(u/256.0); float lo=u-hi*256.0;
  return vec4(hi/255.0, lo/255.0, 0.0, 1.0);
}`;

  const VERT = `attribute vec2 aPos; varying vec2 vUV;
void main(){ vUV=aPos*0.5+0.5; gl_Position=vec4(aPos,0.0,1.0); }`;

  // ---- lazy shared context ------------------------------------------------
  const SIZE = 256;                 // offscreen drawing buffer (max render res)
  let _gl = null, _canvas = null, _quad = null, _failed = false;
  function ctx() {
    if (_gl) return _gl;
    if (_failed) return null;
    try {
      _canvas = document.createElement('canvas');
      _canvas.width = _canvas.height = SIZE;
      _gl = _canvas.getContext('webgl', { preserveDrawingBuffer: true }) ||
            _canvas.getContext('experimental-webgl', { preserveDrawingBuffer: true });
    } catch (e) { _gl = null; }
    if (!_gl) { _failed = true; return null; }
    _canvas.addEventListener('webglcontextlost', (e) => {        // drop everything; callers fall back to CPU
      e.preventDefault(); _gl = null; _quad = null; _failed = true; cache.length = 0;
    }, false);
    _quad = _gl.createBuffer();
    _gl.bindBuffer(_gl.ARRAY_BUFFER, _quad);
    _gl.bufferData(_gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), _gl.STATIC_DRAW);
    return _gl;
  }

  // ---- network -> GLSL ----------------------------------------------------
  // Build the per-genome GLSL pieces shared by the pheno and field shaders. Only
  // ENABLED connections participate (matching cppn.js buildNet); weights stay in
  // the W[] uniform so a weight-only edit needs no recompile.
  function buildDesc(genome) {
    const conns = [];
    for (const c of genome.conns.values()) if (c.enabled) conns.push(c);
    const connIndex = new Map(conns.map((c, i) => [c.i + ',' + c.o, i]));
    const layers = feedForwardLayers(INPUT_KEYS, OUTPUT_KEYS, conns.map((c) => [c.i, c.o]));
    const nodeList = [];
    for (const layer of layers) for (const n of layer) nodeList.push(n);
    const declared = new Set(nodeList);
    for (const k of OUTPUT_KEYS) if (!declared.has(k)) { nodeList.push(k); declared.add(k); }

    const varName = (k) => (k < 0
      ? ({ '-1': 'IN_X', '-2': 'IN_Y', '-3': 'IN_D', '-4': 'IN_B' })[String(k)]
      : `n_${k}`);
    const actName = (k) => { const n = genome.nodes.get(k); return (n && n.activation) ? n.activation : 'identity'; };

    // inputs (with optional transform_inputs)
    const ia = genome.inAct;
    const inLine = (vn, raw, i) => `  float ${vn}=${ia ? GLSL_ACT[glslAct(ia[i])](raw) : raw};`;
    const inputBlock = [
      `  float rawX=(vUV.x*2.0-1.0);`,
      `  float rawY=(1.0-vUV.y*2.0);`,
      `  float rawD=length(vec2(rawX,rawY))*1.41421356;`,
      inLine('IN_X', 'rawX', 0),
      inLine('IN_Y', 'rawY', 1),
      inLine('IN_D', 'rawD', 2),
      inLine('IN_B', '1.0', 3),
    ].join('\n');

    const declBlock = nodeList.map((k) => `  float n_${k}=0.0;`).join('\n');
    let assignBlock = '';
    for (const node of nodeList) {
      const links = conns.filter((c) => c.o === node && (c.i < 0 || declared.has(c.i)));
      const sum = links.length
        ? links.map((c) => `W[${connIndex.get(c.i + ',' + c.o)}]*${varName(c.i)}`).join('+')
        : '0.0';
      assignBlock += `  { float S=(${sum}); n_${node}=${GLSL_ACT[glslAct(actName(node))]('S')}; }\n`;
    }

    // optional transform_outputs (applied only to the final image, not to fields)
    const oa = genome.outAct;
    const outActBlock = oa
      ? [0, 1, 2].map((i) => `  o${i}=${GLSL_ACT[glslAct(oa[i])]('o' + i)};`).join('\n')
      : '';

    // field probe: select the requested node (or input) into outv by uProbe
    const probes = [
      `  if(abs(uProbe-(-1.0))<0.5) outv=IN_X;`,
      `  if(abs(uProbe-(-2.0))<0.5) outv=IN_Y;`,
      `  if(abs(uProbe-(-3.0))<0.5) outv=IN_D;`,
      `  if(abs(uProbe-(-4.0))<0.5) outv=IN_B;`,
    ];
    for (const k of nodeList) probes.push(`  if(abs(uProbe-(${k}.0))<0.5) outv=n_${k};`);
    const probeBlock = probes.join('\n');

    return { conns, connIndex, nodeList, inputBlock, declBlock, assignBlock, outActBlock, probeBlock };
  }

  function phenoFrag(d) {
    return `precision highp float;
varying vec2 vUV;
uniform float W[${Math.max(1, d.conns.length)}];
uniform float uColor;
${HSV2RGB_GLSL}
void main(){
${d.inputBlock}
${d.declBlock}
${d.assignBlock}
  float o0=n_0, o1=n_1, o2=n_2;
${d.outActBlock}
  float h=fract(o0+1.0), s=clamp(o1,0.0,1.0), v=clamp(abs(o2),0.0,1.0);
  vec3 col = uColor>0.5 ? hsv2rgb(h,s,v) : vec3(v);
  gl_FragColor=vec4(col,1.0);
}`;
  }

  function fieldFrag(d) {
    return `precision highp float;
varying vec2 vUV;
uniform float W[${Math.max(1, d.conns.length)}];
uniform float uProbe;
${PACK_GLSL}
void main(){
${d.inputBlock}
${d.declBlock}
${d.assignBlock}
  float outv=0.0;
${d.probeBlock}
  gl_FragColor=packv(outv);
}`;
  }

  function compile(gl, fragSrc) {
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { const log = gl.getShaderInfoLog(s); gl.deleteShader(s); throw new Error(log); }
      return s;
    };
    const prog = gl.createProgram();
    const vs = mk(gl.VERTEX_SHADER, VERT), fs = mk(gl.FRAGMENT_SHADER, fragSrc);
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    gl.deleteShader(vs); gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { const log = gl.getProgramInfoLog(prog); gl.deleteProgram(prog); throw new Error(log); }
    return {
      prog,
      locPos: gl.getAttribLocation(prog, 'aPos'),
      locW: gl.getUniformLocation(prog, 'W'),
      locColor: gl.getUniformLocation(prog, 'uColor'),
      locProbe: gl.getUniformLocation(prog, 'uProbe'),
    };
  }

  // ---- LRU cache of compiled networks, keyed by topology signature --------
  // Signature covers everything baked into the SHADER SOURCE (topology, enabled
  // set, activations, input/output transforms) but NOT weights (a uniform). So a
  // weight scrub keeps hitting the same entry.
  function topoSig(g) {
    let s = '';
    const nk = [...g.nodes.keys()].sort((a, b) => a - b);
    for (const k of nk) { const n = g.nodes.get(k); s += k + ':' + (n.activation || '') + '|'; }
    const ck = [];
    for (const c of g.conns.values()) if (c.enabled) ck.push(c.i + '>' + c.o);
    ck.sort();
    s += '#' + ck.join(',') + '#' + (g.inAct ? g.inAct.join(',') : '') + '#' + (g.outAct ? g.outAct.join(',') : '');
    return s;
  }

  const cache = [];          // [{sig, desc, pheno, field}], most-recent last
  const LRU_MAX = 8;
  function entryFor(genome) {
    const gl = ctx(); if (!gl) return null;
    const sig = topoSig(genome);
    for (let i = 0; i < cache.length; i++) {
      if (cache[i].sig === sig) { const e = cache.splice(i, 1)[0]; cache.push(e); return e; }
    }
    let desc;
    try { desc = buildDesc(genome); } catch (e) { return null; }
    const e = { sig, desc, pheno: null, field: null };
    cache.push(e);
    while (cache.length > LRU_MAX) {
      const old = cache.shift();
      if (old.pheno) gl.deleteProgram(old.pheno.prog);
      if (old.field) gl.deleteProgram(old.field.prog);
    }
    return e;
  }

  function bind(gl, P) {
    gl.useProgram(P.prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, _quad);
    gl.enableVertexAttribArray(P.locPos);
    gl.vertexAttribPointer(P.locPos, 2, gl.FLOAT, false, 0, 0);
  }
  // Read weights from the CURRENT genome, not desc.conns: the cache is keyed by
  // topology only, so a cache hit can return a desc built from a DIFFERENT genome
  // object with the same shape (e.g. after reset()/setGenome swaps in a fresh
  // genome). desc.connIndex is keyed by "i,o" and baked into the shader, so we
  // reuse it but fill each slot from the live connection of the same endpoints —
  // otherwise reset would keep uploading the stale (edited) weights.
  function uploadW(gl, P, desc, genome) {
    const W = new Float32Array(Math.max(1, desc.conns.length));
    for (const c of genome.conns.values()) {
      if (!c.enabled) continue;
      const idx = desc.connIndex.get(c.i + ',' + c.o);
      if (idx !== undefined) W[idx] = c.weight;
    }
    gl.uniform1fv(P.locW, W);
  }

  // ---- public renders -----------------------------------------------------
  function renderPheno(genome, colorMode, dstCtx, size) {
    const gl = ctx(); if (!gl) return false;
    const e = entryFor(genome); if (!e) return false;
    try {
      if (!e.pheno) e.pheno = compile(gl, phenoFrag(e.desc));
      gl.viewport(0, 0, SIZE, SIZE);
      bind(gl, e.pheno);
      uploadW(gl, e.pheno, e.desc, genome);
      gl.uniform1f(e.pheno.locColor, colorMode ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    } catch (err) { return false; }
    const n = size || SIZE;
    dstCtx.clearRect(0, 0, n, n);
    dstCtx.drawImage(_canvas, 0, 0, SIZE, SIZE, 0, 0, n, n);
    return true;
  }

  function renderField(genome, nodeKey, res) {
    const gl = ctx(); if (!gl) return null;
    if (res > SIZE) return null;
    const e = entryFor(genome); if (!e) return null;
    const buf = new Uint8Array(res * res * 4);
    try {
      if (!e.field) e.field = compile(gl, fieldFrag(e.desc));
      gl.viewport(0, 0, res, res);
      bind(gl, e.field);
      uploadW(gl, e.field, e.desc, genome);
      gl.uniform1f(e.field.locProbe, nodeKey);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.readPixels(0, 0, res, res, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    } catch (err) { return null; }
    // decode RG -> scalar; flip rows (GL reads bottom-first, the CPU field is
    // top-row-first) so it lines up with fieldToImageData / the CPU thumbnails.
    const out = new Float32Array(res * res);
    for (let row = 0; row < res; row++) {
      const src = res - 1 - row;
      for (let col = 0; col < res; col++) {
        const si = (src * res + col) * 4;
        const u = buf[si] * 256 + buf[si + 1];
        out[row * res + col] = (u / 65535) * (2 * FIELD_LIM) - FIELD_LIM;
      }
    }
    return out;
  }

  PB.CppnGL = {
    available: () => !!ctx(),
    renderPheno,
    renderField,
    topoSig,
  };
})(typeof window !== 'undefined' ? window : this);
