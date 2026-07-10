/* hero-morph.js — PB.HeroMorph: live GPU CPPN morph engine for the hero banner.
 *
 * Replaces the pre-baked sprite sheets (~44 MB) with live evaluation of each cell's
 * lineage: the CPPN is compiled to a GLSL fragment shader per parent->child edge,
 * and morphing is just interpolating the weight / activation-blend uniforms — so the
 * banner renders any lineage at any resolution from ~0.37 MB of genome JSON.
 *
 * Same math as the sprites: a direct port of render_lineage_animation.py's
 * build_superset + assign_interpolated_values (lerp weights, blend node activations,
 * fade appearing/disappearing conns), using cppn-gl.js's EXACT hsv2rgb + Y-orientation
 * + output-activation transform (NOT morph-demo.html's cheaper IQ/flip variant), so
 * pixels match renderGenome / the baked PNGs.
 *
 * One shared offscreen WebGL context renders each active cell's current frame on
 * demand; the caller blits it into that cell's 2D canvas. Programs + supersets are
 * built lazily per edge and cached. Depends on cppn.js globals (genomeFromJSON,
 * cloneGenome, feedForwardLayers, INPUT_KEYS, OUTPUT_KEYS).
 *
 *   var eng = PB.HeroMorph.create();          // null-safe; eng.ok === WebGL present
 *   var cell = eng.addCell(fig, data);        // data = {color, pubs, genomes:[…]}
 *   eng.render(cell, frac, res);              // draw frac∈[0,1] into shared canvas
 *   ctx.drawImage(eng.canvas, 0,0,res,res, …) // blit to the cell's 2D canvas
 */
(function (root) {
  var PB = root.PB || (root.PB = {});

  // GLSL activations — mirror cppn.js ACT incl. the ±60 clamp (copied from cppn-gl.js
  // so the morph is pixel-consistent with the static GPU renderer).
  var GLSL_ACT = {
    identity: function (s) { return '(' + s + ')'; },
    sin:      function (s) { return 'sin(clamp(' + s + ',-60.0,60.0))'; },
    cos:      function (s) { return 'cos(clamp(' + s + ',-60.0,60.0))'; },
    sigmoid:  function (s) { return '(2.0/(1.0+exp(-clamp(' + s + ',-60.0,60.0)))-1.0)'; },
    gaussian: function (s) { var z = 'clamp(' + s + ',-60.0,60.0)'; return '(2.0*exp(-(' + z + ')*(' + z + '))-1.0)'; }
  };
  function glslAct(n) { return GLSL_ACT[n] ? n : 'identity'; }

  var HSV2RGB_GLSL = '\n' +
    'vec3 hsv2rgb(float h, float s, float v){\n' +
    '  if(s<=0.0) return vec3(v);\n' +
    '  float i6=floor(h*6.0); float f=h*6.0-i6;\n' +
    '  float p=v*(1.0-s), q=v*(1.0-s*f), t=v*(1.0-s*(1.0-f));\n' +
    '  float i=mod(i6,6.0);\n' +
    '  if(i<0.5) return vec3(v,t,p);\n' +
    '  else if(i<1.5) return vec3(q,v,p);\n' +
    '  else if(i<2.5) return vec3(p,v,t);\n' +
    '  else if(i<3.5) return vec3(p,q,v);\n' +
    '  else if(i<4.5) return vec3(t,p,v);\n' +
    '  return vec3(v,p,q);\n' +
    '}';

  var VERT = 'attribute vec2 aPos; varying vec2 vUV;\n' +
             'void main(){ vUV=aPos*0.5+0.5; gl_Position=vec4(aPos,0.0,1.0); }';

  // ---- superset interpolation (port of build_superset + assign_interpolated_values) ----
  // Operates on cppn.js Genome objects. Returns the superset genome, per-node activation
  // pair [startAct,endAct], and per-conn interpolation stats.
  function buildSuperset(parent, child) {
    var sup = cloneGenome(child);              // start from child structure
    var nodePairs = {};                        // key -> [startAct, endAct]
    child.nodes.forEach(function (n, k) {
      var pn = parent.nodes.get(k);
      nodePairs[k] = [pn ? pn.activation : n.activation, n.activation];
    });
    parent.nodes.forEach(function (n, k) {     // parent-only nodes
      if (!sup.nodes.has(k)) {
        sup.nodes.set(k, { activation: n.activation, affinity: n.affinity, input: n.input });
        nodePairs[k] = [n.activation, n.activation];
      }
    });
    var connStats = {};                        // "i,o" -> {sw,ew,se,ee,i,o}
    var keys = {};
    sup.conns.forEach(function (_c, k) { keys[k] = 1; });
    parent.conns.forEach(function (_c, k) { keys[k] = 1; });
    Object.keys(keys).forEach(function (key) {
      var pc = parent.conns.get(key), cc = child.conns.get(key);
      var sw = pc ? pc.weight : 0, se = pc ? pc.enabled : false;
      // A parent-only conn (absent in child) is DISAPPEARING, not constant: its end
      // state is "gone" (ee=false, ew=0) so it fades out by t=1 instead of persisting.
      var ew = cc ? cc.weight : 0, ee = cc ? cc.enabled : false;
      var ref = cc || pc;
      connStats[key] = { sw: sw, ew: ew, se: se, ee: ee, i: ref.i, o: ref.o };
      if (!sup.conns.has(key)) sup.conns.set(key, { i: ref.i, o: ref.o, weight: sw, enabled: se });
    });
    return { sup: sup, nodePairs: nodePairs, connStats: connStats };
  }

  // Compile a superset (+ start/end node activations + start/end output transforms) into
  // a program. Weights (W), per-node activation blend (BL) and the output-transform blend
  // (OT) are uniforms updated each frame -> one compile per edge.
  function compileEdge(gl, quad, sup, nodePairs, connStats, outPair, color) {
    // Only conns ENABLED at some point in this morph (parent or child) participate:
    // include appearing/disappearing conns (stable order across the scrub) but DROP
    // conns disabled at both endpoints — they carry weight 0 anyway, and feeding their
    // dead edges to feedForwardLayers can create cycles that strand real nodes at 0
    // (cppn.js buildNet likewise walks only enabled conns).
    var conns = [];
    sup.conns.forEach(function (c) { var s = connStats[c.i + ',' + c.o]; if (s && (s.se || s.ee)) conns.push(c); });
    var connIndex = new Map(conns.map(function (c, i) { return [c.i + ',' + c.o, i]; }));
    var allLayers = feedForwardLayers(INPUT_KEYS, OUTPUT_KEYS, conns.map(function (c) { return [c.i, c.o]; }));
    var nodeList = [];
    allLayers.forEach(function (layer) { layer.forEach(function (n) { nodeList.push(n); }); });  // layers are Sets
    var declared = new Set(nodeList);
    for (var oi = 0; oi < OUTPUT_KEYS.length; oi++) { var ok = OUTPUT_KEYS[oi]; if (!declared.has(ok)) { nodeList.push(ok); declared.add(ok); } }
    var blendIndex = new Map(nodeList.map(function (k, i) { return [k, i]; }));

    function varName(k) {
      if (k < 0) return ({ '-1': 'IN_X', '-2': 'IN_Y', '-3': 'IN_D', '-4': 'IN_B' })[String(k)];
      return 'n_' + k;
    }

    var body = '';
    for (var x = 0; x < nodeList.length; x++) {
      var node = nodeList[x];
      var links = conns.filter(function (c) { return c.o === node && (c.i < 0 || declared.has(c.i)); });
      var sum = links.length
        ? links.map(function (c) { return 'W[' + connIndex.get(c.i + ',' + c.o) + ']*' + varName(c.i); }).join('+')
        : '0.0';
      var pair = nodePairs[node] || ['identity', 'identity'];
      var fa = GLSL_ACT[glslAct(pair[0])]('S'), fb = GLSL_ACT[glslAct(pair[1])]('S');
      var bi = blendIndex.get(node);
      body += '  { float S=(' + sum + '); n_' + node + '=mix(' + fa + ',' + fb + ',BL[' + bi + ']); }\n';
    }

    // output transforms (transform_outputs): blend parent vs child per channel by OT.
    var oa = outPair[0], ob = outPair[1];       // each null or [act0,act1,act2]
    var outBlock = '';
    for (var c3 = 0; c3 < 3; c3++) {
      var pa = oa ? oa[c3] : 'identity', pb = ob ? ob[c3] : 'identity';
      if (pa === 'identity' && pb === 'identity') continue;
      outBlock += '  o' + c3 + '=mix(' + GLSL_ACT[glslAct(pa)]('o' + c3) + ',' + GLSL_ACT[glslAct(pb)]('o' + c3) + ',OT);\n';
    }

    var decls = nodeList.map(function (k) { return '  float n_' + k + '=0.0;'; }).join('\n');
    var frag = 'precision highp float;\n' +
      'varying vec2 vUV;\n' +
      'uniform float W[' + Math.max(1, conns.length) + '];\n' +
      'uniform float BL[' + Math.max(1, nodeList.length) + '];\n' +
      'uniform float OT;\n' +
      'uniform float uColor;\n' +
      HSV2RGB_GLSL + '\n' +
      'void main(){\n' +
      '  float rawX=(vUV.x*2.0-1.0);\n' +
      '  float rawY=(1.0-vUV.y*2.0);\n' +
      '  float rawD=length(vec2(rawX,rawY))*1.41421356;\n' +
      '  float IN_X=rawX, IN_Y=rawY, IN_D=rawD, IN_B=1.0;\n' +   // input transforms disabled for VLM lineages
      decls + '\n' +
      body +
      '  float o0=n_0, o1=n_1, o2=n_2;\n' +
      outBlock +
      '  float h=fract(o0+1.0), s=clamp(o1,0.0,1.0), v=clamp(abs(o2),0.0,1.0);\n' +
      '  vec3 col = uColor>0.5 ? hsv2rgb(h,s,v) : vec3(v);\n' +
      '  gl_FragColor=vec4(col,1.0);\n' +
      '}';

    function sh(type, src) {
      var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    var prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, frag));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    return {
      prog: prog, conns: conns, connIndex: connIndex, nodeList: nodeList, blendIndex: blendIndex,
      locW: gl.getUniformLocation(prog, 'W'), locBL: gl.getUniformLocation(prog, 'BL'),
      locOT: gl.getUniformLocation(prog, 'OT'), locColor: gl.getUniformLocation(prog, 'uColor'),
      locPos: gl.getAttribLocation(prog, 'aPos')
    };
  }

  // Per-frame uniform arrays for fraction t within an edge (no recompile).
  function edgeUniforms(P, connStats, nodePairs, t) {
    var W = new Float32Array(P.conns.length);
    for (var i = 0; i < P.conns.length; i++) {
      var c = P.conns[i], s = connStats[c.i + ',' + c.o], w;
      // Mirror cppn.js: a conn contributes only while ENABLED. Disabled in both
      // endpoints -> weight 0 (renderGenome skips it); otherwise lerp / fade.
      if (s.se === s.ee)      w = s.se ? ((1 - t) * s.sw + t * s.ew) : 0;
      else if (s.se && !s.ee) w = s.sw * (1 - t);       // disappearing
      else                    w = s.ew * t;             // appearing
      W[P.connIndex.get(c.i + ',' + c.o)] = w;
    }
    var BL = new Float32Array(P.nodeList.length);
    for (var j = 0; j < P.nodeList.length; j++) {
      var k = P.nodeList[j], pr = nodePairs[k];
      BL[j] = (pr && pr[0] === pr[1]) ? 1.0 : t;         // equal acts -> mix picks either
    }
    return { W: W, BL: BL };
  }

  // ---- shared context -----------------------------------------------------
  var SIZE = 256;                                        // offscreen buffer = max render res
  function makeEngine() {
    var canvas = document.createElement('canvas');
    canvas.width = canvas.height = SIZE;
    var gl = null;
    try {
      gl = canvas.getContext('webgl', { antialias: false, preserveDrawingBuffer: true }) ||
           canvas.getContext('experimental-webgl', { antialias: false, preserveDrawingBuffer: true });
    } catch (e) { gl = null; }
    var eng = { ok: !!gl, canvas: canvas };
    if (!gl) return eng;
    var quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    var lost = false;
    canvas.addEventListener('webglcontextlost', function (e) { e.preventDefault(); lost = true; eng.ok = false; }, false);

    // A cell = its lineage's edge list; supersets + programs built lazily per edge.
    eng.addCell = function (fig, data) {
      var raw = data.genomes;
      var objs = raw.map(function (g) { return genomeFromJSON(g); });
      var outActs = raw.map(function (g) { return g.outAct || null; });
      var edges = [];
      for (var i = 0; i < objs.length - 1; i++) {
        edges.push({ pa: objs[i], ch: objs[i + 1], outPair: [outActs[i], outActs[i + 1]], sup: null, P: null });
      }
      return { fig: fig, color: !!data.color, edges: edges, nEdges: edges.length, pubs: data.pubs || [] };
    };

    function ensureEdge(cell, ei) {
      var e = cell.edges[ei];
      if (!e.sup) e.sup = buildSuperset(e.pa, e.ch);
      if (!e.P) e.P = compileEdge(gl, quad, e.sup.sup, e.sup.nodePairs, e.sup.connStats, e.outPair, cell.color);
      return e;
    }

    // frac in [0,1] across the whole lineage -> render that morphed frame at `res` px
    // into the shared canvas (top-left res×res block). Returns true on success.
    eng.render = function (cell, frac, res) {
      if (!eng.ok || !cell.nEdges) return false;
      var x = frac * cell.nEdges;
      var ei = Math.min(cell.nEdges - 1, Math.max(0, Math.floor(x)));
      var t = x - ei;
      var e;
      try { e = ensureEdge(cell, ei); }
      catch (err) { eng.ok = eng.ok && !lost; if (window.console) console.warn('hero-morph compile failed', cell.fig, err); return false; }
      var P = e.P, u = edgeUniforms(P, e.sup.connStats, e.sup.nodePairs, t);
      res = Math.min(SIZE, res | 0);
      gl.viewport(0, SIZE - res, res, res);             // render into the TOP-left block (GL origin is bottom-left)
      gl.useProgram(P.prog);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.enableVertexAttribArray(P.locPos);
      gl.vertexAttribPointer(P.locPos, 2, gl.FLOAT, false, 0, 0);
      gl.uniform1fv(P.locW, u.W);
      gl.uniform1fv(P.locBL, u.BL);
      gl.uniform1f(P.locOT, t);
      gl.uniform1f(P.locColor, cell.color ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return true;
    };
    // The rendered res×res block sits at the TOP of the SIZE canvas (y in [0,res)).
    eng.blockTop = 0;
    return eng;
  }

  PB.HeroMorph = {
    create: function () {
      if (typeof genomeFromJSON !== 'function' || typeof feedForwardLayers !== 'function') return { ok: false };
      try { return makeEngine(); } catch (e) { return { ok: false }; }
    }
  };
})(typeof window !== 'undefined' ? window : this);
