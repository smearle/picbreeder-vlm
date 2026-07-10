/* Archive-viewer host adapter — the implementation of the embed protocol (iframe +
   ordering/seed/model toggles + height/ready relay) that archive.html's chrome sits on.
   The caller just says where the iframe and the toggles live; everything about talking
   to the embedded gallery (archive-gallery-sprite.html) is here. */
window.PBVLMArcViewer = window.PBVLMArcViewer || function(opts){
  // opts.body     : element that hosts the <iframe>
  // opts.headEl   : element the ordering toggle is inserted into
  // opts.beforeEl : () => node to insert the toggle before (the × close button)
  var arcFrame = null, arcToggle = null, arcSub = null, arcSeed = null, arcModel = null;
  var nextFrame = null, swapTimer = null;   // hot-swap: a new arc loads hidden here, then replaces arcFrame on ready
  var curArc = null, curSeed = null, curModel = null;   // remembered so the seed/model toggles can re-mount the same arc
  var pendingFocus = null;   // publication index to park the loupe on once the next mount reports ready
  // Last view mode the user picked (meta + sub), persisted across arc/seed swaps so switching
  // experiment condition stays in the same ordering. Passed to the iframe via ?meta=&sub=.
  var savedMeta = null, savedSub = null;
  var growthLbDocY = null, growthPlaying = null;   // latest growth leaderboard doc-Y + play state (for the #trio hand-off)
  function clearChildren(el){ while(el.firstChild) el.removeChild(el.firstChild); }
  function clearToggle(){
    if(arcToggle){ arcToggle.remove(); arcToggle = null; }
    if(arcSub){ arcSub.remove(); arcSub = null; }
    if(arcSeed){ arcSeed.remove(); arcSeed = null; }
    if(arcModel){ arcModel.remove(); arcModel = null; }
  }
  // opts2 (optional): { run: exact run-dir to load (overrides arc resolution),
  //                     model: the model axis value, focus: publication index to park the loupe on }
  function mount(arc, seed, opts2){
    opts2 = opts2 || {};
    curArc = arc; curSeed = (seed == null ? null : seed); curModel = opts2.model || null;
    pendingFocus = (typeof opts2.focus === 'number') ? opts2.focus : null;
    growthLbDocY = null; growthPlaying = null;
    // An explicit view mode (e.g. the trio's Growth hand-off) both drives THIS mount and
    // persists across later arc/seed/model swaps, like a user-picked mode would.
    if(opts2.meta){ savedMeta = opts2.meta; savedSub = null; }
    var growP = (opts2.meta === 'growth' && opts2.grow != null) ? opts2.grow : null;   // 0..1 seek only meaningful on this first mount
    var ab = new URLSearchParams(location.search).get('archiveBase');  // dev override; prod uses HF default
    var f = document.createElement('iframe');
    f.src = 'archive-gallery-sprite.html?embed=1&arc=' + encodeURIComponent(arc) +
            (seed != null ? '&seed=' + encodeURIComponent(seed) : '') +
            (opts2.model ? '&model=' + encodeURIComponent(opts2.model) : '') +
            (opts2.run ? '&run=' + encodeURIComponent(opts2.run) : '') +
            (savedMeta ? '&meta=' + encodeURIComponent(savedMeta) : '') +
            (savedMeta && savedSub ? '&sub=' + encodeURIComponent(savedSub) : '') +
            (growP != null ? '&grow=' + encodeURIComponent(growP) : '') +
            (opts2.meta === 'growth' && opts2.growPlaying === false ? '&growplay=0' : '') +
            (ab ? '&archiveBase=' + encodeURIComponent(ab) : '');
    f.allow = 'fullscreen; pointer-lock';   // fullscreen + pointer lock both need crossing the iframe boundary
    f.setAttribute('allowfullscreen', '');

    // HOT-SWAP: if an archive is already on screen (switching arc/seed/model), keep it
    // visible and load the new one hidden on top of it. We only tear the old one down
    // once the new frame reports pbvlmReady (see the message handler) — so the grid never
    // flashes to white on a switch. The old toggles/grid stay live until that swap.
    if(arcFrame){
      if(nextFrame){ nextFrame.remove(); nextFrame = null; }   // supersede an in-flight switch
      if(swapTimer){ clearTimeout(swapTimer); swapTimer = null; }
      opts.body.style.position = 'relative';
      f.style.position = 'absolute'; f.style.top = '0'; f.style.left = '0'; f.style.width = '100%';
      f.style.height = arcFrame.style.height || '80vh';        // match the current frame while loading
      f.style.visibility = 'hidden';                           // rendered + measurable, just not painted
      nextFrame = f; opts.body.appendChild(f);
      // Safety net: if the new frame errors and never posts pbvlmReady, reveal it anyway
      // (showing its error status) rather than leaving the stale archive up forever.
      swapTimer = setTimeout(promoteFrame, 8000);
      return f;
    }

    // First open — nothing to preserve, so mount straight into the (empty) body.
    clearToggle(); clearChildren(opts.body);
    f.style.height = '80vh';                 // provisional; the gallery posts its true height back
    arcFrame = f; opts.body.appendChild(f);
    return f;
  }
  // Promote the hidden nextFrame to be the visible arcFrame, discarding the old one.
  function promoteFrame(){
    if(!nextFrame) return;
    if(swapTimer){ clearTimeout(swapTimer); swapTimer = null; }
    var incoming = nextFrame; nextFrame = null;
    if(arcFrame) arcFrame.remove();
    incoming.style.position = ''; incoming.style.top = ''; incoming.style.left = '';
    incoming.style.width = ''; incoming.style.visibility = '';
    arcFrame = incoming;
  }
  // Meta-category toggle built from whatever the gallery reports; buttons message the
  // iframe to morph, the iframe owns the actual reorder/animation. Meta-categories that
  // report sub-options get a nested sub-row that appears only while that meta is active:
  //   Grid -> the four grid sorts (each a real ordering); Phylogeny -> radial/twopi/sfdp.
  // A grid sub IS a full ordering (subIsOrder), so clicking one also becomes the meta's
  // "representative" — re-entering Grid later returns to the last sort you picked.
  function buildToggle(d){
    clearToggle();
    var orderings = d.allOrderings || d.orderings;     // full canonical meta set (for greying)
    var available = d.orderings || orderings;          // metas this run actually has
    var labels = d.labels || {}, subOrderings = d.subOrderings || {}, subLabels = d.subLabels || {};
    var subAvail = d.subAvailable || {}, subIsOrder = d.subIsOrder || {};
    var metaRep = d.metaRep || {}, subCur = d.subCurrent || {}, current = d.current;
    var metaReasons = d.metaReasons || {}, subReasons = d.subReasons || {};   // "why greyed" tooltips
    var has = {}; available.forEach(function(k){ has[k] = true; });
    arcToggle = document.createElement('span');
    arcToggle.className = 'arc-order';
    function showSub(meta){
      if(arcSub){ arcSub.remove(); arcSub = null; }
      var subs = subOrderings[meta];
      if(!subs || !subs.length) return;
      var asOrder = !!subIsOrder[meta];
      arcSub = document.createElement('span');
      arcSub.className = 'arc-suborder';
      subs.forEach(function(s){
        var sb = document.createElement('button');
        sb.type = 'button'; sb.setAttribute('data-sub', s);
        sb.textContent = subLabels[s] || s;
        if(Object.prototype.hasOwnProperty.call(subAvail, s) && !subAvail[s]){ sb.disabled = true; if(subReasons[s]) sb.title = subReasons[s]; arcSub.appendChild(sb); return; }   // run lacks this sort -> greyed (hover = why)
        if(s === subCur[meta]) sb.classList.add('on');
        sb.addEventListener('click', function(){
          if(!arcFrame) return;
          arcFrame.contentWindow.postMessage({pbvlmSubOrder:s}, '*');
          subCur[meta] = s;
          savedMeta = meta; savedSub = s;   // persist this exact view mode across arc/seed swaps
          if(asOrder) metaRep[meta] = s;   // remember the chosen grid sort for next time Grid is clicked
          Array.prototype.forEach.call(arcSub.children, function(x){ x.classList.toggle('on', x === sb); });
        });
        arcSub.appendChild(sb);
      });
      opts.headEl.insertBefore(arcSub, opts.beforeEl());   // just right of the main toggle
    }
    orderings.forEach(function(k){
      var b = document.createElement('button');
      b.type = 'button'; b.setAttribute('data-key', k);
      b.textContent = labels[k] || (k.charAt(0).toUpperCase() + k.slice(1));
      if(!has[k]){ b.disabled = true; if(metaReasons[k]) b.title = metaReasons[k]; arcToggle.appendChild(b); return; }   // run lacks this category -> greyed, inert (hover = why)
      if(k === current) b.classList.add('on');
      b.addEventListener('click', function(){
        if(!arcFrame) return;
        arcFrame.contentWindow.postMessage({pbvlmOrder: metaRep[k] || k}, '*');   // load the meta's representative ordering
        savedMeta = k; savedSub = subCur[k] || null;   // persist this view mode across arc/seed swaps
        Array.prototype.forEach.call(arcToggle.children, function(x){ x.classList.toggle('on', x === b); });
        showSub(k);
      });
      arcToggle.appendChild(b);
    });
    opts.headEl.insertBefore(arcToggle, opts.beforeEl());   // sits just left of the × close
    showSub(current);
  }
  // Seed sub-toggle: the same arc was run with several replicate seeds (s3/s4/s5);
  // each carries its own sprite set, so switching seed re-mounts the iframe (not a morph).
  // Sits left of the ordering toggle; hidden when an arc has only one seed (e.g. human).
  function buildSeedToggle(seeds, current){
    if(arcSeed){ arcSeed.remove(); arcSeed = null; }
    if(!Array.isArray(seeds) || seeds.length < 2) return;
    arcSeed = document.createElement('span');
    arcSeed.className = 'arc-runcat arc-seed';            // match the left-side knob style (label + bare-value group)
    var lab = document.createElement('span'); lab.className = 'lab'; lab.textContent = 'Seed';
    var grp = document.createElement('span'); grp.className = 'grp';
    seeds.forEach(function(s){
      var b = document.createElement('button');
      b.type = 'button'; b.setAttribute('data-seed', s);
      b.textContent = s;                                 // bare number, like the Memory/Noise/Agents buttons
      if(String(s) === String(current)) b.classList.add('on');
      b.addEventListener('click', function(){
        if(String(s) === String(curSeed)) return;   // already showing this seed
        mount(curArc, s, {model: curModel});         // re-mount at this seed, keeping the chosen model
      });
      grp.appendChild(b);
    });
    arcSeed.appendChild(lab); arcSeed.appendChild(grp);
    // If the host gives a seedHost (e.g. the intro viewer's knob row), drop the seed
    // group in there — after Agents — so it reads as one more knob. Otherwise it sits
    // leftmost of the header toggle group, just before the ordering toggle.
    var host = (typeof opts.seedHost === 'function') ? opts.seedHost() : null;
    if(host) host.appendChild(arcSeed);
    else opts.headEl.insertBefore(arcSeed, arcToggle || opts.beforeEl());
  }
  // Model sub-toggle: an arc may have been run with more than one VLM (the canonical
  // gemini-2.5-pro plus off-table gemini-3 / flash / qwen replicas). Model is orthogonal
  // to seed, so switching it re-mounts (a new sprite set), keeping the current seed when
  // that seed exists for the new model. Sits just left of the Seed knob (after Agents) so
  // the row reads Noise|Memory|Agents|Model|Seed.
  //   We always show EVERY model present anywhere in the archive (d.allModels), not just the
  //   ones the current arc was run with (d.models) — so e.g. Memory=10 still lists Flash even
  //   though only Pro/Qwen were run at that memory. Clicking a model the current arc lacks
  //   snaps the run knobs to an arc that DOES have it (via opts.onArcModel), rather than
  //   silently falling back to the arc's default model.
  var MODEL_LABEL = { 'gemini-2.5-pro': '2.5 Pro', 'gemini-3-pro-preview': '3 Pro',
    'gemini-2.5-flash': '2.5 Flash', 'gemini-2.5-flash-lite': '2.5 Flash Lite', 'gemini-random': 'Random Gemini',
    'qwen3-vl-30b-fp8': 'Qwen3-VL 30B', 'qwen3-vl-8b': 'Qwen3-VL 8B' };
  function buildModelToggle(d){
    if(arcModel){ arcModel.remove(); arcModel = null; }
    var arcModels = Array.isArray(d.models) ? d.models : [];      // models the CURRENT arc has
    var all = (Array.isArray(d.allModels) && d.allModels.length) ? d.allModels : arcModels;  // all models anywhere
    var modelArcs = d.modelArcs || {};
    var current = d.currentModel;
    if(all.length < 2) return;                                   // only one model in the whole archive -> no toggle
    arcModel = document.createElement('span');
    arcModel.className = 'arc-runcat arc-model';
    var lab = document.createElement('span'); lab.className = 'lab'; lab.textContent = 'Model';
    var grp = document.createElement('span'); grp.className = 'grp';
    all.forEach(function(m){
      var inArc = arcModels.indexOf(m) >= 0;
      var b = document.createElement('button');
      b.type = 'button'; b.setAttribute('data-model', m);
      b.textContent = MODEL_LABEL[m] || m;
      if(String(m) === String(current)) b.classList.add('on');
      if(!inArc){ b.classList.add('off-arc'); b.title = 'Not run with the current settings—selecting resets them.'; }
      b.addEventListener('click', function(){
        if(String(m) === String(curModel)) return;   // already on this model
        if(inArc){ mount(curArc, curSeed, {model: m}); return; }   // in-arc swap: re-mount, keep seed if available
        // Cross-arc model: snap to an arc that has it (prefer 'default', the richest arc),
        // and let the host update its run knobs to match. Fall back to a bare re-mount.
        var arcs = modelArcs[m] || [];
        var target = (arcs.indexOf('default') >= 0) ? 'default' : arcs[0];
        if(!target) return;
        if(typeof opts.onArcModel === 'function') opts.onArcModel(target, m);
        else mount(target, null, {model: m});
      });
      grp.appendChild(b);
    });
    arcModel.appendChild(lab); arcModel.appendChild(grp);
    // Place it just before the Seed knob in the host's knob row, else left of the toggle group.
    var host = (typeof opts.seedHost === 'function') ? opts.seedHost() : null;
    if(host) host.insertBefore(arcModel, arcSeed || null);   // before Seed (appended after), else last
    else opts.headEl.insertBefore(arcModel, arcSeed || arcToggle || opts.beforeEl());
  }
  window.addEventListener('message', function(e){
    if(!e.data) return;
    // Accept messages from the on-screen frame OR the hidden frame loading behind it.
    var isNext = nextFrame && e.source === nextFrame.contentWindow;
    var isCur  = arcFrame && e.source === arcFrame.contentWindow;
    if(!isNext && !isCur) return;
    if(e.data.pbvlmClose){                                  // Esc inside the iframe -> close the host pane
      if(typeof opts.onClose === 'function') opts.onClose();
      return;
    }
    // The hidden frame is still loading — its only relevant signals are height (to size the
    // off-screen frame) and ready (to trigger the swap). Ignore its growth/focus chatter.
    if(isNext){
      if(typeof e.data.pbvlmHeight === 'number'){ nextFrame.style.height = e.data.pbvlmHeight + 'px'; return; }
      if(e.data.pbvlmReady && Array.isArray(e.data.orderings)){
        promoteFrame();                                    // swap the loaded frame in for the old one, then fall
      } else return;                                       // through to build its toggles below (arcFrame is now it)
    }
    if(e.data.pbvlmGrowth && typeof e.data.pbvlmGrowth === 'object'){   // growth leaderboard position + play state
      if(typeof e.data.pbvlmGrowth.lbY === 'number') growthLbDocY = e.data.pbvlmGrowth.lbY;
      if(typeof e.data.pbvlmGrowth.playing === 'boolean') growthPlaying = e.data.pbvlmGrowth.playing;
      return;
    }
    if(typeof e.data.pbvlmHeight === 'number'){
      // Measure BEFORE the resize: only if the user had scrolled down into the
      // viewer (its top above the viewport top) can a shrink strand them below it.
      var preTop = arcFrame.getBoundingClientRect().top;
      arcFrame.style.height = e.data.pbvlmHeight + 'px';   // exact content height; gallery body is overflow:hidden so no inner scrollbar
      // Switching modes (e.g. a tall grid -> a short Phylogeny) shrinks the iframe.
      // If we were scrolled into it, the page can be left showing the text that
      // follows the viewer. Clamp the page scroll back onto the viewer:
      //   (B, secondary) pull the viewer's bottom down to the viewport bottom;
      //   (A, primary)   but never push the viewer's top below the viewport top.
      // A is applied last so it wins for short content (whole viewer flush at top).
      if(preTop < 0){
        var vh = window.innerHeight || document.documentElement.clientHeight;
        var r = arcFrame.getBoundingClientRect();
        if(r.bottom < vh){                          // viewer ends above the fold -> stranded
          window.scrollBy(0, r.bottom - vh);        // (B) scroll up
          r = arcFrame.getBoundingClientRect();
          if(r.top > 0) window.scrollBy(0, r.top);  // (A) snap top flush to viewport top
        }
      }
    } else if(typeof e.data.pbvlmFocusY === 'number'){
      // the iframe parked the loupe on a focused image (hero double-click) and reports
      // its y within the iframe document; scroll the page so it sits mid-viewport.
      var fr = arcFrame.getBoundingClientRect();
      var target = fr.top + window.scrollY + e.data.pbvlmFocusY - window.innerHeight / 2;
      window.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
    } else if(e.data.pbvlmReady && Array.isArray(e.data.orderings)){
      // allOrderings = full canonical meta set; orderings = the subset this run actually has.
      // Metas not in `orderings` (e.g. Phylogeny/Growth for a run with no scatter layout) render
      // greyed-out and inert; absent grid sorts (e.g. Top Rated on the human archive) grey inside the sub-row.
      buildToggle(e.data);
      buildSeedToggle(e.data.seeds, e.data.currentSeed);   // seed sub-toggle (replicate seeds for this arc)
      buildModelToggle(e.data);                             // model sub-toggle (sits just left of Seed); shows ALL models
      if(e.data.currentSeed != null) curSeed = e.data.currentSeed;     // sync after a default (no-seed) mount resolves
      if(e.data.currentModel != null) curModel = e.data.currentModel;  // ditto for model
      if(pendingFocus !== null){                            // hero jump: park the loupe on the target image
        arcFrame.contentWindow.postMessage({ pbvlmFocus: pendingFocus }, '*');
        pendingFocus = null;
      }
    }
  });
  // Forward the "i" info-layout toggle to the embedded gallery. Its own key handler only
  // sees keydowns while the iframe holds focus; a reader scanning the blog has focus on the
  // parent page, so relay the key here whenever a pane is open and actually on screen. The
  // gallery ignores it unless one of the grid orderings is showing.
  document.addEventListener('keydown', function(e){
    if(e.key !== 'i' && e.key !== 'I') return;
    if(!arcFrame || e.metaKey || e.ctrlKey || e.altKey) return;
    var tag = (e.target && e.target.tagName) || '';
    if(tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) return;
    var r = arcFrame.getBoundingClientRect();
    var vh = window.innerHeight || document.documentElement.clientHeight;
    if(r.bottom < 0 || r.top > vh) return;                  // viewer scrolled out of view -> ignore
    arcFrame.contentWindow.postMessage({pbvlmInfo:true}, '*');
  });
  function destroy(){
    if(swapTimer){ clearTimeout(swapTimer); swapTimer = null; }
    nextFrame = null;                                       // clearChildren drops both frames from the DOM
    clearToggle(); clearChildren(opts.body); opts.body.style.position = '';
    arcFrame = null; growthLbDocY = null; growthPlaying = null;
  }
  return {
    mount: mount, destroy: destroy, clearToggle: clearToggle,
    get frame(){ return arcFrame; },
    // viewport Y of the embedded growth leaderboard (iframe body never scrolls in embed mode,
    // so its parent-viewport top is the iframe's top + the leaderboard's doc-Y). null unless in growth.
    get growthLbViewportY(){ return (arcFrame && growthLbDocY != null) ? arcFrame.getBoundingClientRect().top + growthLbDocY : null; },
    get growthPlaying(){ return growthPlaying; }
  };
};
