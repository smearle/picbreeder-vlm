/* Controller for the standalone archive-viewer page (archive.html). It owns the chrome —
   the back arrow, the Human|AI source toggle, and the AI experiment knobs — and hands the
   embedded sprite gallery to the shared PBVLMArcViewer, exactly as the blog's inline panel
   used to. Every entry point in the post (intro thumbnails, results-table rows, hero grid,
   the #trio leaderboard) now arrives here as a URL, so the whole opening state is params:

     src=human|ai   arc=<arc>   run=<exact run dir>   seed=  model=
     focus=<publication index to park the loupe on>
     meta=growth&grow=<0..1>&growplay=0|1     back=<blog anchor to return to>

   Knob changes rewrite the URL (replaceState) down to just src+arc+back, so a reload or a
   shared link reproduces what you are actually looking at rather than the deep link you
   entered through.

   The chrome keeps the × of the old inline panel rather than a back arrow: the viewer should
   still feel like a pane laid over the post, even though dismissing it is now a navigation. */
(function(){
  var params  = new URLSearchParams(location.search);
  var srcEl   = document.getElementById('arc-src');
  var runsEl  = document.getElementById('arc-runs');
  var orderEl = document.getElementById('arc-order-row');
  var body    = document.getElementById('arc-body');
  var closeBtn = document.getElementById('arc-close');

  var viewer = window.PBVLMArcViewer({
    body: body,
    headEl: orderEl,                          // the ordering toggle + sub-row get their own row
    beforeEl: function(){ return null; },     // insertBefore(x, null) === append: toggle, then sub-row
    seedHost: function(){ return runsEl; },   // Model + Seed join the experiment-knob row
    onArcModel: function(arc, model){ selectArcModel(arc, model); },
    onClose: function(){ closePane(); }        // Esc inside the gallery iframe
  });

  // AI run axes. Each available run varies a SINGLE axis from the default run, so
  // picking a value in one axis resets the others to their default ("automatic
  // toggling"). The shared default value of every axis maps to the 'default' arc.
  var AI_RUNS = [
    { key:'noise',  label:'Noise ε', def:'0', opts:[
      {v:'0',arc:'default'},{v:'.05',arc:'noise_0.05'},{v:'.25',arc:'noise_0.25'},
      {v:'.5',arc:'noise_0.5'},{v:'.75',arc:'noise_0.75'},{v:'1',arc:'noise_1.0'}] },
    { key:'memory', label:'Memory', def:'1', opts:[
      {v:'0',arc:'mem_0'},{v:'1',arc:'default'},{v:'2',arc:'mem_2'},
      {v:'10',arc:'mem_10'},{v:'20',arc:'mem_20'}] },
    { key:'agents', label:'Agents', def:'0', opts:[
      {v:'0',arc:'default'},{v:'10',arc:'agents_10'},{v:'100',arc:'agents_100'},{v:'1000',arc:'agents_1000'}] }
  ];
  // Reverse map: arc-string -> the single axis/value it varies from default. 'default'
  // (the shared origin) and the baselines (random/human) are absent by design — they are
  // resolved separately in boot(), since neither corresponds to one knob position.
  var ARC_TO_SEL = {};
  AI_RUNS.forEach(function(c){ c.opts.forEach(function(o){ if(o.arc !== 'default') ARC_TO_SEL[o.arc] = { key:c.key, val:o.v }; }); });

  var source = 'ai';            // 'human' | 'ai'
  var sel = {};                 // axis key -> selected value (null = highlight nothing, for baselines)
  var currentArc = null;
  function resetSel(){ AI_RUNS.forEach(function(c){ sel[c.key] = c.def; }); }
  function clearSel(){ AI_RUNS.forEach(function(c){ sel[c.key] = null; }); }   // baseline: no knob highlighted
  function arcFromSel(){
    for(var i=0;i<AI_RUNS.length;i++){
      var c = AI_RUNS[i];
      if(sel[c.key] !== c.def){
        for(var j=0;j<c.opts.length;j++) if(c.opts[j].v === sel[c.key]) return c.opts[j].arc;
      }
    }
    return 'default';
  }
  function clearChildren(el){ while(el.firstChild) el.removeChild(el.firstChild); }

  // ---- header controls --------------------------------------------------------
  function buildSource(){
    clearChildren(srcEl);
    [['human','Human'],['ai','AI']].forEach(function(p){
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = p[1]; b.dataset.src = p[0];
      b.addEventListener('click', function(){ setSource(p[0]); });
      srcEl.appendChild(b);
    });
  }
  function buildRuns(){
    clearChildren(runsEl);
    AI_RUNS.forEach(function(c){
      var cat = document.createElement('span'); cat.className = 'arc-runcat';
      var lab = document.createElement('span'); lab.className = 'lab'; lab.textContent = c.label;
      var grp = document.createElement('span'); grp.className = 'grp'; grp.dataset.cat = c.key;
      c.opts.forEach(function(o){
        var b = document.createElement('button');
        b.type = 'button'; b.textContent = o.v; b.dataset.val = o.v;
        b.addEventListener('click', function(){ selectAxis(c.key, o.v); });
        grp.appendChild(b);
      });
      cat.appendChild(lab); cat.appendChild(grp); runsEl.appendChild(cat);
    });
  }
  function paintSource(){
    Array.prototype.forEach.call(srcEl.children, function(b){ b.classList.toggle('on', b.dataset.src === source); });
    runsEl.classList.toggle('hidden', source !== 'ai');
  }
  function paintRuns(){
    Array.prototype.forEach.call(runsEl.querySelectorAll('.grp'), function(grp){
      var key = grp.dataset.cat;
      Array.prototype.forEach.call(grp.children, function(b){ b.classList.toggle('on', sel[key] === b.dataset.val); });
    });
  }
  // Rewrite the address bar to the state actually on screen, dropping the deep-link params
  // (run/focus/meta/grow) that only described how we got here.
  function syncUrl(){
    var q = new URLSearchParams();
    q.set('src', source);
    if(source === 'ai' && currentArc) q.set('arc', currentArc);
    if(params.get('back')) q.set('back', params.get('back'));
    if(params.get('archiveBase')) q.set('archiveBase', params.get('archiveBase'));
    history.replaceState(null, '', location.pathname + '?' + q.toString());
  }

  // ---- arc loading ------------------------------------------------------------
  function loadArc(arc){
    if(arc === currentArc && viewer.frame) return;   // already showing it
    currentArc = arc;
    viewer.mount(arc);
  }
  function setSource(src){
    source = src;
    paintSource(); paintRuns();
    loadArc(src === 'human' ? 'human' : arcFromSel());
    syncUrl();
  }
  function selectAxis(key, val){
    AI_RUNS.forEach(function(c){ sel[c.key] = (c.key === key ? val : c.def); });   // toggle others back to default
    source = 'ai';
    paintSource(); paintRuns();
    loadArc(arcFromSel());
    syncUrl();
  }
  // Snap the run knobs to `arc` and mount it at `model`. Called when the Model sub-toggle
  // picks a model the current arc wasn't run with — the viewer resolves which arc has it,
  // and we realign the knobs to match.
  function selectArcModel(arc, model){
    applyArc(arc);
    paintSource(); paintRuns();
    currentArc = arc;
    viewer.mount(arc, null, { model: model });
    syncUrl();
  }
  // Set source + knob highlighting from an arc string, without mounting.
  function applyArc(arc){
    if(arc === 'human'){ source = 'human'; clearSel(); return; }
    source = 'ai';
    var s = ARC_TO_SEL[arc];
    if(s){ resetSel(); sel[s.key] = s.val; }         // single-axis arc (e.g. mem_10) -> highlight that knob
    else if(arc === 'default' || !arc){ resetSel(); } // shared origin -> all knobs at default
    else clearSel();                                  // baseline arc (random) -> no knob highlighted
  }

  // ---- close ------------------------------------------------------------------
  // Dismissing the "pane" means going back to the post. Prefer a real history step when
  // we came from there — it restores the reader's scroll position on the figure they
  // clicked, so the pane appears to lift off the page it was covering. Otherwise (deep
  // link, new tab) jump to the anchor the entry point told us about.
  function cameFromPost(){
    if(!document.referrer || history.length < 2) return false;
    try {
      var u = new URL(document.referrer);
      return u.origin === location.origin && /(\/|index\.html)$/.test(u.pathname);
    } catch(e){ return false; }
  }
  function closePane(){
    if(cameFromPost()){ history.back(); return; }
    var hash = params.get('back');
    location.href = 'index.html' + (hash ? '#' + hash : '');
  }
  closeBtn.addEventListener('click', closePane);
  document.addEventListener('keydown', function(e){
    if(e.key !== 'Escape') return;
    var tag = (e.target && e.target.tagName) || '';
    if(tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    closePane();
  });

  // ---- boot from the URL ------------------------------------------------------
  (function boot(){
    buildSource(); buildRuns();
    var arc   = params.get('arc');
    var run   = params.get('run');
    var seed  = params.get('seed');
    var model = params.get('model');
    var focus = params.get('focus');
    var meta  = params.get('meta');
    var grow  = params.get('grow');

    if(params.get('src') === 'human' || arc === 'human'){ source = 'human'; clearSel(); arc = 'human'; }
    else applyArc(arc);
    paintSource(); paintRuns();

    var mountOpts = {};
    if(run) mountOpts.run = run;
    if(model) mountOpts.model = model;
    if(focus !== null && focus !== '' && isFinite(+focus)) mountOpts.focus = +focus;
    if(meta){
      mountOpts.meta = meta;
      if(grow !== null && grow !== '' && isFinite(+grow)) mountOpts.grow = +grow;
      if(params.get('growplay') === '0') mountOpts.growPlaying = false;
    }
    currentArc = arc || (source === 'human' ? 'human' : arcFromSel());
    viewer.mount(currentArc, seed, mountOpts);
    syncUrl();
  })();
})();
