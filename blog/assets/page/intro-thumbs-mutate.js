/* Intro archive thumbnails. At rest each thumb shows its static representative grid
   (grid_*.png — the k-center max-coverage 6x6 sample). ON HOVER/FOCUS a canvas seeded
   with a PIXEL-IDENTICAL copy of that same grid fades in (so there's no visible jump),
   then mutates ONE cell at a time: every tick a single random cell cross-fades to a new
   real published individual pulled from the HF sprite sheets (the same assets the inline
   gallery reads). On leave the flipping stops but the grid is LEFT in whatever state it
   reached (it does NOT fade back to the static still); re-hovering resumes from there.
   Lazy: nothing is fetched
   until the first hover, and only a few sheets are pulled per archive. If the dataset
   can't be reached (e.g. while still private pre-launch) the static grid stays put. */
(function(){
  var wrap = document.getElementById('intro-arc-thumbs');
  if(!wrap || !window.fetch) return;
  if(!document.createElement('canvas').getContext) return;
  var params      = new URLSearchParams(location.search);
  var HF_BASE     = "https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive/resolve/main";
  var ARCHIVE_BASE= (params.get('archiveBase') || HF_BASE).replace(/\/$/, '');   // dev override; prod = HF
  var ARC_FOR     = { human:'human', ai:'default' };   // featured AI run = the canonical "default" arc
  var GRID = 6, FLIP_EVERY = 650, MAX_SHEETS = 3;

  // index.json is shared by both thumbs -> fetch once.
  var indexPromise = null;
  function getIndex(){
    if(!indexPromise) indexPromise = fetch(ARCHIVE_BASE + '/index.json', {mode:'cors'})
      .then(function(r){ if(!r.ok) throw new Error('index ' + r.status); return r.json(); });
    return indexPromise;
  }
  function resolveArc(arc){           // arc -> {run, base} (a run may live on a different host, e.g. human on gh-pages)
    return getIndex().then(function(idx){
      var m = idx.runs.filter(function(e){ return e.arc === arc && e.has && e.has.sprite; })[0];
      if(!m) throw new Error('no sprite archive for ' + arc);
      return { run: m.run, base: (m.base || ARCHIVE_BASE) };
    });
  }

  function Flipper(fig){
    var src = fig.getAttribute('data-src');
    var arc = ARC_FOR[src];
    var cv  = fig.querySelector('canvas.flip');
    if(!arc || !cv) return;
    var ctx = cv.getContext('2d');
    var still = fig.querySelector('img');   // the static representative (k-center) grid we start from
    var SP = null, base = '', sheets = {}, loaded = [], cells = [];
    var primed = false, hovering = false, ticking = false, seeded = false;

    // Static cell -> publication index map for this grid (assets/thumb_grids/<src>.json),
    // so a click on an un-mutated cell can park the gallery's loupe on that exact image.
    // Mutated cells carry their own index (cell.gidx), so this is just the resting state.
    var focusMap = null;
    fetch('assets/thumb_grids/' + src + '.json').then(function(r){ return r.ok ? r.json() : null; })
      .then(function(j){ focusMap = j; }).catch(function(){});
    // clientX/clientY -> 0-based publication index of the clicked cell's image, or null.
    // The grid box is the (square) <img>; the flip canvas overlays it exactly.
    fig._pbFocusAt = function(clientX, clientY){
      var box = (still || cv).getBoundingClientRect();
      if(!box.width || !box.height) return null;
      var c = Math.floor((clientX - box.left) / box.width  * GRID);
      var r = Math.floor((clientY - box.top ) / box.height * GRID);
      if(r < 0 || r >= GRID || c < 0 || c >= GRID) return null;
      if(seeded){                                   // a mutated cell knows its own individual
        var cell = cells[r * GRID + c];
        if(cell && cell.content && typeof cell.gidx === 'number') return cell.gidx;
      }
      if(focusMap && focusMap.grid && focusMap.grid[r]){
        var v = focusMap.grid[r][c];
        if(typeof v === 'number') return v;
      }
      return null;
    };

    function pack(i){                  // global image index -> {sheet, sx, sy} (mirrors the gallery's packPos)
      var w = i % SP.per_sheet;
      return { sheet: Math.floor(i / SP.per_sheet),
               sx: (w % SP.sheet_cols) * SP.cell,
               sy: Math.floor(w / SP.sheet_cols) * SP.cell };
    }
    function sheetCount(s){ return Math.min(SP.per_sheet, SP.n - s * SP.per_sheet); }
    function randSpriteIdx(){          // global index of a random published individual from the loaded sheets
      var s = loaded[Math.floor(Math.random() * loaded.length)];
      return s * SP.per_sheet + Math.floor(Math.random() * sheetCount(s));   // == its 0-based publication index
    }
    function loadSheet(s, onFirst){
      if(sheets[s]) return;
      var im = new Image();            // NO crossOrigin: we only draw (never read pixels), so no CORS needed
      sheets[s] = im;
      im.onload  = function(){ loaded.push(s); if(loaded.length === 1 && onFirst) onFirst(); };
      im.onerror = function(){ delete sheets[s]; };
      im.src = base + '/' + SP.sheet_tmpl.replace('{s:03d}', String(s).padStart(3, '0'));
    }
    // A cell's content is either null (= still showing its tile of the static grid)
    // or a sprite descriptor {sheet,sx,sy}. The static grids are a clean 6x6 lattice
    // (verified) so a tile is just the matching 1/GRID slice of the still image —
    // which means the seeded canvas is pixel-identical to the static thumbnail.
    function paint(cell, alpha){
      var d = cv.width / GRID;
      ctx.globalAlpha = (alpha == null ? 1 : alpha);
      if(cell.content){
        var sh = sheets[cell.content.sheet];
        if(sh && sh.complete) ctx.drawImage(sh, cell.content.sx, cell.content.sy, SP.cell, SP.cell, cell.c * d, cell.r * d, d, d);
      } else if(still && still.complete && still.naturalWidth){
        var sd = still.naturalWidth / GRID;     // still is square 6x6 -> per-tile source size
        ctx.drawImage(still, cell.c * sd, cell.r * sd, sd, sd, cell.c * d, cell.r * d, d, d);
      }
      ctx.globalAlpha = 1;
    }
    function seed(){                   // canvas := exact copy of the static representative grid
      cv.width = cv.height = GRID * SP.cell;     // native-res grid; CSS scales it down to the 400px box (crisp)
      ctx.imageSmoothingEnabled = true;
      cells = [];
      for(var r = 0; r < GRID; r++) for(var c = 0; c < GRID; c++){
        var cell = { r: r, c: c, content: null }; cells.push(cell); paint(cell);
      }
    }
    function flipOne(){                // swap ONE random cell to a new individual (shown immediately, no fade)
      var idx = Math.floor(Math.random() * cells.length);
      var g   = randSpriteIdx();
      var nu  = { r: cells[idx].r, c: cells[idx].c, content: pack(g), gidx: g };   // gidx = publication index, for click-to-focus
      cells[idx] = nu; paint(nu);
    }
    function tick(){
      setTimeout(function(){
        if(!hovering){ ticking = false; return; }   // pointer left -> stop flipping (grid frozen in place)
        flipOne(); tick();
      }, FLIP_EVERY);
    }
    function show(){                   // reveal the canvas (seamless: it equals the still) + begin flipping
      if(!loaded.length) return;
      if(!seeded){ seed(); seeded = true; }   // seed from the static grid ONCE; later hovers resume from the current state
      if(!ticking){ ticking = true; flipOne(); tick(); }   // swap one cell immediately, then keep flipping (no startup delay)
      cv.classList.add('on');
    }
    function loadMore(){               // pull a few more sheets over time for extra variety
      if(loaded.length >= Math.min(MAX_SHEETS, SP.n_sheets)) return;
      var avail = [];
      for(var s = 0; s < SP.n_sheets; s++) if(!sheets[s]) avail.push(s);
      if(avail.length){ loadSheet(avail[Math.floor(Math.random() * avail.length)]); setTimeout(loadMore, 4000); }
    }
    function prime(){                  // lazy: resolve + fetch metadata + first sheet (on first hover)
      if(primed) return; primed = true;
      resolveArc(arc).then(function(res){
        base = res.base + '/site/' + res.run + '/sprite';
        return fetch(base + '/sprites.json', {mode:'cors'}).then(function(r){
          if(!r.ok) throw new Error('sprites ' + r.status); return r.json();
        });
      }).then(function(sp){
        SP = sp;
        loadSheet(Math.floor(Math.random() * SP.n_sheets), function(){ if(hovering) show(); });
        setTimeout(loadMore, 4000);
      }).catch(function(){ primed = false; /* allow retry on next hover; static grid stays */ });
    }

    function enter(){ hovering = true; prime(); show(); }                 // show() no-ops until a sheet is ready
    function leave(){ hovering = false; }                                 // stop flipping but leave the grid in its current state
    fig.addEventListener('mouseenter', enter);
    fig.addEventListener('mouseleave', leave);
    fig.addEventListener('focus', enter);   // keyboard parity
    fig.addEventListener('blur',  leave);
  }

  Array.prototype.forEach.call(wrap.querySelectorAll('figure[data-src]'), Flipper);
})();
