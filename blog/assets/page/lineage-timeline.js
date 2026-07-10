(function(){
  /* Generation-scaled lineage timeline. Replaces the boustrophedon snake layout
     for the .lineage strips: the thumbnails sit in a single row, and the gap
     between consecutive publications is scaled PROPORTIONAL to the number of
     generations between them (so a long evolutionary stretch reads as a long
     edge). Each edge carries a connector with the generation count.

     Generation deltas come from the morph's pubKf (assets/lineages/<grp>_ride.json,
     the SAME canon-step coordinates the riding tile uses — see lineage-ride.js),
     so the spatial edge lengths and the morph's travel stay in lockstep. If the
     json is unavailable we fall back to uniform spacing.

     The row is fit to the container width per lineage; if even minimal spacing
     overflows (dense strips on narrow screens) the strip scrolls horizontally.
     lineage-ride.js reads each thumbnail's live offset, so it follows whatever
     this lays out, on resize too. */
  var BASE = 'assets/lineages/';
  var CFG = [ {id:'lineage-vlm', grp:'vlm'}, {id:'lineage-human', grp:'human'} ];
  var MINGAP = 20;     // px floor per edge (room for the connector + label)
  var MAXTHUMB = 104;  // px cap on thumbnail size
  var MINTHUMB = 52;   // px floor before we let the strip scroll instead
  var MINSCALE = 2.2;  // px per generation floor, so proportionality stays visible

  function layout(grid, pub, nRef){
    var figs = Array.prototype.slice.call(grid.querySelectorAll(':scope > figure'));
    var N = figs.length;
    if(N < 2) return;
    // Size thumbnails from the densest strip's count (nRef, shared across all
    // lineages) so every strip's thumbnails come out the same size, regardless
    // of how many stations it has; sparser strips just get roomier gaps.
    var Nsz = Math.max(N, nRef || N);

    var d = [];  // generations per edge
    if(pub && pub.length === N){ for(var i=0;i<N-1;i++) d.push(Math.max(1, pub[i+1]-pub[i])); }
    else { for(var i=0;i<N-1;i++) d.push(1); }
    var sumD = d.reduce(function(a,b){ return a+b; }, 0);

    var W = grid.clientWidth || (grid.parentNode && grid.parentNode.clientWidth) || 1000;
    // leave ~18% of width for the proportional gaps; clamp the thumbnail size
    var thumbW = Math.round(Math.min(MAXTHUMB, Math.max(MINTHUMB, (W - (Nsz-1)*MINGAP) / Nsz * 0.82)));
    var prop = W - N*thumbW - (N-1)*MINGAP;
    var scale = prop > 0 ? prop / sumD : MINSCALE;

    var gaps = d.map(function(g){ return MINGAP + scale*g; });
    var xs = [0];
    for(var i=0;i<N-1;i++) xs.push(xs[i] + thumbW + gaps[i]);
    var total = xs[N-1] + thumbW;
    var offset = total <= W ? (W - total)/2 : 0;

    var hasCap = figs.some(function(f){ return f.querySelector('figcaption'); });
    var capH = hasCap ? 42 : 6;  // room for up to ~2 lines of italic title under a thumb

    grid.classList.add('lineage-tl');
    grid.style.height = (thumbW + capH) + 'px';
    grid.style.overflowX = total <= W ? '' : 'auto';

    figs.forEach(function(f, i){
      f.style.left = (offset + xs[i]) + 'px';
      f.style.top = '0';
      f.style.width = thumbW + 'px';
    });

    // rebuild the edge connectors
    Array.prototype.slice.call(grid.querySelectorAll('.lineage-edge')).forEach(function(e){ e.remove(); });
    for(var i=0;i<N-1;i++){
      var e = document.createElement('div');
      e.className = 'lineage-edge';
      e.style.left = (offset + xs[i] + thumbW) + 'px';
      e.style.width = gaps[i] + 'px';
      e.style.top = (thumbW/2) + 'px';
      var lab = document.createElement('span');
      lab.className = 'lineage-edge-label';
      lab.textContent = d[i];
      lab.title = d[i] + ' generations';
      e.appendChild(lab);
      grid.appendChild(e);
    }
  }

  function debounce(fn, ms){ var t; return function(){ clearTimeout(t); t = setTimeout(fn, ms); }; }

  function setup(cfg, nRef){
    var grid = document.getElementById(cfg.id);
    if(!grid) return;
    function go(pub){ layout(grid, pub, nRef); window.addEventListener('resize', debounce(function(){ layout(grid, pub, nRef); }, 150)); }
    fetch(BASE + cfg.grp + '_ride.json').then(function(r){ return r.json(); })
      .then(function(m){ go(m.pubKf); })
      .catch(function(){ go(null); });
  }

  function init(){
    // densest station count across all present strips -> shared thumbnail size
    var nRef = CFG.reduce(function(m, cfg){
      var g = document.getElementById(cfg.id);
      return g ? Math.max(m, g.querySelectorAll(':scope > figure').length) : m;
    }, 0);
    CFG.forEach(function(cfg){ setup(cfg, nRef); });
  }
  if(document.readyState !== 'loading') init();
  else window.addEventListener('DOMContentLoaded', init);
})();
