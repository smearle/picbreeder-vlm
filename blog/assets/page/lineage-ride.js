(function(){
  /* "Riding" lineage morph: instead of a separate looping clip above the strip,
     a single morph tile travels ALONG the static thumbnail strip, gliding from
     one published thumbnail to the next and pausing briefly on each publication.

     Each lineage ships a captionless grayscale frame atlas (<grp>_ride.webp) and
     <grp>_ride.json = {fps, tile, cols, nframes, frameKf, pubKf, ...}, built by
     archive_animations/lineage_ride.py. frameKf[i] is the fractional canon-genome
     coordinate of frame i; pubKf[j] is the canon coordinate of the j-th published
     thumbnail. We bracket each frame's frameKf between two pubKf stations and lerp
     the two matching thumbnails' on-screen centers, so the tile physically rides
     the strip; the dwell frames the renderer holds at each integer pubKf produce
     the pause. Playback ping-pongs (forward then back) so the loop never teleports
     across the strip. */
  var BASE = 'assets/lineages/';
  var CFG = [ {id:'lineage-vlm', grp:'vlm'}, {id:'lineage-human', grp:'human'} ];

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function lerp(a,b,t){ return a + (b-a)*t; }
  function loadImg(src){
    return new Promise(function(res, rej){
      var im = new Image(); im.onload = function(){ res(im); }; im.onerror = rej; im.src = src;
    });
  }

  function setup(cfg){
    var grid = document.getElementById(cfg.id);
    if(!grid) return;
    Promise.all([
      fetch(BASE + cfg.grp + '_ride.json').then(function(r){ return r.json(); }),
      loadImg(BASE + cfg.grp + '_ride.webp')
    ]).then(function(r){ run(grid, r[0], r[1]); })
      .catch(function(){ /* atlas missing -> leave the static strip untouched */ });
  }

  function run(grid, meta, atlas){
    var thumbs = Array.prototype.slice.call(grid.querySelectorAll(':scope > figure'));
    if(thumbs.length < 2 || !meta.pubKf || meta.pubKf.length !== thumbs.length) return;
    grid.style.position = 'relative';

    var cv = document.createElement('canvas');
    cv.className = 'lineage-rider';
    grid.appendChild(cv);
    var ctx = cv.getContext('2d');

    var tile = meta.tile, cols = meta.cols, n = meta.nframes;
    var kf = meta.frameKf, pub = meta.pubKf, fps = meta.fps;

    function station(c){  // {j, frac}: which pubKf segment coord c falls in
      if(c <= pub[0]) return {j:0, frac:0};
      var last = pub.length - 1;
      if(c >= pub[last]) return {j:last-1, frac:1};
      for(var j=0; j<last; j++){
        if(c >= pub[j] && c <= pub[j+1]) return {j:j, frac:(c-pub[j])/(pub[j+1]-pub[j])};
      }
      return {j:last-1, frac:1};
    }
    function center(fig){
      var im = fig.querySelector('img');
      return {x: fig.offsetLeft + im.offsetWidth/2, y: fig.offsetTop + im.offsetHeight/2,
              s: im.offsetWidth};
    }

    var frame = 0, dir = 1, acc = 0, last = 0, visible = true;

    function draw(){
      var f = Math.max(0, Math.min(n-1, frame));
      var st = station(kf[f]);
      var a = center(thumbs[st.j]), b = center(thumbs[st.j+1] ? thumbs[st.j+1] : thumbs[st.j]);
      var sz = a.s;
      var cx = lerp(a.x, b.x, st.frac), cy = lerp(a.y, b.y, st.frac);
      var dpr = window.devicePixelRatio || 1, px = Math.round(sz*dpr);
      if(cv.width !== px){ cv.width = px; cv.height = px; }
      cv.style.width = sz + 'px'; cv.style.height = sz + 'px';
      cv.style.left = (cx - sz/2) + 'px';
      cv.style.top  = (cy - sz/2) + 'px';
      var sx = (f % cols)*tile, sy = Math.floor(f / cols)*tile;
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.drawImage(atlas, sx, sy, tile, tile, 0, 0, cv.width, cv.height);
    }

    function tick(t){
      requestAnimationFrame(tick);
      if(!visible){ last = t; return; }
      if(!last) last = t;
      acc += (t - last)/1000 * fps; last = t;
      var steps = Math.floor(acc);
      if(steps <= 0) return;
      acc -= steps;
      for(var i=0; i<steps; i++){
        frame += dir;
        if(frame >= n-1){ frame = n-1; dir = -1; }       // ping-pong at the tip
        else if(frame <= 0){ frame = 0; dir = 1; }        // ...and back at the root
      }
      draw();
    }

    draw();
    if(reduce) return;  // honor reduced-motion: park the tile on the first station
    if('IntersectionObserver' in window){
      new IntersectionObserver(function(es){ visible = es[0].isIntersecting; }, {threshold:0.04})
        .observe(grid);
    }
    var rzt;
    window.addEventListener('resize', function(){ clearTimeout(rzt); rzt = setTimeout(draw, 180); });
    requestAnimationFrame(tick);
  }

  function init(){ CFG.forEach(setup); }
  if(document.readyState === 'complete') init();
  else window.addEventListener('load', init);
})();
