(function(){
  /* Boustrophedon (snake) layout + arrow indicators for .lineage grids.
     The CSS grid auto-fills tracks at the current width; we measure that
     natural column count, then place each item explicitly so even rows
     read left-to-right and odd rows right-to-left, and tag each item with
     the arrow class (arrow-right / arrow-left / arrow-down / arrow-none)
     that the CSS pseudo-element renders. --img-h vertically centers the
     side arrows on the image. */
  var grids = document.querySelectorAll('.lineage');
  if(!grids.length) return;

  function snake(grid){
    // only the thumbnail <figure>s flow in the grid; the ride overlay canvas
    // (lineage-ride.js) is absolutely positioned and must be ignored here.
    var items = Array.from(grid.querySelectorAll(':scope > figure'));
    if(!items.length) return;
    // reset positions + classes so the next measurement reflects natural auto-flow
    items.forEach(function(it){
      it.classList.remove('arrow-right','arrow-left','arrow-down','arrow-none');
      it.style.gridColumn = '';
      it.style.gridRow = '';
    });
    // measure the column count from the first item whose offsetTop drops to a new row
    var firstTop = items[0].offsetTop, cols = items.length;
    for(var i = 1; i < items.length; i++){
      if(items[i].offsetTop > firstTop){ cols = i; break; }
    }
    if(cols < 1) cols = 1;
    // re-place every item in snake order and tag its arrow
    items.forEach(function(it, i){
      var row = Math.floor(i / cols);
      var ltr = (row % 2 === 0);
      var col = ltr ? (i % cols) : (cols - 1 - (i % cols));
      it.style.gridColumn = (col + 1);
      it.style.gridRow    = (row + 1);
      var img = it.querySelector('img');
      if(img && img.offsetHeight) it.style.setProperty('--img-h', img.offsetHeight + 'px');
      if(i === items.length - 1){ it.classList.add('arrow-none'); return; }
      var nextRow = Math.floor((i + 1) / cols);
      if(nextRow !== row)       it.classList.add('arrow-down');
      else                      it.classList.add(ltr ? 'arrow-right' : 'arrow-left');
    });
  }

  function applyAll(){ grids.forEach(snake); }
  // Wait for images so offsetHeight is accurate
  if(document.readyState === 'complete') applyAll();
  else window.addEventListener('load', applyAll);
  var rzt;
  window.addEventListener('resize', function(){ clearTimeout(rzt); rzt = setTimeout(applyAll, 120); });
})();
