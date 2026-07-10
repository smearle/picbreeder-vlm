/* Results-table archive viewer — a thin trigger. Every table row opens the archive
   viewer page (archive.html) at the arc that row represents; the URL carries the arc,
   and the page maps it back onto its experiment knobs. See archive-links.js. */
(function(){
  var rows = document.querySelectorAll('table.results tbody tr[data-arc]');
  if(!rows.length) return;

  function open(row){
    if(!window.PBVLMArchive) return;          // controller not loaded
    window.PBVLMArchive.openRow(row);
  }

  rows.forEach(function(row){
    row.setAttribute('tabindex', '0');
    row.setAttribute('role', 'button');
    row.addEventListener('click', function(){ open(row); });
    row.addEventListener('keydown', function(e){
      if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); open(row); }
    });
  });
})();
