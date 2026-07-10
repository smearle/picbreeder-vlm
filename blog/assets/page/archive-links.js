/* Every way into the archive viewer, from the blog post. The viewer itself is its own
   page now (archive.html, like breed/), so each entry point just encodes what it wants
   to see as a URL and navigates there — there is no inline panel left to relocate.
   `back` names the anchor to return to when the reader hits the green back arrow.

   The API shape (window.PBVLMArchive / PBVLMOpenHeroArc) is unchanged, so hero-grid.js
   and trio-live.js keep calling it exactly as before. */
(function(){
  function url(p){
    var q = new URLSearchParams();
    Object.keys(p).forEach(function(k){ if(p[k] !== null && p[k] !== undefined && p[k] !== '') q.set(k, p[k]); });
    var ab = new URLSearchParams(location.search).get('archiveBase');   // dev override; prod uses the HF default
    if(ab) q.set('archiveBase', ab);
    return 'archive.html?' + q.toString();
  }
  function go(p){ location.href = url(p); }

  // Intro thumbnails. A mouse click carries a position: resolve which cell of the
  // thumbnail grid was clicked (intro-thumbs-mutate.js owns the cell<->image map) and
  // park the loupe on that image. Keyboard opens (no position) just open the archive.
  function openThumb(fig, ev){
    var focus = (ev && typeof fig._pbFocusAt === 'function') ? fig._pbFocusAt(ev.clientX, ev.clientY) : null;
    go({ src: fig.getAttribute('data-src') === 'human' ? 'human' : 'ai',
         focus: (typeof focus === 'number') ? focus : null, back: 'archives' });
  }
  // Results-table row -> the arc that row represents. archive.html maps the arc back onto
  // its experiment knobs (or onto no knob at all, for the random/human baselines).
  function openRow(row){ go({ arc: row.getAttribute('data-arc'), back: 'results' }); }

  // Hero-grid double-click: open at the exact source run of the clicked image, loupe parked
  // on it. `prov` is one entry from assets/hero_sprites/provenance.json.
  function openHero(prov){
    if(!prov || !prov.run) return;
    go({ arc: prov.arc, run: prov.run, seed: prov.seed, model: prov.model, focus: prov.index });
  }
  // #trio double-click: open this run's Growth triptych at the frame + play state the
  // inline animation was on.
  function openGrowth(prov){
    if(!prov || !prov.run) return;
    go({ arc: prov.arc, run: prov.run, seed: prov.seed, model: prov.model,
         meta: 'growth', grow: prov.progress, growplay: prov.playing === false ? '0' : null, back: 'trio' });
  }

  window.PBVLMArchive = { openThumb: openThumb, openRow: openRow, openHero: openHero, openGrowth: openGrowth };
  window.PBVLMOpenHeroArc = openHero;   // hero-grid.js calls this on double-click

  // This file loads in <head> (so the hero grid, which sits above the prose, can reach the
  // API), hence the ready guard before touching the thumbnails.
  function wireThumbs(){
    var thumbsWrap = document.getElementById('intro-arc-thumbs');
    if(!thumbsWrap) return;
    thumbsWrap.querySelectorAll('figure[data-src]').forEach(function(fig){
      fig.addEventListener('click', function(e){ openThumb(fig, e); });
      fig.addEventListener('keydown', function(e){
        if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); openThumb(fig); }
      });
    });
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wireThumbs);
  else wireThumbs();
})();
