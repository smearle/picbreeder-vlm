/* Picbreeder-VLM community client glue.
 *
 * Talks to the publish API (a HF Space) so the static site can actually publish,
 * list, and (admin) delete user-bred genomes. Everything else — rendering,
 * branching, the DNA editor — stays exactly as it is; community items flow
 * through the same `run::id` plumbing as archive items, with run === 'community'.
 *
 * Admin: pass ?admin=<KEY> once (it's stashed in localStorage and stripped is
 * left to the page) or call PBC.setAdminKey(). Delete buttons appear wherever a
 * key is present; the key is sent as the X-Admin-Key header.
 */
window.PBC = (function () {
  const qp = new URLSearchParams(location.search);
  const API_DEFAULT = 'https://picbreeder-vlm-picbreeder-community-api.hf.space';
  // The dev override may only point at a LOCAL API (same-origin/localhost). This is a
  // POST target (publish/rate), so a crafted ?communityApi= link pointing elsewhere
  // would ship a visitor's uploads and ratings to an attacker — reject any other host.
  function _safeApi(o) {
    if (!o) return API_DEFAULT;
    const withScheme = /^(https?:)?\/\//.test(o) ? o : 'http://' + o;
    try {
      const h = new URL(withScheme, location.href).hostname;
      if (h === location.hostname || h === 'localhost' || h === '127.0.0.1' || h === '[::1]' || h === '::1') return withScheme;
    } catch (e) {}
    return API_DEFAULT;
  }
  const API = _safeApi(qp.get('communityApi')).replace(/\/$/, '');
  const RUN = 'community';
  const ADMIN_LS = 'pbc_admin_key';

  // capture an admin key handed in via ?admin= so the link is shareable to yourself
  if (qp.get('admin')) { try { localStorage.setItem(ADMIN_LS, qp.get('admin')); } catch (e) {} }

  function getAdminKey() { try { return localStorage.getItem(ADMIN_LS) || ''; } catch (e) { return ''; } }
  function setAdminKey(k) { try { k ? localStorage.setItem(ADMIN_LS, k) : localStorage.removeItem(ADMIN_LS); } catch (e) {} }
  function isAdmin() { return !!getAdminKey(); }

  async function _json(url, opts) {
    const r = await fetch(url, opts);
    let body = null;
    try { body = await r.json(); } catch (e) {}
    if (!r.ok) throw new Error((body && body.detail) || ('HTTP ' + r.status));
    return body;
  }

  // POST a bred genome. payload: {genome, title, author, color, parent, gen, png}
  function publish(payload) {
    return _json(API + '/publish', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  // slim manifest (newest last)
  function items() { return _json(API + '/items'); }

  // {id: genome} for the whole community run — one fetch renders every thumbnail
  function genomes() { return _json(API + '/genomes'); }

  function remove(id) {
    return _json(API + '/item/' + encodeURIComponent(id), {
      method: 'DELETE',
      headers: { 'X-Admin-Key': getAdminKey() },
    });
  }

  // ---- human ratings -------------------------------------------------------
  // Opaque per-browser voter id (anonymous identity model). Swap this for an HF
  // OAuth username later and the rest of the rating system is unchanged.
  function voterId() {
    try {
      let v = localStorage.getItem('pbc_voter');
      if (!v) {
        v = (window.crypto && crypto.randomUUID ? crypto.randomUUID().replace(/-/g, '')
          : ('v' + Date.now().toString(36) + Math.random().toString(36).slice(2))).slice(0, 32);
        localStorage.setItem('pbc_voter', v);
      }
      return v;
    } catch (e) { return 'anonymous0000'; }
  }

  function rate(key, value) {
    return _json(API + '/rate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: key, value: value, voter: voterId() }) });
  }
  function getRating(key) {
    return _json(API + '/rating?key=' + encodeURIComponent(key) + '&voter=' + encodeURIComponent(voterId()));
  }
  function ratingsBatch(keys) {
    return _json(API + '/ratings', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keys: keys, voter: voterId() }) });
  }

  // star strip matching the site sprite; `value` (0..5) sets the filled width
  function starBarHTML(value, cls) {
    const pct = value == null ? 0 : Math.max(0, Math.min(5, value)) / 5 * 100;
    return '<span class="inline-rating ' + (cls || '') + '"><ul class="star-rating small-star">' +
      '<li class="current-rating" style="width:' + pct + '%"></li></ul></span>';
  }

  let _cssDone = false;
  function _injectCSS() {
    if (_cssDone) return; _cssDone = true;
    const css =
      '.pbc-rrow{margin:2px 0;font-size:12px;color:#555;display:flex;align-items:center;gap:6px}' +
      '.pbc-rrow .pbc-rlabel{min-width:80px;text-align:right;color:#888}' +
      '.pbc-rrow .pbc-rmeta{color:#777}' +
      '.pbc-human .pbc-rlabel{color:#3a6b1e;font-weight:bold}' +
      '.ratingbig .inline-rating.pbc-live{cursor:pointer}' +
      '.pbc-ratehint{font-size:11px;color:#999;margin:2px 0}';
    const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s);
  }

  function _valueFromEvent(ul, e) {
    const r = ul.getBoundingClientRect();
    let frac = (e.clientX - r.left) / r.width;
    frac = Math.max(0, Math.min(1, frac));
    return Math.max(0.5, Math.min(5, Math.ceil(frac * 10) / 2)); // round up to the half you're over
  }

  // Turn the existing under-image star bar into a rate-by-hover control, and add
  // a human-aggregate row ABOVE it and a "your rating" row BELOW it.
  //   bigEl: the .ratingbig element holding .inline-rating > ul.star-rating > .current-rating
  //   key:   global item key; vlmRating/vlmN: the model's resting rating (may be null)
  function mountRating(bigEl, key, vlmRating, vlmN) {
    if (!bigEl) return;
    _injectCSS();
    const bar = bigEl.querySelector('.inline-rating');
    const ul = bigEl.querySelector('ul.star-rating');
    const li = bigEl.querySelector('.current-rating');
    if (!bar || !ul || !li) return;
    bar.classList.add('pbc-live');
    const restPct = vlmRating == null ? 0 : Math.max(0, Math.min(5, vlmRating)) / 5 * 100;

    // rows: human (above bigEl) and you (after bigEl, or after #ratingtxt if present)
    const human = document.createElement('div');
    human.className = 'pbc-rrow pbc-human'; human.style.display = 'none';
    human.innerHTML = '<span class="pbc-rlabel">Human</span>' + starBarHTML(0) + '<span class="pbc-rmeta"></span>';
    bigEl.parentNode.insertBefore(human, bigEl);

    const you = document.createElement('div');
    you.className = 'pbc-rrow pbc-you'; you.style.display = 'none';
    you.innerHTML = '<span class="pbc-rlabel">Your rating</span>' + starBarHTML(0) + '<span class="pbc-rmeta"></span>';
    const ratingtxt = document.getElementById('ratingtxt');
    const anchor = (ratingtxt && bigEl.parentNode.contains(ratingtxt)) ? ratingtxt : bigEl;
    anchor.parentNode.insertBefore(you, anchor.nextSibling);

    const hint = document.createElement('div');
    hint.className = 'pbc-ratehint'; hint.textContent = 'Hover the stars and click to rate this image.';
    you.parentNode.insertBefore(hint, you);

    function setHumanRow(mean, n) {
      if (!n) { human.style.display = 'none'; return; }
      human.style.display = 'flex';
      human.querySelector('.current-rating').style.width = (mean / 5 * 100) + '%';
      human.querySelector('.pbc-rmeta').textContent = mean.toFixed(2) + ' / 5 (' + n + (n === 1 ? ' vote)' : ' votes)');
    }
    function setYouRow(value) {
      if (value == null) { you.style.display = 'none'; return; }
      you.style.display = 'flex';
      you.querySelector('.current-rating').style.width = (value / 5 * 100) + '%';
      you.querySelector('.pbc-rmeta').textContent = value.toFixed(1) + ' / 5';
      hint.textContent = 'Your rating is saved—hover and click to change it.';
    }

    // hover preview + commit on the VLM bar (resting state stays the VLM rating)
    _bindBar(ul, li, key, restPct, {
      onClick: () => { hint.textContent = 'Saving your rating…'; },
      onCommit: (value, res) => { setHumanRow(res.human_mean, res.human_n); setYouRow(res.your); },
      onError: (err) => { hint.textContent = 'Couldn’t save rating (' + err.message + ').'; },
    });

    // initial state from the server
    getRating(key).then((r) => { setHumanRow(r.human_mean, r.human_n); setYouRow(r.your); }).catch(() => {});
  }

  // Shared hover-to-fill + click-to-commit behaviour for any star bar.
  // ul/li are the .star-rating strip and its .current-rating fill; restPct is the
  // resting width%. After a commit the bar returns to restPct (callers redraw rows).
  function _bindBar(ul, li, key, restPct, hooks) {
    hooks = hooks || {};
    ul.addEventListener('mousemove', (e) => { li.style.width = (_valueFromEvent(ul, e) / 5 * 100) + '%'; });
    ul.addEventListener('mouseleave', () => { li.style.width = restPct + '%'; });
    ul.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation();          // stars may sit inside a link/cell
      const value = _valueFromEvent(ul, e);
      li.style.width = (value / 5 * 100) + '%';
      if (hooks.onClick) hooks.onClick(value);
      rate(key, value)
        .then((res) => { li.style.width = restPct + '%'; if (hooks.onCommit) hooks.onCommit(value, res); })
        .catch((err) => { li.style.width = restPct + '%'; if (hooks.onError) hooks.onError(err); });
    });
  }

  // Make every star row in a gallery rate-by-hover. Compact: the cell's existing
  // star bar becomes the input; human aggregate + your vote ride in the tooltip.
  // Called automatically from PB.fillGallery; `keys` line up 1:1 with .gcell order.
  function enhanceGalleryRatings(galleryEl, keys, items) {
    _injectCSS();
    const cells = galleryEl.querySelectorAll('.gcell');
    // fillGallery emits a cell only for keys with an item, so align to that filtered list
    const present = keys.filter((k) => items[k]);
    const map = {};
    present.forEach((key, idx) => {
      const it = items[key]; const cell = cells[idx];
      if (!it || !cell) return;
      const bar = cell.querySelector('.inline-rating');
      const ul = bar && bar.querySelector('ul.star-rating');
      const li = bar && bar.querySelector('.current-rating');
      if (!bar || !ul || !li) return;
      const restPct = it.rating == null ? 0 : Math.max(0, Math.min(5, it.rating)) / 5 * 100;
      bar.classList.add('pbc-live'); bar.title = 'Click to rate';
      map[key] = { bar };
      _bindBar(ul, li, key, restPct, {
        onCommit: (value, res) => {
          bar.classList.add('pbc-rated');
          bar.title = 'You rated ' + value.toFixed(1) +
            (res.human_n ? ' · human ' + res.human_mean.toFixed(2) + '/5 (' + res.human_n + ')' : '') + ' · click to change';
        },
      });
    });
    const ks = Object.keys(map);
    if (!ks.length) return;
    ratingsBatch(ks).then((r) => ks.forEach((key) => {
      const v = r[key]; if (!v) return; const bar = map[key].bar;
      if (v.your != null) {
        bar.classList.add('pbc-rated');
        bar.title = 'You: ' + v.your.toFixed(1) +
          (v.human_n ? ' · human ' + v.human_mean.toFixed(2) + '/5 (' + v.human_n + ')' : '') + ' · click to change';
      } else if (v.human_n) {
        bar.title = 'Human ' + v.human_mean.toFixed(2) + '/5 (' + v.human_n + ') · click to rate';
      }
    })).catch(() => {});
  }

  function getLineage(id) { return _json(API + '/lineage/' + encodeURIComponent(id)); }

  // {tag: count} across all community items — for typeahead + a tag cloud
  function tags() { return _json(API + '/tags'); }

  return { API, RUN, publish, items, genomes, remove, getAdminKey, setAdminKey, isAdmin,
           voterId, rate, getRating, ratingsBatch, starBarHTML, mountRating,
           enhanceGalleryRatings, getLineage, tags };
})();
