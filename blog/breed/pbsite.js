/* ===========================================================================
 * pbsite.js — shared glue for the Picbreeder replica.
 * Loads data/archive.json, renders every thumbnail client-side from its CPPN
 * genome (via cppn.js — pixel-identical to the published images), builds the
 * Editor's Picks / browse galleries, star ratings, and the branch links.
 * ======================================================================== */
const PB = (function () {
  let _data = null;
  let _pending = null;

  // Master kill-switch for the visitor-publishing feature (the "Community" shared
  // archive + Publish button + live gallery). While false the breed site is
  // read-only: no Community tab, no live community fetches, no publishing. Flip to
  // true to re-enable once publishing is ready to go live again.
  const COMMUNITY_ENABLED = false;

  // ---- shared site chrome (masthead + nav), injected so all pages match ----
  // Share links carry the current page so a branched image / gallery can be
  // posted directly. Built at module load (one full page load == one URL).
  const _shareUrl = encodeURIComponent(location.href);
  const _shareText = encodeURIComponent('AI Picbreeder: agents growing images together');
  const SHARE = {
    twitter: `https://twitter.com/intent/tweet?url=${_shareUrl}&text=${_shareText}`,
    bluesky: `https://bsky.app/intent/compose?text=${_shareText}%20${_shareUrl}`,
    linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${_shareUrl}`,
  };
  const MAST_HTML = `
    <center><table><tr>
      <td valign="top"><a href="index.html"><img src="pb_assets/logo.png" alt="picbreeder" style="border:0"></a></td>
      <td valign="top" align="left"><div style="font-size:10pt;">
        <h5>What was Picbreeder?</h5>
        Picbreeder was a collaborative art application based on an idea called evolutionary art,
        which is a technique that allows pictures to be bred almost like animals. For example,
        you could evolve a butterfly into a bat by selecting parents that looked like bats.
        This is a reconstruction of the <a href="https://web.archive.org/web/20160304023434/http://picbreeder.org/" target="_blank" rel="noopener">original site</a> using AI.
        <a href="../">Read the blog post.</a>
        <br><img src="pb_assets/progression.png" alt="" style="margin-top:6px"><br>
        <div style="margin-top:4px">Share:
          <a href="${SHARE.twitter}" target="_blank" rel="noopener">Twitter</a> &middot;
          <a href="${SHARE.bluesky}" target="_blank" rel="noopener">Bluesky</a> &middot;
          <a href="${SHARE.linkedin}" target="_blank" rel="noopener">LinkedIn</a>
        </div><br>
      </div></td>
    </tr></table></center>`;

  const NAV = [
    ['Home', 'index.html', 'home'],
    ['Getting Started', 'play.html', 'play'],
    ['Highest Rated', 'browse.html?type=RANK', 'RANK'],
    ['Best New', 'browse.html?type=UPANDCOME', 'UPANDCOME'],
    ['Most Branched', 'browse.html?type=MB', 'MB'],
    ['Newest', 'browse.html?type=CREATED', 'CREATED'],
    ['Random', 'browse.html?type=RANDOM', 'RANDOM'],
    ['Artists', 'artists.html', 'artists'],
    ['About', '../', 'about'],
  ];

  // Secondary row: jump across the different archives/sources behind the site.
  // `All Experiments` lists every AI run here; `Human` browses the original
  // human-bred Picbreeder archive in-site (its CPPNs render/branch client-side
  // just like the AI runs). Add more entries here as needed — the row renders
  // automatically on every page.
  const NAV2 = [
    ['VLM Experiments', 'runs.html', 'runs'],
    ['Human Archive', 'browse.html?run=human', 'human'],
  ].concat(COMMUNITY_ENABLED ? [['Community', 'community.html', 'community']] : []);

  function renderNav(el, items, active) {
    el.innerHTML = items.map(([label, href, key]) => {
      const style = key === active ? ' style="text-decoration:underline"' : '';
      return `<a href="${href}"${style}>${label}</a>`;
    }).join(' |\n');
  }

  function chrome(active) {
    const m = document.getElementById('pb-mast');
    if (m) m.innerHTML = MAST_HTML;
    const h = document.getElementById('pb-nav');
    if (h) renderNav(h, NAV, active);
    // second nav row — inject a sub-bar right after #hdr if not already present
    const hdr = document.getElementById('hdr');
    if (hdr) {
      let h2 = document.getElementById('pb-nav2');
      if (!h2) {
        const bar = document.createElement('div');
        bar.id = 'hdr2';
        bar.innerHTML = '<span id="pb-nav2"></span>';
        hdr.parentNode.insertBefore(bar, hdr.nextSibling);
        h2 = bar.firstChild;
      }
      renderNav(h2, NAV2, active);
    }
    propagateBase();
  }

  // Carry a dev `?archiveBase=` across ALL internal navigation so the local
  // mirror/proxy stays selected page-to-page (no-op in production where it's
  // unset). A single capture-phase click listener rewrites internal links the
  // instant they're clicked — so it covers links added dynamically after load
  // (category strips, artist links, run-gallery cells) too. We also append it
  // to same-page button navigations (pbbtn onclick="location.href=…").
  const _LOCAL = /^(?:\.\/)?(index|browse|play|runs|tree|detail|dna|artists|community)\.html/;
  // Blog pages one level up (`../`, `../index.html`, `../archive-gallery*.html`)
  // also read `?archiveBase=`, so propagate the dev base onto links back to the
  // blog ("Check out the blog", "About", "Human") too.
  const _BLOG = /^\.\.\/(?:$|index\.html|archive-gallery(?:-sprite)?\.html)/;
  function _withBase(href, ab) {
    return (_LOCAL.test(href) || _BLOG.test(href)) && href.indexOf('archiveBase=') < 0
      ? href + (href.indexOf('?') < 0 ? '?' : '&') + 'archiveBase=' + encodeURIComponent(ab)
      : href;
  }
  let _baseHooked = false;
  function propagateBase() {
    const ab = new URLSearchParams(location.search).get('archiveBase');
    if (!ab || _baseHooked) return;
    _baseHooked = true;
    // anchors (incl. dynamically-added): rewrite href just before navigation
    document.addEventListener('click', (e) => {
      const a = e.target && e.target.closest && e.target.closest('a[href]');
      if (a) a.setAttribute('href', _withBase(a.getAttribute('href'), ab));
    }, true);
    // buttons that navigate via inline onclick="location.href='…'"
    document.addEventListener('click', (e) => {
      const b = e.target && e.target.closest && e.target.closest('button[onclick]');
      if (!b) return;
      const m = (b.getAttribute('onclick') || '').match(/location\.href=['"]([^'"]+)['"]/);
      if (m && _LOCAL.test(m[1]) && m[1].indexOf('archiveBase=') < 0) {
        e.preventDefault(); e.stopImmediatePropagation();
        location.href = _withBase(m[1], ab);
      }
    }, true);
  }

  function load() {
    if (_data) return Promise.resolve(_data);
    if (_pending) return _pending;
    _pending = fetch('data/archive.json').then((r) => r.json()).then((d) => { _data = d; return d; });
    return _pending;
  }

  // The blog's CPPN explainer bundles a handful of hand-picked genomes (the classic
  // human Butterfly/Apple/Skull and a few AI picks) that live ONLY in this file, not
  // in the archive index. resolveGenome() falls back here so the blog's "analog on the
  // interactive breeding site" links (e.g. dna.html?g=h_butterfly) open the same DNA.
  let _explainer = null;
  function loadExplainer() {
    if (_explainer) return _explainer;
    _explainer = fetch('data/cppn_explainer.json')
      .then((r) => (r.ok ? r.json() : {}))
      .then((d) => d.genomes || {})
      .catch(() => ({}));
    return _explainer;
  }

  // ---- expansive archive, fetched from Hugging Face (whole published set) ----
  // The bundled archive.json stays tiny for instant homepage load; everything
  // else is pulled on demand from the public HF dataset (genomes rendered/branched
  // client-side) via the resolve URL below. `?archiveBase=` is an optional dev
  // override for pointing at a local mirror (`.`). Per LAYOUT v2:
  // <base>/index.json, <base>/site/<run>/genomes.json.gz,
  // <base>/results/<run>/archive_metadata.json.
  const HF_BASE = 'https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive/resolve/main';
  // Only honour the dev override when it points at a LOCAL mirror: `.` (relative),
  // same-origin, or localhost. A crafted `?archiveBase=evil.com` link would otherwise
  // make a visitor's browser load the whole gallery from an attacker's host, so any
  // other host is ignored and we fall back to the public HF dataset.
  function _safeBase(o) {
    if (!o || o === '.') return o || null;            // unset -> HF; '.' -> offline mirror
    const withScheme = /^(https?:)?\/\//.test(o) ? o : 'http://' + o;   // bare host -> add scheme
    let u; try { u = new URL(withScheme, location.href); } catch (e) { return null; }
    const h = u.hostname;
    const local = h === location.hostname || h === 'localhost' || h === '127.0.0.1' || h === '[::1]' || h === '::1';
    return local ? withScheme : null;
  }
  function archiveBase() {
    const o = _safeBase(new URLSearchParams(location.search).get('archiveBase'));
    return (o || HF_BASE).replace(/\/$/, '');
  }
  // Per-run family-tree lineage sidecar. The VLM runs ship their shards bundled in
  // the blog repo (data/tree/<run>.json.gz); the human Picbreeder archive keeps its
  // shard with the rest of its archive data on HF (results/human/tree.json.gz),
  // fetched via archiveBase() like its genomes/metadata.
  function treeShardURL(run) {
    return run === 'human'
      ? archiveBase() + '/results/human/tree.json.gz'
      : 'data/tree/' + run + '.json.gz';
  }
  // Which VLM bred a run. The index entry's config is authoritative, but plenty of
  // callers (resolveGenome from a bare ?run=&id= link, say) have no entry in hand —
  // and silently defaulting those to gemini-2.5-pro miscredits every other model's
  // images. Every VLM run's name carries `model-<id>_tb`, so parse that first; the
  // default is only for the pre-sweep runs that predate the flag.
  function modelOfRun(run, entry) {
    const cfg = (entry && entry.config) || {};
    if (cfg.model) return cfg.model;
    const m = /model-([a-z0-9.\-]+?)_tb/i.exec(run || '');
    return m ? m[1] : 'gemini-2.5-pro';
  }

  let _index = null;
  const _runs = {};      // run -> Promise<{items, keys, label, run, n}>
  const _runData = {};   // run -> resolved {items,...} (sync access for DNA/branch)

  function loadIndex() {
    if (_index) return _index;
    _index = fetch(archiveBase() + '/index.json').then((r) => {
      if (!r.ok) throw new Error('index.json HTTP ' + r.status);
      return r.json();
    });
    return _index;
  }

  async function _fetchGz(url) {            // gzipped JSON → object (HF serves raw .gz)
    const r = await fetch(url);
    if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
    const stream = r.body.pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text());
  }

  function _meanRating(e) {
    const rs = (e.vlm_ratings || []).filter((x) => typeof x === 'number');
    return rs.length ? rs.reduce((a, b) => a + b, 0) / rs.length : null;
  }

  // Fetch one run's whole published archive: genomes (rendered/branched in-browser)
  // joined with its metadata (titles/ratings/agent). Items are keyed `run::id` so
  // branch/DNA can route back through the run cache without the bundled archive.
  // Live community run: published genomes come from the publish API (a HF Space),
  // not the static archive, so just-published items appear without a CDN lag.
  async function _loadCommunity() {
    const apiBase = (window.PBC && PBC.API) ||
      'https://picbreeder-vlm-picbreeder-community-api.hf.space';
    const [list, gen] = await Promise.all([
      fetch(apiBase + '/items').then((r) => r.json()),
      fetch(apiBase + '/genomes').then((r) => r.json()),
    ]);
    const items = {};
    const order = [];
    (list.items || []).forEach((e) => {
      const g = gen[e.id];
      if (!g) return;
      const key = 'community::' + e.id;
      items[key] = {
        g: g, title: e.title || 'Untitled', model: 'community',
        agent: '', user: e.author || 'anon', color: e.color !== false,
        // community items have no VLM score; show the human aggregate on the stars
        rating: e.hr_mean == null ? null : e.hr_mean, nrat: e.hr_n || 0,
        run: 'community', id: e.id, parent: e.parent || null, added_at: e.created_at || null,
        tags: e.tags || [],
      };
      order.push(key);
    });
    const keys = order.slice().reverse();   // newest first for browsing
    const data = { items, keys, run: 'community', label: 'Community', n: keys.length };
    _runData['community'] = data;
    return data;
  }

  // ---- per-image VLM captions ------------------------------------------------
  // Captions live in their own per-run file (results/<run>/captions_<model>.json,
  // keyed by "<id>.png") rather than in archive_metadata.json, because they're
  // generated post-hoc by a fixed captioner. Across the whole sweep that captioner
  // is gemini-2.5-pro for every run; a handful also carry a qwen3-vl-8b file. We
  // probe in preference order and cache the result (incl. {} for uncaptioned runs)
  // so a run's caption file is fetched at most once.
  const _captions = {};   // run -> Promise<{ "<id>.png": caption }>
  const _captionModel = {};   // run -> the captioner whose file we actually loaded
  const CAPTION_MODELS = ['gemini-2.5-pro', 'qwen3-vl-8b'];
  function loadCaptions(run) {
    if (_captions[run]) return _captions[run];
    _captions[run] = (async () => {
      for (const m of CAPTION_MODELS) {
        try {
          const r = await fetch(archiveBase() + '/results/' + run + '/captions_' + m + '.json');
          if (r.ok) { _captionModel[run] = m; return await r.json(); }
        } catch (e) { /* try next captioner */ }
      }
      return {};
    })();
    return _captions[run];
  }
  // The captioner is a post-hoc, fixed model — never the model that bred the run.
  // Valid only once loadCaptions(run) has resolved; falls back to the preferred one.
  function captionModelOf(run) { return _captionModel[run] || CAPTION_MODELS[0]; }

  function loadRun(run, entry) {
    if (_runs[run]) return _runs[run];
    if (run === 'community') {
      // Publishing disabled: never hit the community Space; hand back an empty run so
      // browse/tag pages and liveLists simply show no visitor-published items.
      _runs[run] = COMMUNITY_ENABLED
        ? _loadCommunity()
        : Promise.resolve({ items: {}, keys: [], run: 'community', label: 'Community', n: 0 });
      return _runs[run];
    }
    const base = archiveBase();
    _runs[run] = (async () => {
      const [genomes, meta, caps] = await Promise.all([
        _fetchGz(base + '/site/' + run + '/genomes.json.gz'),
        fetch(base + '/results/' + run + '/archive_metadata.json').then((r) => r.ok ? r.json() : { entries: [] }),
        loadCaptions(run),
      ]);
      const isHuman = run === 'human' || (entry && entry.arc === 'human');
      const model = isHuman ? 'Picbreeder (human)' : modelOfRun(run, entry);
      // Who *bred* the image (model, above) vs who *captioned* it: captions are
      // written post-hoc by a fixed captioner, so a flash-lite run's images are
      // still captioned by gemini-2.5-pro — and human images were bred by people.
      const captionModel = captionModelOf(run);
      const items = {};
      const order = [];
      (meta.entries || []).forEach((e) => {
        const g = genomes[e.id];
        if (!g) return;
        const key = run + '::' + e.id;
        const r = _meanRating(e);
        // Human Picbreeder images have no human-given title — the "title" field
        // is a VLM caption. Show the filename as the title and surface the
        // caption separately (in the detail view).
        items[key] = {
          g: g,
          title: isHuman ? e.id : (e.title || 'Untitled'),
          caption: isHuman ? (e.title || '') : (caps[e.id + '.png'] || ''),
          model: model,
          captionModel: captionModel,
          agent: (String(e.agent_id || '').match(/\d+/) || [''])[0],
          color: e.color_enabled !== false,
          rating: r == null ? null : Math.round(r * 100) / 100,
          nrat: (e.vlm_ratings || []).length, run: run, id: e.id,
        };
        order.push(key);
      });
      // any genome without a metadata entry still shows (title from id)
      Object.keys(genomes).forEach((id) => {
        const key = run + '::' + id;
        if (items[key]) return;
        items[key] = { g: genomes[id], title: id, model: model, agent: '', color: true, rating: null, nrat: 0, run: run, id: id };
        order.push(key);
      });
      const keys = order.slice().sort((a, b) => (items[b].rating || 0) - (items[a].rating || 0));
      const data = { items, keys, run, label: (entry && entry.label) || run, n: keys.length };
      _runData[run] = data;
      return data;
    })();
    return _runs[run];
  }
  function runItems(run) { return _runData[run] || null; }   // sync access after loadRun resolves

  // Metadata-only run loader for browsing. Builds the same item map as loadRun
  // but WITHOUT fetching genomes.json.gz: galleries show sprite-sheet thumbnails
  // (loadSprite) and the genome is fetched lazily only when the visitor opens
  // Evolve / DNA (resolveGenome -> loadRun). Runs whose metadata is empty fall
  // back to the full loadRun (genomes rendered live); runs that have metadata but
  // no sprite sheet fall back to fillGallery's lazy per-run genome render.
  const _runCards = {};
  function loadRunCards(run, entry) {
    if (_runCards[run]) return _runCards[run];
    if (run === 'community') return loadRun(run, entry);   // community has its own combined path
    _runCards[run] = (async () => {
      const meta = await fetch(archiveBase() + '/results/' + run + '/archive_metadata.json')
        .then((r) => (r.ok ? r.json() : { entries: [] }));
      if (!(meta.entries && meta.entries.length)) return loadRun(run, entry);  // no metadata: live-render
      const isHuman = run === 'human' || (entry && entry.arc === 'human');
      const model = isHuman ? 'Picbreeder (human)' : modelOfRun(run, entry);
      const captionModel = CAPTION_MODELS[0];   // cards carry no caption; the captioner is fixed
      const items = {};
      const order = [];
      meta.entries.forEach((e) => {
        const key = run + '::' + e.id;
        const r = _meanRating(e);
        items[key] = {
          title: isHuman ? e.id : (e.title || 'Untitled'),
          caption: isHuman ? (e.title || '') : '',
          model: model,
          captionModel: captionModel,
          agent: (String(e.agent_id || '').match(/\d+/) || [''])[0],
          color: e.color_enabled !== false,
          rating: r == null ? null : Math.round(r * 100) / 100,
          nrat: (e.vlm_ratings || []).length, run: run, id: e.id,
        };
        order.push(key);
      });
      const keys = order.slice().sort((a, b) => (items[b].rating || 0) - (items[a].rating || 0));
      return { items, keys, run, label: (entry && entry.label) || run, n: keys.length };
    })();
    return _runCards[run];
  }

  // ---- per-image reviews (the individual VLM critiques) ----------------------
  // Each rating an image received carries a short written comment and the title
  // the rater "saw" — stored as parallel arrays vlm_ratings / vlm_comments /
  // vlm_reported_titles in the run's archive_metadata.json. We fetch just that
  // metadata (no genomes), zip the arrays into one review record per rater, and
  // cache per run so detail.html can lazily expand a paginated list on demand.
  const _reviews = {};   // run -> Promise<{ id: [{rating, comment, title}, ...] }>
  function _loadRunReviews(run) {
    if (_reviews[run]) return _reviews[run];
    const p = fetch(archiveBase() + '/results/' + run + '/archive_metadata.json')
      .then((r) => {
        if (r.status === 404) return { entries: [] };          // this run simply has no metadata
        if (!r.ok) throw new Error('reviews HTTP ' + r.status); // other HTTP errors are real failures
        return r.json();
      })
      .then((meta) => {
        const byId = {};
        (meta.entries || []).forEach((e) => {
          const rts = e.vlm_ratings || [], cs = e.vlm_comments || [], ts = e.vlm_reported_titles || [];
          const n = Math.max(rts.length, cs.length, ts.length);
          const list = [];
          for (let i = 0; i < n; i++) {
            list.push({ rating: typeof rts[i] === 'number' ? rts[i] : null, comment: cs[i] || '', title: ts[i] || '' });
          }
          byId[e.id] = list;
        });
        return byId;
      });
    // Don't cache a failed fetch (network error / unreachable proxy): drop it so a
    // later click retries rather than permanently showing the empty result.
    p.catch(() => { if (_reviews[run] === p) delete _reviews[run]; });
    _reviews[run] = p;
    return p;
  }
  // reviews for one resolved record (canon, tree, or HF-run item — all carry run+id)
  async function loadReviews(rec) {
    if (!rec || !rec.run || rec.id == null) return [];
    const byId = await _loadRunReviews(rec.run);
    return byId[rec.id] || [];
  }

  // ---- thumbnail-first assets ------------------------------------------------
  // Galleries render from HF sprite sheets (pre-rendered thumbnails) and a slim
  // per-user card file; the full genome is fetched lazily only when the visitor
  // opens Evolve / DNA (resolveGenome -> loadRun, below). So browsing a whole
  // personality costs a few shared sheet images, not megabytes of genomes.

  // breed/data/user_cards.json: { runs:[name...], users:{ name:[[runIdx, idNum,
  // title, rating, nrat, agent, color(0/1), children], ...] } }. Bundled, browse-only.
  let _userCards = null;
  function loadUserCards() {
    if (!_userCards) _userCards = fetch('data/user_cards.json').then((r) => r.json())
      .catch(() => ({ runs: [], users: {} }));
    return _userCards;
  }

  // breed/data/model_cards.json: the same cards, grouped by the VLM that bred them —
  // { runs:[name...], models:{ id:{ n_images, n_rated, n_runs, cards:[[runIdx, idNum,
  // title, rating, nrat, agent, color(0/1), children], ...] } } }, best-rated first and
  // capped per model. Backs browse.html?model=, which otherwise could only filter the
  // bundled canonical run (one model). n_rated is 0 for a model whose runs never went
  // through a rating pass, so the gallery can drop the "best-rated" claim.
  let _modelCards = null;
  function loadModelCards() {
    if (!_modelCards) _modelCards = fetch('data/model_cards.json').then((r) => r.json())
      .catch(() => ({ runs: [], models: {} }));
    return _modelCards;
  }

  // site/<run>/sprite/sprites.json packing manifest, or null if a run has no
  // sprite set yet (the gallery then falls back to rendering its genome).
  const _sprites = {};   // run -> Promise<manifest|null>
  function loadSprite(run) {
    if (!_sprites[run]) _sprites[run] = fetch(archiveBase() + '/site/' + run + '/sprite/sprites.json')
      .then((r) => (r.ok ? r.json() : null)).then((m) => { if (m) m._run = run; return m; }).catch(() => null);
    return _sprites[run];
  }

  // Sprite linear index i (0-based) is the published image img_{i+1:06d}; sheets
  // pack per_sheet thumbs in a sheet_cols grid. Return the CSS background spec to
  // crop one cell into an sz x sz box, or null if the id isn't in this manifest.
  function spriteCell(man, id, sz) {
    if (!man) return null;
    const i = parseInt(String(id).replace(/\D/g, ''), 10) - 1;
    if (!(i >= 0) || i >= man.n) return null;
    const s = Math.floor(i / man.per_sheet);
    if (s >= man.n_sheets) return null;
    const within = i % man.per_sheet;
    const row = Math.floor(within / man.sheet_cols), col = within % man.sheet_cols;
    const scale = sz / man.cell;
    const dims = (man.sheet_dims && man.sheet_dims[s]) || [man.sheet_cols * man.cell, man.sheet_rows * man.cell];
    const sheet = man.sheet_tmpl.replace(/\{s:0*(\d+)d\}/, (m, w) => String(s).padStart(+w, '0'));
    return { sheet: archiveBase() + '/site/' + man._run + '/sprite/' + sheet,
      bgw: Math.round(dims[0] * scale), bgh: Math.round(dims[1] * scale), x: col * sz, y: row * sz };
  }

  // Genomes-only fetch for runs that lack a sprite set: the gallery still shows
  // a real thumbnail by rendering the CPPN, without the heavy archive_metadata.
  const _genomesOnly = {};
  function loadRunGenomes(run) {
    if (!_genomesOnly[run]) _genomesOnly[run] = _fetchGz(archiveBase() + '/site/' + run + '/genomes.json.gz')
      .catch(() => ({}));
    return _genomesOnly[run];
  }

  // Readable label for an index run entry (table arcs already carry a label;
  // other runs get one synthesized from their config).
  function humanizeRun(e) {
    if (!e) return '';
    if (e.arc === 'human') return e.label || 'Human Picbreeder archive';
    if (e.arc && e.label && e.label !== e.run) return e.label;
    const c = e.config || {}, b = [c.model || 'gemini-2.5-pro'];
    if (c.memory_cl != null && c.memory_cl !== 1) b.push(c.memory_cl === -1 ? 'full memory' : 'mem ' + c.memory_cl);
    if (c.noise_eps) b.push('noise ' + c.noise_eps);
    if (c.personalities) b.push(c.personalities + ' personas');
    if (c.seed != null) b.push('seed ' + c.seed);
    return b.join(' · ');
  }

  // ---- cross-page navigation: a query string that names one genome, shared by
  // play.html (branch), detail.html (info page), dna.html (DNA editor). HF-run
  // items route by run+id (re-fetched on demand); bundled items by archive key.
  function navParams(item, key) {
    return item && item.run
      ? 'run=' + encodeURIComponent(item.run) + '&id=' + encodeURIComponent(item.id)
      : 'g=' + encodeURIComponent(key);
  }
  function branchHref(key, item) { return 'play.html?' + navParams(item, key); }
  function detailHref(key, item) { return 'detail.html?' + navParams(item, key); }
  function dnaHref(key, item)    { return 'dna.html?' + navParams(item, key); }

  // ---- resolve ONE genome (+ its metadata) from any page's URL params -------
  // Handles the three address forms used across the site: ?g=<archiveKey>,
  // ?run=<run>&id=<id> (Hugging Face), ?tree=<id> (Family-Tree node, canon run).
  // Returns a normalized record (g = genome JSON, may be undefined for the
  // view-only banner items whose genome lives only on the cluster).
  async function resolveGenome(params) {
    const tree = params.get('tree');
    const run = params.get('run'), id = params.get('id');
    let gkey = params.get('g');
    if (tree) {
      const [genomes, t] = await Promise.all([_fetchGz('data/tree_genomes.json.gz'), _fetchGz('data/tree.json.gz')]);
      const nd = (t.nodes && t.nodes[tree]) || {};
      return { key: 'c_' + tree, id: tree, run: t.run, canon: true, g: genomes[tree],
        title: nd.t || tree, model: t.model, agent: nd.a, user: nd.u, color: nd.col !== false,
        rating: nd.r == null ? null : nd.r, nrat: nd.n || 0, gen: nd.g, added_at: null, children: nd.c || 0 };
    }
    if (gkey && gkey.indexOf('::') > -1) { const s = gkey.split('::'); return resolveGenome(new URLSearchParams('run=' + encodeURIComponent(s[0]) + '&id=' + encodeURIComponent(s[1]))); }
    if (run && id) {
      const rd = await loadRun(run);
      const it = rd.items[run + '::' + id] || {};
      return { key: run + '::' + id, id: id, run: run, canon: false, g: it.g,
        title: it.title || id, caption: it.caption || '', model: it.model, agent: it.agent, user: it.user, color: it.color !== false,
        rating: it.rating == null ? null : it.rating, nrat: it.nrat || 0, gen: null, added_at: it.added_at || null,
        children: 0, tags: it.tags || [] };
    }
    if (gkey) {
      const d = await load();
      if (!d.items[gkey] || !d.items[gkey].g) {         // not in the archive index —
        const ex = await loadExplainer();               // try the blog explainer bundle
        const e = ex[gkey];
        if (e && e.g) {
          return { key: gkey, id: gkey.replace(/^[a-z]_/, ''), run: d.canon_run, canon: true, g: e.g,
            title: e.title || e.label || gkey, caption: e.caption || '', model: e.model, agent: e.agent,
            user: e.user, color: e.color !== false, rating: null, nrat: 0, gen: null, added_at: null,
            children: 0 };
        }
      }
      const it = d.items[gkey] || {};
      return { key: gkey, id: gkey.replace(/^[a-z]_/, ''), run: d.canon_run, canon: true, g: it.g,
        title: it.title || gkey, caption: it.caption || '', model: it.model, agent: it.agent, user: it.user, color: it.color !== false,
        rating: it.rating == null ? null : it.rating, nrat: it.nrat || 0, gen: it.gen, added_at: it.added_at,
        children: it.children || 0, thumb: it.thumb };
    }
    return null;
  }

  // ---- thumbnail rendering (progressive, so big grids don't freeze) ----
  const _queue = [];
  let _running = false;
  function _pump() {
    if (_running) return;
    _running = true;
    function step() {
      const t0 = performance.now();
      while (_queue.length && performance.now() - t0 < 14) {
        const job = _queue.shift();
        try {
          const g = genomeFromJSON(job.item.g);
          const res = job.canvas.width;
          job.canvas.getContext('2d').putImageData(renderGenome(g, res, job.item.color), 0, 0);
        } catch (e) { /* leave blank on error */ }
      }
      if (_queue.length) requestAnimationFrame(step);
      else _running = false;
    }
    requestAnimationFrame(step);
  }
  function renderThumb(canvas, item) { _queue.push({ canvas, item }); _pump(); }

  // ---- star rating widget (rating in [0,5], may be null) ----
  // Original picbreeder.org star.gif look: a fixed 5-star sprite (star_small.gif,
  // empty row over filled row) clipped to the rating fraction. No visible number
  // or rater count on thumbnails (matches the OG grids); the value is in the tooltip.
  function starHTML(rating /*, nrat */) {
    const pct = rating == null ? 0 : Math.max(0, Math.min(5, rating)) / 5 * 100;
    const tip = rating == null ? 'Not yet rated' : `${rating.toFixed(1)} / 5`;
    return `<span class="inline-rating" title="${tip}"><ul class="star-rating small-star"><li class="current-rating" style="width:${pct}%">${tip}</li></ul></span>`;
  }

  function esc(s) { return (s || '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  function artistLabel(item) {
    // item.user can be a community publisher's free-text author string, so escape it:
    // this label is dropped into innerHTML by cellHTML and play.html's branch header.
    if (item.user) return esc(item.user);   // personality-trait "artist" (publication galleries)
    const m = item.model || 'gemini-2.5-pro';
    return item.agent ? `${m} · agent&nbsp;${item.agent}` : m;
  }

  // ---- live, human-adaptive category lists ---------------------------------
  // The bundled `d.lists` (Highest Rated / Best New / Newest) are frozen at build
  // time and rank on the VLM's ratings only. liveLists folds in the live signal
  // the community API collects — human star-ratings (POST /ratings) and freshly
  // PUBLISHED images (the `community` run) — and re-ranks those three categories
  // so the home strips and the matching browse pages adapt to what visitors do.
  //
  // The rank is a confidence-weighted (Bayesian) blend, NOT a plain mean: human
  // votes are sparse, so a lone 5-star click must not leapfrog an image the VLM
  // scored well over many votes. Each score shrinks toward the global mean with a
  // pseudo-count, and one human vote is weighted as HUMAN_W VLM votes so a handful
  // of real ratings can still move the board. (Plain means reward whatever was
  // *measured most*, not what's best — the archive's known rating exposure-bias.)
  const _LL_PRIOR_C = 8;        // prior strength (pseudo-votes toward the global mean)
  const _LL_HUMAN_W = 4;        // one human vote counts as this many VLM votes

  function _tsMs(it) {
    if (!it) return 0;
    const a = it.added_at;
    if (typeof a === 'number') return a < 1e12 ? a * 1000 : a;   // unix s (community) vs ms
    const t = a ? Date.parse(a) : NaN;
    return isFinite(t) ? t : 0;
  }

  // blended (mean, confidence) for one item given its live human aggregate `hr`
  function _blendScore(it, hr, m0) {
    const vr = it && typeof it.rating === 'number' ? it.rating : null;
    const vn = it && it.nrat ? it.nrat : 0;
    const hm = hr && typeof hr.mean === 'number' ? hr.mean : null;
    const hn = hr && hr.n ? hr.n : 0;
    let num = _LL_PRIOR_C * m0, den = _LL_PRIOR_C;
    if (vr != null) { num += vn * vr; den += vn; }
    if (hm != null) { num += _LL_HUMAN_W * hn * hm; den += _LL_HUMAN_W * hn; }
    return { score: num / den, conf: vn + _LL_HUMAN_W * hn };
  }

  // live human aggregates for a key set, chunked under the API's batch cap.
  async function _humanAgg(keys) {
    const out = {};
    if (!(window.PBC && PBC.ratingsBatch)) return out;
    for (let i = 0; i < keys.length; i += 400) {
      try {
        const r = await PBC.ratingsBatch(keys.slice(i, i + 400));
        Object.keys(r || {}).forEach((k) => {
          const v = r[k]; out[k] = { mean: v.human_mean, n: v.human_n || 0 };
        });
      } catch (e) { /* Space asleep / offline → fall back to VLM-only for this chunk */ }
    }
    return out;
  }

  // Returns { lists, items }: `lists` has the three adaptive categories re-ranked
  // (others pass through), `items` is d.items plus any community publishes so the
  // caller can render the new keys. Any failure falls back to the static bundle,
  // so the homepage never blocks on (or breaks with) an unreachable API.
  async function liveLists(d) {
    const fallback = { lists: d.lists, items: d.items };
    if (!(window.PBC && PBC.ratingsBatch)) return fallback;
    try {
      // 1. community publishes (their own run): items keyed community::id, each
      //    carrying its genome (.g) and the human-rating aggregate as its rating.
      let comm = { items: {}, keys: [] };
      try { comm = await loadRun('community'); } catch (e) { /* none published yet */ }
      const items = Object.assign({}, d.items, comm.items);

      // 2. candidate pool = every bundled list key ∪ every community key.
      const cand = new Set();
      ['highest_rated', 'best_new', 'newest', 'most_branched', 'editors']
        .forEach((L) => (d.lists[L] || []).forEach((k) => cand.add(k)));
      (comm.keys || []).forEach((k) => cand.add(k));
      const pool = Array.from(cand).filter((k) => items[k]);

      // 3. live human aggregates + global VLM prior mean.
      const hr = await _humanAgg(pool);
      const vrs = pool.map((k) => items[k].rating).filter((x) => typeof x === 'number');
      const m0 = vrs.length ? vrs.reduce((a, b) => a + b, 0) / vrs.length : 3.8;

      const sc = {}, cf = {};
      pool.forEach((k) => { const b = _blendScore(items[k], hr[k], m0); sc[k] = b.score; cf[k] = b.conf; });
      const byScore = (a, b) => (sc[b] - sc[a]) || (_tsMs(items[b]) - _tsMs(items[a]));
      const cap = (L, n) => (d.lists[L] || []).length || n;

      // Highest Rated: anything with a little evidence, best blended score first.
      const highest_rated = pool.filter((k) => cf[k] >= 5).sort(byScore).slice(0, cap('highest_rated', 60));

      // Best New: community publishes + the bundled "recent" pool, ranked by
      // blended score (a well-liked new upload climbs), newest breaking ties.
      const newPool = new Set(comm.keys || []);
      (d.lists.best_new || []).forEach((k) => newPool.add(k));
      const best_new = Array.from(newPool).filter((k) => items[k] && cf[k] >= 2)
        .sort(byScore).slice(0, cap('best_new', 24));

      // Newest: pure recency across the bundled archive + community uploads, so a
      // fresh publish lands at the front.
      const newest = pool.slice().sort((a, b) => _tsMs(items[b]) - _tsMs(items[a])).slice(0, cap('newest', 60));

      // Most Branched: each image's bundled branch count (`children`, from the VLM
      // archive) PLUS live community branches — every community publish's `parent`
      // points at the image it was bred from, so human branching moves the board
      // too. Ranked over ALL items (not just the list pool) so a canon image that
      // only becomes popular via community branching can still surface.
      const bc = {};
      Object.keys(items).forEach((k) => { bc[k] = items[k].children || 0; });
      (comm.keys || []).forEach((k) => {
        const p = items[k] && items[k].parent;   // parent key shares this key space
        if (p != null && bc[p] != null) bc[p] += 1;
      });
      const most_branched = Object.keys(items).filter((k) => bc[k] > 0)
        .sort((a, b) => (bc[b] - bc[a]) || (_tsMs(items[b]) - _tsMs(items[a])))
        .slice(0, cap('most_branched', 36));

      return { lists: Object.assign({}, d.lists, { highest_rated, best_new, newest, most_branched }), items };
    } catch (e) {
      return fallback;
    }
  }

  // ---- build one gallery cell ----
  // Clicking the thumbnail/title opens the image's detail page (showgenome.php
  // in the original); the buttons are quick shortcuts to Evolve (breeding
  // console), the DNA editor, and — for canon images — the family tree.
  // View-only cells (item.thumb, banner images whose genome lives on the
  // cluster) show the published thumbnail and only link through to detail.
  // `cell` (optional) is a sprite-sheet crop spec from spriteCell(): when present
  // the thumbnail is a CSS-cropped sheet (no genome). Otherwise it's a canvas
  // rendered from the genome (bundled now, or lazily fetched by fillGallery for
  // HF runs), or a published <img> for view-only banner images.
  function cellHTML(key, item, small, cell) {
    const sz = small ? 88 : 128;
    const cls = small ? 'cellimg sm' : 'cellimg';
    const detail = detailHref(key, item);
    const title = esc(item.title);
    // Human Picbreeder images were bred by people (the VLM only captioned them,
    // but no caption is shown on the gallery cell, so credit the breeder here).
    const human = item.run === 'human';
    const byHref = human
      ? 'browse.html?run=human'
      : item.user
        ? `browse.html?user=${encodeURIComponent(item.user)}`
        : `browse.html?model=${encodeURIComponent(item.model || '')}`;
    const byVerb = human ? 'Bred by' : 'By';
    const byTip = item.user ? 'the AI personality that bred it'
      : (human ? 'a human on the original Picbreeder' : 'the VLM that published it');
    const byLabel = human ? 'a human' : artistLabel(item);
    const by = `<span class="by">${byVerb} <a href="${byHref}" title="${byTip}">${byLabel}</a></span>`;
    // The genome is needed only to breed/edit — never just to show the image.
    const breedable = !!(item.g || (item.run && item.id));
    let thumb;
    if (cell) {
      thumb = `<div class="${cls}" style="background-image:url('${cell.sheet}');background-size:${cell.bgw}px ${cell.bgh}px;background-position:-${cell.x}px -${cell.y}px"></div>`;
    } else if (breedable) {
      thumb = `<canvas class="${cls}" width="${sz}" height="${sz}"></canvas>`;
    } else {
      thumb = `<img class="${cls}" src="${item.thumb}" alt="${title}" loading="lazy">`;
    }
    const btns = breedable
      ? `<button class="pbbtn" onclick="location.href='${branchHref(key, item)}'" title="Branch / evolve this image">Evolve</button>
          ${key.indexOf('c_') === 0 ? `<button class="pbbtn" onclick="location.href='tree.html?g=${encodeURIComponent(key)}'" title="See this image's family tree">Tree</button>` : ''}
          <button class="pbbtn" onclick="location.href='${dnaHref(key, item)}'" title="Inspect / edit this image's CPPN DNA">DNA</button>`
      : `<button class="pbbtn" onclick="location.href='${detail}'">Info</button>`;
    return `
      <div class="gcell">
        <span class="title"><a href="${detail}">${title}</a></span>${by}
        <span class="imgwrap"><a href="${detail}" title="Image information">${thumb}</a></span>
        <span class="btns">${btns}</span>
        <div class="inline-rating-wrap">${starHTML(item.rating, item.nrat)}</div>
      </div>`;
  }

  // Fill a container with item keys. `opts.items` overrides the bundled item map;
  // `opts.sprites` is a {run: manifest} map so HF-run thumbnails come from sprite
  // sheets (no genome). Runs absent from that map fall back to rendering the
  // genome client-side, fetched once per run from genomes.json.gz.
  function fillGallery(el, keys, opts) {
    opts = opts || {};
    const items = opts.items || _data.items;
    const sprites = opts.sprites || {};
    const sz = opts.small ? 88 : 128;
    const cellFor = (it) => (it && it.run ? spriteCell(sprites[it.run], it.id, sz) : null);
    el.innerHTML = keys.map((k) => (items[k] ? cellHTML(k, items[k], opts.small, cellFor(items[k])) : '')).join('');
    const canvases = el.querySelectorAll('canvas');
    let i = 0;
    const lazyByRun = {};
    keys.forEach((k) => {
      const it = items[k];
      if (!it || cellFor(it)) return;                 // missing, or sprite div (no canvas)
      if (it.g) { renderThumb(canvases[i++], it); return; }
      if (it.run && it.id) (lazyByRun[it.run] = lazyByRun[it.run] || []).push([canvases[i++], it]);
    });
    Object.keys(lazyByRun).forEach((run) => loadRunGenomes(run).then((gm) =>
      lazyByRun[run].forEach(([cv, it]) => { const g = gm[it.id]; if (g) { it.g = g; renderThumb(cv, it); } })));
    // make each cell's star bar rate-by-hover (no-op if the community glue isn't loaded)
    if (window.PBC && PBC.enhanceGalleryRatings) PBC.enhanceGalleryRatings(el, keys, items);
  }

  function downloadDNA(key) {
    // route composite `run::id` keys through the run cache
    const item = (key.indexOf('::') > -1 && runItems(key.split('::')[0]))
      ? runItems(key.split('::')[0]).items[key] : _data.items[key];
    if (!item) return;
    const g = genomeFromJSON(item.g);
    const json = JSON.stringify(genomeToJSON(g), null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${(item.title || key).replace(/\s+/g, '_')}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return { COMMUNITY_ENABLED, load, chrome, liveLists, renderThumb, starHTML, cellHTML, fillGallery, downloadDNA, artistLabel, esc,
    archiveBase, treeShardURL, loadIndex, loadRun, loadRunCards, runItems, loadReviews, humanizeRun, modelOfRun, resolveGenome, fetchGz: _fetchGz,
    navParams, branchHref, detailHref, dnaHref,
    loadUserCards, loadModelCards, loadSprite, spriteCell, loadRunGenomes,
    get data() { return _data; } };
})();
