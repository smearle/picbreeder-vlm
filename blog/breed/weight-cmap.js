/* ===================================================================
 * PB_CMAP — single source of truth for the CPPN weight & node-value
 * colormaps used across the blog/breed CPPN figures.
 *
 *   • link(w[, alpha]) — diverging ramp for a connection's WEIGHT
 *     (drawn on graph edges). Returns 'rgb(...)' or, if alpha given,
 *     'rgba(...)'.
 *   • node(t)          — diverging ramp for a node's ACTIVATION VALUE /
 *     magnitude (drawn as node fills, grid cells, waypoint dots).
 *   • pos / neg        — the two end-stop swatch colours (for legends,
 *     the drag thermometer, etc.).
 *
 * Consumed by breed/dna-editor.js and assets/page/cppn-explainer.js.
 * Load this BEFORE those scripts on every page that uses them
 * (index.html, breed/dna.html, breed/play.html).
 *
 *            >>> TO REVERT OR SWAP SCHEMES, EDIT `ACTIVE` BELOW <<<
 *   It is the ONLY knob: set it to any key in SCHEMES. Add a new entry
 *   to SCHEMES to try yet another palette.
 * =================================================================== */
(function () {
  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
  function rgb(r, g, b) { return 'rgb(' + r + ',' + g + ',' + b + ')'; }
  function rgba(r, g, b, a) { return a == null ? rgb(r, g, b) : 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')'; }

  // Diverging ramp by HUE with saturation ∝ |w|, at a fixed mid-lightness so the
  // zero point is a clean neutral grey (never white). |w| = 3 saturates fully.
  function hslDiverge(w, posH, negH, sat, light, alpha) {
    var t = clamp(w / 3, -1, 1), m = Math.abs(t), hue = (t >= 0 ? posH : negH), S = m * sat, L = light;
    var c = (1 - Math.abs(2 * L - 1)) * S, hp = hue / 60, x = c * (1 - Math.abs(hp % 2 - 1)), r = 0, g = 0, b = 0;
    if (hp < 1) { r = c; g = x; } else if (hp < 2) { r = x; g = c; }
    else if (hp < 3) { g = c; b = x; } else if (hp < 4) { g = x; b = c; }
    else if (hp < 5) { r = x; b = c; } else { r = c; b = x; }
    var mm = L - c / 2;
    return rgba(Math.round((r + mm) * 255), Math.round((g + mm) * 255), Math.round((b + mm) * 255), alpha);
  }

  // Linear-RGB diverging ramp from a dark mid-slate (#33384a, never white) out to
  // two bright end-stops; magnitude reads as brightness. Used for node values.
  function slateDiverge(t, posRGB, negRGB) {
    var u = clamp(t, -1, 1), a = u < 0 ? -u : u, e = u >= 0 ? posRGB : negRGB;
    return rgb(Math.round(51 + (e[0] - 51) * a), Math.round(56 + (e[1] - 56) * a), Math.round(74 + (e[2] - 74) * a));
  }

  var SCHEMES = {
    // The original shipped palette — diverging blue(+)/red(−) for weights, and
    // dark-slate→orange(+)/blue(−) for node values.
    blue_red: {
      pos: '#2d68d4', neg: '#d24b2d',
      link: function (w, alpha) {
        var t = (clamp(w / 3, -1, 1) + 1) / 2, neg = [210, 75, 45], po = [45, 104, 212];
        return rgba(Math.round(neg[0] + (po[0] - neg[0]) * t),
                    Math.round(neg[1] + (po[1] - neg[1]) * t),
                    Math.round(neg[2] + (po[2] - neg[2]) * t), alpha);
      },
      node: function (t) { return slateDiverge(t, [255, 122, 24], [47, 122, 230]); },
    },
    // PRGn — green(+)/purple(−) for weights AND node values, so the two cmaps
    // stay in one family and don't clash.
    green_purple: {
      pos: '#40bf75', neg: '#ae40bf',
      link: function (w, alpha) { return hslDiverge(w, 145, 292, 0.50, 0.50, alpha); },
      node: function (t) { return slateDiverge(t, [71, 209, 129], [191, 71, 209]); },
    },
  };

  // ----------------------------------------------------------------------------
  var ACTIVE = 'green_purple';   // <<< THE TOGGLE — set to any key in SCHEMES >>>
  // ----------------------------------------------------------------------------

  var s = SCHEMES[ACTIVE] || SCHEMES.green_purple;
  window.PB_CMAP = { scheme: ACTIVE, schemes: SCHEMES, link: s.link, node: s.node, pos: s.pos, neg: s.neg };
})();
