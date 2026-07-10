/* Click the static system-overview figure to expand it into the animated player
   (system_overview/index.html), which drives the same 4-panel layout live. The
   player posts {sysov:"height"} to self-size the iframe and {sysov:"collapse"} when
   the reader clicks the diagram again. */
(function () {
  const fig = document.querySelector(".sysfig");
  if (!fig) return;
  const img = fig.querySelector("img");
  if (!img) return;
  const SRC = "system_overview/index.html";

  fig.style.position = "relative";
  fig.style.cursor = "pointer";

  let iframe = null;

  function expand() {
    if (iframe) return;
    iframe = document.createElement("iframe");
    iframe.src = SRC;
    iframe.title = "Animated system overview";
    iframe.setAttribute("scrolling", "no");
    iframe.style.cssText =
      "display:block;margin:0 auto;width:100%;max-width:900px;height:640px;" +
      "border:0;overflow:hidden;background:#fff;";
    img.style.display = "none";
    fig.style.cursor = "default";
    fig.appendChild(iframe);
  }

  function collapse() {
    if (!iframe) return;
    iframe.remove(); iframe = null;
    img.style.display = "";
    fig.style.cursor = "pointer";
  }

  fig.addEventListener("click", function (e) {
    if (iframe) return;              // once expanded, the player handles clicks
    e.preventDefault();
    expand();
  });

  // The caption's "Click" link does the same thing; its href is the standalone
  // player, so it still works with scripting off.
  const capLink = document.getElementById("sysfig-animate-link");
  if (capLink) capLink.addEventListener("click", function (e) {
    if (iframe) return;
    e.preventDefault();
    expand();
  });

  window.addEventListener("message", function (e) {
    if (!iframe || e.source !== iframe.contentWindow) return;
    const d = e.data || {};
    if (d.sysov === "height" && d.h) iframe.style.height = d.h + "px";
    else if (d.sysov === "collapse") collapse();
  });
})();
