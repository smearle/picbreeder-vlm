"""Render the animated system-overview figure to a GIF.

The figure at picbreeder-vlm-06b0d76d/system_overview/ is a live DOM animation (Web Animations
API + setTimeout + rAF). We drive it in headless Chromium under CDP **virtual time**
(Emulation.setVirtualTimePolicy), advancing the page clock by exactly one frame's worth of
milliseconds at a time and screenshotting between advances. That makes the capture deterministic
and independent of how fast this machine actually renders: every frame lands on an exact
multiple of 1/fps of animation time, with no dropped or duplicated frames.

The show is a loop over the curated epochs, so by default we cut on epoch boundaries (using the
window.__sysov hook the player exposes) and emit exactly one full cycle — a seamless GIF.

    python3 tools/render_system_overview_gif.py --out /tmp/system_overview.gif
    python3 tools/render_system_overview_gif.py --epochs 1 --fps 25 --width 1000

Requires: playwright (+ `playwright install chromium`) and ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import base64
import functools
import http.server
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# the blog checkout that holds the figure, relative to this repo
FIG_DIR = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/system_overview")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a) -> None:  # noqa: D102
        pass


def serve(root: Path) -> tuple[str, socketserver.TCPServer]:
    """The player fetch()es its manifest, which file:// forbids — so serve the dir over HTTP."""
    handler = functools.partial(QuietHandler, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


class VirtualClock:
    """Advance the page's clock in fixed steps, blocking until each step is fully consumed."""

    def __init__(self, page, cdp):
        self.page, self.cdp = page, cdp
        self._expired = False
        cdp.on("Emulation.virtualTimeBudgetExpired", self._on_expired)

    def _on_expired(self, _evt) -> None:
        self._expired = True

    def _policy(self, **kw) -> None:
        self.cdp.send("Emulation.setVirtualTimePolicy", kw)

    def pause(self) -> None:
        self._policy(policy="pause")

    def advance(self, ms: float, timeout: float = 60.0) -> None:
        self._expired = False
        # pauseIfNetworkFetchesPending: a frame that kicks off an <img> load stalls the clock
        # until it resolves, so the animation never runs ahead of assets it is about to draw.
        self._policy(policy="pauseIfNetworkFetchesPending", budget=ms,
                     maxVirtualTimeTaskStarvationCount=100_000)
        # NB: sync Playwright dispatches CDP events only while we are inside one of its calls, so
        # we must PUMP (wait_for_timeout) rather than block on a threading primitive — blocking
        # here would starve the very event loop that delivers virtualTimeBudgetExpired.
        deadline = time.monotonic() + timeout
        while not self._expired:
            if time.monotonic() > deadline:
                raise TimeoutError(f"virtual time did not advance {ms}ms within {timeout}s")
            self.page.wait_for_timeout(2)


def shot(cdp, path: Path, clip: dict, scale: float) -> None:
    """Grab a frame straight off CDP.

    Page.screenshot (Playwright) waits for the compositor to present a NEW frame, which never
    happens across the show's still beats — a publication held on screen, a dwell between
    phases — once the clock is paused. Page.captureScreenshot returns the surface as it stands.
    """
    res = cdp.send("Page.captureScreenshot",
                   {"format": "png", "captureBeyondViewport": False,
                    "clip": {**clip, "scale": scale}})
    path.write_bytes(base64.b64decode(res["data"]))


def capture(args: argparse.Namespace, frame_dir: Path) -> int:
    base, httpd = serve(FIG_DIR)
    url = f"{base}/index.html"
    if args.manifest:
        url += f"?manifest={args.manifest}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--force-color-profile=srgb",
                                              "--disable-lcd-text",
                                              "--hide-scrollbars"])
            page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                    device_scale_factor=args.scale)
            # Load on the real clock — virtual time can't be paused across a navigation without
            # deadlocking the load event — then take the clock over once the player is up.
            page.goto(url, wait_until="load")
            page.wait_for_function("window.__sysov && window.__sysov.nEpochs > 0")
            page.wait_for_timeout(600)   # fonts + the archive sheet decode

            # A fixed clip rather than an element screenshot: Locator.screenshot waits for the
            # element to hold still across two rAFs, and rAF never fires while the clock is
            # paused. The figure's box is static anyway.
            box = page.locator("#fig").bounding_box()
            clip = {"x": int(box["x"]), "y": int(box["y"]),
                    "width": -(-box["width"] // 2) * 2, "height": -(-box["height"] // 2) * 2}

            cdp = page.context.new_cdp_session(page)
            clock = VirtualClock(page, cdp)
            clock.pause()

            dt = 1000.0 / args.fps
            epochs = args.epochs or page.evaluate("window.__sysov.nEpochs")

            # Run up to the START of the next epoch, so frame 0 is a clean cycle boundary. Seek in
            # coarse steps: nothing is captured here, so frame granularity doesn't matter.
            start_at = page.evaluate("window.__sysov.epochsRun") + 1
            for _ in range(int(args.seek_limit * 1000 / 50)):
                if page.evaluate("window.__sysov.epochsRun") >= start_at:
                    break
                clock.advance(50)
            else:
                raise RuntimeError(f"no epoch boundary within {args.seek_limit}s of virtual time")

            stop_at = start_at + epochs
            n = 0
            while n < args.max_frames:
                shot(cdp, frame_dir / f"f{n:05d}.png", clip, args.scale)
                n += 1
                clock.advance(dt)
                if page.evaluate("window.__sysov.epochsRun") >= stop_at:
                    break  # the loop has come back around; this frame would duplicate frame 0
                if n % 25 == 0:
                    print(f"  {n} frames ({n / args.fps:.1f}s)", file=sys.stderr, flush=True)

            if n >= args.max_frames:
                print(f"WARNING: hit --max-frames={args.max_frames}; the cycle was cut short "
                      f"(lower --fps or raise --max-frames)", file=sys.stderr)
            browser.close()
            return n
    finally:
        httpd.shutdown()


def encode(frame_dir: Path, out: Path, fps: int, colors: int, width: int | None) -> None:
    """Two-pass palette encode: a single global palette keeps the flat figure background clean."""
    scale = f"scale={width}:-1:flags=lanczos," if width else ""
    palette = frame_dir / "palette.png"
    common = ["-framerate", str(fps), "-i", str(frame_dir / "f%05d.png")]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *common,
                    "-vf", f"{scale}palettegen=max_colors={colors}:stats_mode=diff",
                    str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", *common, "-i", str(palette),
                    "-lavfi", f"{scale}paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                    "-loop", "0", str(out)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("system_overview.gif"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=0,
                    help="epochs to capture (default: all of them = one seamless cycle)")
    ap.add_argument("--max-frames", type=int, default=1000, help="hard cap on captured frames")
    ap.add_argument("--seek-limit", type=float, default=90.0,
                    help="virtual seconds to spend seeking to the first epoch boundary")
    ap.add_argument("--width", type=int, default=1280, help="browser viewport width")
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--scale", type=float, default=1.0, help="device pixel ratio for capture")
    ap.add_argument("--gif-width", type=int, default=900, help="output width (0 = native)")
    ap.add_argument("--colors", type=int, default=192)
    ap.add_argument("--manifest", default="", help="override the player's ?manifest= URL")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="sysov_frames_"))
    try:
        print(f"capturing at {args.fps} fps (cap {args.max_frames} frames)...", file=sys.stderr)
        n = capture(args, tmp)
        print(f"{n} frames = {n / args.fps:.1f}s; encoding {args.out}", file=sys.stderr)
        encode(tmp, args.out, args.fps, args.colors, args.gif_width or None)
        print(f"{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    finally:
        if args.keep_frames:
            print(f"frames kept in {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
