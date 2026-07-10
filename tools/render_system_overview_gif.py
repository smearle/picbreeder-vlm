"""Render the animated system-overview figure to a GIF.

The figure at picbreeder-vlm-06b0d76d/system_overview/ is a live DOM animation (Web Animations
API + setTimeout + rAF). We play it in headless Chromium and record it with CDP's screencast,
which pushes a frame every time the compositor presents one, each stamped with the time it was
presented. Those frames are irregular — the compositor emits nothing across the show's still
beats — so we RESAMPLE them onto a fixed 1/fps grid afterwards, holding the last frame through
each gap. The result has exact GIF timing regardless of how fast this machine renders.

(An earlier version drove the page under CDP virtual time and screenshotted between advances.
That is deterministic in principle, but Page.captureScreenshot wedges permanently on roughly the
215th call once the clock is paused — the capture waits for a compositor frame that a paused
clock never produces. The screencast has no such failure mode.)

The show loops over the curated epochs, so by default we cut on epoch boundaries (via the
window.__sysov hook the player exposes) and emit exactly one full cycle — a seamless GIF.

    python3 tools/render_system_overview_gif.py --out /tmp/system_overview.gif
    python3 tools/render_system_overview_gif.py --epochs 1 --fps 25 --speed 1.5

Requires: playwright (+ `playwright install chromium`), pillow and ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import base64
import functools
import http.server
import io
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

# the blog checkout that holds the figure
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


class Screencast:
    """Collect (presented_at, png_bytes) for every compositor frame, newest last."""

    def __init__(self, page, cdp):
        self.page, self.cdp = page, cdp
        self.frames: list[tuple[float, bytes]] = []
        self._pending: list[int] = []
        cdp.on("Page.screencastFrame", self._on_frame)

    def _on_frame(self, evt) -> None:
        # Chromium's metadata.timestamp is wall-clock seconds; fall back to now() if it is absent.
        ts = evt.get("metadata", {}).get("timestamp") or time.time()
        self.frames.append((float(ts), base64.b64decode(evt["data"])))
        self._pending.append(evt["sessionId"])

    def _drain_acks(self) -> None:
        # Ack outside the event handler: re-entering cdp.send from inside a dispatched event is
        # not safe in sync Playwright. Chromium stops sending frames until each one is acked.
        while self._pending:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": self._pending.pop(0)})

    def start(self, width: int, height: int) -> None:
        self.cdp.send("Page.startScreencast", {"format": "png", "everyNthFrame": 1,
                                               "maxWidth": width, "maxHeight": height})

    def stop(self) -> None:
        self.cdp.send("Page.stopScreencast")

    def pump(self, ms: int = 20) -> None:
        """Let events land (sync Playwright only dispatches inside its own calls), then ack."""
        self.page.wait_for_timeout(ms)
        self._drain_acks()


def resample(frames: list[tuple[float, bytes]], t0: float, t1: float, fps: int,
             clip: tuple[int, int, int, int], scale: float, out_dir: Path,
             max_frames: int) -> int:
    """Hold-last-frame resampling of irregular presented frames onto a fixed 1/fps grid."""
    n_out = min(max_frames, max(1, int(round((t1 - t0) * fps))))
    left, top, right, bottom = (int(v * scale) for v in clip)
    i, last = 0, None
    written = 0
    for k in range(n_out):
        t = t0 + k / fps
        while i < len(frames) and frames[i][0] <= t:
            last = frames[i][1]
            i += 1
        if last is None:                       # nothing presented yet; wait for the first frame
            if i >= len(frames):
                break
            last = frames[i][1]
        im = Image.open(io.BytesIO(last)).convert("RGB").crop((left, top, right, bottom))
        im.save(out_dir / f"f{written:05d}.png")
        written += 1
    return written


def capture(args: argparse.Namespace, frame_dir: Path) -> int:
    base, httpd = serve(FIG_DIR)
    url = f"{base}/index.html"
    if args.manifest:
        url += f"?manifest={args.manifest}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--force-color-profile=srgb", "--hide-scrollbars"])
            page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                    device_scale_factor=args.scale)
            page.goto(url, wait_until="load")
            page.wait_for_function("window.__sysov && window.__sysov.nEpochs > 0")
            page.wait_for_timeout(800)   # fonts + the archive sheet decode

            # #fig is overflow:visible and its arrows/box borders spill past its own border
            # box, so crop to the union of the figure and its visible descendants.
            box = page.evaluate("""(pad) => {
                const fig = document.querySelector('#fig');
                const r = fig.getBoundingClientRect();
                let [l, t, rt, b] = [r.left, r.top, r.right, r.bottom];
                for (const el of fig.querySelectorAll('*')) {
                    const q = el.getBoundingClientRect();
                    if (q.width <= 0 || q.height <= 0) continue;
                    const s = getComputedStyle(el);
                    if (s.visibility === 'hidden' || s.display === 'none') continue;
                    l = Math.min(l, q.left); t = Math.min(t, q.top);
                    rt = Math.max(rt, q.right); b = Math.max(b, q.bottom);
                }
                return {left: Math.max(0, l - pad), top: Math.max(0, t - pad),
                        right: Math.min(innerWidth, rt + pad),
                        bottom: Math.min(innerHeight, b + pad)};
            }""", 4)
            clip = (int(box["left"]), int(box["top"]),
                    int(box["right"]) + 1, int(box["bottom"]) + 1)

            if args.speed != 1.0:
                page.evaluate(f"window.__sysov.setSpeed({args.speed})")
            epochs = args.epochs or page.evaluate("window.__sysov.nEpochs")

            cdp = page.context.new_cdp_session(page)
            cast = Screencast(page, cdp)
            cast.start(args.width, args.height)

            # Roll until the next epoch starts, so frame 0 is a clean cycle boundary, then again
            # until `epochs` more have gone by. Wall-clock timestamps on the frames are what we
            # resample against, so we just note the boundary times as they pass.
            def wait_for_epoch(target: int, limit: float) -> float:
                deadline = time.monotonic() + limit
                while page.evaluate("window.__sysov.epochsRun") < target:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"epoch {target} not reached within {limit}s")
                    cast.pump()
                return time.time()

            start_at = page.evaluate("window.__sysov.epochsRun") + 1
            t0 = wait_for_epoch(start_at, args.seek_limit)
            cast.frames.clear()          # everything before the boundary is warm-up
            print(f"cycle start; recording {epochs} epoch(s)...", file=sys.stderr, flush=True)
            t1 = wait_for_epoch(start_at + epochs, args.seek_limit)
            cast.pump(120)               # let the last presented frames land
            cast.stop()

            print(f"{len(cast.frames)} presented frames over {t1 - t0:.1f}s; resampling to "
                  f"{args.fps} fps", file=sys.stderr, flush=True)
            n = resample(cast.frames, t0, t1, args.fps, clip, args.scale, frame_dir,
                         args.max_frames)
            if n >= args.max_frames:
                print(f"WARNING: hit --max-frames={args.max_frames}; the cycle was cut short "
                      f"(lower --fps, or raise --speed to shorten the show)", file=sys.stderr)
            browser.close()
            return n
    finally:
        httpd.shutdown()


def encode(frame_dir: Path, out: Path, fps: int, colors: int, width: int | None) -> None:
    """Two-pass palette encode: one global palette keeps the figure's flat background clean."""
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
    ap.add_argument("--max-frames", type=int, default=1000, help="hard cap on emitted frames")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed of the show")
    ap.add_argument("--seek-limit", type=float, default=180.0,
                    help="seconds to wait for an epoch boundary")
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
