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

Two things keep the result from looking choppy. First, the screencast only sustains ~10 PNG
frames/s at --scale 2, well under the 25 fps we want, so we play the show at --capture-speed
(0.25) and sample the wall clock 4x slower to match: each emitted frame is then a *distinct*
presented frame rather than a held duplicate. Second, --scale rasterizes at a higher device
scale factor and the frames are downscaled at encode time, so the GIF is supersampled. Held
duplicates should stay near 3% — the show's genuine still beats — not 27%.

The show loops over the curated epochs, so by default we cut on epoch boundaries (via the
window.__sysov hook the player exposes) and emit exactly one full cycle — a seamless GIF.
The README ships two of the three epochs, which is a shorter, lighter loop.

One beat resists this pipeline: the rated pile that stacks up in the Evaluation node sits STATIC
(cards snap on, nothing animates), so the compositor may present no frame while it is up and the
resampler holds a pre-pile frame — the pile then silently vanishes from an otherwise-fine GIF,
non-deterministically from run to run. So each render is verified (stack_coverage) and re-rolled up
to --attempts times until the pile lands in every epoch; the best attempt is kept regardless. And to
protect a good committed render, an existing --out is never overwritten without --force (the new one
lands beside it as <name>.new.gif).

    python3 tools/render_system_overview_gif.py --out /tmp/system_overview.gif
    python3 tools/render_system_overview_gif.py --epochs 2 --out figures/system_fig/system_overview.gif --force

Requires: playwright (+ `playwright install chromium`), pillow, numpy and ffmpeg on PATH.
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

    def pump(self, ms: int = 2) -> None:
        """Let events land (sync Playwright only dispatches inside its own calls), then ack.

        Chromium withholds the next frame until the last is acked, so a slow pump caps the
        capture rate well below the 60 Hz the page actually presents at. Keep `ms` small.
        """
        self.page.wait_for_timeout(ms)
        self._drain_acks()


def resample(frames: list[tuple[float, bytes]], t0: float, t1: float, fps: float,
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
        im = Image.open(io.BytesIO(last)).convert("RGB")
        if right > im.width or bottom > im.height:   # PIL would pad with black instead of failing
            raise ValueError(f"crop {(left, top, right, bottom)} exceeds the {im.size} frame; "
                             f"--scale does not match the captured device scale factor")
        im.crop((left, top, right, bottom)).save(out_dir / f"f{written:05d}.png")
        written += 1
    return written


def capture(args: argparse.Namespace, frame_dir: Path) -> int:
    base, httpd = serve(FIG_DIR)
    url = f"{base}/index.html"
    if args.manifest:
        url += f"?manifest={args.manifest}"
    try:
        with sync_playwright() as p:
            # Screencast frames come back at the surface size, which a context-level
            # device_scale_factor does NOT change -- only the browser-level flag does. So
            # supersampling has to be requested at launch; `clip` stays in CSS px and is scaled
            # up to frame px in resample().
            browser = p.chromium.launch(args=["--force-color-profile=srgb", "--hide-scrollbars",
                                              f"--force-device-scale-factor={args.scale}"])
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
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

            # Chromium's screencast presents ~28 fps, so sampling a 1x show at 25 fps would hold
            # duplicate frames at irregular intervals -- visible as judder. Play the show at
            # `capture_speed` instead and sample the wall clock that much slower; the GIF still
            # gets `fps` frames per show-second, but each one is a distinct presented frame.
            show_speed = args.speed * args.capture_speed
            if show_speed != 1.0:
                page.evaluate(f"window.__sysov.setSpeed({show_speed})")
            epochs = args.epochs or page.evaluate("window.__sysov.nEpochs")

            cdp = page.context.new_cdp_session(page)
            cast = Screencast(page, cdp)
            # in device pixels: a smaller max would make Chromium downscale away the supersample
            cast.start(int(args.width * args.scale), int(args.height * args.scale))

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

            wall_fps = args.fps * args.capture_speed
            print(f"{len(cast.frames)} presented frames over {t1 - t0:.1f}s "
                  f"({len(cast.frames) / (t1 - t0):.0f}/s); sampling at {wall_fps:g}/s for a "
                  f"{args.fps} fps GIF", file=sys.stderr, flush=True)
            n = resample(cast.frames, t0, t1, wall_fps, clip, args.scale, frame_dir,
                         args.max_frames)
            if n >= args.max_frames:
                print(f"WARNING: hit --max-frames={args.max_frames}; the cycle was cut short "
                      f"(lower --fps, or raise --speed to shorten the show)", file=sys.stderr)
            browser.close()
            return n, epochs
    finally:
        httpd.shutdown()


def stack_coverage(gif: Path, epochs: int) -> list[int]:
    """Per-epoch count of frames whose Evaluation node actually shows the rated pile.

    That pile is the one beat the screencast drops non-deterministically: while it sits (cards
    snapped on, nothing animating) the compositor may present no frames, so the resampler holds a
    pre-pile frame and the pile silently vanishes from an otherwise-fine GIF. We slice the loop into
    its `epochs` equal parts and, in each, count the frames whose bottom-right quadrant (below the
    VLM-critic box) carries real image content — saturated or dark pixels, i.e. rated thumbnails.
    """
    try:
        import numpy as np
    except ImportError:
        return [-1] * max(1, epochs)          # can't verify; caller treats -1 as "unknown, accept"
    if epochs <= 0:
        return [0]
    # Decode with ffmpeg, not PIL's ImageSequence: PIL's GIF frame compositing doesn't reproduce the
    # encoder's disposal, so its frames read blank here.
    vtmp = Path(tempfile.mkdtemp(prefix="sysov_verify_"))
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(gif), str(vtmp / "f%05d.png")],
                       check=True)
        pngs = sorted(vtmp.glob("f*.png"))
        n = len(pngs)
        if not n:
            return [0] * epochs
        counts = [0] * epochs
        for idx, p in enumerate(pngs):
            a = np.asarray(Image.open(p).convert("RGB")).astype(int)
            h, w = a.shape[0], a.shape[1]
            q = a[int(h * 0.54):int(h * 0.965), int(w * 0.60):int(w * 0.985)]
            mx, mn = q.max(2), q.min(2)
            if float((((mx - mn) > 45) | (mx < 200)).mean()) > 0.03:
                counts[min(epochs - 1, idx * epochs // n)] += 1
        return counts
    finally:
        shutil.rmtree(vtmp, ignore_errors=True)


def encode(frame_dir: Path, out: Path, fps: int, colors: int, width: int | None,
           dither: str) -> None:
    """Two-pass palette encode: one global palette keeps the figure's flat background clean.

    The figure is mostly flat fills, and the frames arrive supersampled, so `dither=none` is
    both the cleanest and the smallest choice -- dithering the flat panels only stipples them
    with a crosshatch that shimmers from frame to frame.
    """
    scale = f"scale={width}:-1:flags=lanczos," if width else ""
    palette = frame_dir / "palette.png"
    common = ["-framerate", str(fps), "-i", str(frame_dir / "f%05d.png")]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *common,
                    "-vf", f"{scale}palettegen=max_colors={colors}:stats_mode=diff",
                    str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", *common, "-i", str(palette),
                    "-lavfi", f"{scale}paletteuse=dither={dither}:diff_mode=rectangle",
                    "-loop", "0", str(out)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("system_overview.gif"))
    ap.add_argument("--fps", type=int, default=25,
                    help="GIF frame rate (25 and 20 land on exact GIF centisecond delays)")
    ap.add_argument("--epochs", type=int, default=0,
                    help="epochs to capture (default: all of them = one seamless cycle)")
    ap.add_argument("--max-frames", type=int, default=1000, help="hard cap on emitted frames")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed of the show")
    ap.add_argument("--capture-speed", type=float, default=0.25,
                    help="record the show this much slower than real time, to get a distinct "
                         "presented frame per emitted frame (output timing is unaffected)")
    ap.add_argument("--seek-limit", type=float, default=180.0,
                    help="seconds to wait for an epoch boundary")
    ap.add_argument("--width", type=int, default=1280, help="browser viewport width")
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--scale", type=float, default=2.0,
                    help="device pixel ratio to rasterize at; frames are downscaled to "
                         "--gif-width, so >1 supersamples")
    ap.add_argument("--gif-width", type=int, default=900, help="output width (0 = native)")
    ap.add_argument("--colors", type=int, default=256)
    ap.add_argument("--dither", default="none", help="paletteuse dither (none, bayer, ...)")
    ap.add_argument("--manifest", default="", help="override the player's ?manifest= URL")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--attempts", type=int, default=4,
                    help="re-render up to this many times until the rated pile lands in every "
                         "epoch (the screencast drops that static beat non-deterministically); "
                         "the best attempt is kept if none is clean")
    ap.add_argument("--min-stack-frames", type=int, default=25,
                    help="an epoch passes verification only if at least this many of its frames "
                         "show the rated pile")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the rated-pile check and keep the first render")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists; otherwise the render is written "
                         "beside it as <name>.new.gif and the committed file is left untouched")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH", file=sys.stderr)
        return 1

    # rename-pin: never silently clobber an existing render (e.g. the committed figure). Land beside
    # it and let the user promote it by hand, unless they opt in with --force.
    out = args.out
    if out.exists() and not args.force:
        out = out.with_suffix(".new.gif")
        print(f"{args.out} exists — writing to {out} instead (review it, then move it over "
              f"manually, or pass --force to overwrite).", file=sys.stderr)

    attempts = args.attempts if args.verify else 1
    best_min = -2            # best (over attempts) of the worst-covered epoch
    for attempt in range(1, attempts + 1):
        tmp = Path(tempfile.mkdtemp(prefix="sysov_frames_"))
        cand = out.with_suffix(f".try{attempt}.gif") if attempts > 1 else out
        try:
            n, epochs = capture(args, tmp)
            print(f"{n} frames = {n / args.fps:.1f}s; encoding {cand}", file=sys.stderr)
            encode(tmp, cand, args.fps, args.colors, args.gif_width or None, args.dither)
        finally:
            if args.keep_frames:
                print(f"frames kept in {tmp}", file=sys.stderr)
            else:
                shutil.rmtree(tmp, ignore_errors=True)

        if not args.verify:
            break
        cov = stack_coverage(cand, epochs)
        worst = min(cov)
        print(f"attempt {attempt}/{attempts}: rated-pile frames per epoch = {cov}"
              + (" (numpy missing → not verified)" if worst < 0 else ""), file=sys.stderr)
        if worst > best_min:                        # promote this attempt to the winner
            best_min = worst
            if cand != out:
                shutil.move(str(cand), str(out))
        elif cand != out:
            cand.unlink(missing_ok=True)             # a worse attempt: discard
        if worst < 0 or worst >= args.min_stack_frames:
            print("rated pile present in every epoch ✓" if worst >= 0 else "", file=sys.stderr)
            break
    else:
        print(f"WARNING: no clean render in {attempts} attempts; kept the best "
              f"({best_min} pile-frame(s) in its weakest epoch). Re-run, raise --attempts, or "
              f"eyeball {out}.", file=sys.stderr)

    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
