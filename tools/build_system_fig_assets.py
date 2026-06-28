#!/usr/bin/env python3
"""Build the four panel assets for paper/system_fig/system_overview.tex from an
arbitrary run + featured agent.

Outputs (into ``--out-dir``):

    archive_grid_small.png   white-padded mosaic of the whole archive
    branching_subsample.png  5 subset rows x 5 cols, branched parent in red
    candidate_grid.png       3x5 candidate grid of the agent's gen --gen (default 10;
                             real picks + bred siblings, diverged from the seed)
    rated_examples.png       2x3 grid of rated archive entries with star ratings
    _featured_meta.tex       \\def\\pubTitle / \\def\\parentTitle for the .tex
    manifest.json            featured agent, parent id, ratings source, seeds

Example:
    .venv/bin/python tools/build_system_fig_assets.py \\
        --run-dir sweep_logs/sweep/th2_ag20_model-gemini-3-pro-preview_..._s4 \\
        --agent agent_909 \\
        --ratings-run sweep_logs_bkp/th1_ag20_model-gemini-2.5-pro_..._s0 \\
        --out-dir paper/system_fig

If ``--agent random`` is passed, picks one (deterministic w/ --seed) from the
run's agents/ dir. If ``--ratings-run`` is omitted, falls back to the main run;
if that run also lacks vlm_ratings, the rated-examples panel is skipped.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "archive_animations"))
sys.path.insert(0, str(REPO / "archive_animations" / "agent_life"))

# Pillow's default cap (~89 Mpx) is below some run grids; bump it.
Image.MAX_IMAGE_PIXELS = 400_000_000

from rendering import create_numbered_grid  # noqa: E402
import archive_animations.agent_life.common as C  # noqa: E402
from archive_animations.agent_life.build_assets import build_grid  # noqa: E402


# ---------------------------------------------------------------------------- helpers


def _find_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _ensure_agent_workdir(agent_id: str, run_dir: Path, work_root: Path) -> Path:
    """Extract agents/<id>.zip into work_root/<id>/ if not already there."""
    dest = work_root / agent_id
    if (dest / "populations").exists():
        return dest
    zp = run_dir / "agents" / f"{agent_id}.zip"
    if not zp.exists():
        raise FileNotFoundError(f"agent zip not found: {zp}")
    work_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(work_root)
    return dest


def _list_agent_ids(run_dir: Path) -> list[str]:
    return sorted(p.stem for p in (run_dir / "agents").glob("agent_*.zip"))


def _pick_branching_agent(run_dir: Path, rng: random.Random) -> str:
    """Pick a random agent that actually branched (so the branching panel is real)."""
    work_root = REPO / "archive_animations" / "_work"
    ids = _list_agent_ids(run_dir)
    rng.shuffle(ids)
    for aid in ids:
        # peek inside the zip without extracting
        with zipfile.ZipFile(run_dir / "agents" / f"{aid}.zip", "r") as zf:
            try:
                with zf.open(f"{aid}/logs/branching_selection.json") as h:
                    sel = json.loads(h.read().decode())
            except KeyError:
                continue
        if sel.get("choice") == "branch":
            return aid
    raise RuntimeError(f"no branching agent found in {run_dir}")


# ---------------------------------------------------------------------------- panels


def render_archive_grid(run_dir: Path, out_path: Path, *,
                        agent_dir: Optional[Path] = None,
                        seed: int = 0,
                        side: int = 10,
                        side_px: int = 800) -> dict:
    """Mosaic of an archive subsample as the agent would have seen it at branching time.

    Filters ``archive_metadata.json`` entries to those added strictly before this
    agent's ``branching_selection.timestamp``, then samples ``side*side - 1``
    of them at random plus the branched parent (always included). Cells are
    placed in a fixed random order so re-runs with the same seed are identical;
    if fewer entries are available than the cap, trailing cells stay white.
    """
    cap = side * side
    cell = side_px // side
    canvas = Image.new("RGB", (side_px, side_px), (255, 255, 255))

    meta_path = run_dir / "archive" / "archive_metadata.json"
    meta = json.loads(meta_path.read_text())
    entries_by_id = {e["id"]: e for e in meta["entries"]}

    parent_id = None
    if agent_dir is not None:
        sel_path = agent_dir / "logs" / "branching_selection.json"
        if sel_path.exists():
            sel = json.loads(sel_path.read_text())
            ts = sel.get("timestamp")
            pids = sel.get("selected_entry_ids") or []
            parent_id = pids[0] if pids else None
            if ts:
                eligible = [eid for eid, e in entries_by_id.items()
                            if e.get("added_at") and e["added_at"] < ts]
            else:
                eligible = list(entries_by_id.keys())
        else:
            eligible = list(entries_by_id.keys())
    else:
        eligible = list(entries_by_id.keys())

    rng = random.Random(seed)
    others = [eid for eid in eligible if eid != parent_id]
    rng.shuffle(others)
    picks = others[: cap - (1 if parent_id else 0)]
    if parent_id and parent_id in entries_by_id:
        picks.append(parent_id)
    rng.shuffle(picks)

    imgs_dir = run_dir / "archive" / "images"
    for k, eid in enumerate(picks):
        r, c = divmod(k, side)
        try:
            thumb = Image.open(imgs_dir / f"{eid}.png").convert("RGB")
        except Exception:
            continue
        thumb = thumb.resize((cell, cell), Image.LANCZOS)
        canvas.paste(thumb, (c * cell, r * cell))

    canvas.save(out_path)
    return {"n_eligible": len(eligible),
            "n_placed": len(picks),
            "cap": cap,
            "grid_side": side,
            "cell_px": cell,
            "parent_id": parent_id,
            "parent_included": parent_id in picks if parent_id else False}


def render_branching_subsample(agent_dir: Path, run_dir: Path, out_path: Path,
                                seed: int, cell_px: int = 128,
                                samples_per_row: int = 5) -> dict:
    """Pure 5-row x N-col thumbnail grid (no labels, no margin).

    The branched parent (branching_selection.selected_images[0]) is force-included
    in its native subset's row and highlighted with a red border. Row labels are
    NOT drawn here -- they're overlaid in TikZ so they render in LaTeX font.
    """
    snap_path = agent_dir / "logs" / "branching_snapshot.json"
    sel_path = agent_dir / "logs" / "branching_selection.json"
    snap = json.loads(snap_path.read_text())
    sel = json.loads(sel_path.read_text())
    parent_idx = (sel.get("selected_images") or [None])[0]
    parent_entry_id = (sel.get("selected_entry_ids") or [None])[0]

    archive_imgs_dir = run_dir / "archive" / "images"
    rng = random.Random(seed)

    rows = snap["subset_ranges"]
    n_rows = len(rows)
    W = samples_per_row * cell_px
    H = n_rows * cell_px
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Row labels emitted as-is into \branchingRowLabels (no escaping); they are
    # consumed inside a TikZ node with align=right, so LaTeX `\\` produces a
    # line break.
    short_labels = {
        "top_rated": "top-rated", "best_new": "best-new",
        "most_branched": "most-\\\\branched", "newest": "latest", "random": "random",
    }

    parent_row_idx = None
    parent_col_idx = None
    fill = []   # number of (non-empty) thumbnails actually placed in each row
    for ri, sub in enumerate(rows):
        start, end = sub["start"], sub["end"] + 1
        idxs = list(range(start, end))
        if parent_idx is not None and start <= parent_idx < end:
            parent_row_idx = ri
            others = [i for i in idxs if i != parent_idx]
            rng.shuffle(others)
            pick = [parent_idx] + others[: samples_per_row - 1]
            rng.shuffle(pick)
            parent_col_idx = pick.index(parent_idx)
        else:
            rng.shuffle(idxs)
            pick = idxs[:samples_per_row]
        fill.append(len(pick))

        for ci, snap_idx in enumerate(pick):
            eid = snap["entries"][snap_idx]["id"]
            thumb = Image.open(archive_imgs_dir / f"{eid}.png").convert("RGB").resize(
                (cell_px, cell_px), Image.LANCZOS)
            x = ci * cell_px
            y = ri * cell_px
            canvas.paste(thumb, (x, y))
            if snap_idx == parent_idx:
                draw.rectangle([x + 1, y + 1, x + cell_px - 2, y + cell_px - 2],
                               outline=(220, 30, 30), width=4)

    canvas.save(out_path)
    return {"parent_snapshot_index": parent_idx,
            "parent_entry_id": parent_entry_id,
            "parent_row": parent_row_idx,
            "parent_col": parent_col_idx,
            "samples_per_row": samples_per_row,
            "n_rows": n_rows,
            "fill": fill,
            "row_labels": [short_labels.get(r["key"], r["key"]) for r in rows]}


def render_candidate_grid(agent_dir: Path, run_dir: Path, out_path: Path,
                          seed: int, gen: int = 10, thumb_px: int = 140) -> dict:
    """Reconstructed candidate grid for generation ``gen`` (real picks + bred siblings).

    Defaults to a *later* generation rather than gen 0: by gen 0 the offspring are
    bred directly off the branched seed and look nearly identical to it, which
    makes the figure's candidate grid read as a copy of the branched image. A
    mid/late generation has visibly diverged from the seed while still cohering
    as a population the VLM selects from.
    """
    config = C.build_config()
    hist = C.load_selection_history(agent_dir)
    color_enabled = C.detect_color(agent_dir)
    variant = "color" if color_enabled else "gray"

    # branched_parents only seed gen 0's initial population; later gens breed
    # from the previous generation's selections, so skip the (expensive) load.
    branched_parents = []
    if gen == 0:
        branching_sel = C.load_branching(agent_dir)
        if branching_sel and branching_sel.get("choice") == "branch":
            archive_dir = run_dir / "archive"
            work_root = REPO / "archive_animations" / "_work"
            for eid in (branching_sel.get("selected_entry_ids") or []):
                gn = C.load_archive_genome(eid, archive_dir, run_dir, work_root)
                if gn is not None:
                    branched_parents.append(gn)

    grid, sel_idx = build_grid(agent_dir, gen, hist, config, thumb_px,
                                color_enabled, variant, seed,
                                branched_parents=branched_parents or None,
                                bg=(255, 255, 255, 255), draw_numbers=False)
    grid.save(out_path)
    return {"gen": gen,
            "selected_indices": sel_idx,
            "n_branched_parents": len(branched_parents),
            "color_enabled": color_enabled}


def render_rated_examples(ratings_run: Path, out_dir: Path, seed: int,
                          thumb_px: int = 256,
                          cutoff_added_at: Optional[str] = None,
                          n_cells: int = 6) -> Optional[dict]:
    """Emit ``n_cells`` thumbnail files (rated_thumb_0..N-1.png) spanning the
    rating range max->least, drawn from the archive as it stood around the
    featured agent's time.

    "Relevant round" is approximated, because per-round rating logs aren't
    persisted by the collab pipeline (only the cumulative ``vlm_ratings`` list
    per entry survives). We therefore:
      * restrict to entries that already existed at the featured agent's archive
        position (``cutoff_added_at``, mapped by archive fraction in main), so
        the examples are contemporaneous and not images the agent never saw; and
      * score each entry by its *most-recent raw rating* (last element of
        ``vlm_ratings``) -- what a rater actually said in its latest pass, not a
        cumulative average.

    Cells read 5*, 4*, 3*, 2*, 1* (max->least) plus one bonus drawn from the
    densest bucket; an empty star bucket falls back to the nearest populated
    score so all cells fill and the spread still spans the range.
    """
    meta = json.loads((ratings_run / "archive" / "archive_metadata.json").read_text())
    rated = []
    for e in meta["entries"]:
        rs = e.get("vlm_ratings") or []
        if not rs:
            continue
        if cutoff_added_at and e.get("added_at") and e["added_at"] > cutoff_added_at:
            continue
        rated.append((e, float(rs[-1]), len(rs)))
    if not rated:
        return None

    buckets: dict[int, list] = defaultdict(list)
    for e, last, n in rated:
        r = min(5, max(1, int(round(last))))
        buckets[r].append((e, last, n))
    rng = random.Random(seed)
    for k in buckets:
        rng.shuffle(buckets[k])  # deterministic-random tie-break within a bucket

    used_ids: set[str] = set()

    def take(star: int):
        for e, last, n in buckets.get(star, []):
            if e["id"] not in used_ids:
                used_ids.add(e["id"])
                return (star, e)
        return None

    cells: list = []
    for star in (5, 4, 3, 2, 1):
        pick = take(star)
        if pick is None:  # nearest populated bucket
            for d in range(1, 5):
                for s2 in (star - d, star + d):
                    if 1 <= s2 <= 5 and pick is None:
                        pick = take(s2)
                if pick is not None:
                    break
        cells.append(pick)
    while len(cells) < n_cells:  # bonus cell(s) from the densest remaining bucket
        pick = None
        for s in sorted(buckets, key=lambda k: -len(buckets[k])):
            pick = take(s)
            if pick is not None:
                break
        cells.append(pick)
        if pick is None:
            break
    cells = cells[:n_cells]
    # read left->right, top->bottom in descending rating (bonus slots merge in,
    # e.g. 5,4,4,3,2,1 rather than the bonus trailing after the 1-star).
    cells.sort(key=lambda it: (it is not None, it[0] if it else 0), reverse=True)

    imgs_dir = ratings_run / "archive" / "images"
    picked = []
    for i, item in enumerate(cells):
        out = out_dir / f"rated_thumb_{i}.png"
        if item is None:
            # blank placeholder so TikZ can still \includegraphics it
            Image.new("RGB", (thumb_px, thumb_px), (255, 255, 255)).save(out)
            picked.append(None)
            continue
        star, e = item
        thumb = Image.open(imgs_dir / f"{e['id']}.png").convert("RGB").resize(
            (thumb_px, thumb_px), Image.LANCZOS)
        thumb.save(out)
        picked.append({"id": e["id"], "title": e.get("title") or e["id"],
                       "rating": star})
    return {"picked": picked, "cutoff_added_at": cutoff_added_at,
            "n_eligible": len(rated)}


# ---------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="run directory containing agents/ and archive/")
    ap.add_argument("--agent", default="random",
                    help="agent id (e.g. agent_909) or 'random'")
    ap.add_argument("--ratings-run", type=Path, default=None,
                    help="separate run providing vlm_ratings (default: --run-dir)")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "paper" / "system_fig")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen", type=int, default=10,
                    help="generation to show in the candidate grid (default 10; "
                         "later than 0 so the grid has diverged from the branched seed)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = REPO / "archive_animations" / "_work"
    rng = random.Random(args.seed)

    agent_id = args.agent
    if agent_id == "random":
        agent_id = _pick_branching_agent(run_dir, rng)
        print(f"[random] picked {agent_id}")
    agent_dir = _ensure_agent_workdir(agent_id, run_dir, work_root)

    pub = C.load_publication(agent_dir)
    pub_title = (pub or {}).get("title") or agent_id

    # The agent's own published image -- shown below the "VLM publication" node
    # in the figure. Looked up from the archive by agent_id so it works even when
    # the publication log lacks the entry id.
    pub_entry_id = None
    try:
        run_entries = json.loads(
            (run_dir / "archive" / "archive_metadata.json").read_text())["entries"]
        pub_entry_id = next((e["id"] for e in run_entries
                             if e.get("agent_id") == agent_id), None)
    except Exception:  # pylint: disable=broad-except
        pass
    if pub_entry_id:
        src = run_dir / "archive" / "images" / f"{pub_entry_id}.png"
        if src.exists():
            Image.open(src).convert("RGB").save(out_dir / "publication.png")
            print(f"[publication] {pub_entry_id} ({pub_title!r}) -> publication.png")

    print(f"[archive] mosaic from {run_dir}/archive/images")
    arch_info = render_archive_grid(run_dir, out_dir / "archive_grid_small.png",
                                     agent_dir=agent_dir, seed=args.seed)

    print(f"[branching] subsample for {agent_id}")
    branch_info = render_branching_subsample(
        agent_dir, run_dir, out_dir / "branching_subsample.png", seed=args.seed)

    print(f"[grid] candidate grid (gen {args.gen}) for {agent_id}")
    gen0_info = render_candidate_grid(agent_dir, run_dir,
                                      out_dir / "candidate_grid.png",
                                      seed=args.seed, gen=args.gen)

    ratings_run = (args.ratings_run or run_dir).resolve()
    # Map the featured agent's archive position onto a contemporaneous cutoff in
    # the ratings run. The featured agent may live in a different run than the
    # one carrying vlm_ratings, so we align by archive *fraction* rather than
    # wall-clock (the two sweeps don't share a timeline).
    cutoff_added_at = None
    try:
        run_entries = json.loads(
            (run_dir / "archive" / "archive_metadata.json").read_text())["entries"]
        pub_idx = next((i for i, e in enumerate(run_entries)
                        if e.get("agent_id") == agent_id), None)
        frac = 1.0 if pub_idx is None else (pub_idx + 1) / len(run_entries)
        rate_added = sorted(
            e["added_at"] for e in json.loads(
                (ratings_run / "archive" / "archive_metadata.json").read_text())["entries"]
            if e.get("added_at"))
        if rate_added:
            ci = min(len(rate_added) - 1, max(0, round(frac * len(rate_added)) - 1))
            cutoff_added_at = rate_added[ci]
            print(f"[rated] featured frac={frac:.3f} -> ratings cutoff {cutoff_added_at}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  ! could not compute ratings cutoff ({exc}); using full archive")

    print(f"[rated] examples from {ratings_run}")
    rated_info = render_rated_examples(ratings_run, out_dir, seed=args.seed,
                                       cutoff_added_at=cutoff_added_at)
    if rated_info is None:
        print(f"  ! no vlm_ratings in {ratings_run}; rated_thumb_*.png NOT written")

    parent_title = None
    if branch_info["parent_entry_id"]:
        meta = json.loads((run_dir / "archive" / "archive_metadata.json").read_text())
        for e in meta["entries"]:
            if e["id"] == branch_info["parent_entry_id"]:
                parent_title = e.get("title")
                break

    # Macros consumed by paper/system_fig/system_overview.tex
    def _esc(s: str) -> str:
        if s is None:
            return ""
        return (s.replace("\\", "\\textbackslash{}")
                 .replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
                 .replace("#", "\\#").replace("_", "\\_")
                 .replace("{", "\\{").replace("}", "\\}"))

    lines = [
        "% auto-generated by tools/build_system_fig_assets.py",
        f"\\def\\pubTitle{{{_esc(pub_title)}}}",
        f"\\def\\parentTitle{{{_esc(parent_title) if parent_title else ''}}}",
        f"\\def\\featuredAgent{{{_esc(agent_id)}}}",
        # Sampling row labels, in row order (top -> bottom).
        # Used via: \foreach \i/\lbl in \branchingRowLabels {...}
        # Each label is brace-wrapped so commas/backslashes inside it don't
        # confuse the TikZ \foreach parser; labels are not escaped because we
        # control their content (e.g. "most-\\branched" carries a TeX `\\`).
        "\\def\\branchingRowLabels{"
        + ", ".join(f"{i}/{{{lab}}}" for i, lab in enumerate(branch_info["row_labels"]))
        + "}",
        f"\\def\\branchingRows{{{branch_info['n_rows']}}}",
        f"\\def\\branchingCols{{{branch_info['samples_per_row']}}}",
    ]
    if rated_info:
        # Rated cells: row-major (top row, then bottom row), 3 cols each.
        # \ratedCells expanded as \foreach \i/\title/\rating in \ratedCells
        cells = rated_info["picked"]
        parts = []
        for i, c in enumerate(cells):
            if c is None:
                parts.append(f"{i}//0")
            else:
                parts.append(f"{i}/{_esc(c['title'])}/{c['rating']}")
        lines.append("\\def\\ratedCells{" + ", ".join(parts) + "}")
    (out_dir / "_featured_meta.tex").write_text("\n".join(lines) + "\n")

    manifest = {
        "run_dir": str(run_dir),
        "ratings_run": str(ratings_run),
        "agent": agent_id,
        "publication_title": pub_title,
        "parent_title": parent_title,
        "seed": args.seed,
        "archive": arch_info,
        "branching": branch_info,
        "gen0": gen0_info,
        "rated_examples": rated_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote: {out_dir}")
    expected = ["archive_grid_small.png", "branching_subsample.png",
                "candidate_grid.png", "_featured_meta.tex", "manifest.json"]
    if rated_info:
        expected += [f"rated_thumb_{i}.png" for i in range(6)]
    for k in expected:
        p = out_dir / k
        print(f"  {k:30s} {'OK' if p.exists() else 'MISSING'}")
    print(f"\npub: {pub_title!r}   parent: {parent_title!r}   agent: {agent_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
