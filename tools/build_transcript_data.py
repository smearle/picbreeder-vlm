#!/usr/bin/env python3
"""Export per-agent *transcript* data for the blog "everything every agent saw and
said" newsreel viewer (transcripts/index.html).

For each agent in a run we reconstruct, in chronological order, what the agent SAW
(the archive branching sample, then each generation's 3x5 candidate grid) and what
it SAID (its branching rationale + per-generation rationale + the cells it picked),
through to publication. The reconstruction reuses the proven helpers in
``archive_animations/agent_life`` (``build_grid`` / ``build_branching`` /
``common.py``): the agent zips only saved the *selected* genomes per generation, so
real picks are dropped into their recorded grid slots and the remaining cells are
bred as plausible siblings using the recorded mutation mode/strength + a fixed seed.
Those non-selected cells are flagged ``reconstructed`` so the viewer can mark them
(and the viewer shows a global disclaimer).

Per agent we emit, under ``assets/transcripts/<run-key>/<agent_NNN>/``:

  atlas.webp        a single sprite sheet stacking every block top-to-bottom: the
                    branching grid (if any) then each generation's candidate grid.
                    The page makes one image request per agent, not one per gen.
  transcript.json   { agent, run, title, color, grid_rows, grid_cols, publication,
                      atlas:{file,w,h},
                      timeline:[ {kind:"branching"|"gen"|"publication", ...,
                                  block:{x,y,w,h,rows,cols,cell,margin}} ] }
                    Each block's geometry lets the viewer locate cell i at
                    (x + margin + col*(cell+margin), y + margin + row*(cell+margin)).

Per run we emit ``assets/transcripts/<run-key>/index.json`` listing the built
agents (id, title, n_gen, published) plus the parsed arc/seed/model.

Usage:
    .venv/bin/python tools/build_transcript_data.py \
        --run sweep_logs/sweep/th1_ag20_model-gemini-2.5-pro_..._s3 \
        --agent all --max-agents 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "archive_animations" / "agent_life"))
sys.path.insert(0, str(REPO / "archive_animations"))
sys.path.insert(0, str(REPO))

import common as C  # noqa: E402  (archive_animations/agent_life/common.py)
from build_assets import build_grid, build_branching  # noqa: E402
from tools.hf_archive_push import parse_config, canonical_arc  # noqa: E402

BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
GRID_MARGIN = 10    # create_numbered_grid margin used by build_grid (hard-coded there)
BRANCH_MARGIN = 4   # create_numbered_grid margin used by build_branching


def _grid_dims(rows: int, cols: int, cell: int, margin: int) -> tuple[int, int]:
    """Exact pixel size of a create_numbered_grid canvas (see rendering.py)."""
    return (cols * cell + (cols + 1) * margin, rows * cell + (rows + 1) * margin)


def build_agent(agent_dir: Path, run: Path, run_dir: Path, work_root: Path,
                grid_thumb: int, branch_thumb: int, seed: int) -> dict | None:
    """Reconstruct one agent's full transcript -> (timeline, atlas image). Returns a
    payload dict ready to serialize, or None if the agent has no generations."""
    n_gen = C.num_generations(agent_dir)
    if n_gen == 0:
        return None

    config = C.build_config()
    hist = C.load_selection_history(agent_dir)
    pub = C.load_publication(agent_dir)
    color = C.detect_color(agent_dir)
    variant = "auto"
    title = (pub.get("title") if pub else None) or agent_dir.name
    pub_gen = pub.get("generation") if pub else None
    archive_dir = run / "archive"

    # Resolve branched parent genomes so gen-0 fillers mutate off the branched seed.
    branching_sel = C.load_branching(agent_dir)
    branched_parents = []
    if branching_sel and branching_sel.get("choice") == "branch":
        for eid in (branching_sel.get("selected_entry_ids") or []):
            gn = C.load_archive_genome(eid, archive_dir if archive_dir.exists() else None,
                                       run_dir, work_root)
            if gn is not None:
                branched_parents.append(gn)

    blocks: list[Image.Image] = []   # block image, in atlas (= timeline) order
    timeline: list[dict] = []

    # ---- branching block (the archive sample shown before gen 0) ----
    tmp_branch = Path(tempfile.mkdtemp(prefix="tx_branch_"))
    try:
        (tmp_branch / "grids").mkdir(parents=True, exist_ok=True)
        bblock = build_branching(agent_dir, tmp_branch,
                                 archive_dir if archive_dir.exists() else None,
                                 skip_existing=False, branch_thumb=branch_thumb)
        if bblock.get("present"):
            entry = {
                "kind": "branching",
                "rationale": bblock.get("rationale", ""),
                "choice": bblock.get("choice"),
                "archive_empty": bool(bblock.get("archive_empty")),
                "selected_indices": bblock.get("selected_indices", []),
            }
            if bblock.get("archive_empty"):
                entry["empty_text"] = bblock.get("empty_text", "")
            if bblock.get("grid"):
                img = Image.open(tmp_branch / bblock["grid"]).convert("RGB")
                rows, cols = bblock["grid_rows"], bblock["grid_cols"]
                entry["block"] = {"rows": rows, "cols": cols, "cell": branch_thumb,
                                  "margin": BRANCH_MARGIN, "w": img.width, "h": img.height}
                blocks.append(img)
                entry["_has_block"] = True
            timeline.append(entry)
    finally:
        shutil.rmtree(tmp_branch, ignore_errors=True)

    # ---- per-generation candidate grids ----
    for g in range(n_gen):
        row = hist[g] if g < len(hist) else {}
        grid, sel_idx = build_grid(agent_dir, g, hist, config, grid_thumb, color,
                                   variant, seed,
                                   branched_parents=branched_parents if g == 0 else None,
                                   bg=(255, 255, 255, 255),
                                   draw_numbers=False, draw_selection=False)
        reconstructed = [i not in set(sel_idx) for i in range(C.POP_SIZE)]
        w, h = grid.size
        entry = {
            "kind": "gen",
            "generation": g,
            "rationale": (row.get("rationale") or "").strip(),
            "selected": sel_idx,
            "select_k": len(sel_idx),
            "mutation_mode": row.get("mutation_mode", "all"),
            "mutation_strength": row.get("mutation_strength", 0.5),
            "reconstructed": reconstructed,
            "block": {"rows": C.GRID_ROWS, "cols": C.GRID_COLS, "cell": grid_thumb,
                      "margin": GRID_MARGIN, "w": w, "h": h},
            "_has_block": True,
        }
        blocks.append(grid)
        timeline.append(entry)
        if pub_gen is not None and g == pub_gen:
            timeline.append({"kind": "publication", "generation": g,
                             "title": title, "reason": (pub or {}).get("reason", "")})

    # ---- compose vertical atlas, recording each block's y-offset ----
    gap = 6
    atlas_w = max((b.width for b in blocks), default=1)
    atlas_h = sum(b.height for b in blocks) + gap * max(0, len(blocks) - 1)
    atlas = Image.new("RGB", (atlas_w, max(1, atlas_h)), (255, 255, 255))
    y = 0
    bi = 0
    for entry in timeline:
        if not entry.pop("_has_block", False):
            continue
        b = blocks[bi]; bi += 1
        atlas.paste(b, (0, y))
        entry["block"]["x"] = 0
        entry["block"]["y"] = y
        y += b.height + gap

    return {
        "agent": agent_dir.name,
        "title": title,
        "color": color,
        "grid_rows": C.GRID_ROWS,
        "grid_cols": C.GRID_COLS,
        "n_gen": n_gen,
        "published": pub_gen is not None,
        "publication": ({"generation": pub_gen, "title": title,
                         "reason": (pub or {}).get("reason", "")} if pub else None),
        "timeline": timeline,
        "_atlas": atlas,
    }


def _build_one(task):
    """Worker: extract one agent zip into its own temp dir, reconstruct + write
    atlas.webp + transcript.json. Returns (status, built_entry_or_None)."""
    (zp_str, run_str, out_root_str, grid_thumb, branch_thumb, seed, skip_existing) = task
    zp = Path(zp_str); run = Path(run_str); out_root = Path(out_root_str)
    agent_id = zp.stem
    agent_out = out_root / agent_id
    if skip_existing and (agent_out / "transcript.json").exists():
        try:
            meta = json.loads((agent_out / "transcript.json").read_text())
            return ("skip", {"id": agent_id, "title": meta.get("title"),
                             "n_gen": meta.get("n_gen"), "published": meta.get("published")})
        except Exception:
            pass
    work_root = Path(tempfile.mkdtemp(prefix="tx_w_"))
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(work_root)
        payload = build_agent(work_root / agent_id, run, run, work_root,
                              grid_thumb, branch_thumb, seed)
        if payload is None:
            return ("empty", None)
        atlas = payload.pop("_atlas")
        agent_out.mkdir(parents=True, exist_ok=True)
        atlas.save(agent_out / "atlas.webp", method=6, quality=82)
        payload["run"] = run.name
        payload["atlas"] = {"file": "atlas.webp", "w": atlas.width, "h": atlas.height}
        (agent_out / "transcript.json").write_text(json.dumps(payload))
        return ("ok", {"id": agent_id, "title": payload["title"],
                       "n_gen": payload["n_gen"], "published": payload["published"]})
    except Exception as e:  # skip-and-log a broken agent, keep the run going
        return ("err:%s: %s" % (type(e).__name__, e), None)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def main() -> None:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="sweep run dir (has agents/ + archive/)")
    ap.add_argument("--agent", default="all", help="'all' or a single agent id (e.g. agent_005)")
    ap.add_argument("--max-agents", type=int, default=20,
                    help="when --agent all, build the first N agents (the parallel-column wave)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--grid-thumb", type=int, default=96)
    ap.add_argument("--branch-thumb", type=int, default=44)
    ap.add_argument("--seed", type=int, default=0, help="must match agent_life default for stable grids")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <blog>/assets/transcripts/<run-key>)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip agents whose transcript.json already exists")
    ap.add_argument("--total-agents", type=int, default=0,
                    help="the run's true agent count (else agents/.total_agents, else local zips)")
    args = ap.parse_args()

    run = args.run
    agents_dir = run / "agents"
    if not agents_dir.exists():
        raise SystemExit(f"no agents/ dir in {run}")

    run_key = run.name
    out_root = args.out or (BLOG / "assets" / "transcripts" / run_key)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.agent == "all":
        zips = sorted(agents_dir.glob("agent_*.zip"))[: args.max_agents]
    else:
        zp = agents_dir / f"{args.agent}.zip"
        if not zp.exists():
            raise SystemExit(f"missing {zp}")
        zips = [zp]

    cfg = parse_config(run_key)
    arc = canonical_arc(run_key, cfg)
    tasks = [(str(zp), str(run), str(out_root), args.grid_thumb, args.branch_thumb,
              args.seed, args.skip_existing) for zp in zips]
    built = []

    def handle(status, entry, label):
        if entry:
            built.append(entry)
        tag = status if status in ("ok", "skip", "empty") else status
        print(f"  {label}: {tag}" + (f" | {entry['title']!r}" if entry and status == 'ok' else ""))

    jobs = max(1, min(args.jobs, len(tasks) or 1))
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_build_one, t): Path(t[0]).stem for t in tasks}
            done = 0
            for fut in as_completed(futs):
                status, entry = fut.result()
                done += 1
                handle(status, entry, f"[{done}/{len(tasks)}] {futs[fut]}")
    else:
        for t in tasks:
            status, entry = _build_one(t)
            handle(status, entry, Path(t[0]).stem)

    built.sort(key=lambda e: e["id"])  # as_completed is unordered; keep chronological
    # When only the first few zips were fetched (tools/fetch_agent_zips.py), the
    # local count understates the run; .total_agents carries the real one.
    total_file = agents_dir / ".total_agents"
    if args.total_agents:
        total_agents = args.total_agents
    elif total_file.exists():
        total_agents = int(total_file.read_text().strip())
    else:
        total_agents = len(list(agents_dir.glob("agent_*.zip")))
    index = {
        "run": run_key, "arc": arc, "seed": cfg.get("seed"), "model": cfg.get("model"),
        "grid_rows": C.GRID_ROWS, "grid_cols": C.GRID_COLS,
        "n_agents": len(built), "total_agents": total_agents, "agents": built,
    }
    (out_root / "index.json").write_text(json.dumps(index))
    print(f"\nwrote {out_root}/  ({len(built)} agents, arc={arc}, model={cfg.get('model')})")


if __name__ == "__main__":
    main()
