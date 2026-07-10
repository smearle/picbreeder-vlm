#!/usr/bin/env python3
"""Export per-MODEL publication CARDS for the blog breed/ browse pages.

browse.html?model=<id> is the "By <model>" link on every AI-bred image's detail
page. It used to filter the bundled archive.json, which only ever holds the
canonical run (always gemini-2.5-pro) plus the hero banner picks — so every other
VLM's gallery was however many banner images it happened to own (one, for
gemini-2.5-flash-lite). This builds the real thing: for each model, the best-rated
images it published across all of its canonical, sprite-sheeted runs.

Same shape and lazy-loading contract as user_cards.json — a cell renders from its
card (title/rating/agent) plus the run's HF sprite sheet, and the genome is fetched
only when the visitor clicks Evolve / DNA. Output: breed/data/model_cards.json

  { "runs":   [<full run name>, ...],                      # de-duped run table
    "models": { "<model id>": { "n_images": N, "n_rated": R, "n_runs": K,
                                "cards": [[runIdx, idNum, title,
                                          rating, nrat, agent, color, children], ...] } } }

idNum is the integer N from img_{N:06d}; sprite linear index is N-1. Cards are
best-rated first and capped at CARD_CAP; ties break by rank-within-run, which
interleaves the seeds/conditions rather than letting one run's 5.0s fill the page.
Unrated images (rating null) sort last but are kept — the qwen3-vl-30b runs were
never put through a community-rating pass, and dropping them would leave that model
with no gallery at all. browse.html?run= shows unrated cells the same way.

Reads only JSON (no NEAT / rendering deps), so it re-runs cheaply whenever new
runs land. build_breed_data.py performs the same build inline.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
OUT = BLOG / "breed" / "data"
SWEEP_DIR = REPO / "sweep_logs" / "sweep"
MIRROR = REPO / "archive_animations" / "_archive_mirror" / "index.json"
CARD_CAP = 400          # images listed per model (keep in sync with the browse subhead)


def agent_num(agent_id: str) -> str:
    m = re.search(r"(\d+)", agent_id or "")
    return str(int(m.group(1))) if m else (agent_id or "?")


def mean_rating(entry):
    rs = [r for r in (entry.get("vlm_ratings") or []) if isinstance(r, (int, float))]
    return (sum(rs) / len(rs), len(rs)) if rs else (None, 0)


def model_of(entry: dict) -> str:
    """The VLM that bred a run, from the archive index entry."""
    return (entry.get("config") or {}).get("model") or entry.get("model") or "gemini-2.5-pro"


def browseable_runs() -> list[tuple[str, str]]:
    """(run, model) for every canonical run with a published sprite sheet and local
    metadata — the same set browse.html?run= can already open. Human Picbreeder has
    no VLM breeder, so it is not a model gallery."""
    idx = json.load(open(MIRROR))
    out = []
    for e in idx.get("runs", []):
        if e.get("arc") == "human" or not e.get("canonical"):
            continue
        if not (e.get("has") or {}).get("sprite"):
            continue
        if not (SWEEP_DIR / e["run"] / "archive" / "archive_metadata.json").is_file():
            continue
        out.append((e["run"], model_of(e)))
    return sorted(out)


def build(verbose: bool = True) -> dict:
    run_table: list[str] = []
    run_idx: dict[str, int] = {}

    def run_i(run: str) -> int:
        if run not in run_idx:
            run_idx[run] = len(run_table)
            run_table.append(run)
        return run_idx[run]

    # model -> (sort key, card); n_images counts EVERY published image, not just the
    # capped cards, so the gallery can say "the 400 best-rated of 3,017". n_rated is
    # what makes "best-rated" true: a model whose runs never saw a rating pass (all of
    # qwen3-vl-30b) is listed in run/publication order and must not claim otherwise.
    pool: dict[str, list] = {}
    totals: dict[str, list] = {}     # model -> [n_images, n_rated, n_runs]
    for run, model in browseable_runs():
        meta = json.load(open(SWEEP_DIR / run / "archive" / "archive_metadata.json"))
        entries = meta.get("entries", [])
        t = totals.setdefault(model, [0, 0, 0])
        t[0] += len(entries)
        t[2] += 1
        ranked = sorted(((mean_rating(e)[0], e) for e in entries),
                        key=lambda x: (x[0] is None, -(x[0] or 0.0)))
        ri = run_i(run)
        for rank, (r, e) in enumerate(ranked):
            card = [
                ri,
                int(e["id"].split("_")[-1]),                # img_000940 -> 940
                e.get("title") or "Untitled",
                None if r is None else round(r, 2),
                len(e.get("vlm_ratings") or []),
                agent_num(e.get("agent_id", "")),
                0 if e.get("color_enabled") is False else 1,
                int(e.get("n_published_children") or 0),
            ]
            pool.setdefault(model, []).append(((r is None, -(r or 0.0), rank, ri), card))
        n_rated = sum(1 for r, _e in ranked if r is not None)
        t[1] += n_rated
        if verbose:
            print(f"  {model:<22} {run}  ({n_rated}/{len(entries)} rated)")

    models = {}
    for model, cards in pool.items():
        cards.sort(key=lambda p: p[0])
        n_images, n_rated, n_runs = totals[model]
        models[model] = {"n_images": n_images, "n_rated": n_rated, "n_runs": n_runs,
                         "cards": [c for _k, c in cards[:CARD_CAP]]}
    return {"runs": run_table, "models": models}


if __name__ == "__main__":
    data = build()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "model_cards.json"
    json.dump(data, open(out_path, "w"), separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"[done] {out_path}  ({len(data['runs'])} runs, {len(data['models'])} models, "
          f"{sum(len(m['cards']) for m in data['models'].values())} cards, {size_kb:.0f} KB)")
    for m, d in sorted(data["models"].items(), key=lambda kv: -kv[1]["n_images"]):
        print(f"        {m:<22} {len(d['cards']):>4} cards of {d['n_images']:>6} images "
              f"across {d['n_runs']:>2} runs ({d['n_rated']} rated)")
