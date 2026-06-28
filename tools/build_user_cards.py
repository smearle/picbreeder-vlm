#!/usr/bin/env python3
"""Export per-user publication CARDS for the blog breed/ browse pages.

browse.html?user=<name> needs, for every image a personality published, just
enough to render a gallery cell WITHOUT downloading any genome: title, mean
rating, rater count, agent, colour flag and branch count. The thumbnail itself
comes from the run's HF sprite sheet (site/<run>/sprite/), and the genome is
fetched lazily only when the visitor clicks Evolve / DNA.

This is the same artist/publication attribution build_breed_data.py performs
(traits archive_metadata.json joined with the per-run {agent_id: trait} maps),
but it reads only JSON — no NEAT / rendering deps — so it can be re-run cheaply
whenever new traits runs land. Output: breed/data/user_cards.json

  { "runs":  [<full run name>, ...],                       # de-duped run table
    "users": { "<username>": [[runIdx, idNum, title,
                               rating, nrat, agent, color, children], ...] } }

idNum is the integer N from img_{N:06d}; sprite linear index is N-1. Each 89-char
run name repeats across ~140 of a user's images, so the run table keeps the file
small (it is otherwise dominated by the keys). Mirrors the artist filter
(>= ARTIST_MIN_IMGS rated images) so the user set matches the leaderboard.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
OUT = BLOG / "breed" / "data"
SWEEP_DIR = REPO / "sweep_logs" / "sweep"
TRAITS_META = REPO / "traits_meta_cluster"
USERNAMES = json.load(open(REPO / "traits_usernames.json"))
ARTIST_MIN_IMGS = 50    # keep in sync with build_breed_data.py


def agent_num(agent_id: str) -> str:
    m = re.search(r"(\d+)", agent_id or "")
    return str(int(m.group(1))) if m else (agent_id or "?")


def mean_rating(entry):
    rs = [r for r in (entry.get("vlm_ratings") or []) if isinstance(r, (int, float))]
    return (sum(rs) / len(rs), len(rs)) if rs else (None, 0)


def slugless(trait: str) -> str:
    return USERNAMES.get(trait) or re.sub(r"\W+", "", trait.title())[:18] or "Artist"


# trait -> [card dicts]; one entry per rated published image attributed to the trait
trait_entries: dict[str, list] = {}
n_runs = 0
for amap in sorted(TRAITS_META.glob("*/agent_traits.json")):
    run = amap.parent.name
    arch = SWEEP_DIR / run / "archive" / "archive_metadata.json"
    if not arch.exists():
        print(f"  [skip] no local archive for {run}")
        continue
    n_runs += 1
    agent_to_trait = json.load(open(amap))
    for e in json.load(open(arch))["entries"]:
        trait = agent_to_trait.get(e.get("agent_id"))
        if not trait:
            continue
        r, nr = mean_rating(e)
        if r is None:
            continue
        trait_entries.setdefault(trait, []).append({
            "run": run, "id": e["id"], "rating": r, "nrat": nr,
            "children": int(e.get("n_published_children") or 0),
            "title": e.get("title") or "Untitled",
            "color": bool(e.get("color_enabled", True)),
            "agent": agent_num(e.get("agent_id", "")),
        })

# Assemble per-user cards (rating-sorted, like the gallery), de-duping runs.
run_table: list[str] = []
run_idx: dict[str, int] = {}


def run_i(run: str) -> int:
    if run not in run_idx:
        run_idx[run] = len(run_table)
        run_table.append(run)
    return run_idx[run]


users: dict[str, list] = {}
for trait, ents in trait_entries.items():
    if len(ents) < ARTIST_MIN_IMGS:        # mirror the leaderboard's threshold
        continue
    user = slugless(trait)
    cards = []
    for c in sorted(ents, key=lambda x: -x["rating"]):
        cards.append([
            run_i(c["run"]),
            int(c["id"].split("_")[-1]),                 # img_000940 -> 940
            c["title"],
            round(c["rating"], 2) if c["rating"] is not None else None,
            c["nrat"],
            c["agent"],
            1 if c["color"] else 0,
            c["children"],
        ])
    users[user] = cards

OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "user_cards.json"
json.dump({"runs": run_table, "users": users}, open(out_path, "w"), separators=(",", ":"))
size_kb = out_path.stat().st_size / 1024
print(f"[done] {out_path}  ({n_runs} runs scanned, {len(run_table)} in table, "
      f"{len(users)} users, {sum(len(v) for v in users.values())} cards, {size_kb:.0f} KB)")
