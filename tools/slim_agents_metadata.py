#!/usr/bin/env python3
"""Shrink existing `agents_metadata.json` files in place.

Each agent record embeds its branching decision, and that decision used to carry
`archive_elite_names` (the entire archive's titles as of the agent's branch — an
O(archive size) snapshot, never read back) plus a fat `input_parts` list (the
candidate display, with image_path/title/caption/ratings duplicated from the
archive). Summed over agents that made the orchestrator file grow as
O(n_agents^2): ~170 MB for a 2000-agent run.

This rewrites each record's branching decision to keep only what isn't
reconstructable from the final archive: the archive-sample *membership, order
and subset* the agent saw, plus its selection. The runner rebuilds image_path /
caption from `archive_index` on resume (see agent_runner.slim_branching_decision
and `_set_archive_sample_reference`); no analysis/animation tool reads the
dropped fields. Self-contained (stdlib only) so it can run on the cluster.

Usage:
  python tools/slim_agents_metadata.py PATH [PATH ...]      # dry-run (report only)
  python tools/slim_agents_metadata.py --apply PATH [...]   # rewrite in place
PATH may be an agents_metadata.json, a run dir, or a dir to search recursively.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# Mirror of agent_runner._BRANCH_INPUT_PART_KEEP / slim_branching_decision so this
# stays runnable without importing the (heavy) project package on the cluster.
_KEEP = ("index", "archive_index", "archive_sample_index",
         "subset", "subset_label", "kind", "leading_index")


def slim_decision(decision):
    if not isinstance(decision, dict):
        return decision
    slim = dict(decision)
    slim.pop("archive_elite_names", None)
    parts = slim.get("input_parts")
    if isinstance(parts, list):
        slim["input_parts"] = [
            {k: p[k] for k in _KEEP if k in p}
            for p in parts if isinstance(p, dict)
        ]
    return slim


def _membership(decision):
    """Provenance we must preserve: ordered (archive_index, subset) + selection."""
    parts = (decision or {}).get("input_parts") or []
    mem = [(p.get("archive_index"), p.get("subset_label"))
           for p in parts if isinstance(p, dict) and p.get("archive_index") is not None]
    sel = ((decision or {}).get("choice"),
           tuple((decision or {}).get("selected_entry_ids") or []),
           tuple((decision or {}).get("selected_images") or []))
    return mem, sel


def find_files(paths):
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            direct = os.path.join(p, "agents_metadata.json")
            if os.path.isfile(direct):
                out.append(direct)
            else:
                for root, _dirs, files in os.walk(p):
                    if "agents_metadata.json" in files:
                        out.append(os.path.join(root, "agents_metadata.json"))
    return sorted(set(out))


def process(path, apply):
    before = os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    agents = meta.get("agents") or {}
    n_changed = 0
    for rec in agents.values():
        if not isinstance(rec, dict):
            continue
        bd = rec.get("branching_decision")
        if not bd:
            continue
        old_mem, old_sel = _membership(bd)
        slim = slim_decision(bd)
        new_mem, new_sel = _membership(slim)
        # Hard invariant: never silently lose provenance.
        if (new_mem, new_sel) != (old_mem, old_sel):
            raise RuntimeError(f"{path}: membership/selection changed for an agent — aborting")
        rec["branching_decision"] = slim
        n_changed += 1

    after_bytes = len(json.dumps(meta, separators=(",", ":")))
    pct = 100 * after_bytes / before if before else 100
    print(f"  {path}")
    print(f"    {len(agents)} agents, {n_changed} decisions slimmed; "
          f"{before/1e6:.1f} MB -> ~{after_bytes/1e6:.1f} MB ({pct:.0f}%)")
    if apply:
        d = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, separators=(",", ":"))
        os.replace(tmp, path)
        print(f"    written ({os.path.getsize(path)/1e6:.1f} MB on disk)")
    return before, after_bytes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="files / run dirs / dirs to search")
    ap.add_argument("--apply", action="store_true", help="rewrite in place (default: dry-run)")
    args = ap.parse_args(argv)

    files = find_files(args.paths)
    if not files:
        print("no agents_metadata.json found under given paths", file=sys.stderr)
        return 1
    print(f"{'APPLYING to' if args.apply else 'DRY-RUN over'} {len(files)} file(s):")
    tot_b = tot_a = 0
    for f in files:
        try:
            b, a = process(f, args.apply)
            tot_b += b
            tot_a += a
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"  ERROR {f}: {exc}", file=sys.stderr)
    print(f"total: {tot_b/1e6:.1f} MB -> ~{tot_a/1e6:.1f} MB "
          f"({100*tot_a/tot_b if tot_b else 100:.0f}%)"
          + ("" if args.apply else "   [dry-run; pass --apply to write]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
