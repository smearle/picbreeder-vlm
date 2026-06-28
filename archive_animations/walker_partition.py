"""Partition a Picbreeder-VLM phylogeny into k equal-length walker tours.

Two algorithms:

* ``build_walks`` (Euler-tour slice) -- a global rating-ordered DFS Euler tour
  of the phylogeny forest, sliced into k equal-length contiguous segments.
  Tiles all edges, equal length, **but each cross-root transition is a jump**
  in the emitted sequence (the virtual root is invisible), so walks straddling
  a root boundary teleport. Use for whole-archive structural visualization.

* ``best_first_walks`` (continuous, recommended) -- k independent walkers, each
  anchored in a distinct connected component, each walking K continuous edges
  (no jumps ever). At every step each walker picks the neighbor maximizing
  ``alpha*rating + beta*global_unvisited_bonus - gamma*walker_retread_count``.
  Beta drives coverage via a shared edge-claim; gamma is the "explore new
  territory" knob. Walks have identical length K by construction.

A "walk" is a list of directed (from_id, to_id) edges; consecutive edges share
their pivot vertex (the walker is at ``edges[i][1] == edges[i+1][0]`` between
steps), so there are no jumps within a walk by construction.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

VIRTUAL_ROOT = "__root__"          # invisible NEAT-style root (not a node)
NEUTRAL_ROOT_ID = "__neutral__"    # synthetic super-root: solid-grey CPPN
Edge = Tuple[str, str]   # (from_id, to_id), directed: walker moves from→to


def load_forest(
    run: Path,
    *,
    synthetic_root: bool = True,
    max_n: Optional[int] = None,
    require_genome: bool = False,
) -> Tuple[List[str], Dict[str, str], Dict[str, int]]:
    """Return (publication_order, parent_map, score_map).

    parent_map[id] = parent_id or VIRTUAL_ROOT.
    score_map[id] = n_published_children (used to sort siblings by rating).
    Only entries with an on-disk image are kept (so the layout/render code can
    match against `archive/images/{id}.png`).

    ``max_n`` (optional) caps the phylogeny to the first N publications by
    ``added_at`` -- useful when later genomes don't have on-disk pickles.
    ``require_genome`` additionally drops entries that lack ``archive/genomes/{id}.pkl``
    (needed for the CPPN morph grid renderer).

    ``synthetic_root=True`` (default) inserts ``NEUTRAL_ROOT_ID`` as the parent
    of every real archive root, turning the 76-tree forest into a single
    connected tree -- so walkers can traverse between lineages by morphing
    through solid grey and the global graph has zero jumps. The ``NEUTRAL_ROOT``
    node has score 0 and is placed first in publication order.
    """
    meta = json.loads((run / "archive" / "archive_metadata.json").read_text())
    entries = meta["entries"] if isinstance(meta, dict) else meta

    def ts(e):
        try:
            return datetime.fromisoformat(e["added_at"])
        except Exception:
            return datetime.min

    entries.sort(key=ts)
    img_dir = run / "archive" / "images"
    entries = [e for e in entries if (img_dir / f"{e['id']}.png").exists()]
    if require_genome:
        gen_dir = run / "archive" / "genomes"
        entries = [e for e in entries if (gen_dir / f"{e['id']}.pkl").exists()]
    if max_n is not None:
        entries = entries[:max_n]
    ids = {e["id"] for e in entries}

    parent: Dict[str, str] = {}
    score: Dict[str, int] = {}
    for e in entries:
        src = e.get("source_entry_ids") or []
        p = src[0] if src and src[0] in ids else VIRTUAL_ROOT
        parent[e["id"]] = p
        score[e["id"]] = int(e.get("n_published_children") or 0)
    order = [e["id"] for e in entries]

    if synthetic_root:
        # Reassign every real archive root to descend from NEUTRAL_ROOT_ID.
        for nid, par in list(parent.items()):
            if par == VIRTUAL_ROOT:
                parent[nid] = NEUTRAL_ROOT_ID
        parent[NEUTRAL_ROOT_ID] = VIRTUAL_ROOT
        score[NEUTRAL_ROOT_ID] = 0
        order = [NEUTRAL_ROOT_ID] + order

    return order, parent, score


def _global_euler_tour(order, parent, score) -> List[Edge]:
    """Rating-ordered DFS Euler tour of the forest.

    Returns a list of directed edges (parent->child to descend, child->parent to
    backtrack). Length = 2 * (number of real edges). Children at every node are
    visited in descending order of `score` (n_published_children), with
    publication-order as a stable tiebreaker.
    """
    children: Dict[str, List[str]] = defaultdict(list)
    for c, p in parent.items():
        children[p].append(c)
    pos_in_order = {n: i for i, n in enumerate(order)}
    for p in children:
        children[p].sort(key=lambda n: (-score.get(n, 0), pos_in_order.get(n, 0)))

    # Iterative DFS Euler tour. We treat VIRTUAL_ROOT as an invisible super-root:
    # we DO descend "edges" from it to each real root (in score order), so all
    # 76 real roots get visited, but we don't emit those virtual edges -- they
    # don't exist in the phylogeny so a walker shouldn't morph along them.
    tour: List[Edge] = []
    # Stack entry: (node, iter_over_children, is_virtual_root)
    stack = [(VIRTUAL_ROOT, iter(children[VIRTUAL_ROOT]), True)]
    path = [VIRTUAL_ROOT]
    while stack:
        node, it, is_virt = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            stack.pop()
            if path:
                popped = path.pop()
                # emit backtrack edge child->parent (unless we just backtracked
                # past the virtual root)
                if path and path[-1] != VIRTUAL_ROOT:
                    tour.append((popped, path[-1]))
                elif path and path[-1] == VIRTUAL_ROOT:
                    # popping a real root back up to virtual root -- skip
                    pass
            continue
        # descend into child `nxt`
        if not is_virt:
            tour.append((node, nxt))   # real edge
        # else: descending from virtual root to a real root -- no emit
        stack.append((nxt, iter(children.get(nxt, [])), False))
        path.append(nxt)
    return tour


def build_walks(run: Path, k: int) -> Tuple[List[List[Edge]], Dict[str, str], Dict[str, int], List[str]]:
    """Compute k equal-length walker tours over the phylogeny at `run`.

    Returns (walks, parent_map, score_map, publication_order).
    walks: list of length k; each walk is a list of directed edges (Edge).
    """
    order, parent, score = load_forest(run)
    tour = _global_euler_tour(order, parent, score)
    if not tour:
        return [[] for _ in range(k)], parent, score, order

    # Slice into k contiguous equal-length segments. The last segment absorbs
    # the remainder so total tour length is preserved exactly.
    n = len(tour)
    base = n // k
    rem = n - base * k
    walks: List[List[Edge]] = []
    i = 0
    for w in range(k):
        size = base + (1 if w < rem else 0)
        walks.append(tour[i:i + size])
        i += size
    assert sum(len(w) for w in walks) == n
    return walks, parent, score, order


def walk_start_node(walk: List[Edge]) -> str | None:
    """The node a walker is positioned on before its first traversal."""
    return walk[0][0] if walk else None


# ---------- continuous best-first walks with retread penalty ----------

def _neighbor_map(parent: Dict[str, str]) -> Dict[str, set]:
    """Undirected neighbor map over **real** edges only (virtual root excluded)."""
    nbrs: Dict[str, set] = defaultdict(set)
    for c, p in parent.items():
        if p != VIRTUAL_ROOT:
            nbrs[c].add(p)
            nbrs[p].add(c)
    return nbrs


def _components(parent: Dict[str, str], nbrs: Dict[str, set]) -> List[List[str]]:
    """Connected components of the phylogeny (real edges only)."""
    seen: set = set()
    comps: List[List[str]] = []
    for n in parent:
        if n in seen:
            continue
        comp: List[str] = []
        dq: deque = deque([n])
        while dq:
            u = dq.popleft()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for v in nbrs[u]:
                if v not in seen:
                    dq.append(v)
        comps.append(comp)
    return comps


def _allocate_walkers(sizes: List[int], k: int) -> List[int]:
    """Distribute k walkers across components with edge-counts ``sizes``.

    Singletons (size 0) get 0 walkers. Every non-singleton gets >=1 if possible.
    Remaining walkers are assigned by largest-remainder proportional allocation
    so big components get many walkers and edge-load per walker stays balanced.
    """
    n = len(sizes)
    alloc = [0] * n
    nonzero = [i for i in range(n) if sizes[i] > 0]
    nonzero.sort(key=lambda i: -sizes[i])
    if not nonzero:
        return alloc
    # Step 1: floor 1 walker per non-singleton, up to k.
    for i in nonzero:
        if sum(alloc) >= k:
            break
        alloc[i] = 1
    # Step 2: distribute the rest by "biggest load-per-walker" greedy. Equivalent
    # to repeatedly snapping the walker to whatever component would most reduce
    # the max load (size_i / (alloc_i + 1)).
    while sum(alloc) < k:
        best_i = max(nonzero, key=lambda i: sizes[i] / (alloc[i] + 1))
        alloc[best_i] += 1
    # If we overshot k (because we ran past in step 1), trim from smallest comps.
    while sum(alloc) > k:
        # Trim from the component with smallest size whose alloc > 1
        candidates = [i for i in nonzero if alloc[i] > 1]
        if not candidates:
            candidates = [i for i in nonzero if alloc[i] > 0]
        i = min(candidates, key=lambda i: sizes[i])
        alloc[i] -= 1
    return alloc


def _pick_starts(comps: List[List[str]], score: Dict[str, int], k: int) -> List[str]:
    """Pick k start nodes spread across components proportional to component size."""
    sizes = [max(0, len(c) - 1) for c in comps]   # edges
    alloc = _allocate_walkers(sizes, k)
    starts: List[str] = []
    for ci, n_w in enumerate(alloc):
        if n_w == 0:
            continue
        comp_sorted = sorted(comps[ci], key=lambda n: (-score.get(n, 0), n))
        starts.extend(comp_sorted[:n_w])
    return starts[:k]


def best_first_walks(
    run: Path,
    k: int,
    K: int,
    *,
    alpha: float = 0.5,
    beta: float = 2.0,
    gamma: float = 1.0,
    start_nodes: Optional[Sequence[str]] = None,
    max_n: Optional[int] = None,
    require_genome: bool = False,
    synthetic_root: bool = True,
) -> Tuple[List[List[Edge]], Dict[str, str], Dict[str, int], List[str], List[str]]:
    """k continuous walks of length K each, with the score blend documented above.

    Neighbor score: ``alpha*norm_rating(u) + beta*[edge globally unvisited]
                    - gamma*walker_visits[edge]``.

    Ratings are normalized to ``[0, 1]`` (rating(u) / max_rating) so the three
    knobs operate on the same scale. With defaults the order is
    fresh > visited-by-other > walker-once > walker-twice for any rating, so
    coverage is strongly preferred but rating breaks ties (and biases choice
    among same-class neighbors). Lower beta or higher alpha to weight rating
    more strongly; raise gamma to push harder against retreading.

    Returns (walks, parent_map, score_map, publication_order, starts).
    """
    order, parent, score = load_forest(
        run, synthetic_root=synthetic_root, max_n=max_n, require_genome=require_genome,
    )
    nbrs = _neighbor_map(parent)
    comps = _components(parent, nbrs)
    starts = list(start_nodes) if start_nodes is not None else _pick_starts(comps, score, k)
    if len(starts) != k:
        raise ValueError(f"got {len(starts)} starts, need {k}")
    max_rating = max(score.values()) if score else 1
    max_rating = max(max_rating, 1)

    global_visits: Dict[frozenset, int] = defaultdict(int)
    walker_visits: List[Dict[frozenset, int]] = [defaultdict(int) for _ in range(k)]
    walks: List[List[Edge]] = []
    for w in range(k):
        walk: List[Edge] = []
        cur = starts[w]
        for _ in range(K):
            cands = nbrs.get(cur)
            if not cands:
                break   # isolated node: walker holds still
            best_u = None
            best_s = float("-inf")
            for u in cands:
                e = frozenset((cur, u))
                s = (
                    alpha * (score.get(u, 0) / max_rating)
                    + beta * (1.0 if global_visits[e] == 0 else 0.0)
                    - gamma * walker_visits[w][e]
                )
                # tie-break: stable by node id so output is deterministic
                if s > best_s or (s == best_s and (best_u is None or u < best_u)):
                    best_s = s
                    best_u = u
            assert best_u is not None
            e = frozenset((cur, best_u))
            walk.append((cur, best_u))
            walker_visits[w][e] += 1
            global_visits[e] += 1
            cur = best_u
        walks.append(walk)

    return walks, parent, score, order, starts


def edge_coverage(walks: Sequence[Sequence[Edge]], parent: Dict[str, str]) -> Tuple[int, int]:
    """Returns (edges_covered, total_real_edges) -- how much of the phylogeny
    the union of these walks touches."""
    real_edges = {frozenset((c, p)) for c, p in parent.items() if p != VIRTUAL_ROOT}
    visited = {frozenset((a, b)) for w in walks for (a, b) in w}
    visited &= real_edges
    return len(visited), len(real_edges)


if __name__ == "__main__":   # quick sanity check
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("-k", type=int, default=36)
    ap.add_argument("--method", choices=["euler-slice", "best-first"], default="best-first")
    ap.add_argument("-K", type=int, default=200, help="(best-first only) edges per walker")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()
    if args.method == "euler-slice":
        walks, parent, score, order = build_walks(args.run, args.k)
        starts = [walk_start_node(w) for w in walks]
    else:
        walks, parent, score, order, starts = best_first_walks(
            args.run, args.k, args.K, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    n_real_edges = sum(1 for c, p in parent.items() if p != VIRTUAL_ROOT)
    covered, total = edge_coverage(walks, parent)
    print(f"forest: {len(order)} nodes, {n_real_edges} real edges; method={args.method}")
    print(f"  coverage: {covered}/{total} unique real edges = {100*covered/total:.1f}%")
    for i, (w, s) in enumerate(zip(walks[:6], starts[:6])):
        print(f"  walker {i:2d}: len={len(w):5d}, start={s} (score={score.get(s, 0)})")
    sizes = [len(w) for w in walks]
    print(f"  walker lengths: min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.1f}")
    # Verify continuity (no jumps within any walk)
    n_jumps = 0
    for w in walks:
        for i in range(1, len(w)):
            if w[i-1][1] != w[i][0]:
                n_jumps += 1
    print(f"  jumps within walks: {n_jumps}")
