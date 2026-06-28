"""Picbreeder-VLM community publish API (Hugging Face Space).

A tiny FastAPI gateway that lets the static `breed/` site actually *publish* user-bred
genomes — the one thing a static GitHub Pages site can't do itself. The Space holds the
HF write token (as a secret) and is the only thing allowed to commit to the private
community dataset.

Storage (in `COMMUNITY_DATASET`, a HF *dataset* repo):
  items/<id>.json   full record: genome + metadata
  images/<id>.png   512px representative thumbnail (optional; client also renders genomes)
  index.json        slim manifest the site reads: [{id,title,color,parent,author,created_at,...}]

Endpoints:
  GET    /                 health + counts
  GET    /items            slim manifest (the browse gallery reads this)
  GET    /item/{id}        full record incl. genome (branch/DNA read this)
  POST   /publish          validate + commit a new item, returns {id}
  DELETE /item/{id}        admin-only (X-Admin-Key), for dev/debug cleanup

The index is kept in memory and rebuilt from the dataset on startup; every write is
persisted back to `index.json` so it survives Space restarts.
"""
import base64
import binascii
import gzip
import io
import json
import os
import re
import threading
import time
from collections import deque

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.utils import HfHubHTTPError

DATASET = os.environ.get("COMMUNITY_DATASET", "picbreeder-vlm/picbreeder-vlm-community")
HF_TOKEN = os.environ.get("HF_TOKEN")                  # write token (Space secret)
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")            # admin secret for DELETE
RUN_NAME = "community"                                 # the site's "run" key for these items

# Validation / abuse limits ---------------------------------------------------
MAX_NODES = 600
MAX_CONNS = 4000
MAX_TITLE = 80
MAX_AUTHOR = 40
MAX_TAGS = 8
MAX_TAG_LEN = 30
MAX_PNG_BYTES = 600_000
MAX_BODY_BYTES = 16_000_000               # large: a publish may carry the full session lineage
MAX_LINEAGE_GENS = 2000                   # cap generations in a saved lineage
MAX_LINEAGE_BYTES = 12_000_000            # cap serialized lineage before gzip
RATE_N = 30                                             # publishes per window per IP
RATE_WINDOW = 3600                                     # seconds
RATE_R_N = 300                                          # ratings per window per IP
MAX_KEY = 200                                           # item-key length cap
MAX_VOTER = 64                                          # voter-id length cap
MAX_BATCH = 500                                         # keys per /ratings batch call

api = HfApi(token=HF_TOKEN)
app = FastAPI(title="Picbreeder-VLM Community API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # public read/publish; admin is gated by the header secret
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- in-memory state ---------------------------------------------------------
_lock = threading.Lock()          # serialize commits (single-worker Space)
_index = []                       # list of slim entries (newest last)
_by_id = {}                       # id -> slim entry
_genomes = {}                     # id -> genome JSON (served to the gallery for in-browser render)
# Human ratings span the WHOLE site, not just community items: keyed by the global
# item key ("<run>::<id>", "c_<id>", "community::<id>") -> {voter_id: value(0.5..5)}.
_ratings = {}                     # key -> {voter: value}
_rate = {}                        # ip -> deque[timestamps]  (publish limit)
_rate_r = {}                      # ip -> deque[timestamps]  (rating limit)
_ctr = 0                          # monotonic counter for id uniqueness within a process


def _load_json(name, default):
    try:
        with open(api.hf_hub_download(DATASET, name, repo_type="dataset")) as f:
            return json.load(f)
    except Exception:
        return default


def _load_state():
    global _index, _by_id, _genomes, _ratings
    _index = _load_json("index.json", [])
    _genomes = _load_json("genomes.json", {})
    _ratings = _load_json("ratings.json", {})
    _by_id = {e["id"]: e for e in _index}


@app.on_event("startup")
def _startup():
    _load_state()


# --- helpers -----------------------------------------------------------------
_ID_RE = re.compile(r"^c[0-9a-z]{4,32}$")


def _new_id():
    """Time-sortable, filesystem-safe, collision-resistant id (no Math.random reliance)."""
    global _ctr
    _ctr += 1
    return "c%s%03x" % (format(int(time.time() * 1000), "x"), _ctr % 0xFFF)


def _client_ip(request: Request) -> str:
    # HF Spaces sit behind a proxy; trust the first forwarded hop.
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def _rate_limit(store, ip, cap) -> bool:
    now = time.time()
    dq = store.setdefault(ip, deque())
    while dq and now - dq[0] > RATE_WINDOW:
        dq.popleft()
    if len(dq) >= cap:
        return False
    dq.append(now)
    return True


def _rate_ok(ip: str) -> bool:
    return _rate_limit(_rate, ip, RATE_N)


_VOTER_RE = re.compile(r"^[A-Za-z0-9_.\-]{8,64}$")


def _agg(key):
    """(mean, n) of human ratings for an item key; mean rounded to 2dp."""
    votes = _ratings.get(key)
    if not votes:
        return None, 0
    vals = list(votes.values())
    return round(sum(vals) / len(vals), 2), len(vals)


def _norm_value(v):
    """Accept 0.5..5 in 0.5 steps; snap & validate, else 400."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, "value must be a number")
    snapped = round(f * 2) / 2
    if snapped < 0.5 or snapped > 5 or snapped != f:
        raise HTTPException(400, "value must be 0.5..5 in steps of 0.5")
    return snapped


def _rating_view(key, voter):
    mean, n = _agg(key)
    your = None
    if voter and key in _ratings:
        your = _ratings[key].get(voter)
    return {"key": key, "human_mean": mean, "human_n": n, "your": your}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and -1e9 < float(x) < 1e9


def _validate_genome(g):
    """Mirror cppn.js genomeToJSON shape; reject anything malformed or oversized."""
    if not isinstance(g, dict):
        raise HTTPException(400, "genome must be an object")
    if str(g.get("format", "")).split("/")[0] != "picbreeder-vlm-cppn":
        raise HTTPException(400, "unrecognized genome format")
    nodes = g.get("nodes")
    conns = g.get("connections")
    if not isinstance(nodes, list) or not isinstance(conns, list):
        raise HTTPException(400, "genome needs nodes[] and connections[]")
    if not nodes or len(nodes) > MAX_NODES:
        raise HTTPException(400, f"node count out of range (1..{MAX_NODES})")
    if len(conns) > MAX_CONNS:
        raise HTTPException(400, f"too many connections (>{MAX_CONNS})")
    keys = set()
    for n in nodes:
        if not isinstance(n, dict) or "key" not in n or "activation" not in n:
            raise HTTPException(400, "bad node")
        if not isinstance(n["activation"], str) or len(n["activation"]) > 24:
            raise HTTPException(400, "bad activation")
        keys.add(n["key"])
    for c in conns:
        if not isinstance(c, dict) or not _finite(c.get("weight")):
            raise HTTPException(400, "bad connection weight")
    return g


_TAG_RE = re.compile(r"[^a-z0-9 \-]")


def _norm_tags(tags):
    """Creator tags entered at publish (original Picbreeder model): free-text,
    lowercased, de-duped, capped. Returns an ordered unique list."""
    if not tags:
        return []
    out, seen = [], set()
    for t in tags:
        if not isinstance(t, str):
            continue
        t = _TAG_RE.sub("", t.strip().lower())
        t = re.sub(r"\s+", " ", t).strip()[:MAX_TAG_LEN]
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= MAX_TAGS:
            break
    return out


def _gz_lineage(lineage):
    """Validate the session lineage loosely and return gzipped JSON bytes (or None).

    Provenance/analysis data, not security-critical (the published genome is validated
    separately), so we only guard shape + size, then store it compressed."""
    if not lineage:
        return None
    hist = lineage.get("history")
    if not isinstance(hist, list):
        raise HTTPException(400, "lineage.history must be a list")
    if len(hist) > MAX_LINEAGE_GENS:
        raise HTTPException(400, f"lineage too long (>{MAX_LINEAGE_GENS} generations)")
    blob = json.dumps(lineage).encode("utf-8")
    if len(blob) > MAX_LINEAGE_BYTES:
        raise HTTPException(400, "lineage too large")
    return gzip.compress(blob, compresslevel=6)


def _decode_png(data_url):
    if not data_url:
        return None
    b64 = data_url.split(",", 1)[-1] if isinstance(data_url, str) and "," in data_url else data_url
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "png is not valid base64")
    if len(raw) > MAX_PNG_BYTES:
        raise HTTPException(400, f"png too large (>{MAX_PNG_BYTES} bytes)")
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise HTTPException(400, "png magic bytes missing")
    return raw


def _slim(rec):
    """Project a full item record down to the manifest entry the site reads."""
    return {
        "id": rec["id"],
        "title": rec.get("title") or "Untitled",
        "tags": rec.get("tags", []),
        "color": rec.get("color", True),
        "parent": rec.get("parent"),
        "author": rec.get("author") or "anon",
        "gen": rec.get("gen"),
        "created_at": rec["created_at"],
        "has_png": rec.get("has_png", False),
        "has_lineage": rec.get("has_lineage", False),
        "gens": rec.get("gens"),     # number of generations in the saved lineage
        "hr_mean": None,        # human-rating aggregate, filled in as votes arrive
        "hr_n": 0,
    }


def _state_ops():
    """Commit ops that rewrite the public manifests from in-memory state."""
    return [
        CommitOperationAdd("index.json", json.dumps(_index).encode("utf-8")),
        CommitOperationAdd("genomes.json", json.dumps(_genomes).encode("utf-8")),
        CommitOperationAdd("ratings.json", json.dumps(_ratings).encode("utf-8")),
    ]


def _persist_state():
    api.create_commit(DATASET, _state_ops(), repo_type="dataset",
                      commit_message=f"state: {len(_index)} items")


def _persist_ratings():
    # a vote changes ratings.json (+ a community slim entry in index.json); genomes
    # are untouched, so don't rewrite the (growing) genomes.json on every vote.
    ops = [
        CommitOperationAdd("ratings.json", json.dumps(_ratings).encode("utf-8")),
        CommitOperationAdd("index.json", json.dumps(_index).encode("utf-8")),
    ]
    api.create_commit(DATASET, ops, repo_type="dataset",
                      commit_message=f"rate: {sum(len(v) for v in _ratings.values())} votes")


# --- models ------------------------------------------------------------------
class PublishIn(BaseModel):
    genome: dict
    title: str = Field(default="Untitled")
    author: str = Field(default="anon")
    tags: list[str] | None = None      # creator's free-text tags, entered at publish
    color: bool = True
    parent: str | None = None          # parent key: "community::<id>" or "<run>::<id>" or canon key
    gen: int | None = None
    png: str | None = None             # base64 or data: URL of 512px representative
    lineage: dict | None = None        # full session trajectory: {founder, history[], rep_gen, rep_idx, ...}


class RateIn(BaseModel):
    key: str                           # global item key being rated
    value: float                       # 0.5..5 in 0.5 steps
    voter: str                         # opaque per-browser id (or, later, HF username)


class RatingsIn(BaseModel):
    keys: list[str]
    voter: str | None = None


# --- routes ------------------------------------------------------------------
@app.get("/")
def health():
    return {"ok": True, "dataset": DATASET, "items": len(_index), "run": RUN_NAME}


@app.get("/items")
def items():
    return {"run": RUN_NAME, "n": len(_index), "items": _index}


@app.get("/tags")
def tags():
    """Tag -> count across all community items (powers typeahead + a tag cloud)."""
    counts = {}
    for e in _index:
        for t in e.get("tags", []) or []:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@app.get("/genomes")
def genomes():
    # {id: genome} for the whole community run, so the gallery renders every
    # thumbnail client-side in one fetch (mirrors the archive's genomes.json.gz).
    return _genomes


@app.get("/item/{item_id}")
def item(item_id: str):
    if not _ID_RE.match(item_id):
        raise HTTPException(400, "bad id")
    slim = _by_id.get(item_id)
    if slim is None or item_id not in _genomes:
        raise HTTPException(404, "not found")
    return {**slim, "g": _genomes[item_id]}


@app.get("/lineage/{item_id}")
def lineage(item_id: str):
    """The full saved session trajectory (founder + every generation's population/picks
    up to the published genome) — the breed-site analogue of the original Picbreeder's
    per-series generation chain, kept as a superset (whole populations, not just the chain)."""
    if not _ID_RE.match(item_id):
        raise HTTPException(400, "bad id")
    slim = _by_id.get(item_id)
    if slim is None or not slim.get("has_lineage"):
        raise HTTPException(404, "no lineage for this item")
    try:
        path = api.hf_hub_download(DATASET, f"lineages/{item_id}.json.gz", repo_type="dataset")
    except Exception:
        raise HTTPException(404, "lineage file missing")
    with open(path, "rb") as f:
        return json.loads(gzip.decompress(f.read()))


@app.post("/publish")
async def publish(body: PublishIn, request: Request):
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(413, "request too large")
    ip = _client_ip(request)
    if not _rate_ok(ip):
        raise HTTPException(429, "rate limit: too many publishes, try later")

    genome = _validate_genome(body.genome)
    title = (body.title or "Untitled").strip()[:MAX_TITLE] or "Untitled"
    author = (body.author or "anon").strip()[:MAX_AUTHOR] or "anon"
    tags = _norm_tags(body.tags)
    png = _decode_png(body.png)
    lineage_gz = _gz_lineage(body.lineage)
    gens = len(body.lineage["history"]) if lineage_gz is not None else None

    with _lock:
        item_id = _new_id()
        rec = {
            "id": item_id,
            "title": title,
            "author": author,
            "tags": tags,
            "color": bool(body.color),
            "parent": body.parent,
            "gen": body.gen,
            "created_at": int(time.time()),
            "ip": ip,                      # kept private (items/ not public-readable during dev)
            "has_png": png is not None,
            "has_lineage": lineage_gz is not None,
            "gens": gens,
            "genome": genome,
        }
        slim = _slim(rec)
        _index.append(slim)
        _by_id[item_id] = slim
        _genomes[item_id] = genome

        # one atomic commit: the audit record, optional png, the lineage, and the manifests
        ops = [CommitOperationAdd(f"items/{item_id}.json",
                                  json.dumps(rec).encode("utf-8"))]
        if png is not None:
            ops.append(CommitOperationAdd(f"images/{item_id}.png", png))
        if lineage_gz is not None:
            ops.append(CommitOperationAdd(f"lineages/{item_id}.json.gz", lineage_gz))
        ops += _state_ops()
        try:
            api.create_commit(DATASET, ops, repo_type="dataset",
                              commit_message=f"publish {item_id}: {title[:40]}")
        except HfHubHTTPError as e:
            _index.pop(); _by_id.pop(item_id, None); _genomes.pop(item_id, None)
            raise HTTPException(502, f"storage commit failed: {e}")

    return {"ok": True, "id": item_id, "key": f"{RUN_NAME}::{item_id}"}


@app.delete("/item/{item_id}")
def delete(item_id: str, x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "admin key required")
    if not _ID_RE.match(item_id):
        raise HTTPException(400, "bad id")
    with _lock:
        if item_id not in _by_id:
            raise HTTPException(404, "not found")
        had_png = _by_id[item_id].get("has_png")
        had_lineage = _by_id[item_id].get("has_lineage")
        _index[:] = [e for e in _index if e["id"] != item_id]
        _by_id.pop(item_id, None)
        _genomes.pop(item_id, None)
        _ratings.pop(f"{RUN_NAME}::{item_id}", None)
        ops = [CommitOperationDelete(f"items/{item_id}.json")]
        if had_png:
            ops.append(CommitOperationDelete(f"images/{item_id}.png"))
        if had_lineage:
            ops.append(CommitOperationDelete(f"lineages/{item_id}.json.gz"))
        ops += _state_ops()                       # refreshed manifests, same commit
        try:
            api.create_commit(DATASET, ops, repo_type="dataset",
                              commit_message=f"admin delete {item_id}")
        except HfHubHTTPError as e:
            raise HTTPException(502, f"delete failed: {e}")
    return {"ok": True, "deleted": item_id}


# --- ratings -----------------------------------------------------------------
@app.get("/rating")
def rating(key: str, voter: str = ""):
    """Human-rating aggregate for one item, plus this voter's own vote (if any)."""
    if not key or len(key) > MAX_KEY:
        raise HTTPException(400, "bad key")
    return _rating_view(key, voter or None)


@app.post("/ratings")
def ratings_batch(body: RatingsIn):
    """Aggregates (+ this voter's votes) for many keys at once — for gallery rows."""
    keys = body.keys[:MAX_BATCH]
    voter = (body.voter or None)
    return {k: _rating_view(k, voter) for k in keys if k and len(k) <= MAX_KEY}


@app.post("/rate")
def rate(body: RateIn, request: Request):
    if not _rate_limit(_rate_r, _client_ip(request), RATE_R_N):
        raise HTTPException(429, "rate limit: too many ratings, try later")
    key = body.key
    if not key or len(key) > MAX_KEY:
        raise HTTPException(400, "bad key")
    voter = (body.voter or "").strip()
    if not _VOTER_RE.match(voter):
        raise HTTPException(400, "bad voter id")
    value = _norm_value(body.value)

    with _lock:
        _ratings.setdefault(key, {})[voter] = value
        mean, n = _agg(key)
        # mirror the aggregate into the community slim entry so the gallery shows it
        if key.startswith(f"{RUN_NAME}::"):
            slim = _by_id.get(key.split("::", 1)[1])
            if slim is not None:
                slim["hr_mean"], slim["hr_n"] = mean, n
        try:
            _persist_ratings()
        except HfHubHTTPError as e:
            raise HTTPException(502, f"rating commit failed: {e}")
    return {"ok": True, "key": key, "human_mean": mean, "human_n": n, "your": value}
