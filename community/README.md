---
title: Picbreeder-VLM Community API
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Picbreeder-VLM Community API

Write/read gateway that lets the static `breed/` site publish, browse, branch, and
(admin) delete user-bred CPPN genomes. Storage lives in the private dataset
`picbreeder-vlm/picbreeder-vlm-community`; this Space holds the only write token.

## Endpoints
- `GET /` — health + item count
- `GET /items` — slim manifest (browse gallery)
- `GET /item/{id}` — full record incl. genome (branch / DNA)
- `POST /publish` — `{genome, title, author, color, parent, gen, png}` → `{id, key}`
- `DELETE /item/{id}` — admin only, requires `X-Admin-Key`

## Secrets (set in Space settings)
- `HF_TOKEN` — write-scoped token for the community dataset
- `ADMIN_KEY` — secret string required by `DELETE`
- `COMMUNITY_DATASET` — optional override (default `picbreeder-vlm/picbreeder-vlm-community`)

Local run: `HF_TOKEN=... ADMIN_KEY=dev uvicorn app:app --reload`
