---
title: "Cooc - Working Spec"
type: spec
status: draft
last_updated: 2026-06-06
entity: project
related: []
---

# CLAUDE.md - Cooc

Prototype recipe generator built on the epicure-cooc ingredient co-occurrence
embedding. This file is a thin index; substance lives in the files it points at.

## What this project is

Two deliverables, one model:

- `epicure_query.py` - CLI for querying the embedding directly (neighbors,
  cuisine SLERP, clusters, vocab search).
- `recipe-generator/` - two-pane web app: chat prompt in, recipe out.
  Ingredient pairing by epicure-cooc, recipe prose by Claude.

## Where the rules live

- `recipe-generator/SPEC.md` - architecture, API contract, open questions.
  Iterate on the spec before iterating on code.
- `recipe-generator/README.md` - setup and run instructions.
- `epicure-cooc/README.md` - the model card (operators, limits, license).

## Working agreements

- Develop specs iteratively with two-way socratic questioning before coding.
- `ANTHROPIC_API_KEY` lives in the environment only; never in code or files.
- The model is vendored (CC BY 4.0, attribution in `epicure-cooc/LICENSE`);
  pinned deliberately - do not auto-update from Hugging Face.
- Keep recipe costs realistic: Sonnet default, consider Haiku; no Opus.
