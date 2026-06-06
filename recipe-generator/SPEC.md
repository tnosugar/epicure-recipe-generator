# Recipe Generator — Prototype Spec

A two-pane web app that generates recipes. You type a prompt in the side-pane
chat; a recipe renders in the main window. Ingredient selection is driven by the
**epicure-cooc** co-occurrence embedding; the recipe prose is written by Claude.

## Why two engines

`epicure-cooc` is an ingredient *recommender* — it returns ingredient names and
similarity scores, with no quantities, steps, or prose. It answers "what goes
with what." Claude turns the recommended ingredient set into an actual recipe.
Neither alone is a recipe generator; together they are.

## Architecture

```
Browser (index.html)                 Flask backend (app.py)
┌───────────────┬───────────┐        ┌──────────────────────────────┐
│  RECIPE       │  CHAT     │  POST  │ 1. parse prompt (model_layer) │
│  (main pane)  │ (side)    │ ─────► │ 2. epicure-cooc → brief       │
│               │           │ /gen   │ 3. Claude writes recipe       │
│  pairing panel│  input    │ ◄───── │ 4. return structured JSON     │
└───────────────┴───────────┘  JSON  └──────────────────────────────┘
                                              │ loads once
                                              ▼
                                       ../epicure-cooc/
```

## Layer 1 — prompt parsing (`model_layer.py`)

- **Ingredients:** scan all 1–3 word n-grams of the prompt, normalise
  (lowercase, spaces→underscores), match against the 1,790-word vocab; fall back
  to fuzzy match for near-misses.
- **Cuisine/theme:** match prompt words against a synonym map → one of the seven
  cuisine poles (East_Asian, Japanese, Southeast_Asian, South_Asian,
  Mediterranean, Western_Atlantic, Latin_American).
- **Constraints:** lightweight flags passed through to Claude (vegetarian, vegan,
  spicy, quick, etc.) — Claude enforces them; the embedding does not.

## Layer 2 — ingredient brief (`model_layer.py`, the model's real job)

Given parsed seeds + optional cuisine, build a ranked candidate set with scores:

- **neighbors(seed):** top co-occurring partners for each named ingredient.
- **bridges(seeds):** when ≥2 seeds, ingredients close to their mean vector
  (things that connect them).
- **cuisine pull:** when a cuisine is named, `slerp(seed → cuisine, 30°)` and the
  members of the closest cuisine/factor mode.
- Output: `{seeds, cuisine, candidates:[{name, score, source}], rationale}`.

This brief is injected into Claude's prompt, and surfaced in the UI's "why these
pairings" panel, so the embedding visibly drives the recipe.

## Layer 3 — recipe writing (`app.py` → Claude)

Claude receives: the user message, conversation history, the ingredient brief,
and the current recipe (for refinement). It must build the recipe **around** the
brief's ingredients and return JSON only:

```json
{
  "title": "string",
  "description": "1-2 sentence intro",
  "servings": 4,
  "time": "e.g. 45 min",
  "ingredients": [{"item": "beef chuck", "quantity": "500 g"}],
  "steps": ["string", "..."],
  "pairing_notes": "how the epicure-cooc suggestions shaped this dish"
}
```

## Chat behaviour — iterative refinement

The frontend keeps `messages[]` and the `current_recipe`. Each new message
(e.g. "make it spicier", "swap beef for lamb", "vegetarian") is sent with both,
so Claude edits the existing recipe rather than starting over.

## API contract

`POST /generate`
```json
{ "messages": [{"role":"user","content":"..."}], "current_recipe": {…}|null }
```
Response:
```json
{ "recipe": {…}, "brief": {…}, "assistant_message": "short chat reply" }
```

## What the user provides

- `ANTHROPIC_API_KEY` env var (the recipe-writing layer).
- Python 3 + `flask`, `anthropic` (see requirements.txt). Model deps already in
  `../epicure-cooc`.

## Defaults (changeable)

- Claude model: `claude-sonnet-4-6` (env `CLAUDE_MODEL`).
- Servings: 4. Pairing panel: shown.

## Out of scope (prototype)

Accounts, persistence, images, nutrition facts, multi-recipe history. The
embedding's known weakness (cuisine SLERP is loosest on the cooc sibling) is
accepted; Claude compensates with culinary knowledge.

## Open questions / iteration log

- [ ] Should constraints (allergies/diet) be first-class UI inputs, not just chat?
- [ ] Show candidate ingredients the recipe *didn't* use, as swap suggestions?
- [ ] Switch to epicure-core/chem for sharper cuisine pulls?
