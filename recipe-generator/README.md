# Epicure Recipe Generator (prototype)

A two-pane web app: type a prompt in the side-pane chat, a recipe appears in the
main window. Ingredient pairings come from the **epicure-cooc** co-occurrence
embedding; the recipe itself is written by Claude.

See `SPEC.md` for the design.

## Layout

```
Cooc/
├── epicure-cooc/          # the model (must sit next to this folder)
└── recipe-generator/
    ├── app.py             # Flask backend
    ├── model_layer.py     # epicure-cooc -> ingredient brief
    ├── index.html         # two-pane UI
    ├── requirements.txt
    └── SPEC.md
```

## Setup

```bash
cd recipe-generator
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # required: the recipe-writing layer
# optional: export CLAUDE_MODEL=claude-opus-4-6
python app.py
```

Open http://127.0.0.1:5000 and start chatting.

## How it works

1. Your message is parsed for ingredients, cuisine, and dietary constraints.
2. `epicure-cooc` expands that into a ranked ingredient brief (neighbours,
   bridges between your ingredients, and a cuisine pull).
3. Claude writes a recipe built around the brief and returns structured JSON.
4. Follow-up messages ("make it spicier", "swap beef for lamb") edit the
   current recipe.

The "Why these pairings" panel shows the actual ingredients and similarity
scores the model contributed.

## Notes & known limits

- Needs the `epicure-cooc` model folder one level up (override with `EPICURE_DIR`).
- The cooc sibling is the weakest at cuisine direction arithmetic; Claude's
  culinary knowledge compensates. Swapping in `epicure-core`/`epicure-chem`
  would sharpen cuisine pulls (`EPICURE_DIR=../epicure-core`).
- Prototype only: no accounts, persistence, or images.
