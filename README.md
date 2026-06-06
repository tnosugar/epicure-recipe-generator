# Epicure Recipe Generator

A prototype that turns "what should I cook?" prompts into recipes, driven by
[epicure-cooc](https://huggingface.co/Kaikaku/epicure-cooc) - a 300-dimensional
ingredient embedding trained on co-occurrence statistics from 4.14M recipes.

The embedding decides *what goes together* (ingredient pairings, bridges,
cuisine directions); Claude writes the actual recipe around its suggestions.

## Contents

| Path | What it is |
|---|---|
| `recipe-generator/` | Two-pane web app: chat in the side pane, recipe in the main window. See its README for setup. |
| `epicure_query.py` | CLI for querying the embedding directly: `neighbors`, `slerp`, `mode`, `find`, `cuisines`, `info`. |
| `epicure-cooc/` | The vendored model (weights, vocab, mode atlas, loader). |
| `CLAUDE.md` | Working spec index for Claude-assisted development. |

## Quick start

CLI (no API key needed):

```bash
pip install huggingface_hub safetensors numpy
python epicure_query.py neighbors chicken
python epicure_query.py slerp corn Latin_American --theta 30
```

Web app (needs an Anthropic API key for the recipe-writing layer):

```bash
cd recipe-generator
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

Open http://127.0.0.1:5000.

## How the generator works

1. Your message is parsed for ingredients, cuisine, and dietary constraints.
2. epicure-cooc expands it into a ranked "ingredient brief": co-occurrence
   neighbors, bridge ingredients connecting what you named, and a cuisine pull
   via spherical interpolation toward one of seven cuisine directions.
3. Claude writes a structured recipe built around the brief.
4. Follow-up chat messages refine the current recipe.

Design details and open questions: `recipe-generator/SPEC.md`.

## Model attribution

The `epicure-cooc/` folder vendors the model by Radzikowski & Chen, KAIKAKU.AI,
licensed CC BY 4.0 (see `epicure-cooc/LICENSE`). Pinned deliberately; the
canonical source is the
[Hugging Face repo](https://huggingface.co/Kaikaku/epicure-cooc). Paper:
[Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings](https://arxiv.org/abs/2605.22391).

Known model limits (from the model card): the corpus skews East Asian; the cooc
sibling is the weakest of the three Epicure models at cuisine direction
arithmetic. Not for nutritional or medical use.
