---
license: cc-by-4.0
language:
  - en
  - zh
  - ru
  - vi
  - es
  - tr
  - id
  - de
library_name: custom
tags:
  - ingredient-embeddings
  - food
  - computational-gastronomy
  - metapath2vec
  - skip-gram
  - word2vec
  - retrieval
  - feature-extraction
pipeline_tag: feature-extraction
datasets:
  - Kaikaku/epicure-corpus-resources
arxiv: 2605.22391
---

# Epicure-Cooc

A 300-dimensional skip-gram ingredient embedding over a 1,790-ingredient canonical vocabulary, trained on pure ingredient-ingredient random walks weighted by normalised pointwise mutual information (NPMI) over a 4.14M-recipe multilingual corpus.

Cooc is the recipe-context extreme of the three Epicure siblings. It sees no chemistry signal: walks traverse only the ingredient-ingredient NPMI graph (203,508 edges over 1,790 ingredients). Its neighbour structure answers "what else do I cook this with" rather than "what shares its flavour profile."

Companions in the family: [epicure-core](https://huggingface.co/Kaikaku/epicure-core) (blended) and [epicure-chem](https://huggingface.co/Kaikaku/epicure-chem) (chemistry only).

Paper: [Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings](https://arxiv.org/abs/2605.22391)

## Quick start

```python
from epicure import Epicure

m = Epicure.from_pretrained("Kaikaku/epicure-cooc")

# nearest-neighbour pairings
m.neighbors("chicken", k=5)
# -> [('garlic', 0.39), ('onion', 0.37), ('black_pepper', 0.36),
#     ('turkey', 0.35), ('carrot', 0.34)]

# SLERP toward a supervised pole
m.slerp("corn", "cuisine:Latin_American", theta_deg=30, k=5)
# -> [('salsa_verde', 0.45), ('red_onion', 0.45), ('chorizo', 0.44), ...]

# closest emergent factor mode
m.closest_mode("miso", kind="factor", k=3)
```

`epicure.py` is shipped inside this repo. Install requirements with `pip install huggingface_hub safetensors numpy`.

## What is in this repo

| File | Description |
| --- | --- |
| `embeddings.safetensors` | (1,790 x 300) float32 matrix, key `embeddings`. Raw skip-gram outputs (not L2-normalised on disk). |
| `vocab.json` | `name -> int` mapping for the 1,790 canonical ingredients. |
| `itos.json` | inverse vocabulary. |
| `config.json` | model metadata (schema, training hyperparameters, graph statistics, corpus statistics). |
| `modes.json` | GMM mode atlas. 150 modes for Cooc, each with members, label, mode-pole vector, and kind (`factor` for emergent, `continuous`/`binary` for supervised properties). |
| `factor_poles.npy` | (60, 300) stack of all factor-mode poles for the 20 emergent ICA factors. |
| `factor_pole_index.json` | parallel list of mode ids matching `factor_poles.npy`. |
| `supervised_poles.json` | per-mode pole vectors keyed by `{property}/{mode_label}`, plus seven heuristic `cuisine:{Region}` poles built by averaging modes whose Claude-generated labels reference each macro-region. |
| `cuisine_pole_provenance.json` | per-cuisine, the list of mode ids and labels that were averaged to build the cuisine direction. |
| `paper_slerp_results.csv` | the 720 rows of `direction_arithmetic_full.csv` filtered to `model=cooc` so users can verify their reconstruction against the paper's reported tables. |
| `epicure.py` | the loader and operator implementation. |
| `LICENSE` | CC BY 4.0 (matches the arXiv submission). |

## Reported numbers (this sibling)

From [the paper](https://arxiv.org/abs/2605.22391):

- Isotropy: participation ratio `PR = 173.6` of 300; average pairwise cosine in the 0.10-0.12 band. Cooc is one of the two isotropic siblings (Chem `PR=183.1` is the other; Core is concentrated at `PR=94.2`).
- Direction quality (5-fold repeated CV Spearman rho, mean over probes): baked-in compound-feature 0.28; held-out basic-taste compound-feature 0.32; USDA macros 0.41. Cuisine macro-region Cohen's d mean 2.43 across the eight regions.
- Emergent modes: 150 modes across 41 properties. Mean within-mode pairwise cosine 0.611 against random-pair baseline 0.097 (a 6x tightness margin).
- Normalised mutual information against USDA food groups: 0.20-0.25 band. Soft NMI against cuisine macro-regions: 0.43-0.46 band.

The Cooc <= Core <= Chem ordering holds on every supervised direction-quality probe (basic taste, USDA macros, cuisine). For Cooc that means: it is the cleanest recipe-context model but the weakest supervised-direction extractor of the three; if you want a chemistry-aware navigation operator pick Core or Chem.

## Operator semantics

Three operator families live on this embedding:

1. **Nearest-neighbour pairings** (`m.neighbors(seed, k)`). Top-K cosine neighbours over the L2-normalised matrix. Excludes the seed by default.
2. **Closest-mode lookup** (`m.closest_mode(seed, kind=..., k)`). Cosine of the seed to each mode pole; returns the top-K modes. `kind` filters to `factor` (emergent), `continuous` (sensory and USDA), or `binary` (food group, NOVA).
3. **SLERP direction arithmetic** (`m.slerp(seed, direction, theta_deg, k)`). Rotates the unit-normalised seed toward a unit direction by angle theta on the unit sphere. The direction is either a supervised pole key from `supervised_poles.json` or a raw 300-D numpy array. At theta=0 the rotated query equals the seed; at theta=60 deg cosine-to-seed = 0.5; by 90 deg the target neighbourhood dominates. The seed is excluded from results by default.

## Honesty about cuisine pole reconstruction

The paper's SLERP-toward-cuisine results in Section 4.2 use cuisine-macro-region pole vectors built from the 8-region distinctive-marker tags introduced in Section 2.2. Those tag-to-ingredient mappings (the `canonical_vocabulary.csv` with `cuisine_tags` column) are **not** in the arXiv ancillary bundle, so this repo reconstructs cuisine pole vectors heuristically: the unit-mean of every mode-atlas pole whose Claude-generated label contains a cuisine macro-region keyword (e.g. "South Asian", "Mexican", "Italian"). Provenance is logged per cuisine in `cuisine_pole_provenance.json`.

Empirically the heuristic poles produce paper-genre results in two of three siblings:

| Hero query | Paper §4.2 result (target tokens) | Reconstructed Cooc | Reconstructed Core | Reconstructed Chem |
| --- | --- | --- | --- | --- |
| corn +30 deg -> Latin American | salsa verde, tomatillo, queso fresco, fajita seasoning, corn tortilla | salsa_verde, red_onion, chorizo, black_pepper, tomatillo | red_onion, bell_pepper, cayenne_pepper, corn_tortilla, monterey_jack_cheese | poblano_pepper, corn_tortilla, salsa, queso_fresco, chipotle_pepper |
| rice +30 deg -> South Asian | curry leaf, masoor dal, urad dal, chana dal, fenugreek seed | salt, onion, black_pepper, garlic, carrot | turmeric, mustard_seed, fenugreek_seed, coriander, cumin | fenugreek_seed, cardamom, saffron, fennel_seed, curry |

The Cooc reconstruction matches the genre less tightly than Core or Chem because the Cooc sibling does not encode chemistry-mediated cuisine clustering, so the heuristic cuisine pole picks up generic Cooc co-occurrence companions. To exactly reproduce the paper's Cooc results, build the cuisine pole from the original per-ingredient cuisine tag set. We will add `canonical_vocabulary.csv` to the corpus-resources dataset in a follow-up; until then, `paper_slerp_results.csv` ships the full paper-reported SLERP table for offline verification.

## Limitations

From Section 5.3 of the paper:

- **Corpus imbalance.** The 4.14M-recipe corpus is roughly half East Asian and a tenth Mediterranean; single-digit shares for South Asian, Eastern European, Latin American. Held-out-cuisine confidence intervals widen accordingly in smaller regions; the cross-region ranking of the three siblings is nevertheless stable.
- **Hub coverage.** Only 523 of 1,790 ingredients are chemistry hubs (those with active typed I-C edges after the min_compound_degree=2 filter). The remaining 1,267 non-hubs reach compound context only indirectly via the N-H-C[x]-H-N metapath. For pure-Cooc Cooc this is irrelevant; it matters for Core and Chem.
- **LLM dependence.** The canonical vocabulary, cuisine tags, sensory ground-truth scores, and mode/factor labels are LLM-generated under deterministic decoding. The embeddings themselves are LLM-free.

## Intended use

Research and prototyping for chef-facing or recipe-assistant tools: ingredient pairing, cross-cuisine navigation, sensory- or nutrient-aware exploration. Not intended for nutritional or medical recommendation: USDA values are looked up against FoodData Central but the embedding only encodes their continuous projection, not the absolute numbers.

## Citation

```bibtex
@article{radzikowski2026epicure,
  title   = {Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings},
  author  = {Radzikowski, Jakub and Chen, Josef},
  journal = {arXiv preprint arXiv:2605.22391},
  year    = {2026}
}
```

## License

CC BY 4.0. Matches the arXiv submission. Attribution to Radzikowski and Chen, KAIKAKU.AI.
