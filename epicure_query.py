#!/usr/bin/env python3
"""
epicure_query.py - a friendly CLI wrapper around the Kaikaku/epicure-cooc
ingredient-embedding model.

The model is a 1,790-ingredient x 300-dim co-occurrence embedding: it knows
"what gets cooked with what" from a 4.14M-recipe corpus. This script wraps the
three operators the model ships with.

SETUP (one time)
    pip install huggingface_hub safetensors numpy
    # Place this script in the SAME folder as the downloaded "epicure-cooc"
    # model folder (the one containing epicure.py, embeddings.safetensors, ...).

USAGE
    # Nearest co-occurrence neighbours - "what goes with X"
    python epicure_query.py neighbors chicken
    python epicure_query.py neighbors basil -k 10

    # SLERP - nudge an ingredient toward a cuisine direction
    python epicure_query.py slerp corn Latin_American
    python epicure_query.py slerp rice South_Asian --theta 45 -k 8

    # Closest emergent cluster ("mode") for an ingredient
    python epicure_query.py mode miso

    # Helpers
    python epicure_query.py cuisines              # list cuisine directions
    python epicure_query.py find choc             # search the vocabulary
    python epicure_query.py info                  # model summary

Notes
- Ingredient names use underscores: "black_pepper", "olive_oil". Use `find`
  if you're unsure of the exact token.
- Cuisine names accept either "Latin_American" or "cuisine:Latin_American".
- The cooc sibling is the cleanest co-occurrence model but the WEAKEST at
  cuisine direction arithmetic (the model card is explicit about this). For
  sharper cuisine results use epicure-core or epicure-chem.
"""

import argparse
import difflib
import os
import sys

# Locate the model folder (sibling "epicure-cooc" dir, or override via env).
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("EPICURE_DIR", os.path.join(HERE, "epicure-cooc"))


def _load():
    if not os.path.isdir(MODEL_DIR):
        sys.exit(
            f"Model folder not found at: {MODEL_DIR}\n"
            "Download it first, e.g.:\n"
            "  python -c \"from huggingface_hub import snapshot_download as d; "
            "d(repo_id='Kaikaku/epicure-cooc', local_dir='epicure-cooc')\"\n"
            "or set EPICURE_DIR to its location."
        )
    sys.path.insert(0, MODEL_DIR)
    try:
        from epicure import Epicure
    except ImportError:
        sys.exit("Could not import epicure.py from the model folder. "
                 "Install deps: pip install huggingface_hub safetensors numpy")
    return Epicure.from_pretrained(MODEL_DIR)


def _resolve(m, name):
    """Map a user term to a real vocab token, suggesting close matches."""
    if name in m.vocab:
        return name
    close = difflib.get_close_matches(name, m.vocab.keys(), n=8, cutoff=0.6)
    contains = [k for k in m.vocab if name.lower() in k.lower()][:8]
    hints = list(dict.fromkeys(close + contains))
    msg = f"'{name}' is not in the vocabulary."
    if hints:
        msg += " Did you mean: " + ", ".join(hints) + " ?"
    else:
        msg += " Try: python epicure_query.py find <part_of_name>"
    sys.exit(msg)


def _norm_cuisine(m, c):
    key = c if c.startswith("cuisine:") else f"cuisine:{c}"
    if key not in m.supervised_poles:
        avail = [k.split(":", 1)[1] for k in m.supervised_poles if k.startswith("cuisine:")]
        sys.exit(f"Unknown cuisine '{c}'. Available: {', '.join(avail)}")
    return key


def _print_pairs(rows):
    for name, score in rows:
        print(f"  {name:<28} {score:.3f}")


def cmd_neighbors(m, a):
    seed = _resolve(m, a.ingredient)
    print(f"Cooked-with neighbours of '{seed}':")
    _print_pairs(m.neighbors(seed, k=a.k))


def cmd_slerp(m, a):
    seed = _resolve(m, a.ingredient)
    cuisine = _norm_cuisine(m, a.cuisine)
    label = cuisine.split(":", 1)[1]
    print(f"'{seed}' rotated +{a.theta:g} deg toward {label}:")
    _print_pairs(m.slerp(seed, cuisine, theta_deg=a.theta, k=a.k))


def cmd_mode(m, a):
    seed = _resolve(m, a.ingredient)
    kind = None if a.kind == "all" else a.kind
    print(f"Closest {a.kind} modes for '{seed}':")
    for mid, label, score in m.closest_mode(seed, kind=kind, k=a.k):
        print(f"  [{mid:<8}] {label:<46} {score:.3f}")


def cmd_cuisines(m, a):
    print("Cuisine directions available for `slerp`:")
    for k in m.supervised_poles:
        if k.startswith("cuisine:"):
            print("  " + k.split(":", 1)[1])


def cmd_find(m, a):
    hits = sorted(k for k in m.vocab if a.term.lower() in k.lower())
    if not hits:
        print(f"No ingredients matching '{a.term}'.")
        return
    print(f"{len(hits)} ingredient(s) matching '{a.term}':")
    for h in hits:
        print("  " + h)


def cmd_info(m, a):
    print(repr(m))
    print(f"  Model folder: {MODEL_DIR}")
    print(f"  Ingredients : {len(m.vocab)}")
    print(f"  Modes       : {len(m.modes)}")
    n_cuisine = sum(1 for k in m.supervised_poles if k.startswith('cuisine:'))
    print(f"  Cuisines    : {n_cuisine}")


def main():
    p = argparse.ArgumentParser(description="Query the epicure-cooc ingredient embedding.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("neighbors", help="nearest cooked-with ingredients")
    s.add_argument("ingredient")
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(func=cmd_neighbors)

    s = sub.add_parser("slerp", help="rotate an ingredient toward a cuisine")
    s.add_argument("ingredient")
    s.add_argument("cuisine")
    s.add_argument("--theta", type=float, default=30.0, help="rotation angle in degrees (default 30)")
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(func=cmd_slerp)

    s = sub.add_parser("mode", help="closest emergent cluster(s)")
    s.add_argument("ingredient")
    s.add_argument("--kind", default="factor",
                   help="factor (default), cuisine, food_group, nova_level, "
                        "cf_sensory, usda_nutrient, or all")
    s.add_argument("-k", type=int, default=3)
    s.set_defaults(func=cmd_mode)

    s = sub.add_parser("cuisines", help="list cuisine directions"); s.set_defaults(func=cmd_cuisines)
    s = sub.add_parser("find", help="search the vocabulary"); s.add_argument("term"); s.set_defaults(func=cmd_find)
    s = sub.add_parser("info", help="model summary"); s.set_defaults(func=cmd_info)

    a = p.parse_args()
    a.func(_load(), a)


if __name__ == "__main__":
    main()
