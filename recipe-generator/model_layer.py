"""
model_layer.py - turns a free-text prompt into an "ingredient brief" using the
epicure-cooc co-occurrence embedding.

This is the part of the recipe generator that the embedding actually drives:
it parses the prompt for ingredients and cuisine, then uses the model's
neighbors / bridge / cuisine-SLERP operators to assemble a ranked candidate set.
The brief is later handed to Claude, which writes the recipe around it.
"""

from __future__ import annotations

import difflib
import os
import re
import sys

import numpy as np

# Locate the model folder (sibling ../epicure-cooc by default).
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    "EPICURE_DIR", os.path.normpath(os.path.join(HERE, "..", "epicure-cooc"))
)

# Map user-facing cuisine words to the seven supervised cuisine poles.
CUISINE_SYNONYMS = {
    "east_asian": "East_Asian", "chinese": "East_Asian", "korean": "East_Asian",
    "cantonese": "East_Asian", "sichuan": "East_Asian", "szechuan": "East_Asian",
    "japanese": "Japanese", "japan": "Japanese", "izakaya": "Japanese",
    "sushi": "Japanese", "ramen": "Japanese",
    "southeast_asian": "Southeast_Asian", "thai": "Southeast_Asian",
    "vietnamese": "Southeast_Asian", "filipino": "Southeast_Asian",
    "indonesian": "Southeast_Asian", "malaysian": "Southeast_Asian",
    "south_asian": "South_Asian", "indian": "South_Asian",
    "pakistani": "South_Asian", "sri_lankan": "South_Asian",
    "bangladeshi": "South_Asian", "curry": "South_Asian",
    "mediterranean": "Mediterranean", "italian": "Mediterranean",
    "greek": "Mediterranean", "spanish": "Mediterranean",
    "lebanese": "Mediterranean", "turkish": "Mediterranean",
    "moroccan": "Mediterranean", "middle_eastern": "Mediterranean",
    "western_atlantic": "Western_Atlantic", "american": "Western_Atlantic",
    "british": "Western_Atlantic", "french": "Western_Atlantic",
    "german": "Western_Atlantic", "western": "Western_Atlantic",
    "latin_american": "Latin_American", "mexican": "Latin_American",
    "latin": "Latin_American", "peruvian": "Latin_American",
    "brazilian": "Latin_American", "tex_mex": "Latin_American",
    "cuban": "Latin_American", "argentinian": "Latin_American",
}

# Lightweight dietary / style flags passed through to Claude.
CONSTRAINT_KEYWORDS = {
    "vegetarian": "vegetarian", "veggie": "vegetarian",
    "vegan": "vegan", "plant based": "vegan", "plant-based": "vegan",
    "pescatarian": "pescatarian",
    "gluten free": "gluten-free", "gluten-free": "gluten-free",
    "dairy free": "dairy-free", "dairy-free": "dairy-free",
    "spicy": "spicy", "mild": "mild",
    "quick": "quick", "fast": "quick", "weeknight": "quick",
    "healthy": "healthy", "low carb": "low-carb", "low-carb": "low-carb",
    "keto": "low-carb", "comfort": "comfort food",
}

_STOPWORDS = {
    "a", "an", "the", "with", "and", "or", "of", "for", "in", "on", "to",
    "some", "make", "made", "want", "like", "please", "recipe", "dish",
    "meal", "dinner", "lunch", "cook", "cooking", "using", "use", "give",
    "me", "something", "i", "id", "we", "my", "add", "more", "less", "it",
}


class RecipeModel:
    """Wraps epicure-cooc with prompt-parsing and brief-building helpers."""

    def __init__(self, model_dir: str = MODEL_DIR):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"epicure-cooc model not found at {model_dir}. "
                "Set EPICURE_DIR or place the model folder beside this app."
            )
        sys.path.insert(0, model_dir)
        from epicure import Epicure  # shipped inside the model folder

        self.m = Epicure.from_pretrained(model_dir)
        self.vocab = set(self.m.vocab.keys())
        self.cuisine_poles = {
            k.split(":", 1)[1]
            for k in self.m.supervised_poles
            if k.startswith("cuisine:")
        }

    # ---------- parsing ----------

    def parse(self, text: str) -> dict:
        """Extract ingredients, cuisine, and constraints from free text."""
        low = " " + text.lower().strip() + " "

        constraints = sorted({
            tag for kw, tag in CONSTRAINT_KEYWORDS.items() if kw in low
        })

        cuisine = None
        for syn, pole in CUISINE_SYNONYMS.items():
            phrase = syn.replace("_", " ")
            if phrase in low and pole in self.cuisine_poles:
                cuisine = pole
                break

        ingredients = self._extract_ingredients(text)
        return {
            "ingredients": ingredients,
            "cuisine": cuisine,
            "constraints": constraints,
        }

    def _extract_ingredients(self, text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]*", text.lower())
        found, seen = [], set()
        n = len(words)

        def singular(w: str) -> str:
            if w.endswith("ies") and len(w) > 4:
                return w[:-3] + "y"
            if w.endswith("es") and len(w) > 4:
                return w[:-2]
            if w.endswith("s") and len(w) > 3:
                return w[:-1]
            return w

        covered = set()  # word indices already claimed by a longer match
        # Prefer longer n-grams first (e.g. "black pepper" before "pepper").
        for size in (3, 2, 1):
            for i in range(n - size + 1):
                idxs = range(i, i + size)
                if any(j in covered for j in idxs):
                    continue
                gram = words[i:i + size]
                if size == 1 and gram[0] in _STOPWORDS:
                    continue
                # try the n-gram, then with its last word singularised
                variants = ["_".join(gram)]
                variants.append("_".join(gram[:-1] + [singular(gram[-1])]))
                for token in variants:
                    if token in self.vocab and token not in seen:
                        found.append(token)
                        seen.add(token)
                        covered.update(idxs)
                        break
        # Fuzzy fallback only if nothing matched, on content words.
        if not found:
            for w in words:
                if w in _STOPWORDS or len(w) < 4:
                    continue
                m = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.85)
                if m and m[0] not in seen:
                    found.append(m[0]); seen.add(m[0])
        return found

    # ---------- brief building ----------

    def build_brief(self, parsed: dict, max_candidates: int = 14) -> dict:
        """Build a ranked candidate ingredient set with provenance + scores."""
        seeds = parsed["ingredients"]
        cuisine = parsed["cuisine"]
        scored: dict[str, dict] = {}

        def add(name, score, source):
            if name in seeds:
                return
            cur = scored.get(name)
            if cur is None or score > cur["score"]:
                scored[name] = {"name": name, "score": round(float(score), 3),
                                "source": source}

        # 1) neighbours of each named ingredient
        for s in seeds:
            for name, sc in self.m.neighbors(s, k=8):
                add(name, sc, f"pairs with {s}")

        # 2) bridges between multiple seeds
        if len(seeds) >= 2:
            vecs = [self.m.vec(s) for s in seeds]
            mid = np.sum(vecs, axis=0)
            mid = mid / (np.linalg.norm(mid) + 1e-9)
            sims = self.m.E @ mid
            for i in np.argsort(-sims)[:20]:
                name = self.m.itos[int(i)]
                # require a real link to *each* seed, not just the average
                if all(float(self.m.vec(s) @ self.m.vec(name)) > 0.10 for s in seeds):
                    add(name, sims[i], "bridges your ingredients")

        # 3) cuisine pull via SLERP + closest cuisine cluster
        cuisine_members: list[str] = []
        if cuisine:
            key = f"cuisine:{cuisine}"
            slerp_seeds = seeds or self._cuisine_anchor(key)
            for s in slerp_seeds:
                try:
                    for name, sc in self.m.slerp(s, key, theta_deg=30, k=6):
                        add(name, sc, f"{cuisine.replace('_', ' ')} direction")
                except KeyError:
                    pass
            cuisine_members = self._cuisine_cluster_members(cuisine)
            for name in cuisine_members[:8]:
                if name in self.vocab:
                    base = max((float(self.m.vec(s) @ self.m.vec(name))
                                for s in seeds), default=0.3)
                    add(name, base, f"{cuisine.replace('_', ' ')} staple")

        candidates = sorted(scored.values(), key=lambda x: -x["score"])[:max_candidates]

        rationale = self._rationale(seeds, cuisine, candidates)
        return {
            "seeds": seeds,
            "cuisine": cuisine,
            "constraints": parsed["constraints"],
            "candidates": candidates,
            "rationale": rationale,
        }

    def _cuisine_anchor(self, key: str) -> list[str]:
        """Pick a couple of vocab ingredients nearest the cuisine pole as seeds."""
        pole = self.m.supervised_poles[key]
        pole = pole / (np.linalg.norm(pole) + 1e-9)
        sims = self.m.E @ pole
        return [self.m.itos[int(i)] for i in np.argsort(-sims)[:2]]

    def _cuisine_cluster_members(self, cuisine: str) -> list[str]:
        key = cuisine.lower().replace("_", " ")
        best, members = -1.0, []
        for mode in self.m.modes:
            if key in mode.label.lower():
                if mode.n_members > best:
                    best, members = mode.n_members, list(mode.members)
        return members

    def _rationale(self, seeds, cuisine, candidates) -> str:
        bits = []
        if seeds:
            bits.append("Seeded with " + ", ".join(s.replace("_", " ") for s in seeds))
        else:
            bits.append("No specific ingredient named")
        if cuisine:
            bits.append(f"pulled toward {cuisine.replace('_', ' ')}")
        if candidates:
            top = ", ".join(c["name"].replace("_", " ") for c in candidates[:5])
            bits.append(f"top co-occurring partners: {top}")
        return "; ".join(bits) + "."


if __name__ == "__main__":
    rm = RecipeModel()
    for prompt in [
        "a comforting beef and cinnamon stew",
        "something spicy with tofu, korean style",
        "quick vegetarian pasta with mushrooms",
        "mexican dinner using corn",
    ]:
        p = rm.parse(prompt)
        b = rm.build_brief(p)
        print("PROMPT:", prompt)
        print("  parsed:", p)
        print("  candidates:", [(c["name"], c["score"], c["source"]) for c in b["candidates"][:8]])
        print("  rationale:", b["rationale"])
        print()
