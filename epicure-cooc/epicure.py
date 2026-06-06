"""
Epicure: minimal loader for the three sibling ingredient embeddings.

Usage
-----
    from epicure import Epicure
    m = Epicure.from_pretrained("Kaikaku/epicure-cooc")
    m.neighbors("chicken", k=5)
    m.slerp("rice", "cuisine:South_Asian/South Asian", theta_deg=30, k=5)
    m.closest_mode("miso", kind="factor", k=3)

The three repos (epicure-cooc, epicure-core, epicure-chem) ship the same loader.
Paper: https://arxiv.org/abs/2605.22391
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _try_hf_download(repo_id: str, filename: str, revision: str | None = None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for from_pretrained(). "
            "Install with: pip install huggingface_hub safetensors numpy"
        ) from exc
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _load_safetensors(path: str) -> np.ndarray:
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise ImportError("safetensors required. pip install safetensors") from exc
    return load_file(path)["embeddings"]


def _unit(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


@dataclass
class ModeEntry:
    mode_id: str
    kind: str
    property: str
    label: str
    n_members: int
    members: list[str]
    pole: np.ndarray  # (d_model,) unit-normalised


class Epicure:
    """Lookup-table embedding with neighbour, SLERP, and closest-mode operators."""

    def __init__(
        self,
        E: np.ndarray,
        vocab: dict[str, int],
        modes: list[ModeEntry],
        supervised_poles: dict[str, np.ndarray],
        config: dict,
    ):
        self.E_raw = E.astype(np.float32)
        self.E = _unit(self.E_raw)
        self.vocab = vocab
        self.itos = {i: n for n, i in vocab.items()}
        self.modes = modes
        self.supervised_poles = supervised_poles
        self.config = config

    # ----- constructors -----

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, revision: str | None = None) -> "Epicure":
        if os.path.isdir(repo_id_or_path):
            base = repo_id_or_path
            getp = lambda fn: os.path.join(base, fn)
        else:
            getp = lambda fn: _try_hf_download(repo_id_or_path, fn, revision=revision)
        E = _load_safetensors(getp("embeddings.safetensors"))
        with open(getp("vocab.json")) as f:
            vocab = json.load(f)
        with open(getp("modes.json")) as f:
            modes_raw = json.load(f)
        with open(getp("supervised_poles.json")) as f:
            sup_raw = json.load(f)
        with open(getp("config.json")) as f:
            config = json.load(f)
        modes = [
            ModeEntry(
                mode_id=m["mode_id"],
                kind=m["kind"],
                property=m["property"],
                label=m["label"],
                n_members=m["n_members"],
                members=m["members"],
                pole=np.array(m["pole"], dtype=np.float32),
            )
            for m in modes_raw
        ]
        supervised_poles = {k: np.array(v, dtype=np.float32) for k, v in sup_raw.items()}
        return cls(E, vocab, modes, supervised_poles, config)

    # ----- core operators -----

    def vec(self, name: str, normalised: bool = True) -> np.ndarray:
        i = self.vocab[name]
        return self.E[i] if normalised else self.E_raw[i]

    def neighbors(self, name: str, k: int = 5, exclude_self: bool = True) -> list[tuple[str, float]]:
        v = self.vec(name)
        sims = self.E @ v
        order = np.argsort(-sims)
        start = 1 if exclude_self else 0
        return [(self.itos[int(i)], float(sims[i])) for i in order[start:start + k]]

    def slerp(
        self,
        seed: str,
        direction: str | np.ndarray,
        theta_deg: float,
        k: int = 5,
        exclude_seed: bool = True,
    ) -> list[tuple[str, float]]:
        """Rotate the seed vector toward a unit direction by angle theta on the unit sphere.

        ``direction`` is either a supervised pole key (e.g.
        ``"cuisine:South_Asian"``) or a raw (d_model,) np.ndarray.
        At theta=0 the query is the seed. At theta=60deg cosine to seed = 0.5.
        With ``exclude_seed=True`` (default) the seed ingredient is removed from results
        (the paper's reported tables also exclude it).
        """
        seed_idx = self.vocab[seed]
        v = self.E[seed_idx]
        d = self.supervised_poles[direction] if isinstance(direction, str) else direction
        d = np.asarray(d, dtype=np.float32)
        d = _unit(d)
        # Gram-Schmidt: orthogonal component of d relative to v
        d_perp = d - (d @ v) * v
        n_perp = np.linalg.norm(d_perp)
        if n_perp < 1e-9:
            # d is colinear with v: rotation has no defined plane; return seed neighbours
            return self.neighbors(seed, k=k)
        d_perp = d_perp / n_perp
        theta = np.deg2rad(float(theta_deg))
        q = np.cos(theta) * v + np.sin(theta) * d_perp
        q = _unit(q)
        sims = self.E @ q
        if exclude_seed:
            sims[seed_idx] = -np.inf
        order = np.argsort(-sims)
        return [(self.itos[int(i)], float(sims[i])) for i in order[:k]]

    def closest_mode(
        self,
        name: str,
        kind: str | None = None,
        k: int = 3,
    ) -> list[tuple[str, str, float]]:
        """Return the top-k closest modes to the named ingredient.

        ``kind`` filters by mode kind: 'factor', 'cuisine', 'food_group',
        'nova_level', 'cf_sensory', 'usda_nutrient' or None for all.
        """
        v = self.vec(name)
        scored = []
        for m in self.modes:
            if kind is not None and m.kind != kind:
                continue
            scored.append((m.mode_id, m.label, float(_unit(m.pole) @ v)))
        scored.sort(key=lambda x: -x[2])
        return scored[:k]

    def mode_members(self, mode_id: str, k: int | None = None) -> list[str]:
        for m in self.modes:
            if m.mode_id == mode_id:
                return m.members[:k] if k is not None else m.members
        raise KeyError(mode_id)

    # ----- introspection -----

    def list_supervised_poles(self, prefix: str | None = None) -> list[str]:
        if prefix is None:
            return list(self.supervised_poles.keys())
        return [k for k in self.supervised_poles if k.startswith(prefix)]

    def list_modes(self, kind: str | None = None) -> list[tuple[str, str]]:
        if kind is None:
            return [(m.mode_id, m.label) for m in self.modes]
        return [(m.mode_id, m.label) for m in self.modes if m.kind == kind]

    def __repr__(self) -> str:
        return (
            f"Epicure(schema={self.config.get('schema')!r}, "
            f"d_model={self.config.get('d_model')}, "
            f"vocab_size={self.config.get('vocab_size')}, "
            f"modes={len(self.modes)}, "
            f"supervised_poles={len(self.supervised_poles)})"
        )
