"""Prompt sources for preference-data construction.

Every loader returns a list of ``{"prompt", "source", "offline_unsafe"}`` dicts, where
``offline_unsafe`` is an optional human-annotated unsafe response (only PKU-SafeRLHF has one)
that can serve as an off-policy ``rejected`` fallback.
"""

from __future__ import annotations

import json
import logging
import random
import re

log = logging.getLogger(__name__)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _load_hf(spec: dict):
    from datasets import load_dataset

    return load_dataset(spec["hf_path"], spec.get("hf_config"), split=spec.get("split", "train"))


def _pku_saferlhf(spec: dict) -> list[dict]:
    rows = []
    for r in _load_hf(spec):
        s0, s1 = r["is_response_0_safe"], r["is_response_1_safe"]
        if s0 and s1:
            continue  # keep prompts where at least one annotated response was unsafe
        unsafe = r["response_0"] if not s0 else r["response_1"]
        rows.append({"prompt": r["prompt"], "offline_unsafe": unsafe})
    return rows


def _or_bench(spec: dict) -> list[dict]:
    return [{"prompt": r["prompt"]} for r in _load_hf(spec)]


def _alpaca(spec: dict) -> list[dict]:
    rows = []
    for r in _load_hf(spec):
        if r.get("input"):
            if not spec.get("include_with_input", False):
                continue
            prompt = f"{r['instruction']}\n\n{r['input']}"
        else:
            prompt = r["instruction"]
        rows.append({"prompt": prompt})
    return rows


def _generic_hf(spec: dict) -> list[dict]:
    field = spec.get("prompt_field", "prompt")
    return [{"prompt": r[field]} for r in _load_hf(spec)]


def _jsonl(spec: dict) -> list[dict]:
    rows = []
    with open(spec["path"], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"prompt": r["prompt"], "offline_unsafe": r.get("unsafe_response")})
    return rows


LOADERS = {
    "pku_saferlhf": _pku_saferlhf,
    "or_bench": _or_bench,
    "alpaca": _alpaca,
    "hf": _generic_hf,
    "jsonl": _jsonl,
}


def load_source(spec: dict, seed: int) -> list[dict]:
    loader = LOADERS[spec.get("loader", spec["name"])]
    rows = loader(spec)
    min_len, max_len = spec.get("min_chars", 15), spec.get("max_chars", 1500)
    seen, out = set(), []
    for r in rows:
        p = (r["prompt"] or "").strip()
        key = normalize(p)
        if not (min_len <= len(p) <= max_len) or key in seen:
            continue
        seen.add(key)
        out.append({"prompt": p, "source": spec["name"], "offline_unsafe": r.get("offline_unsafe")})
    random.Random(seed).shuffle(out)
    n = spec.get("max_prompts")
    log.info("source %s: %d unique prompts, keeping %s", spec["name"], len(out), n or "all")
    return out[:n] if n else out
