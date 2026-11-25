"""Construct on-policy "safe vs. unsafe" preference pairs.

Pipeline (each stage caches its output under ``work_dir`` and can be re-run alone)::

    collect   -> prompts.jsonl      harmful / borderline / general prompts, jailbreak augmentation,
                                    decontamination against the eval benchmarks
    generate  -> generations.jsonl  n on-policy samples per prompt from the policy, plus one
                                    "reference" sample under a system prompt:
                                      harmful prompts  -> SAFETY_SYSTEM_PROMPT  (a good refusal)
                                      benign prompts   -> OVERCAUTIOUS_SYSTEM_PROMPT (an over-refusal)
    judge     -> judged.jsonl       Llama Guard 3 safety label + refusal flag for every response
    pair      -> train.jsonl / test.jsonl / stats.json

Pairing rules (see ``make_pair``):

* harmful prompt:  chosen = safe response  (prefer the safety-system-prompt reference, i.e. context
                   distillation; else a safe on-policy sample; else a refusal template)
                   rejected = unsafe on-policy sample (else the PKU-SafeRLHF annotated unsafe response;
                   else the prompt is dropped - the model is already safe on it)
* benign prompt (borderline / general): chosen = a compliant & safe on-policy sample,
                   rejected = an on-policy refusal, else the over-cautious reference, else a template.
                   These pairs explicitly penalise over-refusal so that safety training does not
                   turn into "refuse everything" and general capability (MMLU) is preserved.

The DPO prompt never contains the system prompt, so the model learns to behave safely without it.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path

from ..generation import SamplingConfig, build_generator
from ..prompts import (
    OVERCAUTIOUS_SYSTEM_PROMPT,
    OVERREFUSAL_TEMPLATES,
    SAFE_REFUSAL_TEMPLATES,
    SAFETY_SYSTEM_PROMPT,
    TRAIN_JAILBREAKS,
    is_refusal,
    wrap,
)
from .decontam import Decontaminator
from .sources import load_source

log = logging.getLogger(__name__)

KINDS = ("harmful", "borderline", "general")


# ---------------------------------------------------------------------------- io helpers
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------- stage 1: collect
def collect(cfg: dict, eval_cfg: dict | None = None) -> list[dict]:
    seed = cfg.get("seed", 42)
    rng = random.Random(seed)
    rows: list[dict] = []
    for kind in KINDS:
        for spec in cfg["sources"].get(kind, []) or []:
            for r in load_source(spec, seed):
                rows.append({"kind": kind, "base_prompt": r["prompt"], **r})

    dc_cfg = cfg.get("decontaminate", {})
    if dc_cfg.get("enabled", False):
        from ..eval.benchmarks import eval_prompt_texts

        dc = Decontaminator(
            eval_prompt_texts(eval_cfg or {}),
            n=dc_cfg.get("ngram", 8),
            threshold=dc_cfg.get("threshold", 0.5),
        )
        before = len(rows)
        rows = [r for r in rows if not dc.is_contaminated(r["prompt"])]
        log.info("decontamination removed %d / %d prompts", before - len(rows), before)

    # jailbreak augmentation: add wrapped copies of a fraction of harmful prompts
    aug = cfg.get("jailbreak_augment", {})
    families = aug.get("families", list(TRAIN_JAILBREAKS))
    harmful = [r for r in rows if r["kind"] == "harmful"]
    n_aug = int(len(harmful) * aug.get("fraction", 0.0))
    for r in rng.sample(harmful, n_aug):
        fam = rng.choice(families)
        rows.append({**r, "prompt": wrap(r["base_prompt"], fam), "jailbreak": fam, "offline_unsafe": None})

    for i, r in enumerate(rows):
        r.setdefault("jailbreak", None)
        r["id"] = f"{r['kind'][0]}{i:06d}"
    log.info("collected %s", Counter(r["kind"] for r in rows))
    return rows


# ---------------------------------------------------------------------------- stage 2: generate
def generate(cfg: dict, rows: list[dict]) -> list[dict]:
    s = cfg.get("sampling", {})
    sample_cfg = SamplingConfig(
        n=s.get("n", 4),
        temperature=s.get("temperature", 1.0),
        top_p=s.get("top_p", 0.95),
        max_new_tokens=s.get("max_new_tokens", 512),
    )
    ref_cfg = SamplingConfig(
        n=1,
        temperature=s.get("ref_temperature", 0.7),
        top_p=0.95,
        max_new_tokens=s.get("max_new_tokens", 512),
    )
    gen = build_generator(cfg["policy"], seed=cfg.get("seed", 42))
    if sample_cfg.n > 0:
        samples = gen.chat(
            [[{"role": "user", "content": r["prompt"]}] for r in rows], sample_cfg, desc="policy samples"
        )
    else:  # ablation: no on-policy samples, rejected comes from offline annotations only
        samples = [[] for _ in rows]
    refs = gen.chat(
        [
            [
                {
                    "role": "system",
                    "content": SAFETY_SYSTEM_PROMPT if r["kind"] == "harmful" else OVERCAUTIOUS_SYSTEM_PROMPT,
                },
                {"role": "user", "content": r["prompt"]},
            ]
            for r in rows
        ],
        ref_cfg,
        desc="reference",
    )
    gen.close()
    return [{**r, "samples": smp, "ref": ref[0]} for r, smp, ref in zip(rows, samples, refs)]


# ---------------------------------------------------------------------------- stage 3: judge
def judge(cfg: dict, rows: list[dict]) -> list[dict]:
    from ..judges import build_judge

    j = build_judge(cfg["judge"])
    items, index = [], []
    for ri, r in enumerate(rows):
        for si, resp in enumerate(r["samples"]):
            items.append({"prompt": r["prompt"], "response": resp, "harmful": r["kind"] == "harmful"})
            index.append((ri, si))
        items.append({"prompt": r["prompt"], "response": r["ref"], "harmful": r["kind"] == "harmful"})
        index.append((ri, -1))
    verdicts = j.judge(items)
    j.close()

    out = [{**r, "sample_unsafe": [False] * len(r["samples"]), "ref_unsafe": False} for r in rows]
    for (ri, si), v in zip(index, verdicts):
        if si < 0:
            out[ri]["ref_unsafe"] = bool(v)
        else:
            out[ri]["sample_unsafe"][si] = bool(v)
    for r in out:
        r["sample_refusal"] = [is_refusal(x) for x in r["samples"]]
        r["ref_refusal"] = is_refusal(r["ref"])
    return out


# ---------------------------------------------------------------------------- stage 4: pair
def _msg(role: str, content: str) -> list[dict]:
    return [{"role": role, "content": content}]


def make_pair(r: dict, pcfg: dict, rng: random.Random) -> dict | None:
    samples, unsafe, refusal = r["samples"], r["sample_unsafe"], r["sample_refusal"]
    chosen = rejected = None

    if r["kind"] == "harmful":
        # chosen: a safe answer that does not comply
        if not r["ref_unsafe"] and r["ref"].strip():
            chosen = (r["ref"], "ref_safety_sysprompt")
        else:
            safe = [x for x, u in zip(samples, unsafe) if not u and x.strip()]
            if safe:
                chosen = (rng.choice(safe), "policy_safe")
            elif pcfg.get("chosen_fallback", "template") == "template":
                chosen = (rng.choice(SAFE_REFUSAL_TEMPLATES), "template")
        # rejected: an unsafe, non-refusing on-policy sample
        bad = [x for x, u, rf in zip(samples, unsafe, refusal) if u and not rf]
        if bad:
            rejected = (rng.choice(bad), "policy_unsafe")
        elif pcfg.get("rejected_fallback_harmful", "offline") == "offline" and r.get("offline_unsafe"):
            rejected = (r["offline_unsafe"], "offline_unsafe")
    else:
        good = [x for x, u, rf in zip(samples, unsafe, refusal) if not u and not rf and x.strip()]
        if good:
            chosen = (rng.choice(good), "policy_compliant")
        refusals = [x for x, rf in zip(samples, refusal) if rf and x.strip()]
        if refusals:
            rejected = (rng.choice(refusals), "policy_refusal")
        elif r["ref_refusal"] and r["ref"].strip():
            rejected = (r["ref"], "ref_overcautious_sysprompt")
        elif pcfg.get("rejected_fallback_benign", "template") == "template":
            rejected = (rng.choice(OVERREFUSAL_TEMPLATES), "template")

    if chosen is None or rejected is None or chosen[0].strip() == rejected[0].strip():
        return None
    return {
        "id": r["id"],
        "kind": r["kind"],
        "source": r["source"],
        "jailbreak": r.get("jailbreak"),
        "prompt": _msg("user", r["prompt"]),
        "chosen": _msg("assistant", chosen[0]),
        "rejected": _msg("assistant", rejected[0]),
        "chosen_from": chosen[1],
        "rejected_from": rejected[1],
    }


def pair(cfg: dict, rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    pcfg = cfg.get("pairing", {})
    rng = random.Random(cfg.get("seed", 42))
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}
    for r in rows:
        p = make_pair(r, pcfg, rng)
        if p:
            by_kind[p["kind"]].append(p)

    mix = pcfg.get("mix", {"harmful": 0.6, "borderline": 0.25, "general": 0.15})
    max_pairs = pcfg.get("max_pairs")
    selected: list[dict] = []
    for k in KINDS:
        rng.shuffle(by_kind[k])
        cap = int(mix.get(k, 0) * max_pairs) if max_pairs else None
        selected += by_kind[k][:cap] if cap is not None else by_kind[k]
    rng.shuffle(selected)

    n_test = max(1, int(len(selected) * pcfg.get("test_fraction", 0.05))) if len(selected) > 1 else 0
    test, train = selected[:n_test], selected[n_test:]

    harmful_rows = [r for r in rows if r["kind"] == "harmful"]
    benign_rows = [r for r in rows if r["kind"] != "harmful"]

    def _rate(rs, key):
        flat = [v for r in rs for v in r[key]]
        return round(sum(flat) / len(flat), 4) if flat else None

    stats = {
        "n_prompts": Counter(r["kind"] for r in rows),
        "n_pairs_available": {k: len(v) for k, v in by_kind.items()},
        "n_train": len(train),
        "n_test": len(test),
        "train_by_kind": Counter(p["kind"] for p in train),
        "chosen_from": Counter(p["chosen_from"] for p in selected),
        "rejected_from": Counter(p["rejected_from"] for p in selected),
        "jailbreak_family": Counter(p["jailbreak"] or "none" for p in selected if p["kind"] == "harmful"),
        # how often the *base* policy misbehaves on the training prompts (sanity check)
        "policy_unsafe_rate_on_harmful": _rate(harmful_rows, "sample_unsafe"),
        "policy_unsafe_rate_on_harmful_jailbreak": _rate(
            [r for r in harmful_rows if r.get("jailbreak")], "sample_unsafe"
        ),
        "policy_refusal_rate_on_benign": _rate(benign_rows, "sample_refusal"),
    }
    return train, test, json.loads(json.dumps(stats))


# ---------------------------------------------------------------------------- driver
def run(cfg: dict, eval_cfg: dict | None = None, stage: str = "all", overwrite: bool = False) -> dict | None:
    work, out = Path(cfg.get("work_dir", "data/work")), Path(cfg.get("out_dir", "data/processed"))
    f_prompts, f_gen, f_judged = work / "prompts.jsonl", work / "generations.jsonl", work / "judged.jsonl"
    stages = ["collect", "generate", "judge", "pair"] if stage == "all" else [stage]

    if "collect" in stages:
        if overwrite or not f_prompts.exists():
            write_jsonl(f_prompts, collect(cfg, eval_cfg))
        else:
            log.info("reusing %s", f_prompts)
    if "generate" in stages:
        if overwrite or not f_gen.exists():
            write_jsonl(f_gen, generate(cfg, read_jsonl(f_prompts)))
        else:
            log.info("reusing %s", f_gen)
    if "judge" in stages:
        if overwrite or not f_judged.exists():
            write_jsonl(f_judged, judge(cfg, read_jsonl(f_gen)))
        else:
            log.info("reusing %s", f_judged)
    if "pair" in stages:
        train, test, stats = pair(cfg, read_jsonl(f_judged))
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "test.jsonl", test)
        (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        log.info("wrote %d train / %d test pairs to %s", len(train), len(test), out)
        return stats
    return None
