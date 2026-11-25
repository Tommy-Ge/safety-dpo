"""Loaders for evaluation benchmarks. Each accepts a local-file override for offline tests."""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

HARMBENCH_TEST_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_test.csv"
)
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_harmbench(cfg: dict) -> list[dict]:
    """HarmBench text test set. Copyright behaviors are excluded by default because they need
    the hash-based copyright judge rather than the classifier."""
    import pandas as pd

    df = pd.read_csv(cfg.get("behaviors_path") or cfg.get("behaviors_url", HARMBENCH_TEST_URL))
    cats = set(cfg.get("functional_categories", ["standard", "contextual"]))
    df = df[df["FunctionalCategory"].isin(cats)]
    rows = []
    for _, r in df.iterrows():
        ctx = r.get("ContextString")
        ctx = ctx if isinstance(ctx, str) and ctx.strip() else None
        rows.append(
            {
                "id": r["BehaviorID"],
                "behavior": r["Behavior"],
                "context": ctx,
                "category": r["SemanticCategory"],
                "functional_category": r["FunctionalCategory"],
                # HarmBench convention for contextual behaviors
                "prompt": f"{ctx}\n\n---\n\n{r['Behavior']}" if ctx else r["Behavior"],
            }
        )
    return rows


def load_jbb(cfg: dict, split: str = "harmful") -> list[dict]:
    if cfg.get("path"):
        return [r for r in _read_jsonl(cfg["path"]) if r.get("split", "harmful") == split]
    from datasets import load_dataset

    ds = load_dataset(cfg.get("hf_path", "JailbreakBench/JBB-Behaviors"), "behaviors", split=split)
    return [
        {
            "id": f"jbb-{split}-{i}",
            "index": i,
            "goal": r["Goal"],
            "behavior": r["Behavior"],
            "category": r["Category"],
        }
        for i, r in enumerate(ds)
    ]


JBB_ARTIFACTS_BASE = "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"


def load_jbb_artifact(method: str, model_name: str) -> dict[int, str | None]:
    """Transfer-attack prompts from the official JailbreakBench artifacts repo, keyed by the 0-based
    behavior index (goal strings differ slightly between the artifacts and the HF dataset).

    Fetched directly from GitHub (the ``jailbreakbench`` pip package pins an old litellm API).
    """
    import requests

    info = requests.get(f"{JBB_ARTIFACTS_BASE}/{method}/attack-info.json", timeout=60).json()
    attack_type = info["models"][model_name][0]
    url = f"{JBB_ARTIFACTS_BASE}/{method}/{attack_type}/{model_name}.json"
    art = requests.get(url, timeout=60).json()
    return {int(j["index"]): j.get("prompt") for j in art["jailbreaks"]}


def load_xstest(cfg: dict) -> list[dict]:
    """XSTest (450 prompts: 250 safe look-alikes + 200 unsafe contrasts). Defaults to the official,
    ungated CSV on GitHub; ``hf_path`` (e.g. the gated walledai/XSTest) is also accepted."""
    if cfg.get("path"):
        return _read_jsonl(cfg["path"])
    if cfg.get("hf_path"):
        from datasets import load_dataset

        rows = list(load_dataset(cfg["hf_path"], split=cfg.get("split", "test")))
    else:
        import pandas as pd

        rows = pd.read_csv(cfg.get("csv_url", XSTEST_URL)).to_dict("records")
    return [
        {"id": f"xstest-{i}", "prompt": r["prompt"], "label": r["label"], "type": r["type"]}
        for i, r in enumerate(rows)
    ]


def eval_prompt_texts(eval_cfg: dict) -> list[str]:
    """All prompts of the enabled eval benchmarks, for training-data decontamination."""
    texts: list[str] = []
    b = eval_cfg.get("benchmarks", {})
    if b.get("harmbench", {}).get("enabled", True):
        texts += [r["behavior"] for r in load_harmbench(b.get("harmbench", {}))]
    if b.get("jbb", {}).get("enabled", True):
        for split in ("harmful", "benign"):
            texts += [r["goal"] for r in load_jbb(b.get("jbb", {}), split)]
    if b.get("xstest", {}).get("enabled", True):
        texts += [r["prompt"] for r in load_xstest(b.get("xstest", {}))]
    return texts
