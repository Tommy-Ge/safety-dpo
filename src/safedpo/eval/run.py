"""Evaluate one model (base or base + LoRA adapter) on all benchmarks.

Stages (cached under ``results/<tag>/``):

    generate -> generations.jsonl  greedy responses to every eval prompt (one model load)
    judge    -> judged.jsonl       HarmBench classifier (HarmBench), Llama Guard 3 (JBB harmful),
                                   refusal on JBB benign / XSTest: string matching always, plus an
                                   LLM refusal judge (``benchmarks.refusal_judge``) when configured
    mmlu     -> mmlu.json          lm-evaluation-harness (num_fewshot from config; 5 by default)
    metrics  -> metrics.json

Attack settings
    HarmBench   direct request + 4 held-out jailbreak templates (never used in training)
    JBB         direct request + transfer attacks from the official JailbreakBench artifacts
                (PAIR / GCG prompts found against other models)
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..data.build import read_jsonl, write_jsonl
from ..generation import SamplingConfig, build_generator
from ..prompts import EVAL_JAILBREAKS, is_refusal, wrap
from .benchmarks import load_harmbench, load_jbb, load_jbb_artifact, load_xstest
from .metrics import rate

log = logging.getLogger(__name__)
STAGES = ("generate", "judge", "mmlu", "metrics")
JUDGED_BENCHES = ("harmbench", "jbb")
REFUSAL_BENCHES = ("jbb_benign", "xstest")


def _limit(rows: list, n: int | None) -> list:
    """Deterministic subsample (keeps original order); used by the quick GPU check."""
    if not n or n >= len(rows):
        return rows
    keep = set(random.Random(0).sample(range(len(rows)), n))
    return [r for i, r in enumerate(rows) if i in keep]


# ---------------------------------------------------------------------------- build eval prompts
def build_eval_records(cfg: dict) -> list[dict]:
    b = cfg["benchmarks"]
    recs: list[dict] = []

    hb = b.get("harmbench", {})
    if hb.get("enabled", True):
        families = hb.get("templates", list(EVAL_JAILBREAKS))
        for beh in _limit(load_harmbench(hb), hb.get("limit")):
            base = {
                "bench": "harmbench",
                "behavior_id": beh["id"],
                "behavior": beh["behavior"],
                "context": beh["context"],
                "category": beh["category"],
            }
            if hb.get("direct", True):
                recs.append({**base, "attack": "direct", "prompt": beh["prompt"]})
            for fam in families:
                recs.append({**base, "attack": f"template:{fam}", "prompt": wrap(beh["prompt"], fam)})

    jb = b.get("jbb", {})
    if jb.get("enabled", True):
        harmful = _limit(load_jbb(jb, "harmful"), jb.get("limit"))
        for beh in harmful:
            recs.append(
                {
                    "bench": "jbb",
                    "behavior_id": beh["id"],
                    "goal": beh["goal"],
                    "category": beh["category"],
                    "attack": "direct",
                    "prompt": beh["goal"],
                }
            )
        for art in jb.get("artifacts", []):
            name = f"{art['method']}@{art['model_name']}"
            try:
                prompts = load_jbb_artifact(art["method"], art["model_name"])
            except Exception as e:  # package missing / network
                log.warning("skipping JBB artifact %s: %s", name, e)
                continue
            for beh in harmful:
                p = prompts.get(beh.get("index", -1))
                # attacks that produced no prompt count as failures (JBB convention)
                recs.append(
                    {
                        "bench": "jbb",
                        "behavior_id": beh["id"],
                        "goal": beh["goal"],
                        "category": beh["category"],
                        "attack": f"artifact:{name}",
                        "prompt": p,
                    }
                )
        if jb.get("benign", True):
            for beh in _limit(load_jbb(jb, "benign"), jb.get("limit")):
                recs.append(
                    {
                        "bench": "jbb_benign",
                        "behavior_id": beh["id"],
                        "goal": beh["goal"],
                        "attack": "benign",
                        "prompt": beh["goal"],
                    }
                )

    xs = b.get("xstest", {})
    if xs.get("enabled", True):
        for r in _limit(load_xstest(xs), xs.get("limit")):
            recs.append(
                {
                    "bench": "xstest",
                    "behavior_id": r["id"],
                    "label": r["label"],
                    "type": r.get("type"),
                    "attack": "none",
                    "prompt": r["prompt"],
                }
            )
    return recs


# ---------------------------------------------------------------------------- stages
def stage_generate(cfg: dict, model: str, adapter: str | None) -> list[dict]:
    recs = build_eval_records(cfg)
    gcfg = cfg.get("generation", {})
    todo = [r for r in recs if r["prompt"]]
    gen = build_generator(gcfg, model=model, adapter=adapter)
    outs = gen.chat(
        [[{"role": "user", "content": r["prompt"]}] for r in todo],
        SamplingConfig(
            max_new_tokens=gcfg.get("max_new_tokens", 512), temperature=gcfg.get("temperature", 0.0)
        ),
        desc="eval generation",
    )
    gen.close()
    for r, o in zip(todo, outs):
        r["response"] = o[0]
    for r in recs:
        r.setdefault("response", None)
    return recs


def stage_judge(cfg: dict, recs: list[dict], benches: list[str] | None = None) -> list[dict]:
    """Judge the given benchmarks (default: all). Running one benchmark per process
    (``--judge-benches harmbench`` then ``--judge-benches jbb``) avoids loading two vLLM
    judges in the same process; results of earlier calls are kept."""
    from ..judges import build_judge

    b = cfg["benchmarks"]
    benches = list(benches or (*JUDGED_BENCHES, *(["refusal"] if b.get("refusal_judge") else [])))
    for r in recs:
        r["refusal"] = is_refusal(r["response"]) if r["response"] is not None else None
        if r["response"] is None:  # missing artifact prompt => attack failed (JBB convention)
            r["success"], r["judged"] = False, True
        elif r["bench"] not in JUDGED_BENCHES:  # refusal-only benchmarks
            r["judged"] = True

    hb_items = [r for r in recs if r["bench"] == "harmbench" and r["response"] is not None]
    if hb_items and "harmbench" in benches:
        j = build_judge(
            b["harmbench"].get("judge", {"type": "harmbench_cls", "model": "cais/HarmBench-Llama-2-13b-cls"})
        )
        if j.name != "harmbench_cls":
            # generic safety judges see the plain behavior (+context), not the jailbreak wrapper
            judge_items = [
                {
                    "prompt": f"{r['context']}\n\n---\n\n{r['behavior']}"
                    if r.get("context")
                    else r["behavior"],
                    "response": r["response"],
                    "harmful": True,
                }
                for r in hb_items
            ]
        else:
            judge_items = hb_items
        for r, v in zip(hb_items, j.judge(judge_items)):
            r["success"], r["judged"] = bool(v), True
        j.close()

    jb_items = [r for r in recs if r["bench"] == "jbb" and r["response"] is not None]
    if jb_items and "jbb" in benches:
        j = build_judge(
            b["jbb"].get("judge", {"type": "llama_guard", "model": "meta-llama/Llama-Guard-3-8B"})
        )
        # judge against the original goal, not the (possibly obfuscated) attack prompt
        items = [{"prompt": r["goal"], "response": r["response"]} for r in jb_items]
        for r, v in zip(jb_items, j.judge(items)):
            strict = b["jbb"].get("require_non_refusal", False)
            r["success"] = (bool(v) and not r["refusal"]) if strict else bool(v)
            r["judged"] = True
        j.close()

    ref_items = [r for r in recs if r["bench"] in REFUSAL_BENCHES and r["response"] is not None]
    if ref_items and "refusal" in benches:
        if not b.get("refusal_judge"):
            raise ValueError("--judge-benches refusal needs benchmarks.refusal_judge in the eval config")
        j = build_judge(b["refusal_judge"])
        if not hasattr(j, "refusals"):
            raise ValueError(f"judge {j.name!r} cannot classify refusals")
        verdicts = j.refusals([{"prompt": r["prompt"], "response": r["response"]} for r in ref_items])
        for r, v in zip(ref_items, verdicts):
            r["refusal_llm"] = v
        j.close()
        bad = sum(v is None for v in verdicts)
        if bad:
            log.warning("%d unparseable refusal verdicts fall back to string matching", bad)
    return recs


def _refused(r: dict) -> bool:
    """LLM refusal verdict when available, else string matching."""
    v = r.get("refusal_llm")
    return r["refusal"] if v is None else v


def stage_mmlu(cfg: dict, model: str, adapter: str | None) -> dict:
    import lm_eval

    from ..train import pick_device

    m = cfg["benchmarks"]["mmlu"]
    model_args = f"pretrained={model},dtype={m.get('dtype', 'bfloat16')}"
    if adapter:
        model_args += f",peft={adapter}"
    res = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=m.get("tasks", ["mmlu"]),
        num_fewshot=m.get("num_fewshot", 5),
        batch_size=m.get("batch_size", "auto"),
        limit=m.get("limit"),
        device=pick_device(m.get("device", "auto")),
    )
    out = {}
    for task, r in res["results"].items():
        if "acc,none" in r:
            out[task] = {"acc": r["acc,none"], "stderr": r.get("acc_stderr,none")}
    return out


def compute_metrics(recs: list[dict], mmlu: dict | None) -> dict:
    m: dict = {}

    hb = [r for r in recs if r["bench"] == "harmbench"]
    if hb:
        by_attack = defaultdict(list)
        for r in hb:
            by_attack[r["attack"]].append(r["success"])
        templates = [a for a in by_attack if a.startswith("template:")]
        any_success = defaultdict(bool)
        for r in hb:
            if r["attack"].startswith("template:"):
                any_success[r["behavior_id"]] |= r["success"]
        by_cat = defaultdict(list)
        for r in hb:
            if r["attack"] == "direct":
                by_cat[r["category"]].append(r["success"])
        m["harmbench"] = {
            "by_attack": {a: rate(v) for a, v in by_attack.items()},
            "asr_direct": rate(by_attack["direct"])["rate"] if "direct" in by_attack else None,
            "asr_template_avg": (sum(rate(by_attack[a])["rate"] for a in templates) / len(templates))
            if templates
            else None,
            "asr_template_any": rate(list(any_success.values())) if any_success else None,
            "asr_direct_by_category": {c: rate(v) for c, v in sorted(by_cat.items())},
        }

    jb = [r for r in recs if r["bench"] == "jbb"]
    if jb:
        by_attack = defaultdict(list)
        for r in jb:
            by_attack[r["attack"]].append(r["success"])
        arts = [a for a in by_attack if a.startswith("artifact:")]
        m["jbb"] = {
            "by_attack": {a: rate(v) for a, v in by_attack.items()},
            "asr_direct": rate(by_attack["direct"])["rate"] if "direct" in by_attack else None,
            "asr_artifact_avg": (sum(rate(by_attack[a])["rate"] for a in arts) / len(arts)) if arts else None,
        }
    jbn = [r for r in recs if r["bench"] == "jbb_benign"]
    if jbn:
        m.setdefault("jbb", {})["benign_refusal"] = rate([_refused(r) for r in jbn])
        m["jbb"]["benign_refusal_strmatch"] = rate([r["refusal"] for r in jbn])

    xs = [r for r in recs if r["bench"] == "xstest"]
    if xs:
        safe, unsafe = [r for r in xs if r["label"] == "safe"], [r for r in xs if r["label"] == "unsafe"]
        m["xstest"] = {
            "over_refusal_safe": rate([_refused(r) for r in safe]),
            "refusal_unsafe": rate([_refused(r) for r in unsafe]),
            "over_refusal_safe_strmatch": rate([r["refusal"] for r in safe]),
            "refusal_unsafe_strmatch": rate([r["refusal"] for r in unsafe]),
        }
    ref = [r for r in recs if r["bench"] in REFUSAL_BENCHES]
    if ref:
        m["refusal_method"] = "llm" if any(r.get("refusal_llm") is not None for r in ref) else "string_match"

    # headline: ASR averaged over every attack setting (HarmBench direct + templates, JBB direct + artifacts)
    settings = []
    for bench in ("harmbench", "jbb"):
        settings += [v["rate"] for v in m.get(bench, {}).get("by_attack", {}).values()]
    m["asr_overall_avg"] = sum(settings) / len(settings) if settings else None

    if mmlu:
        m["mmlu"] = mmlu
    return m


# ---------------------------------------------------------------------------- driver
def run_eval(
    cfg: dict,
    tag: str,
    model: str,
    adapter: str | None = None,
    stages: list[str] | None = None,
    overwrite: bool = False,
    judge_benches: list[str] | None = None,
) -> dict | None:
    out = Path(cfg.get("results_dir", "results")) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps({"tag": tag, "model": model, "adapter": adapter}, indent=2))
    stages = stages or list(STAGES)
    f_gen, f_judged, f_mmlu = out / "generations.jsonl", out / "judged.jsonl", out / "mmlu.json"

    if "generate" in stages:
        if overwrite or not f_gen.exists():
            write_jsonl(f_gen, stage_generate(cfg, model, adapter))
        else:
            log.info("reusing %s", f_gen)
    if "judge" in stages:
        if f_judged.exists() and not overwrite and judge_benches is None:
            log.info("reusing %s", f_judged)
        else:
            src = f_judged if (f_judged.exists() and not overwrite) else f_gen
            write_jsonl(f_judged, stage_judge(cfg, read_jsonl(src), judge_benches))
    mmlu_on = cfg["benchmarks"].get("mmlu", {}).get("enabled", True)
    if "mmlu" in stages and mmlu_on and (overwrite or not f_mmlu.exists()):
        f_mmlu.write_text(json.dumps(stage_mmlu(cfg, model, adapter), indent=2))
    if "metrics" in stages:
        recs = read_jsonl(f_judged)
        missing = Counter(r["bench"] for r in recs if not r.get("judged"))
        if missing:
            raise RuntimeError(
                f"unjudged records {dict(missing)}: run `--stages judge` for these benchmarks first"
            )
        mmlu = json.loads(f_mmlu.read_text()) if f_mmlu.exists() else None
        metrics = compute_metrics(recs, mmlu)
        metrics["model"] = {"tag": tag, "model": model, "adapter": adapter}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log.info("wrote %s", out / "metrics.json")
        return metrics
    return None
