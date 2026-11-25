import random

from safedpo.data.build import make_pair, pair
from safedpo.prompts import OVERREFUSAL_TEMPLATES, SAFE_REFUSAL_TEMPLATES

PCFG = {
    "chosen_fallback": "template",
    "rejected_fallback_harmful": "offline",
    "rejected_fallback_benign": "template",
}


def rec(kind, samples, unsafe, refusal, ref="", ref_unsafe=False, ref_refusal=False, offline=None, i=0):
    return {
        "id": f"{kind[0]}{i}",
        "kind": kind,
        "source": "t",
        "jailbreak": None,
        "prompt": f"p{i}",
        "samples": samples,
        "sample_unsafe": unsafe,
        "sample_refusal": refusal,
        "ref": ref,
        "ref_unsafe": ref_unsafe,
        "ref_refusal": ref_refusal,
        "offline_unsafe": offline,
    }


def test_harmful_prefers_safety_reference_and_unsafe_sample():
    r = rec(
        "harmful",
        ["Sure, step 1...", "I can't help."],
        [True, False],
        [False, True],
        ref="I can't, but here is a safe alternative.",
    )
    p = make_pair(r, PCFG, random.Random(0))
    assert p["chosen_from"] == "ref_safety_sysprompt"
    assert p["rejected_from"] == "policy_unsafe" and p["rejected"][0]["content"] == "Sure, step 1..."
    assert p["prompt"] == [{"role": "user", "content": "p0"}]  # no system prompt in the DPO prompt


def test_harmful_falls_back_to_policy_safe_then_template():
    r = rec("harmful", ["Sure...", "I won't."], [True, False], [False, True], ref="Sure...", ref_unsafe=True)
    assert make_pair(r, PCFG, random.Random(0))["chosen_from"] == "policy_safe"
    r = rec("harmful", ["Sure...", "Ok..."], [True, True], [False, False], ref="Sure...", ref_unsafe=True)
    p = make_pair(r, PCFG, random.Random(0))
    assert p["chosen_from"] == "template" and p["chosen"][0]["content"] in SAFE_REFUSAL_TEMPLATES


def test_harmful_already_safe_uses_offline_or_is_dropped():
    r = rec(
        "harmful", ["I can't.", "No."], [False, False], [True, True], ref="I can't.", offline="unsafe text"
    )
    assert make_pair(r, PCFG, random.Random(0))["rejected_from"] == "offline_unsafe"
    r["offline_unsafe"] = None
    assert make_pair(r, PCFG, random.Random(0)) is None
    assert (
        make_pair(
            {**r, "offline_unsafe": "x"}, {**PCFG, "rejected_fallback_harmful": "skip"}, random.Random(0)
        )
        is None
    )


def test_unsafe_but_refusing_sample_is_not_rejected():
    # flagged unsafe by the judge but actually a refusal -> not a useful negative
    r = rec("harmful", ["I'm sorry, I can't."], [True], [True], ref="I can't.")
    assert make_pair(r, PCFG, random.Random(0)) is None


def test_benign_pairs_penalise_over_refusal():
    r = rec("borderline", ["Run kill -9 <pid>.", "I'm sorry, I can't."], [False, False], [False, True])
    p = make_pair(r, PCFG, random.Random(0))
    assert (p["chosen_from"], p["rejected_from"]) == ("policy_compliant", "policy_refusal")
    r = rec("general", ["Here you go."], [False], [False], ref="I'm sorry.", ref_refusal=True)
    assert make_pair(r, PCFG, random.Random(0))["rejected_from"] == "ref_overcautious_sysprompt"
    r = rec("general", ["Here you go."], [False], [False], ref="Here.", ref_refusal=False)
    p = make_pair(r, PCFG, random.Random(0))
    assert p["rejected_from"] == "template" and p["rejected"][0]["content"] in OVERREFUSAL_TEMPLATES


def test_benign_without_good_answer_is_dropped():
    r = rec("borderline", ["I can't."], [False], [True], ref="I can't.", ref_refusal=True)
    assert make_pair(r, PCFG, random.Random(0)) is None


def test_mix_and_split():
    rows = [rec("harmful", ["bad"], [True], [False], ref="I can't.", i=i) for i in range(20)]
    rows += [rec("general", ["ok"], [False], [False], i=100 + i) for i in range(20)]
    cfg = {
        "seed": 0,
        "pairing": {**PCFG, "mix": {"harmful": 0.5, "general": 0.25}, "max_pairs": 20, "test_fraction": 0.1},
    }
    train, test, stats = pair(cfg, rows)
    assert len(train) + len(test) == 15
    assert stats["n_test"] == 1
    assert not {p["id"] for p in train} & {p["id"] for p in test}
    assert stats["policy_unsafe_rate_on_harmful"] == 1.0
