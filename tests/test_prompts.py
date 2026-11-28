import pytest

from safedpo.prompts import (
    EVAL_JAILBREAKS,
    OVERREFUSAL_TEMPLATES,
    SAFE_REFUSAL_TEMPLATES,
    TRAIN_JAILBREAKS,
    is_refusal,
    wrap,
)


def test_train_and_eval_jailbreaks_are_disjoint():
    assert not set(TRAIN_JAILBREAKS) & set(EVAL_JAILBREAKS)
    assert not set(TRAIN_JAILBREAKS.values()) & set(EVAL_JAILBREAKS.values())


@pytest.mark.parametrize("family", list(TRAIN_JAILBREAKS) + list(EVAL_JAILBREAKS))
def test_wrap_contains_prompt(family):
    assert "QQQ-test-prompt" in wrap("QQQ-test-prompt", family)


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "I can’t assist with that request.",
        "As an AI language model, I cannot provide this.",
        "I won't help with this.",
        "I must decline.",
        "I'm unable to assist with that.",
        "I am not able to help with this request.",
        "",
    ],
)
def test_refusals_detected(text):
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Sure! Here are three tips for staying healthy: 1. Sleep ...",
        "To kill a Python process, run `kill -9 <pid>` or use Task Manager.",
        "Binary search halves the search interval each step." + " x" * 200 + " I cannot stress this enough.",
    ],
)
def test_compliance_not_flagged(text):
    assert not is_refusal(text)


def test_templates_are_classified_consistently():
    # the over-refusal templates used as `rejected` must look like refusals
    assert all(is_refusal(t) for t in OVERREFUSAL_TEMPLATES)
    assert all(is_refusal(t) for t in SAFE_REFUSAL_TEMPLATES)
