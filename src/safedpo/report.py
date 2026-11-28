"""Aggregate ``results/<tag>/metrics.json`` into Markdown tables + a figure, and optionally
write them into README.md between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->``.

Every number in the generated tables comes from metrics.json; nothing is hard-coded.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .eval.metrics import relative_reduction

log = logging.getLogger(__name__)


def markers(name: str = "RESULTS") -> tuple[str, str]:
    return f"<!-- {name}:START -->", f"<!-- {name}:END -->"


def _pct(x, digits=1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}"


def _get(m: dict, *path):
    for p in path:
        if not isinstance(m, dict) or p not in m:
            return None
        m = m[p]
    return m


def _rate(m, *path):
    v = _get(m, *path)
    return v["rate"] if isinstance(v, dict) else v


MAIN_COLUMNS = [
    ("HarmBench Direct ↓", lambda m: _rate(m, "harmbench", "asr_direct"), "asr"),
    ("HarmBench Held-out Templates ↓", lambda m: _rate(m, "harmbench", "asr_template_avg"), "asr"),
    ("HarmBench Any-Template ↓", lambda m: _rate(m, "harmbench", "asr_template_any"), "asr"),
    ("JBB Direct ↓", lambda m: _rate(m, "jbb", "asr_direct"), "asr"),
    ("JBB Transfer (PAIR/GCG) ↓", lambda m: _rate(m, "jbb", "asr_artifact_avg"), "asr"),
    ("Avg ASR ↓", lambda m: m.get("asr_overall_avg"), "asr"),
    ("XSTest Over-refusal ↓", lambda m: _rate(m, "xstest", "over_refusal_safe"), "util"),
    ("JBB Benign Refusal ↓", lambda m: _rate(m, "jbb", "benign_refusal"), "util"),
    ("MMLU ↑", lambda m: _get(m, "mmlu", "mmlu", "acc"), "util"),
]


def load_metrics(results_dir: Path, tags: list[str]) -> dict[str, dict]:
    out = {}
    for t in tags:
        f = results_dir / t / "metrics.json"
        if f.exists():
            out[t] = json.loads(f.read_text())
        else:
            log.warning("missing %s", f)
    return out


def main_table(ms: dict[str, dict]) -> str:
    tags = list(ms)
    cols = [c for c in MAIN_COLUMNS if any(c[1](m) is not None for m in ms.values())]
    lines = ["| Model | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for t in tags:
        lines.append(f"| {t} | " + " | ".join(_pct(c[1](ms[t])) for c in cols) + " |")
    for other in tags[1:]:
        base, new = ms[tags[0]], ms[other]
        cells = []
        for _, fn, kind in cols:
            b, a = fn(base), fn(new)
            if b is None or a is None:
                cells.append("–")
            elif kind == "asr":
                rr = relative_reduction(b, a)
                cells.append("–" if rr is None else f"**{-100 * rr:+.0f}%** rel.".replace("-", "−"))
            else:
                cells.append(f"{100 * (a - b):+.1f} pt")
        lines.append(f"| Δ {other} vs {tags[0]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def attack_table(ms: dict[str, dict]) -> str:
    rows = []
    for bench in ("harmbench", "jbb"):
        attacks = []
        for m in ms.values():
            for a in _get(m, bench, "by_attack") or {}:
                if a not in attacks:
                    attacks.append(a)
        for a in attacks:
            cells = []
            for m in ms.values():
                v = _get(m, bench, "by_attack", a)
                cells.append(
                    "–" if v is None else f"{_pct(v['rate'])} [{_pct(v['ci95'][0])}, {_pct(v['ci95'][1])}]"
                )
            rows.append(f"| {bench} | {a} | " + " | ".join(cells) + " |")
    head = "| Benchmark | Attack | " + " | ".join(ms) + " |\n|---|---|" + "---|" * len(ms)
    return head + "\n" + "\n".join(rows)


def category_table(ms: dict[str, dict]) -> str:
    cats = []
    for m in ms.values():
        for c in _get(m, "harmbench", "asr_direct_by_category") or {}:
            if c not in cats:
                cats.append(c)
    if not cats:
        return ""
    head = "| HarmBench category (direct) | " + " | ".join(ms) + " |\n|---|" + "---|" * len(ms)
    rows = [
        f"| {c} | "
        + " | ".join(_pct(_rate(m, "harmbench", "asr_direct_by_category", c)) for m in ms.values())
        + " |"
        for c in cats
    ]
    return head + "\n" + "\n".join(rows)


def refusal_table(ms: dict[str, dict]) -> str:
    """LLM-judged vs string-matched refusal rates, shown when any model has LLM verdicts."""
    if not any(m.get("refusal_method") == "llm" for m in ms.values()):
        return ""
    rows = [
        ("XSTest 安全问题被拒（过度拒答）", ("xstest", "over_refusal_safe")),
        ("XSTest 不安全问题被拒", ("xstest", "refusal_unsafe")),
        ("JBB 良性行为被拒", ("jbb", "benign_refusal")),
    ]
    tags = list(ms)
    lines = [
        "| 指标 | " + " | ".join(f"{t}（LLM / 字符串）" for t in tags) + " |",
        "|---|" + "---|" * len(tags),
    ]
    for name, (bench, key) in rows:
        cells = [
            f"{_pct(_rate(ms[t], bench, key))} / {_pct(_rate(ms[t], bench, key + '_strmatch'))}" for t in tags
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(ms: dict[str, dict]) -> str:
    tags = list(ms)
    if len(tags) < 2:
        return ""
    base, new = ms[tags[0]], ms[tags[1]]  # headline: baseline vs the main model (2nd tag)
    parts = []
    b, a = base.get("asr_overall_avg"), new.get("asr_overall_avg")
    rr = relative_reduction(b, a)
    if rr is not None:
        parts.append(
            f"平均越狱成功率（所有攻击设置）从 **{_pct(b)}%** 降到 **{_pct(a)}%**，相对降低 **{100 * rr:.0f}%**"
        )
    mb, ma = _get(base, "mmlu", "mmlu", "acc"), _get(new, "mmlu", "mmlu", "acc")
    if mb is not None and ma is not None:
        parts.append(
            f"MMLU 从 {_pct(mb)} 变为 {_pct(ma)}（{100 * (ma - mb):+.1f} pt，相对 {100 * (ma - mb) / mb:+.2f}%）"
        )
    xb, xa = _rate(base, "xstest", "over_refusal_safe"), _rate(new, "xstest", "over_refusal_safe")
    if xb is not None and xa is not None:
        parts.append(f"XSTest 过度拒答率 {_pct(xb)}% → {_pct(xa)}%")
    return "；".join(parts) + "。" if parts else ""


def plot(ms: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attacks = []
    for m in ms.values():
        for bench in ("harmbench", "jbb"):
            for a in _get(m, bench, "by_attack") or {}:
                if (bench, a) not in attacks:
                    attacks.append((bench, a))
    if not attacks:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(attacks)), 3.6))
    width = 0.8 / len(ms)
    for i, (tag, m) in enumerate(ms.items()):
        vals = [100 * (_rate(m, b, "by_attack", a) or 0) for b, a in attacks]
        ax.bar([x + i * width for x in range(len(attacks))], vals, width, label=tag)
    ax.set_xticks([x + width * (len(ms) - 1) / 2 for x in range(len(attacks))])
    ax.set_xticklabels(
        [f"{b}\n{a.split(':', 1)[-1]}" for b, a in attacks], fontsize=7, rotation=30, ha="right"
    )
    ax.set_ylabel("Attack success rate (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def data_section(stats_path: Path) -> str:
    st = json.loads(stats_path.read_text())

    def fmt(d: dict) -> str:
        return "，".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

    rows = [
        ("训练 / 验证偏好对", f"{st['n_train']} / {st['n_test']}"),
        ("训练集按类型", fmt(st.get("train_by_kind", {}))),
        ("chosen 来源", fmt(st.get("chosen_from", {}))),
        ("rejected 来源", fmt(st.get("rejected_from", {}))),
        ("有害样本的越狱包装", fmt(st.get("jailbreak_family", {}))),
        (
            "基座在训练有害提示上的不安全率（全部 / 带越狱包装）",
            f"{_pct(st.get('policy_unsafe_rate_on_harmful'))}% / {_pct(st.get('policy_unsafe_rate_on_harmful_jailbreak'))}%",
        ),
        ("基座在良性训练提示上的拒答率", f"{_pct(st.get('policy_refusal_rate_on_benign'))}%"),
    ]
    return "### 构造出的偏好数据\n\n| 项目 | 值 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def plot_training(log_path: Path, out: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = json.loads(log_path.read_text())
    panels = [
        ("loss", "eval_loss", "DPO loss"),
        ("rewards/margins", "eval_rewards/margins", "reward margin (chosen − rejected)"),
        ("rewards/accuracies", "eval_rewards/accuracies", "reward accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    drawn = False
    for ax, (tk, ek, title) in zip(axes, panels):
        for key, label, style in ((tk, "train", "-"), (ek, "eval", "o--")):
            pts = [(h["step"], float(h[key])) for h in hist if key in h and "step" in h]
            if pts:
                ax.plot(*zip(*pts), style, label=label, markersize=3)
                drawn = True
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    if drawn:
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return drawn


def build_report(
    results_dir: str,
    tags: list[str],
    readme: str | None = None,
    data_stats: str | None = None,
    train_log: str | None = None,
    marker: str = "RESULTS",
) -> str:
    rd = Path(results_dir)
    ms = load_metrics(rd, tags)
    if not ms:
        raise SystemExit(f"no metrics.json found under {rd} for tags {tags}")
    plot(ms, rd / "asr_by_attack.png")
    extra = []
    if data_stats and Path(data_stats).exists():
        extra.append(data_section(Path(data_stats)))
    if train_log and Path(train_log).exists() and plot_training(Path(train_log), rd / "training_curves.png"):
        extra.append(f"### 训练曲线\n\n![training curves]({rd.as_posix()}/training_curves.png)")
    md = "\n\n".join(
        s
        for s in [
            headline(ms),
            "### 主结果（%）\n\n" + main_table(ms),
            "### 各攻击设置的 ASR（%，括号内为 95% Wilson 置信区间）\n\n" + attack_table(ms),
            category_table(ms) and "### HarmBench 各语义类别（直接请求，ASR %）\n\n" + category_table(ms),
            f"![ASR by attack]({rd.as_posix()}/asr_by_attack.png)",
            refusal_table(ms)
            and "### 拒答率：LLM 判定 vs 字符串匹配（%）\n\n"
            "主表里的拒答率用 LLM 判定（Qwen3Guard 的 `Refusal` 字段）。字符串匹配只认“I'm sorry / I cannot”这类固定说法，"
            "会漏掉换了说法的拒答，下表把两者并列。\n\n"
            + refusal_table(ms),
            *extra,
        ]
        if s
    )
    (rd / "summary.md").write_text(md + "\n", encoding="utf-8")
    if readme:
        p = Path(readme)
        text = p.read_text(encoding="utf-8")
        START, END = markers(marker)
        if START in text and END in text:
            text = re.sub(
                re.escape(START) + r".*?" + re.escape(END), f"{START}\n{md}\n{END}", text, flags=re.S
            )
            p.write_text(text, encoding="utf-8")
            log.info("updated %s", p)
        else:
            log.warning("README has no %s / %s markers; not updated", START, END)
    return md
