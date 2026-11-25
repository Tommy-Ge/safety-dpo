"""Remove training prompts that overlap with evaluation benchmarks (HarmBench / JBB / XSTest)."""

from __future__ import annotations

from .sources import normalize


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = normalize(text).split()
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class Decontaminator:
    """Flags a prompt if it equals an eval prompt after normalisation, or if more than
    ``threshold`` of the eval prompt's word n-grams appear in it (containment)."""

    def __init__(self, eval_texts: list[str], n: int = 8, threshold: float = 0.5):
        self.n = n
        self.threshold = threshold
        self.exact = {normalize(t) for t in eval_texts}
        self.eval_grams = [ngrams(t, n) for t in eval_texts]
        self.index: dict[tuple, list[int]] = {}
        for i, gs in enumerate(self.eval_grams):
            for g in gs:
                self.index.setdefault(g, []).append(i)

    def is_contaminated(self, text: str) -> bool:
        if normalize(text) in self.exact:
            return True
        grams = ngrams(text, self.n)
        hits: dict[int, int] = {}
        for g in grams:
            for i in self.index.get(g, ()):
                hits[i] = hits.get(i, 0) + 1
        return any(c / max(1, len(self.eval_grams[i])) > self.threshold for i, c in hits.items())
