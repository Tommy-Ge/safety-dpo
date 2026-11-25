"""A thin generation wrapper with two backends.

* ``vllm`` - fast, supports LoRA adapters natively; Linux + CUDA only.
* ``hf``   - plain ``transformers.generate``; works everywhere (used by the CPU smoke test).

Both backends render chat conversations with the *model's own* chat template, so the
policy, Llama Guard and the HarmBench classifier all see exactly the format they expect.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

from tqdm import tqdm

log = logging.getLogger(__name__)

# Minimal Llama-3-style template for tiny test models that ship without one.
FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@dataclass
class SamplingConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1


class Generator:
    def __init__(
        self,
        model: str,
        adapter: str | None = None,
        backend: str = "hf",
        dtype: str = "bfloat16",
        batch_size: int = 16,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        max_lora_rank: int = 64,
        seed: int = 0,
    ):
        self.model_name = model
        self.adapter = adapter
        self.backend = backend
        self.batch_size = batch_size
        self.seed = seed

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.chat_template is None:
            log.warning("%s has no chat template; using a minimal fallback template", model)
            self.tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if backend == "vllm":
            from vllm import LLM

            self._llm = LLM(
                model=model,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_lora=adapter is not None,
                max_lora_rank=max_lora_rank,
                seed=seed,
            )
            self._lora = None
            if adapter:
                from vllm.lora.request import LoRARequest

                self._lora = LoRARequest("adapter", 1, adapter)
        elif backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM

            torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch_dtype in (torch.bfloat16, torch.float16):
                torch_dtype = torch.float32
            from .train import dtype_kwargs

            self._model = AutoModelForCausalLM.from_pretrained(model, **dtype_kwargs(torch_dtype)).to(device)
            if adapter:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, adapter).merge_and_unload()
            self._model.eval()
            self._device = device
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # ------------------------------------------------------------------ rendering
    def render_chat(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ public API
    def chat(
        self, conversations: list[list[dict]], sampling: SamplingConfig, desc: str = "generate"
    ) -> list[list[str]]:
        """Returns ``n`` completions per conversation."""
        prompts = [self.render_chat(c) for c in conversations]
        return self.complete(prompts, sampling, desc=desc)

    def complete(
        self, prompts: list[str], sampling: SamplingConfig, desc: str = "generate"
    ) -> list[list[str]]:
        if not prompts:
            return []
        if self.backend == "vllm":
            return self._complete_vllm(prompts, sampling)
        return self._complete_hf(prompts, sampling, desc)

    def close(self) -> None:
        """Free GPU memory so another model (e.g. a judge) can be loaded in the same process."""
        import torch

        for attr in ("_llm", "_model"):
            if hasattr(self, attr):
                delattr(self, attr)
        if self.backend == "vllm":
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:  # pragma: no cover - best effort
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ backends
    def _complete_vllm(self, prompts: list[str], s: SamplingConfig) -> list[list[str]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=s.n,
            temperature=s.temperature,
            top_p=s.top_p if s.temperature > 0 else 1.0,
            max_tokens=s.max_new_tokens,
            seed=self.seed if s.temperature > 0 else None,
        )
        # prompts already contain BOS from the chat template
        outs = self._llm.generate(
            [{"prompt_token_ids": self.tokenizer(p, add_special_tokens=False).input_ids} for p in prompts],
            params,
            lora_request=self._lora,
        )
        return [[o.text for o in out.outputs] for out in outs]

    def _complete_hf(self, prompts: list[str], s: SamplingConfig, desc: str) -> list[list[str]]:
        import torch

        results: list[list[str]] = []
        do_sample = s.temperature > 0
        if do_sample:
            torch.manual_seed(self.seed)
        for i in tqdm(range(0, len(prompts), self.batch_size), desc=desc, leave=False):
            batch = prompts[i : i + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
                self._device
            )
            gen_kwargs = dict(
                max_new_tokens=s.max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=s.n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if do_sample:
                gen_kwargs.update(temperature=s.temperature, top_p=s.top_p)
            with torch.no_grad():
                out = self._model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc.input_ids.shape[1] :]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for j in range(len(batch)):
                results.append([t.strip() for t in texts[j * s.n : (j + 1) * s.n]])
        return results


def build_generator(
    cfg: dict, model: str | None = None, adapter: str | None = None, seed: int = 0
) -> Generator:
    """Construct from a config block like ``{model, backend, dtype, batch_size, ...}``."""
    return Generator(
        model=model or cfg["model"],
        adapter=adapter if adapter is not None else cfg.get("adapter"),
        backend=cfg.get("backend", "hf"),
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=cfg.get("batch_size", 16),
        max_model_len=cfg.get("max_model_len", 4096),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        max_lora_rank=cfg.get("max_lora_rank", 64),
        seed=seed,
    )
