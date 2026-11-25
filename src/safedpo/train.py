"""LoRA + DPO training with TRL.

The reference model is the frozen base model: with a PEFT model TRL computes reference
log-probs by temporarily disabling the adapter, so only one copy of the 8B weights is in memory.

Works across TRL versions: config keys unknown to the installed ``DPOConfig`` are dropped with a
warning (e.g. ``max_prompt_length`` / ``rpo_alpha`` exist in TRL 0.x but not in TRL 1.x).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def dtype_kwargs(dtype) -> dict:
    """transformers renamed ``torch_dtype`` -> ``dtype`` in v4.56."""
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


def _filter_fields(cls, kwargs: dict, what: str) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(kwargs) - names)
    if dropped:
        log.warning("%s: ignoring keys not supported by the installed version: %s", what, dropped)
    return {k: v for k, v in kwargs.items() if k in names}


def train(cfg: dict) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from .generation import FALLBACK_CHAT_TEMPLATE

    mcfg, tcfg, lcfg = cfg["model"], cfg["training"], cfg["lora"]
    out_dir = Path(tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2))

    # ---------------------------------------------------------------- tokenizer
    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # Llama-3 has no pad token; loss masking does not rely on it

    # ---------------------------------------------------------------- model
    use_cuda = torch.cuda.is_available()
    dtype = getattr(torch, mcfg.get("dtype", "bfloat16")) if use_cuda else torch.float32
    model_kwargs = dtype_kwargs(dtype)
    if mcfg.get("attn_implementation") and use_cuda:
        model_kwargs["attn_implementation"] = mcfg["attn_implementation"]
    if mcfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(mcfg["name"], **model_kwargs)
    if mcfg.get("load_in_4bit"):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tcfg.get("gradient_checkpointing", True)
        )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lcfg.get("r", 16),
        lora_alpha=lcfg.get("alpha", 32),
        lora_dropout=lcfg.get("dropout", 0.05),
        target_modules=lcfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---------------------------------------------------------------- data
    dcfg = cfg["data"]
    files = {"train": dcfg["train_file"]}
    if dcfg.get("test_file") and Path(dcfg["test_file"]).exists() and Path(dcfg["test_file"]).stat().st_size:
        files["test"] = dcfg["test_file"]
    ds = load_dataset("json", data_files=files)
    keep = ["prompt", "chosen", "rejected"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    if dcfg.get("max_train_samples"):
        ds["train"] = ds["train"].select(range(min(dcfg["max_train_samples"], len(ds["train"]))))

    # ---------------------------------------------------------------- trainer
    args_kwargs = dict(tcfg)
    args_kwargs.setdefault("bf16", use_cuda)
    if not use_cuda:
        args_kwargs["bf16"] = False
        args_kwargs.setdefault("use_cpu", True)
    if "test" not in files:
        args_kwargs["eval_strategy"] = "no"
    args = DPOConfig(**_filter_fields(DPOConfig, args_kwargs, "DPOConfig"))

    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("test"),
        peft_config=peft_config,
    )
    params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tok
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.model.print_trainable_parameters() if hasattr(
        trainer.model, "print_trainable_parameters"
    ) else None

    trainer.train()
    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    metrics = {k: v for k, v in trainer.state.log_history[-1].items()} if trainer.state.log_history else {}
    if "test" in files:
        metrics.update(trainer.evaluate())
    (out_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    log.info("saved LoRA adapter to %s", final)
    return str(final)
