PY ?= python
BASE ?= meta-llama/Meta-Llama-3-8B-Instruct
TRAIN_CFG ?= configs/dpo_llama3_8b.yaml
ADAPTER ?= outputs/llama3-8b-safety-dpo/final
SMOKE_MODEL = trl-internal-testing/tiny-LlamaForCausalLM-3.2

.PHONY: install data train eval-base eval-dpo report all smoke test lint clean-smoke

install:
	$(PY) -m pip install -e ".[vllm,eval,dev]"

# 1. build preference pairs (policy sampling + Llama Guard labelling)
data:
	safedpo build-data --config configs/data.yaml --eval-config configs/eval.yaml

# 2. LoRA + DPO
train:
	safedpo train --config $(TRAIN_CFG)

# 3. evaluate base and aligned model
eval-base:
	safedpo eval --config configs/eval.yaml --tag base --model $(BASE)

eval-dpo:
	safedpo eval --config configs/eval.yaml --tag dpo --model $(BASE) --adapter $(ADAPTER)

# 4. tables + figure, written into README.md
report:
	safedpo report --results results --tags base dpo --update-readme README.md

all: data train eval-base eval-dpo report

# CPU-only end-to-end check with a tiny random model (plumbing, not quality)
smoke:
	safedpo build-data --config configs/smoke/data.yaml --eval-config configs/smoke/eval.yaml --overwrite
	safedpo train --config configs/smoke/dpo.yaml
	safedpo eval --config configs/smoke/eval.yaml --tag base --model $(SMOKE_MODEL) --overwrite
	safedpo eval --config configs/smoke/eval.yaml --tag dpo --model $(SMOKE_MODEL) --adapter smoke_out/outputs/tiny-dpo/final --overwrite
	safedpo report --results smoke_out/results --tags base dpo

test:
	pytest -q

lint:
	ruff check src tests

clean-smoke:
	rm -rf smoke_out
