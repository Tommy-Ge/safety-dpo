#!/usr/bin/env bash
# One-command reproduction on a single CUDA GPU (A100 / A800 / H100 80GB recommended).
#
#   QUICK=1 bash scripts/run_gpu.sh   # ~20-40 min end-to-end check on a tiny subset - run this first
#   bash scripts/run_gpu.sh           # full run; resumable: re-run the same command after any interruption
#   QLORA=1 bash scripts/run_gpu.sh   # 24GB GPU: QLoRA training + fp8 HarmBench judge
#   SMOKE=1 bash scripts/run_gpu.sh   # CPU-only rehearsal of this script with a tiny model (CI / laptops)
#   MAC=1   bash scripts/run_gpu.sh   # laptop-scale run on Apple silicon: Qwen2.5-1.5B + Qwen3Guard (configs/mac)
#
# Every finished step leaves runs/<run>/logs/<step>.done; finished steps are skipped on re-run.
# At the end, runs/<run>/bundle.tar.gz holds everything needed to update the GitHub repo
# (metrics, figures, README, data stats, training logs - but no model generations).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-meta-llama/Meta-Llama-3-8B-Instruct}
QUICK=${QUICK:-0}
QLORA=${QLORA:-0}
SMOKE=${SMOKE:-0}
MAC=${MAC:-0}
DATA_CFG=configs/data.yaml
EVAL_CFG=configs/eval.yaml
export TOKENIZERS_PARALLELISM=false

# AutoDL "academic acceleration" (GitHub / Hugging Face proxy) if available
if [[ -f /etc/network_turbo && -z "${NO_TURBO:-}" ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo >/dev/null 2>&1 || true
fi

if [[ "$QLORA" == 1 ]]; then
  TRAIN_CFG=configs/dpo_llama3_8b_qlora.yaml
  OUT_DIR=outputs/llama3-8b-safety-dpo-qlora
else
  TRAIN_CFG=configs/dpo_llama3_8b.yaml
  OUT_DIR=outputs/llama3-8b-safety-dpo
fi

if [[ "$MAC" == 1 ]]; then
  BASE=Qwen/Qwen2.5-1.5B-Instruct
  DATA_CFG=configs/mac/data.yaml; EVAL_CFG=configs/mac/eval.yaml; TRAIN_CFG=configs/mac/dpo.yaml
  RUN=runs/mac; DATA_DIR=data/mac; OUT_DIR=outputs/qwen2.5-1.5b-safety-dpo; RESULTS=results/mac
  DATA_O=(); TRAIN_O=(); EVAL_O=(); README_ARG=(--update-readme README.md --marker RESULTS-MAC)
elif [[ "$SMOKE" == 1 ]]; then
  BASE=trl-internal-testing/tiny-LlamaForCausalLM-3.2
  DATA_CFG=configs/smoke/data.yaml; EVAL_CFG=configs/smoke/eval.yaml; TRAIN_CFG=configs/smoke/dpo.yaml
  RUN=runs/smoke; DATA_DIR=$RUN/data; OUT_DIR=$RUN/outputs/dpo; RESULTS=$RUN/results
  DATA_O=(); TRAIN_O=(); EVAL_O=(); README_ARG=()
elif [[ "$QUICK" == 1 ]]; then
  RUN=runs/quick
  DATA_DIR=$RUN/data
  OUT_DIR=$RUN/outputs/dpo
  RESULTS=$RUN/results
  DATA_O=(-o sources.harmful.0.max_prompts=40 -o sources.borderline.0.max_prompts=15
          -o sources.general.0.max_prompts=10 -o sampling.n=2 -o sampling.max_new_tokens=128
          -o pairing.max_pairs=null)
  TRAIN_O=(-o training.max_steps=4 -o training.eval_steps=4 -o training.save_strategy=no)
  EVAL_O=(-o benchmarks.harmbench.limit=10 -o benchmarks.jbb.limit=10 -o benchmarks.xstest.limit=20
          -o 'benchmarks.mmlu.tasks=[mmlu_abstract_algebra]' -o benchmarks.mmlu.limit=20
          -o generation.max_new_tokens=128)
  README_ARG=()
else
  RUN=runs/full
  DATA_DIR=data
  RESULTS=results
  DATA_O=(); TRAIN_O=(); EVAL_O=()
  README_ARG=(--update-readme README.md)
fi
DATA_O+=(-o work_dir=$DATA_DIR/work -o out_dir=$DATA_DIR/processed)
TRAIN_O+=(-o training.output_dir=$OUT_DIR -o data.train_file=$DATA_DIR/processed/train.jsonl
          -o data.test_file=$DATA_DIR/processed/test.jsonl)
EVAL_O+=(-o results_dir=$RESULTS)
ADAPTER=$OUT_DIR/final
LOGS=$RUN/logs
mkdir -p "$LOGS"

step() {
  local name=$1; shift
  if [[ -f $LOGS/$name.done ]]; then echo "[skip] $name"; return; fi
  echo; echo "==================== $name  ($(date '+%F %T'))"
  "$@" 2>&1 | tee "$LOGS/$name.log"
  touch "$LOGS/$name.done"
}

# ------------------------------------------------------------------ preflight
preflight() {
  nvidia-smi --query-gpu=name,memory.total --format=csv
  python - "$BASE" <<'PY'
import sys, torch
assert torch.cuda.is_available(), "CUDA not available"
import vllm, trl, peft, lm_eval  # noqa: F401
from huggingface_hub import hf_hub_download
models = [sys.argv[1], "meta-llama/Llama-Guard-3-8B", "cais/HarmBench-Llama-2-13b-cls"]
for m in models:
    try:
        hf_hub_download(m, "config.json")
        print("ok  ", m)
    except Exception as e:
        sys.exit(f"cannot access {m}: {type(e).__name__}. Request access on huggingface.co and run `huggingface-cli login`.")
PY
}
if [[ "$SMOKE" != 1 && "$MAC" != 1 ]]; then
  step preflight preflight
  GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
else
  GPU_MB=999999
fi
if [[ "$GPU_MB" -lt 40000 ]]; then
  echo "GPU has ${GPU_MB}MB < 40GB: loading the 13B HarmBench judge in fp8"
  EVAL_O+=(-o benchmarks.harmbench.judge.quantization=fp8)
fi

# ------------------------------------------------------------------ 1. data
for st in collect generate judge pair; do
  step "data_$st" safedpo build-data --config $DATA_CFG --eval-config $EVAL_CFG --stage $st "${DATA_O[@]}"
done

# ------------------------------------------------------------------ 2. train
step train safedpo train --config $TRAIN_CFG "${TRAIN_O[@]}"

# ------------------------------------------------------------------ 3. eval (one vLLM model per process)
for tag in base dpo; do
  AD=(); [[ $tag == dpo ]] && AD=(--adapter "$ADAPTER")
  E=(safedpo eval --config $EVAL_CFG --tag $tag --model "$BASE" ${AD[@]+"${AD[@]}"} "${EVAL_O[@]}")
  step "eval_${tag}_generate"        "${E[@]}" --stages generate
  step "eval_${tag}_judge_harmbench" "${E[@]}" --stages judge --judge-benches harmbench
  step "eval_${tag}_judge_jbb"       "${E[@]}" --stages judge --judge-benches jbb
  step "eval_${tag}_mmlu"            "${E[@]}" --stages mmlu
  step "eval_${tag}_metrics"         "${E[@]}" --stages metrics
done

# ------------------------------------------------------------------ 4. report + bundle
safedpo report --results $RESULTS --tags base dpo ${README_ARG[@]+"${README_ARG[@]}"} \
  --data-stats $DATA_DIR/processed/stats.json --train-log $OUT_DIR/log_history.json | tee "$LOGS/report.log"

FILES=$(ls -d README.md $RESULTS/summary.md $RESULTS/*.png $RESULTS/*/metrics.json $RESULTS/*/mmlu.json \
          $RESULTS/*/model.json $DATA_DIR/processed/stats.json $OUT_DIR/*.json "$LOGS" 2>/dev/null || true)
tar czf $RUN/bundle.tar.gz $FILES
tar czf $RUN/adapter.tar.gz -C "$(dirname "$ADAPTER")" "$(basename "$ADAPTER")"
echo
echo "done. results:  $RUN/bundle.tar.gz   (download it and unpack in the repo root)"
echo "      adapter:  $RUN/adapter.tar.gz  (LoRA weights, optional)"
