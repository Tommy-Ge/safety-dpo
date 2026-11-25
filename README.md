# safety-dpo：用 LoRA + DPO 做安全对齐

**Safety alignment of Llama-3-8B-Instruct with LoRA + DPO on self-constructed safe / unsafe preference pairs**

在 Llama-3-8B-Instruct 上用 LoRA 和 DPO 做安全对齐，偏好数据由本项目自己构造：让模型自己采样回答，再用 Llama Guard 3 标注安全 / 不安全，从中组成“安全回答 vs. 不安全回答”的偏好对。评测覆盖越狱基准 **HarmBench** 和 **JailbreakBench**，过度拒答基准 **XSTest**，以及通用能力基准 **MMLU**。目标是在大幅降低越狱成功率（ASR）的同时，让 MMLU 下降不超过 1%。

本项目提供：

1. **偏好数据构造流水线**：数据是 on-policy 的。有害提示用 6 类越狱模板做增强，另外加入“看似有害、实则无害”的样本来约束过度拒答。训练集按 n-gram 与所有评测集去重。
2. **LoRA + DPO 训练**：基于 TRL `DPOTrainer`。参考模型就是关掉 adapter 的基座，显存里只放一份 8B 权重。80GB 单卡用 bf16 LoRA，24GB 单卡用 QLoRA。
3. **评测**：HarmBench 用官方分类器，测直接请求和 4 个**训练时没见过的**越狱模板；JailbreakBench 用 Llama Guard 3 判定，测直接请求和官方 PAIR / GCG 迁移攻击；另外测 XSTest 过度拒答率和 MMLU 5-shot。每个 ASR 都附 95% 置信区间。
4. **自动报告**：`make report` 读取每个模型的 `metrics.json`，生成表格和图并写回本 README。README 里的结果数字全部由脚本生成，没有手填。

---

## 方法

### 1. 偏好数据构造

```
                       ┌─────────────── 有害提示 PKU-SafeRLHF（至少一条人工标注为不安全）
  collect ─────────────┼─────────────── 边界提示 OR-Bench-80k（看似有害、实则无害）
   │ 去重 + 与评测集去污染   └─────────────── 普通指令 Alpaca
   │ 60% 有害提示再套一层训练用越狱模板（6 类）
   ▼
  generate   策略模型 Llama-3-8B-Instruct（就是要对齐的那个模型）
   │  · 每个提示采样 n=4 条回答（T=1.0）                     ← on-policy
   │  · 再采 1 条“参考回答”：有害提示加安全系统提示，良性提示加“过度谨慎”系统提示
   ▼
  judge      Llama-Guard-3-8B 给每条回答打 safe / unsafe 标签 + 拒答检测
   ▼
  pair       按下表规则组成 (prompt, chosen, rejected)，按 60/25/15 混合
```

| 提示类型 | chosen（优先级从高到低） | rejected（优先级从高到低） | 作用 |
|---|---|---|---|
| 有害（含越狱包装） | ① 安全系统提示下的回答，经 Llama Guard 判为安全<br>② 判为安全的 on-policy 采样<br>③ 拒答模板 | ① 判为不安全且未拒答的 on-policy 采样<br>② PKU-SafeRLHF 人工标注的不安全回答<br>③ 都没有则丢弃（模型在这条上已经安全） | 学会拒绝，包括带越狱包装的请求 |
| 边界 / 普通 | 未拒答且判为安全的 on-policy 采样（没有则丢弃） | ① on-policy 拒答<br>② 过度谨慎系统提示下的拒答<br>③ 拒答模板 | 惩罚过度拒答，保住有用性和 MMLU |

几个关键设计：

- **On-policy rejected**：rejected 是模型自己真会说出来的不安全回答。DPO 直接压低这些回答的概率，比用别的模型写的不安全文本更有针对性（off-policy 数据的问题见 Xu et al. 2024）。
- **上下文蒸馏（context distillation）**：chosen 优先取“加了安全系统提示”时模型自己的回答，但 DPO 训练用的 prompt 里**不带**系统提示。这样模型学到的是不靠系统提示也能安全地拒绝，而且拒绝时会解释原因、给出安全替代，比一句 “I can't” 好。
- **过度拒答对**：只训练拒绝，模型会变得“见到敏感词就拒”。每个良性样本都把拒答放在 rejected，这是保住 MMLU 和 XSTest 的关键，消融里可以单独关掉验证。
- **训练 / 评测模板分离**：训练用的 6 类越狱模板（`roleplay` 角色扮演、`fiction` 小说框架、`prefix_injection` 前缀注入、`refusal_suppression` 拒答抑制、`authority` 冒充权威、`hypothetical` 假设情境）与评测用的 4 类（`developer_mode`、`evil_confidant`、`text_continuation`、`translation_pretext`）完全不重叠（`tests/test_prompts.py` 会检查），HarmBench 上的越狱 ASR 衡量的是泛化。JBB 的 PAIR / GCG 提示是针对别的模型优化出来的，同样没有参与训练。
- **去污染**：训练提示与 HarmBench、JBB、XSTest 的全部 890 条评测提示比较，完全重合或 8-gram 覆盖率超过 50% 的样本会被剔除。

### 2. 训练

DPO 目标（Rafailov et al. 2023），π_ref 为冻结的基座：

$$\mathcal{L}_\text{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_c|x)}{\pi_\text{ref}(y_c|x)} - \beta\log\tfrac{\pi_\theta(y_r|x)}{\pi_\text{ref}(y_r|x)}\Big)\Big]$$

| 超参 | 值 |
|---|---|
| 基座 | `meta-llama/Meta-Llama-3-8B-Instruct` |
| LoRA | r=16, α=32, dropout=0.05，作用于 q/k/v/o/gate/up/down |
| β | 0.1 |
| 学习率 | 5e-6，cosine，warmup 10% |
| 有效 batch | 32（4 × 8 梯度累积） |
| epoch | 2 |
| 最大长度 | 1024 tokens |
| 硬件 | 1×A100/H100 80GB（bf16）；或 1×24GB（QLoRA，`configs/dpo_llama3_8b_qlora.yaml`） |

想加 RPO 式的 NLL 正则（防止 chosen 的似然一起下降），在 TRL ≥ 1.0 下设 `loss_type: [sigmoid, sft]`、`loss_weights: [1.0, 0.2]` 即可。

### 3. 评测

| 基准 | 规模 | 攻击设置 | 判定 | 指标 |
|---|---|---|---|---|
| HarmBench（test，standard + contextual） | 240 条行为 | 直接请求；4 个预留越狱模板 | 官方分类器 `cais/HarmBench-Llama-2-13b-cls`，官方 prompt | ASR，并分别报告逐模板平均和“任一模板成功”两种口径 |
| JailbreakBench（JBB-Behaviors） | 100 有害 + 100 良性 | 直接请求；PAIR@vicuna-13b、GCG@llama-2-7b 迁移攻击（官方 artifacts） | Llama-Guard-3-8B | ASR；良性样本拒答率 |
| XSTest | 250 安全 + 200 不安全 | — | 拒答字符串匹配 | 安全提示的过度拒答率 |
| MMLU | 57 科 | 5-shot | lm-evaluation-harness | acc |

所有回答用贪心解码，`max_new_tokens=512`，与 HarmBench 设置一致。copyright 类行为需要基于哈希的专用判定器，因此不计入。

---

## 快速开始

```bash
git clone https://github.com/Tommy-Ge/safety-dpo && cd safety-dpo
pip install -e ".[vllm,eval,dev]"          # 24GB 卡再加 [qlora]
huggingface-cli login                      # Llama-3 和 Llama Guard 3 需要先在 HF 上申请访问
```

### 不需要 GPU 的检查

```bash
make test     # 37 个单元测试：配对规则、去污染、指标、报告
make smoke    # 用 tiny 随机模型在 CPU 上跑通 构造数据 → DPO → 评测 → 报告（只验证流程，不代表效果）
SMOKE=1 bash scripts/run_gpu.sh   # 在 CPU 上预演下面的 GPU 一键脚本（CI 每次都会跑）
```

### 在云 GPU 上一键复现（推荐）

`scripts/run_gpu.sh` 把整条流程串成一个命令：

- **可断点续跑**：每完成一步会在 `runs/<run>/logs/` 留下 `.done` 标记。实例被中断或关机后，重跑同一条命令就会跳过已完成的步骤。
- **一个进程只加载一个模型**：策略模型、Llama Guard、HarmBench 分类器分别在独立进程里加载，避免 vLLM 在同一进程反复初始化出问题。
- **自动打包结果**：跑完后生成 `runs/full/bundle.tar.gz`，包含指标、图、更新后的 README、数据统计和训练日志，**不含**任何模型生成文本。下载到本地，在仓库根目录解压后提交即可。

```bash
# 1) 先在 Hugging Face 申请 Meta-Llama-3-8B-Instruct 和 Llama-Guard-3-8B 的访问权限，然后：
git clone https://github.com/Tommy-Ge/safety-dpo && cd safety-dpo
pip install -e ".[vllm,eval,dev]"
huggingface-cli login

# 2) 小样本试跑（约 20–40 分钟）：用几十条数据把 数据构造 → 训练 → 全部评测 → 报告 走一遍，先排除环境问题
QUICK=1 bash scripts/run_gpu.sh

# 3) 正式运行（建议放在 tmux / screen 里）
bash scripts/run_gpu.sh

# 4) 本地：解压结果并提交
tar xzf bundle.tar.gz && git add README.md results data/processed/stats.json && git commit -m "Results" && git push
```

| 显卡 | 用法 | 说明 |
|---|---|---|
| A100 / A800 / H100 80GB | `bash scripts/run_gpu.sh` | 推荐；bf16 LoRA |
| A100 40GB / L40S 48GB | 同上 | 一般够用；训练时如果 OOM，加 `-o training.per_device_train_batch_size=2 -o training.gradient_accumulation_steps=16` |
| RTX 4090 / 3090 24GB | `QLORA=1 bash scripts/run_gpu.sh` | 4-bit QLoRA 训练；显存不足 40GB 时脚本会自动把 13B 的 HarmBench 分类器改用 fp8 加载。如果你的 vLLM 版本或显卡不支持 fp8，评测这一步就换一张 40GB 以上的卡 |

国内平台（如 AutoDL）：脚本检测到 `/etc/network_turbo` 时会自动开启学术加速，以便访问 GitHub 和 Hugging Face。也可以设置 `export HF_ENDPOINT=https://hf-mirror.com`，gated 模型仍然需要先登录。整个流程的大头是数据构造时的采样、两次完整评测和 MMLU。具体耗时取决于显卡和网速，第一次跑之前先用 `QUICK=1` 看看每一步大概要多久。

### 分步运行（单张 80GB GPU）

```bash
make data        # 构造偏好数据 → data/processed/{train,test}.jsonl + stats.json
make train       # LoRA + DPO → outputs/llama3-8b-safety-dpo/final
make eval-base   # 评测原始模型 → results/base/metrics.json
make eval-dpo    # 评测对齐后模型 → results/dpo/metrics.json
make report      # 生成表格 + 图，写入下面的“结果”一节
```

每个阶段都会缓存中间结果，可以单独重跑，例如 `safedpo build-data --stage judge`、`safedpo eval --stages judge,metrics`。在同一进程里先后加载两个 vLLM 模型偶尔会不稳定，遇到这种情况就分阶段运行。任何配置项都可以用 `-o` 在命令行覆盖：

```bash
safedpo train -o training.beta=0.05 -o training.output_dir=outputs/beta005
safedpo build-data -o pairing.mix.borderline=0 -o pairing.mix.general=0   # 消融：去掉过度拒答对
```

---

## 结果

### A. 笔记本规模复现：Qwen2.5-1.5B-Instruct（Apple M4，24GB）

整条流程完全相同，只是规模缩小，并换成不需要申请权限的 Apache-2.0 模型，在一台 M4 MacBook 的 GPU（MPS）上跑完：`MAC=1 bash scripts/run_gpu.sh`。

| | 完整配置（8B，云 GPU） | 本节（M4 笔记本） |
|---|---|---|
| 策略模型 | Llama-3-8B-Instruct | Qwen2.5-1.5B-Instruct |
| 训练数据标注 | Llama-Guard-3-8B | Qwen3Guard-Gen-0.6B |
| 评测判定 | HarmBench 官方分类器 / Llama-Guard-3-8B | Qwen3Guard-Gen-4B（与标注模型不同、更大，避免“自己判自己”） |
| 有害 / 边界 / 普通提示 | 4000 / 1500 / 800，每个提示采样 4 次 | 500 / 250 / 120，每个提示采样 3 次 |
| 训练 | bf16 LoRA，有效 batch 32，lr 5e-6 | bf16 LoRA（MPS），有效 batch 16，lr 5e-5 |
| 生成长度上限 | 512 | 256 |
| MMLU | 5-shot | 0-shot，全部 14,042 题 |

评测集与完整配置相同：HarmBench 240 条行为 × 5 种攻击设置，JBB 100 条有害行为的直接请求加 PAIR / GCG 迁移攻击，外加 100 条良性行为、XSTest 450 条和 MMLU。

<!-- RESULTS-MAC:START -->
> 正在 M4 上运行，跑完后由 `safedpo report --marker RESULTS-MAC` 自动填入。
<!-- RESULTS-MAC:END -->

### B. 完整配置：Llama-3-8B-Instruct（云 GPU）

<!-- RESULTS:START -->
> 还没有在 GPU 上跑出结果。运行 `bash scripts/run_gpu.sh`（或分步运行后执行 `make report`）后，这里会自动填入：主结果表（各基准 ASR、过度拒答率、MMLU，以及相对基座的变化）、各攻击设置的 ASR 及 95% 置信区间、HarmBench 分类别结果、构造出的偏好数据统计和训练曲线。
<!-- RESULTS:END -->

### 建议的消融

| 实验 | 命令 | 想回答的问题 |
|---|---|---|
| 去掉过度拒答对 | `-o pairing.mix.borderline=0 -o pairing.mix.general=0` | 良性样本对 MMLU 和 XSTest 的贡献有多大 |
| 去掉越狱增强 | `-o jailbreak_augment.fraction=0` | 训练时见过越狱包装，能否泛化到没见过的模板 |
| 只用 off-policy 数据 | `-o sampling.n=0`，rejected 全部来自 PKU 标注 | on-policy 数据的作用 |
| β ∈ {0.05, 0.1, 0.3} | `-o training.beta=...` | 安全性和能力保持之间的权衡 |
| LoRA r ∈ {8, 16, 64} | `-o lora.r=...` | 容量 |

---

## 项目结构

```
configs/
  data.yaml                  数据源、采样、配对规则、混合比例、去污染
  dpo_llama3_8b.yaml         bf16 LoRA（80GB）
  dpo_llama3_8b_qlora.yaml   4-bit QLoRA（24GB）
  eval.yaml                  基准、攻击设置、判定模型
  smoke/                     CPU 冒烟测试用配置
src/safedpo/
  prompts.py                 安全 / 过度谨慎系统提示、训练和评测越狱模板、拒答检测
  generation.py              vLLM / HF 两种推理后端（支持 LoRA）
  judges.py                  Llama Guard 3、HarmBench 分类器、关键词判定
  data/sources.py            PKU-SafeRLHF / OR-Bench / Alpaca 加载
  data/decontam.py           n-gram 去污染
  data/build.py              collect → generate → judge → pair
  train.py                   LoRA + DPO（兼容 TRL 0.x 和 1.x）
  eval/benchmarks.py         HarmBench / JBB（含 artifacts） / XSTest 加载
  eval/run.py                generate → judge → mmlu → metrics
  report.py                  表格、图，写回 README
scripts/run_gpu.sh           云 GPU 一键复现（断点续跑、小样本试跑、结果打包）
tests/                       单元测试 + fixtures
```

## 负责任使用

- `data/work/`、`data/processed/*.jsonl`、`results/*/generations.jsonl`、`results/*/judged.jsonl` 里有模型生成的不安全文本，都已写进 `.gitignore`，**不要公开上传**。仓库只提交指标文件。
- 越狱模板都是公开越狱手法的简短改写，只用于训练和评测防御。

## 局限

- Llama Guard 3 既给训练数据打标签，又是 JBB 的判定器，可能让结果偏乐观。HarmBench 用的是独立的分类器，可以交叉验证。
- 拒答检测用的是字符串匹配，对“部分拒答”和“先拒绝再照做”的识别不够准。XSTest 原论文用的是人工标注或 GPT-4 判定。
- 评测的越狱攻击是固定模板和迁移攻击，没有针对对齐后模型重新做自适应攻击（比如重跑 PAIR / GCG），所以 ASR 是下界意义上的稳健性。
- MMLU 只反映知识类能力，生成质量、指令遵循等方面最好再补 MT-Bench / AlpacaEval。

## 参考文献

- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. / Dettmers et al. *QLoRA.* NeurIPS 2023.
- Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024.
- Chao et al. *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* NeurIPS 2024 D&B.
- Röttger et al. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs.* NAACL 2024.
- Ji et al. *PKU-SafeRLHF / BeaverTails.* 2023–2024. / Cui et al. *OR-Bench: An Over-Refusal Benchmark.* 2024.
- Inan et al. *Llama Guard.* 2023. / Askell et al. *A General Language Assistant as a Laboratory for Alignment*（context distillation）. 2021.
- Xu et al. *Is DPO Superior to PPO for LLM Alignment?* ICML 2024. / Pang et al. *Iterative Reasoning Preference Optimization*（RPO）. 2024.

## License

MIT。使用 Llama 3 和 Llama Guard 3 需遵守 Meta Llama 3 Community License。
