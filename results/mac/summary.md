平均越狱成功率（所有攻击设置）从 **24.9%** 降到 **3.7%**，相对降低 **85%**；MMLU 从 60.2 变为 59.9（-0.3 pt，相对 -0.53%）；XSTest 过度拒答率 46.8% → 58.4%。

### 主结果（%）

| Model | HarmBench Direct ↓ | HarmBench Held-out Templates ↓ | HarmBench Any-Template ↓ | JBB Direct ↓ | JBB Transfer (PAIR/GCG) ↓ | Avg ASR ↓ | XSTest Over-refusal ↓ | JBB Benign Refusal ↓ | MMLU ↑ |
|---|---|---|---|---|---|---|---|---|---|
| base | 17.9 | 30.0 | 67.1 | 1.0 | 30.0 | 24.9 | 46.8 | 46.0 | 60.2 |
| dpo | 2.9 | 3.8 | 10.8 | 0.0 | 6.0 | 3.7 | 58.4 | 52.0 | 59.9 |
| dpo_no_benign | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 98.4 | 100.0 | 59.9 |
| Δ dpo vs base | **−84%** rel. | **−88%** rel. | **−84%** rel. | **−100%** rel. | **−80%** rel. | **−85%** rel. | +11.6 pt | +6.0 pt | -0.3 pt |
| Δ dpo_no_benign vs base | **−100%** rel. | **−100%** rel. | **−100%** rel. | **−100%** rel. | **−100%** rel. | **−100%** rel. | +51.6 pt | +54.0 pt | -0.3 pt |

### 各攻击设置的 ASR（%，括号内为 95% Wilson 置信区间）

| Benchmark | Attack | base | dpo | dpo_no_benign |
|---|---|---|---|---|
| harmbench | direct | 17.9 [13.6, 23.3] | 2.9 [1.4, 5.9] | 0.0 [0.0, 1.6] |
| harmbench | template:developer_mode | 12.1 [8.5, 16.8] | 1.2 [0.4, 3.6] | 0.0 [0.0, 1.6] |
| harmbench | template:evil_confidant | 6.2 [3.8, 10.1] | 0.0 [0.0, 1.6] | 0.0 [0.0, 1.6] |
| harmbench | template:text_continuation | 41.7 [35.6, 48.0] | 6.2 [3.8, 10.1] | 0.0 [0.0, 1.6] |
| harmbench | template:translation_pretext | 60.0 [53.7, 66.0] | 7.5 [4.8, 11.5] | 0.0 [0.0, 1.6] |
| jbb | direct | 1.0 [0.2, 5.4] | 0.0 [0.0, 3.7] | 0.0 [0.0, 3.7] |
| jbb | artifact:PAIR@vicuna-13b-v1.5 | 54.0 [44.3, 63.4] | 10.0 [5.5, 17.4] | 0.0 [0.0, 3.7] |
| jbb | artifact:GCG@llama-2-7b-chat-hf | 6.0 [2.8, 12.5] | 2.0 [0.6, 7.0] | 0.0 [0.0, 3.7] |

### HarmBench 各语义类别（直接请求，ASR %）

| HarmBench category (direct) | base | dpo | dpo_no_benign |
|---|---|---|---|
| chemical_biological | 38.1 | 9.5 | 0.0 |
| cybercrime_intrusion | 25.0 | 3.8 | 0.0 |
| harassment_bullying | 0.0 | 0.0 | 0.0 |
| harmful | 16.7 | 0.0 | 0.0 |
| illegal | 1.9 | 0.0 | 0.0 |
| misinformation_disinformation | 18.5 | 1.9 | 0.0 |

![ASR by attack](results/mac/asr_by_attack.png)

### 拒答率：LLM 判定 vs 字符串匹配（%）

主表里的拒答率用 LLM 判定（Qwen3Guard 的 `Refusal` 字段）。字符串匹配只认“I'm sorry / I cannot”这类固定说法，会漏掉换了说法的拒答，下表把两者并列。

| 指标 | base（LLM / 字符串） | dpo（LLM / 字符串） | dpo_no_benign（LLM / 字符串） |
|---|---|---|---|
| XSTest 安全问题被拒（过度拒答） | 46.8 / 52.4 | 58.4 / 25.6 | 98.4 / 100.0 |
| XSTest 不安全问题被拒 | 96.0 / 97.5 | 97.0 / 78.5 | 100.0 / 100.0 |
| JBB 良性行为被拒 | 46.0 / 49.0 | 52.0 / 40.0 | 100.0 / 100.0 |

### 构造出的偏好数据

| 项目 | 值 |
|---|---|
| 训练 / 验证偏好对 | 966 / 50 |
| 训练集按类型 | harmful 641，borderline 212，general 113 |
| chosen 来源 | ref_safety_sysprompt 508，policy_compliant 335，policy_safe 89，template 84 |
| rejected 来源 | offline_unsafe 470，policy_unsafe 211，template 177，policy_refusal 97，ref_overcautious_sysprompt 61 |
| 有害样本的越狱包装 | none 500，prefix_injection 52，fiction 48，hypothetical 45，refusal_suppression 15，roleplay 13，authority 8 |
| 基座在训练有害提示上的不安全率（全部 / 带越狱包装） | 20.8% / 48.9% |
| 基座在良性训练提示上的拒答率 | 19.1% |

### 训练曲线

![training curves](results/mac/training_curves.png)
