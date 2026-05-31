# 阶段 A：答案层评测脚本设计

本文件给出"答案层评测 + 按 question_type 分组评测"的完整脚本架构。设计完成后可交给 Codex 落地实现，或由本人实现。所有路径、文件名、Split 口径、`should_abstain` 处理方式都遵循当前 `aiops-docs/experiment/` 已冻结的实验纪律，不破坏既有 retrieval 结果。

---

## 1. 设计目标

补足当前 RAG 实验只有检索层指标的空白，引入下面四类答案层指标，并支持按 `question_type` 分组：

| 指标 | 定义 | 主要用途 |
| --- | --- | --- |
| Faithfulness | 答案中每个声明是否被检索到的证据所支持 | 反映"幻觉率"的反面 |
| Answer Correctness | 答案与 gold_answer 的语义一致性 | 端到端正确率 |
| Citation Accuracy | 答案声称引用的 chunk_id 是否真的支持对应声明 | 引用真实性 |
| Hallucination Rate | Faithfulness 的反向衍生指标 | 论文中常用 |

辅助维度：

- 按 `question_type` 分组（troubleshooting、alarm_fault_code、parameter_and_configuration、safety_and_constraint、maintenance_procedure）
- 按 `should_abstain` 区分 answerable / should_abstain 子集
- 记录 generator 模型、judge 模型、运行日期、token 消耗（用于论文实施细节章节）

---

## 2. 与现有资产的对齐

下面这些资产**只读不改**：

| 资产 | 路径 | 用途 |
| --- | --- | --- |
| Validated dataset | `aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl` | 提供 question / gold_answer / evidence_chunk_ids / question_type / should_abstain |
| Splits | `aiops-docs/experiment/rag/splits/expanded/rag_{build,dev,test,reserve}.jsonl` | 沿用 build/dev/test 切分 |
| Retrieval result（推荐 backbone） | `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_test.json` | 提供每条 query 的 top10 chunk_ids 与 chunk text |
| Chunk store | `aiops-docs/experiment/chunks/experiment_chunks.jsonl` | 提供 chunk text 全文，用于补齐证据原文 |
| Confidence test result | `aiops-docs/experiment/results/confidence/final/confidence_final_dense_current_original_test.json` | 标识每条样本的最终 confidence bucket（high/medium/low） |

新增产物全部落到下面两个目录：

```
aiops-docs/experiment/results/answer/expanded/
aiops-docs/experiment/results/thesis_tables/answer/expanded/
```

不要写入 retrieval/、不要修改 retrieval JSON。

---

## 3. 总体流水线

整个 Stage A 由 **三个独立可重入脚本** 串联：

```
[Stage A.1] generate_rag_answers.py
   ├─ in : dataset + retrieval JSON + chunks
   └─ out: answers JSONL（每条样本一条记录，含生成答案、引用、用到的证据）

[Stage A.2] judge_rag_answers.py
   ├─ in : answers JSONL + dataset + chunks
   └─ out: judge JSONL（每条样本对应 4 类指标的细分评分 + 评判 reasoning）

[Stage A.3] aggregate_answer_evaluation.py
   ├─ in : judge JSONL + split 信息 + question_type 信息
   └─ out: 总览 JSON + thesis_tables CSV + Markdown 报告
```

每个脚本都支持 `--split build|dev|test|reserve`，先在 build/dev 跑通流程并校准 prompt，再到 test 上一次性跑出最终数据。这一原则与现有实验纪律一致。

---

## 4. 数据契约

### 4.1 Stage A.1 输出：`{backbone}_{query_mode}_{split}_answers.jsonl`

每行一条 JSON，字段约定如下（字段全部以 snake_case，便于直接读入 pandas）：

```json
{
  "sample_id": "rag-058",
  "split": "test",
  "question_type": "troubleshooting_procedure",
  "should_abstain": false,
  "question": "When the rotary unit reports a servo alarm, what should be checked first?",
  "gold_answer": "First check the brake release pressure and the cable connections to the rotary motor.",
  "gold_evidence_chunk_ids": [
    "haascnc_com_rotary_troubleshooting_guide_ngc::p12-p13::c0008"
  ],
  "retrieval": {
    "experiment_name": "dense_current_rerank",
    "split": "test",
    "candidate_top_k": 50,
    "final_top_k": 10,
    "rerank_provider": "local",
    "final_chunk_ids": ["...", "...", "..."],
    "final_chunk_scores": [0.92, 0.81, ...]
  },
  "generation": {
    "generator_model": "gpt-4o-mini-2024-07-18",
    "generator_prompt_version": "v1.0",
    "system_prompt_hash": "sha256:...",
    "generated_answer": "When a servo alarm appears on the rotary unit, the first check should be the brake release pressure ... [chunk_id=haascnc_com_rotary_troubleshooting_guide_ngc::p12-p13::c0008]",
    "cited_chunk_ids": [
      "haascnc_com_rotary_troubleshooting_guide_ngc::p12-p13::c0008"
    ],
    "abstained": false,
    "abstention_reason": null,
    "generation_latency_ms": 1840,
    "generation_tokens": {"prompt": 2150, "completion": 142},
    "generation_finish_reason": "stop"
  },
  "meta": {
    "run_id": "stage_a1_20260508_001",
    "git_sha": "<filled at runtime>",
    "timestamp": "2026-05-08T10:00:00+08:00"
  }
}
```

要点：
- `cited_chunk_ids` 必须由生成端按统一 prompt 模板**显式输出**，禁止从 `final_chunk_ids` 兜底推断
- `abstained` 用于刻画 generator 是否选择拒答；与 retrieval 层 confidence bucket 解耦
- `generation_tokens` 用于在论文中报告平均提示长度和成本

### 4.2 Stage A.2 输出：`{backbone}_{query_mode}_{split}_judge.jsonl`

每行一条记录，**每条记录对应一条样本的 4 类指标的全部细分判定**：

```json
{
  "sample_id": "rag-058",
  "split": "test",
  "question_type": "troubleshooting_procedure",
  "should_abstain": false,
  "judge_model": "claude-sonnet-4-6",
  "judge_prompt_version": "v1.0",
  "judge_run_id": "stage_a2_20260508_001",

  "claims": [
    {
      "claim_id": "rag-058::c1",
      "text": "the first check should be brake release pressure",
      "cited_chunk_ids": ["haascnc_com_rotary_troubleshooting_guide_ngc::p12-p13::c0008"],
      "supported_by_retrieved": true,
      "supported_by_cited": true,
      "supporting_evidence_quote": "Check brake release pressure first ...",
      "judge_reasoning": "The chunk explicitly states ..."
    }
  ],

  "faithfulness": {
    "n_claims": 3,
    "n_supported_by_retrieved": 3,
    "n_supported_by_cited": 2,
    "score_supported_by_retrieved": 1.0,
    "score_supported_by_cited": 0.667
  },

  "answer_correctness": {
    "verdict": "correct",
    "score": 1.0,
    "judge_reasoning": "The generated answer captures the same key actions ..."
  },

  "citation_accuracy": {
    "n_citations": 2,
    "n_correct_citations": 2,
    "n_missing_citations": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },

  "hallucination": {
    "n_unsupported_claims": 0,
    "rate": 0.0,
    "any_hallucination": false
  },

  "abstention_check": {
    "should_abstain_label": false,
    "model_abstained": false,
    "abstention_correct": true
  }
}
```

`verdict` 三档：`correct / partially_correct / incorrect`，对应 score 1.0 / 0.5 / 0.0，便于做 strict 与 lenient 两种聚合。

### 4.3 Stage A.3 输出

| 文件 | 用途 |
| --- | --- |
| `answer_eval_summary_{backbone}_{query_mode}_{split}.json` | 单 split 全局指标 |
| `answer_eval_by_qtype_{backbone}_{query_mode}_{split}.json` | 分 question_type 指标 |
| `thesis_tables/answer/expanded/answer_eval_main_{split}.csv` | 论文主表（行：实验名，列：四类指标） |
| `thesis_tables/answer/expanded/answer_eval_by_qtype_{split}.csv` | 论文分组表 |
| `aiops-docs/experiment/reports/answer_evaluation_{split}_report.md` | Markdown 报告，含分析段落与表格 |

---

## 5. Stage A.1：Answer Generation 设计

### 5.1 输入参数

```
python -m scripts.experiment.generate_rag_answers \
  --dataset aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl \
  --split-file aiops-docs/experiment/rag/splits/expanded/rag_test.jsonl \
  --retrieval-results aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_test.json \
  --chunks aiops-docs/experiment/chunks/experiment_chunks.jsonl \
  --confidence-results aiops-docs/experiment/results/confidence/final/confidence_final_dense_current_original_test.json \
  --generator-model gpt-4o-mini \
  --abstention-policy confidence_bucket \
  --output aiops-docs/experiment/results/answer/expanded/dense_current_original_test_answers.jsonl \
  --max-context-chunks 10 \
  --temperature 0.0 \
  --seed 20260508 \
  --batch-size 8 \
  --resume
```

### 5.2 关键设计点

**Abstention 路径与 confidence 解耦**：
- `--abstention-policy confidence_bucket`：当 confidence bucket 为 `low` 时，generator 收到一个 "abstain"  指令，输出固定模板答案与 `abstained=true`
- `--abstention-policy never_abstain`：所有样本都生成（包括 should_abstain=true）。用于诊断 generator 在缺证据时是否会胡编
- `--abstention-policy gold_label`：用 gold `should_abstain` 直接决定是否生成。**仅作消融，不能作为主结果**

至少应跑 `confidence_bucket` 与 `never_abstain` 两组，在 build/dev 上能对比不同策略下的幻觉率，论文里可作为单独消融。

**确定性**：
- temperature=0.0、固定 seed
- 把 retrieved chunks 按 final_chunk_scores 降序、稳定排序

**`--resume`**：
- 把 sample_id 持久化到 checkpoint，运行被打断后从断点续跑
- 避免重复计费和数据漂移

**chunk 文本注入**：
- 不要直接把 chunk_id 拼进 prompt，要用 `[chunk:{idx}]` 这样的索引标识，附录给 chunk_id ↔ idx 映射，避免 generator 把 chunk_id 字符串当成内容
- 示例 prompt 段落见 §7.1

**should_abstain 处理**：
- 即使 should_abstain=true 也要生成一条记录，但 generator 应在收到 0 retrieved 或 confidence=low 的 prompt 后输出固定 abstain 模板
- 这一行为后面 judge 阶段会用 `abstention_check` 单独评估

### 5.3 错误恢复

- API 429 / 5xx → 指数退避重试，最多 5 次
- 单条失败不中断 batch，写到 `*_errors.jsonl`
- 全部完成后输出统计：成功/失败/abstain 计数

---

## 6. Stage A.2：LLM-as-Judge 设计

### 6.1 评判模型选择

| 角色 | 默认模型 | 替代 | 备注 |
| --- | --- | --- | --- |
| Generator | `gpt-4o-mini-2024-07-18` | 系统中现用 generator | 写论文时也要在系统现有 generator 上跑一组，便于声称"在生产配置下" |
| Judge（主） | `claude-sonnet-4-6` | gpt-4o | 与 generator 不同家族，规避自评偏置 |
| Judge（仲裁） | `claude-opus-4-6` 或 `gpt-4o` | — | 仅在主 judge 标 `unsure` 或两次跑出现分歧时调用 |

**抗自评偏置**：generator 为 OpenAI 系，judge 默认选 Anthropic 系；若 generator 为 Anthropic，则 judge 改 OpenAI。

### 6.2 输入参数

```
python -m scripts.experiment.judge_rag_answers \
  --answers aiops-docs/experiment/results/answer/expanded/dense_current_original_test_answers.jsonl \
  --dataset aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl \
  --chunks aiops-docs/experiment/chunks/experiment_chunks.jsonl \
  --judge-model claude-sonnet-4-6 \
  --judge-prompt-version v1.0 \
  --output aiops-docs/experiment/results/answer/expanded/dense_current_original_test_judge.jsonl \
  --temperature 0.0 \
  --batch-size 4 \
  --enable-claim-extraction \
  --resume
```

### 6.3 评判子任务

每条样本的评判分四步串行（同一次 API 内合并以省 token，但**逻辑上独立**）：

**Step 1 — Claim Extraction（声明抽取）**：
- 将 `generated_answer` 拆成原子声明（atomic claims）
- 每个声明保留其在原文中的引用标号
- 输出：`claims[]`

**Step 2 — Faithfulness（忠实度）**：
对每条 claim 判断：
- `supported_by_retrieved`：retrieved chunks 是否能支持
- `supported_by_cited`：claim 自带的 cited_chunk_ids 是否能支持
- 同时给出 `supporting_evidence_quote`（必须是原文片段，不可改写）和 `judge_reasoning`

**Step 3 — Answer Correctness（正确性）**：
- 输入：`question` + `gold_answer` + `generated_answer`
- 输出三档判定：correct / partially_correct / incorrect
- 给出 `judge_reasoning`，至少 2 句

**Step 4 — Citation Accuracy（引用准确性）**：
- 对每个 cited_chunk_id：是否在该 chunk 中确实能找到支持
- 同时统计 `n_missing_citations`：模型应当 cite 但没 cite 的（基于 gold_evidence_chunk_ids 与 retrieved 的交集判定）

### 6.4 关键防偏措施

**先抽 claim 再判定，避免一次性问太多**。Anthropic 的 RAGAS / TruLens 工程经验表明：把"答案对不对"和"声明对不对"耦合在一个 prompt 里，judge 容易给出整体印象分而不是逐项判定。

**为每个判定要求"逐字证据"**：要求 judge 输出 `supporting_evidence_quote`（必须是 retrieved chunks 中的原文连续片段），且让脚本做后处理校验该 quote 是否真的存在于 retrieved chunks 中。如果不存在，把对应 claim 的 `supported_by_retrieved` 强制改为 `false` 并记录 `judge_quote_invalid=true`，避免 judge 自己幻觉。

**温度=0**：所有 judge 调用 temperature=0，判断稳定优先。

**采样人工复核**：脚本应支持 `--sample-for-human-review N`，把 N 条 judge 结果导出到独立 CSV，人工复核 judge 是否一致。建议 build 上抽 5–10 条做 prompt 校准，dev 上抽 5 条做最终 judge 信度抽检。这一抽检结果应在论文实验细节中报告（"Cohen's κ between judge and human = X"）。

### 6.5 成本估算

```
Test set 60 条样本
× 平均 prompt 长度 ~3500 tokens（含 retrieved chunks）
× 每条样本生成 ~600 tokens（含 reasoning）
× judge_model = claude-sonnet-4-6（按当前价格）
≈ 单次 judge 成本 ~$0.5–1.5
```

build/dev/test 全跑下来一次约 $2–5。考虑 prompt 迭代和重跑，预算 $10 左右。这一数字应作为论文实施细节中的"实验成本"信息记录。

---

## 7. Prompt 模板（v1.0）

为了使 prompt 可版本化且复现，所有 prompt 单独存放在：

```
scripts/experiment/prompts/answer_eval/
├── generator_v1.txt
├── judge_claim_extract_v1.txt
├── judge_faithfulness_v1.txt
├── judge_correctness_v1.txt
└── judge_citation_v1.txt
```

每个 prompt 文件首行注释写版本号。脚本读取时计算 SHA-256 写入结果 JSON 的 `*_prompt_version_hash`，便于追溯。

### 7.1 Generator Prompt 设计要点

```
[System]
You are a precise question-answering assistant for industrial machine fault troubleshooting.
You answer ONLY using the provided evidence chunks. If the evidence is insufficient, you must abstain by replying exactly:
"INSUFFICIENT_EVIDENCE: <one-sentence reason>"

When you can answer, format the answer in <= 4 sentences and append citation markers in this exact form: [chunk:{idx}]. Cite every chunk you used. Do not cite chunks you did not use.

[User]
Question:
{question}

Evidence chunks (sorted by relevance):
[chunk:1] (id={chunk_id_1})
{chunk_text_1}

[chunk:2] (id={chunk_id_2})
{chunk_text_2}
...

Answer:
```

设计要点：
- 用 `[chunk:idx]` 而非 `chunk_id` 字符串，避免 generator 把长 ID 当成自然语言写入答案
- 强制 abstain 模板 `INSUFFICIENT_EVIDENCE: ...`，便于 parser 自动识别
- 限制句数，防止生成过长答案稀释 faithfulness 计算
- "Cite every chunk you used" 与 "Do not cite chunks you did not use" 同时强调，便于评测 citation precision/recall

后处理：
- parser 把 `[chunk:idx]` 还原为 `chunk_id` 写入 `cited_chunk_ids`
- 移除答案文本中的 `[chunk:N]` 标记后保留干净答案 `clean_answer`

### 7.2 Judge Prompt 设计要点

每个 judge prompt 严格遵循下面三段结构：

```
[Task]
You are evaluating <metric_name>. You must reply ONLY in the JSON schema given below.

[Schema]
{
  "verdict": "...",
  "reasoning": "...",
  ...
}

[Inputs]
question: ...
gold_answer: ...
generated_answer: ...
retrieved_chunks: [...]
```

要点：
- 用 JSON schema 强约束输出格式
- 评判前要求 judge 给出 `reasoning`（chain-of-thought 提升稳定性，但 reasoning 不要超过 100 词，避免 cost 失控）
- 任何引用必须是原文连续片段，写入 `supporting_evidence_quote` 字段；脚本后处理校验

---

## 8. Stage A.3：聚合与报告

### 8.1 输入参数

```
python -m scripts.experiment.aggregate_answer_evaluation \
  --judge-files aiops-docs/experiment/results/answer/expanded/dense_current_original_test_judge.jsonl \
  --dataset aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl \
  --split-files aiops-docs/experiment/rag/splits/expanded/rag_test.jsonl \
  --output-json aiops-docs/experiment/results/answer/expanded/answer_eval_summary_dense_current_original_test.json \
  --output-csv aiops-docs/experiment/results/thesis_tables/answer/expanded/answer_eval_main_test.csv \
  --output-report aiops-docs/experiment/reports/answer_evaluation_test_report.md \
  --by-question-type \
  --bootstrap-ci 1000
```

### 8.2 主表（CSV）字段

```
experiment_name, split, evaluated_samples, abstain_samples,
faithfulness_supported_by_retrieved_mean,
faithfulness_supported_by_cited_mean,
correctness_strict_mean,        # 仅 correct=1，partial=0
correctness_lenient_mean,       # correct=1, partial=0.5
citation_precision_mean,
citation_recall_mean,
citation_f1_mean,
hallucination_rate_mean,
abstention_precision,
abstention_recall,
generator_model, judge_model, prompt_version, run_date
```

每个 mean 列同时输出 `_ci_low_95` 与 `_ci_high_95`（bootstrap 1000 次）。这是为了应对 60 条 test 样本的小样本问题，给论文表格补充置信区间。

### 8.3 分组表（CSV）字段

```
experiment_name, split, question_type, n_samples,
faithfulness_supported_by_retrieved_mean,
correctness_strict_mean,
citation_f1_mean,
hallucination_rate_mean
```

### 8.4 报告模板

`answer_evaluation_test_report.md` 应自动生成下列段落：

1. **数据规模与拆分**：evaluated samples、abstain samples、按 question_type 分布
2. **总体指标表**：主表 CSV 的 Markdown 渲染
3. **按 question_type 分组**：分组表 + 一段自动生成的弱项归纳（如 "alarm_fault_code 表现最高，safety_and_constraint 最低，相差 X 个百分点"）
4. **置信区间说明**：bootstrap 方法、N=1000、解释为何在 60 条小样本下需要置信区间
5. **抽检结果**：human spot-check 的 κ 值与不一致样例数量
6. **诚实性 caveat**（必填段落）：
   - judge 不是 ground truth，仅作离线代理评测
   - generator 模型版本、API 日期与温度
   - should_abstain 子集的特殊处理口径
   - 60 条样本下小差距不一定显著

---

## 9. 实验纪律与红线

写入流水线的每个脚本都应在 `--help` 里复述这些约束：

```
不要重新生成 expanded dataset
不要重新切 chunk
不要重新索引 Milvus
不要覆盖 retrieval 已有 JSON
不要把 current_rerank 写成 pure Cohere rerank
generator/judge 模型与版本必须落到结果 JSON
test 只用于最终冻结后的运行，先在 build/dev 上跑通 prompt
should_abstain=true 样本以独立子集报告，不要混进 answerable 平均
```

---

## 10. 推荐执行顺序

```
Phase 1: 准备 prompt 模板 v1.0，提交到 scripts/experiment/prompts/answer_eval/
Phase 2: 在 build (30) 上跑 generate → judge → aggregate；人工复核 judge 输出 ~5 条
Phase 3: 调整 prompt 至 v1.1 / v1.2，再跑 dev (35) 验证稳定性
Phase 4: prompt 冻结后，在 test (60) 上一次性跑出最终结果
Phase 5: 生成 thesis_tables CSV 与 markdown 报告
Phase 6: （可选）切换 generator 或 judge 模型，做敏感性分析消融
Phase 7: 把数据汇入论文第四章实验结果分析
```

---

## 11. 论文实验章节中的呈现方式

按 `dissertation-writing` skill 的 `experiment-reporting.md` 建议，本阶段产出在论文中应单独成节，建议放在第四章 "实验与结果分析" 的 4.X 节："答案层评测与幻觉控制"。结构：

1. **4.X.1 评测设置**：generator / judge 模型、prompt 版本、判定流程、人工抽检方法
2. **4.X.2 总体结果**：主表 + 置信区间，文字解读 2–3 段
3. **4.X.3 按问题类型的细分**：分组表 + 弱项分析
4. **4.X.4 幻觉案例分析**：抽 2–3 条幻觉率高的样本做 case study
5. **4.X.5 局限性**：judge 一致性、样本规模、generator 版本依赖等

附录中给出 prompt 全文（v1.0）、人工抽检明细、bootstrap 方法说明。

---

## 12. 风险清单

| 风险 | 缓解 |
| --- | --- |
| Judge LLM 自评偏置 | generator 与 judge 选不同家族；保留 reasoning 可审计 |
| Judge 引用幻觉（声称的 supporting_quote 不在原文） | 后处理校验 quote 必须是 retrieved chunks 子串 |
| 60 条 test 样本统计噪声大 | 报告 95% bootstrap CI；不要包装小差距为"显著优于" |
| Cohere 429 fallback 影响 generator 上下文 | 在结果 JSON 记录 rerank_provider；论文实施细节如实报告 |
| Generator API 版本漂移 | model 字段记录精确版本（如 `gpt-4o-mini-2024-07-18`），不要写 `gpt-4o-mini` |
| Prompt 微改导致结果跳变 | prompt 版本化 + SHA-256 哈希；论文中固化为 v1.0 |
| Cost 失控 | `--resume` + checkpoint；先跑 build 估算单条成本 |

---

## 13. 落地清单（给 Codex 的执行 checklist）

```
□ 创建 scripts/experiment/prompts/answer_eval/ 目录与五个 prompt 文件
□ 实现 generate_rag_answers.py（含 abstention policy 与 resume）
□ 实现 judge_rag_answers.py（含 4 类指标 + quote 校验）
□ 实现 aggregate_answer_evaluation.py（含 bootstrap CI 与 by-qtype）
□ 实现 sample_for_human_review.py（导出 CSV 给人工复核）
□ 在 build 上跑通三阶段，调整 prompt
□ dev 验证稳定性
□ test 一次性运行
□ 输出 thesis_tables CSV 与 Markdown 报告
□ 在论文第四章新增 4.X 节
```
