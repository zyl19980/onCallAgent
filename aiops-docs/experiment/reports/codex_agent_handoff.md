# Codex Agent Handoff Report

本报告面向接手当前毕业论文 RAG 实验的新 Codex agent。目标不是重复论文叙述，而是让新的 agent 能在最短时间内理解：

- 当前实验做到哪一步了
- 现在哪些文件是主入口
- 当前哪些结论已经成立
- 下一步应从哪里继续，而不是重复旧工作

---

## 1. One-Screen Status

当前实验已经完成：

- 独立实验数据流水线搭建
- 两批候选题生成与人工审核
- 扩展版正式 RAG dataset 构建与校验
- expanded build/dev/test/reserve 切分
- expanded dense retrieval / current rerank 实验
- 初步 retrieval confidence baseline 实验
- 实验目录重组与阶段报告归档

当前主实验资产是：

- validated dataset  
  [experiment_rag_dataset_expanded.validated.jsonl](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl)
- expanded split  
  [rag/splits/expanded](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/splits/expanded)
- unified evidence base  
  [experiment_chunks.jsonl](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/chunks/experiment_chunks.jsonl)
- Milvus collection  
  `experiment_manuals_all`

当前不要再做的事：

- 不要恢复旧 `context_catalog_v1.jsonl`
- 不要恢复旧评测系统
- 不要重新切 PDF 构造新的 chunk，除非用户明确要求重做切片策略实验
- 不要覆盖 batch1 / batch2 / merged / pilot / expanded 现有产物

---

## 2. Current Directory Contract

当前实验目录已经按阶段和用途重组，新的 agent 应优先使用这些路径。

### 2.1 RAG Assets

- 候选题
  - [rag/candidates/batch1](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/candidates/batch1)
  - [rag/candidates/batch2](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/candidates/batch2)
- 审核结果
  - [rag/reviews/batch1](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/reviews/batch1)
  - [rag/reviews/batch2](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/reviews/batch2)
  - [rag/reviews/merged](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/reviews/merged)
- 正式数据集
  - [rag/datasets/pilot](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/datasets/pilot)
  - [rag/datasets/expanded](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/datasets/expanded)
- split
  - [rag/splits/pilot](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/splits/pilot)
  - [rag/splits/expanded](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/splits/expanded)

### 2.2 Experiment Results

- retrieval JSON
  - [results/retrieval/pilot](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/retrieval/pilot)
  - [results/retrieval/expanded](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/retrieval/expanded)
- confidence JSON
  - [results/confidence/baseline](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/confidence/baseline)
  - [results/confidence/tuning](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/confidence/tuning)
- 诊断与建索引
  - [results/diagnostics](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/diagnostics)
  - [results/indexing](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/indexing)
- 论文表格 CSV
  - [results/thesis_tables/retrieval](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/thesis_tables/retrieval)
  - [results/thesis_tables/confidence](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/thesis_tables/confidence)

### 2.3 Reports

- 总入口  
  [aiops-docs/experiment/README.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/README.md)
- 当前阶段报告  
  [expanded_experiment_stage_report.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/reports/expanded_experiment_stage_report.md)
- 后续任务列表  
  [next_steps.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/reports/next_steps.md)

---

## 3. Dataset Lineage

当前数据集不是一次性生成的，而是按以下脉络演化出来：

1. 源文档清点  
   `source_manifest.json`
2. PDF -> chunk  
   `experiment_chunks.jsonl`
3. chunk -> annotation pool  
   `experiment_annotation_pool.jsonl`
4. annotation pool -> candidate questions  
   batch1 `80` 条，batch2 `160` 条
5. 人工审核导回  
   batch1 reviewed usable `64` 条，batch2 reviewed usable `74` 条
6. 合并 reviewed candidates  
   merged usable `138` 条
7. reviewed candidates -> expanded RAG dataset  
   `138` 条
8. validate  
   `valid=138`, `invalid=0`
9. split  
   `build=30`, `dev=35`, `test=60`, `reserve=13`

如果新的 agent 需要继续扩样，应该从：

- [rag/reviews/merged/rag_candidate_questions.merged.reviewed.jsonl](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/reviews/merged/rag_candidate_questions.merged.reviewed.jsonl)
- [rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl)

继续往下推进，而不是回退到 batch1 单独资产。

---

## 4. Current Dataset Facts

### 4.1 Evidence Base

- source 文档数：`5`
- total chunks：`4503`
- 当前统一 evidence base：
  [experiment_chunks.jsonl](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/chunks/experiment_chunks.jsonl)

各 source chunk 数量：

- ABB: `504`
- Grundfos: `265`
- Haas: `124`
- Rockwell: `971`
- Siemens S7-1200: `2639`

### 4.2 Expanded Formal Dataset

- total validated samples: `138`
- normal answerable: `121`
- `abstention_insufficient_evidence`: `17`
- `cross_doc_multi`: `10`

question_type 分布：

- `parameter_or_fault_code`: `37`
- `troubleshooting_step`: `30`
- `safety_or_constraint`: `27`
- `abstention_insufficient_evidence`: `17`
- `symptom_cause`: `13`
- `cross_doc_multi`: `10`
- `definition_or_component_lookup`: `4`

### 4.3 Expanded Split

- build: `30`
- dev: `35`
- test: `60`
- reserve: `13`

`should_abstain` 分布：

- build: `false 27 / true 3`
- dev: `false 30 / true 5`
- test: `false 52 / true 8`
- reserve: `false 12 / true 1`

当前 split 有 2 条 leakage warning，但只是页级 overlap，不是 `reference_chunk_id` 跨 split：

- `rockwell ... page_145: build,test`
- `rockwell ... page_244: dev,test`

---

## 5. How This Dataset Evaluates The RAG Module

当前数据集主要用来评估 RAG 模块里的：

- dense retrieval
- current rerank
- retrieval-level confidence

不做的事：

- 不在这一步生成最终答案质量实验
- 不重新切片
- 不把旧 testset 兼容回实验链路

### 5.1 Retrieval Evaluation Protocol

统一设置：

- dataset: expanded split 中的 build/dev/test
- chunks: `experiment_chunks.jsonl`
- collection: `experiment_manuals_all`
- retrieval strategy: `dense`
- `candidate_top_k = 50`
- `final_top_k = 10`
- `ks = 1,3,5,10`

`should_abstain=true` 的样本默认跳过 retrieval metrics。

### 5.2 Why Candidate vs Final Are Split

评估脚本故意区分：

- `candidate_results`
- `final_results`

因为当前实验已经证明：

- dense retrieval 的 candidate top50 里经常已经有 gold chunk
- 问题主要不是“召不回来”，而是“排不进 final top10”

因此后续所有 rerank / hybrid 实验都应保留这种 candidate-vs-final 分析结构。

---

## 6. Retrieval Results You Can Trust Right Now

当前 expanded retrieval 实验已经跑完 6 组：

- `dense_no_rerank_expanded_build/dev/test`
- `dense_current_rerank_expanded_build/dev/test`

主对比文件：

- [retrieval_experiment_comparison_expanded.json](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded.json)
- [retrieval_experiment_comparison_expanded.csv](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/thesis_tables/retrieval/expanded/retrieval_experiment_comparison_expanded.csv)

### 6.1 No Rerank Baseline

- build
  - `evaluated_samples = 27`
  - `Hit@10 = 0.407407`
  - `MRR = 0.288272`
  - `candidate Hit@50 = 0.740741`
- dev
  - `evaluated_samples = 30`
  - `Hit@10 = 0.433333`
  - `MRR = 0.153651`
  - `candidate Hit@50 = 0.700000`
- test
  - `evaluated_samples = 52`
  - `Hit@10 = 0.442308`
  - `MRR = 0.302564`
  - `candidate Hit@50 = 0.750000`

### 6.2 Current Rerank

- build
  - `Hit@10 = 0.666667`
  - `MRR = 0.545062`
  - `gold_in_candidate_not_final_count: 9 -> 2`
- dev
  - `Hit@10 = 0.466667`
  - `MRR = 0.208889`
  - `gold_in_candidate_not_final_count: 8 -> 7`
- test
  - `Hit@10 = 0.596154`
  - `MRR = 0.429884`
  - `gold_in_candidate_not_final_count: 16 -> 8`

### 6.3 Stable Interpretation

这些结论当前已经足够稳定，可以继续沿用：

- dense baseline 的 candidate recall 不低
- final top10 排序仍是主要瓶颈
- current rerank 在 build/dev/test 都提升了 `Hit@10` 和 `MRR`
- rerank 的主要作用是把 candidate top50 中已有的 gold 推进 final top10

### 6.4 Important Caveat

当前 `current rerank` 不是“纯 Cohere rerank”，而是：

- online rerank
- + local fallback

实验日志中已经出现多次 Cohere `429 Too Many Requests`，但系统自动回退到本地 rerank，没有中断实验。后续 agent 在写报告或做结论时，必须沿用这个表述。

---

## 7. Confidence Status

confidence 暂时不是下一步主线，但新 agent 需要知道当前结论边界。

当前已评估：

- `rank_and_margin`
- `system_top3_support`
- `system_top3_support` tuning

当前可用结论：

- `rank_and_margin` 是更稳的通用 baseline
- `system_top3_support` 更像强低置信度拦截策略
- `system_top3_support` 当前阈值还没有最终冻结
- confidence 参数预计等正式 reviewed samples 扩展到 `150-170+` 后再重新确定

不要现在继续把主要精力投入到 confidence 微调，除非用户明确要求。

---

## 8. Canonical Files New Agent Should Open First

如果新 agent 只看 10 个文件，建议按这个顺序：

1. [aiops-docs/experiment/README.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/README.md)
2. [aiops-docs/experiment/rag/README.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/README.md)
3. [aiops-docs/experiment/results/README.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/README.md)
4. [expanded_experiment_stage_report.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/reports/expanded_experiment_stage_report.md)
5. [next_steps.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/reports/next_steps.md)
6. [experiment_rag_dataset_expanded.validated.jsonl](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl)
7. [rag_split_report.json](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/rag/splits/expanded/rag_split_report.json)
8. [retrieval_experiment_comparison_expanded.json](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded.json)
9. [confidence_results_summary.csv](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/reports/confidence_results_summary.csv)
10. [experiment_findings.json](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/reports/experiment_findings.json)

---

## 9. Safe Next Actions

如果没有新的用户约束，新的 agent 最安全、最自然的延续工作是：

1. 跑 `hybrid_no_rerank`
2. 跑 `hybrid_current_rerank`
3. 生成 hybrid vs dense 对比表
4. 在 expanded build/dev 上复验最终策略
5. 冻结 test，只保留最终策略做终评
6. 最后才回到 confidence 参数最终确定与服务层集成

---

## 10. Unsafe Actions To Avoid

- 不要把新产物再写回旧平铺路径
- 不要覆盖现有 expanded JSON / CSV 结果
- 不要重新索引 Milvus，除非 chunk 或 collection 发生变化
- 不要把 `system_top3_support` 当前 tuning 结论写成最终结论
- 不要把 `current rerank` 写成纯第三方在线 rerank

---

## 11. Handoff Bottom Line

一句话概括当前局面：

当前实验已经从 pilot 走到了可用于正式 retrieval/rerank 对比的 expanded 阶段，主数据集为 `138` 条 validated samples，主结论是“dense 的 candidate recall 还可以，主要瓶颈在 final 排序，而 current rerank 能稳定改善这一点”。新的 agent 不应重复造数据或回退到旧链路，而应在现有 expanded 资产之上继续完成 hybrid 对比、最终策略冻结和工程集成。
