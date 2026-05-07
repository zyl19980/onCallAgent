# Expanded RAG Experiment Stage Report

## Scope

当前阶段实验基于扩展版数据集与扩展版 split，目标是形成一套可用于论文写作和后续实验扩展的独立 RAG 实验资产。实验约束如下：

- 固定 `aiops-docs/experiment/chunks/experiment_chunks.jsonl` 作为统一 evidence base。
- 当前不重新验证 chunk 切片策略，只在固定 evidence base 上评估检索、重排和检索级置信度。
- 当前检索使用 Milvus 实验 collection `experiment_manuals_all`。
- 当前结果属于 `build / dev / test` 扩展阶段实验，不依赖旧 `context_catalog_v1.jsonl`，也不恢复旧评测系统。

## How The Dataset Was Built

扩展版数据集按“文档整理 -> chunk 构建 -> annotation pool -> 候选题 -> 人工审核 -> 正式 RAG dataset -> validated split”流水线构建：

1. 原始文档整理  
   使用 `inspect_experiment_sources.py` 扫描 `docs/experiment_doc`，生成 `source_manifest.json`，确认 5 份工业手册 PDF 均可直接提取文本，无需 OCR。

2. Chunk 构建  
   使用 `build_experiment_chunks.py` 生成 `experiment_chunks.jsonl`。总 chunk 数为 `4503`。  
   各 source chunk 数量：
   - ABB: `504`
   - Grundfos: `265`
   - Haas: `124`
   - Rockwell: `971`
   - Siemens S7-1200: `2639`

   各 chunk_type 数量：
   - `parameter_and_configuration`: `1746`
   - `safety_and_constraint`: `566`
   - `front_matter`: `468`
   - `troubleshooting_procedure`: `365`
   - `installation_or_wiring`: `313`
   - `alarm_fault_code`: `287`
   - `other`: `669`
   - `maintenance_procedure`: `52`
   - `concept_and_component`: `37`

3. Annotation Pool  
   使用 `build_annotation_pool.py` 从全部 chunk 中筛出适合人工标注和问题生成的候选片段，得到 `728` 条 annotation pool。  
   annotation priority 分布：
   - `high`: `491`
   - `medium`: `218`
   - `low`: `19`

4. 候选题生成与人工审核  
   第一批候选题 `80` 条，第二批候选题 `160` 条。  
   第二批重点补齐：
   - Grundfos 样本
   - `symptom_cause`
   - `safety_or_constraint`
   - `cross_doc_multi`
   - `abstention_insufficient_evidence`

   两批人工审核后，合并 reviewed candidates 共 `138` 条，其中：
   - 普通可回答样本：`121`
   - `abstention_insufficient_evidence`: `17`
   - `cross_doc_multi`: `10`

5. 正式 RAG Dataset 构建与校验  
   使用 `build_rag_dataset_from_candidates.py` 生成扩展版正式数据集，再用 `validate_rag_dataset.py` 校验。  
   结果：
   - `valid_samples = 138`
   - `invalid_samples = 0`

6. Expanded Split  
   使用 `split_rag_dataset.py` 重新切分为：
   - `build = 30`
   - `dev = 35`
   - `test = 60`
   - `reserve = 13`

   `should_abstain` 分布：
   - build: `false 27 / true 3`
   - dev: `false 30 / true 5`
   - test: `false 52 / true 8`
   - reserve: `false 12 / true 1`

## Current Dataset Status

当前扩展版正式数据集已经可以作为后续论文实验的主数据集：

- 正式 validated samples: `138`
- 问题类型分布：
  - `parameter_or_fault_code`: `37`
  - `troubleshooting_step`: `30`
  - `safety_or_constraint`: `27`
  - `abstention_insufficient_evidence`: `17`
  - `symptom_cause`: `13`
  - `cross_doc_multi`: `10`
  - `definition_or_component_lookup`: `4`
- 当前还有 `duplicate_user_input_candidates:19` 的 warning，说明后续冻结最终 test 前仍可继续做小规模表述去重。

## How The Dataset Is Used To Evaluate The RAG Module

当前数据集主要用于评估系统 RAG 模块中的 **retrieval** 与 **rerank**，暂不生成答案，也不重新构造 evidence base。

### 1. Evidence Base

所有检索实验统一使用：

- chunks: `aiops-docs/experiment/chunks/experiment_chunks.jsonl`
- Milvus collection: `experiment_manuals_all`

这意味着变量被收敛到：

- dense retrieval 能否把 gold chunk 召回进 candidate set
- rerank 能否把 gold chunk 从 candidate top50 推进到 final top10

### 2. Retrieval Evaluation

使用 `evaluate_rag_retrieval.py` 对每个 split 运行 live retrieval：

- query: `user_input`
- retrieval strategy: `dense`
- `candidate_top_k = 50`
- `final_top_k = 10`
- `ks = 1,3,5,10`

`should_abstain=true` 的样本默认跳过 retrieval metrics，因此当前 build/dev/test 的 `evaluated_samples` 分别为：

- build: `27`，skipped abstain `3`
- dev: `30`，skipped abstain `5`
- test: `52`，skipped abstain `8`

### 3. Retrieval Metrics

当前主要观察两类指标：

- candidate 指标：`candidate_hit_at_10/20/50`
  用于判断 dense baseline 的召回上界。
- final 指标：`Hit@K / Recall@K / MRR / Evidence Coverage / Source Accuracy / Page Accuracy`
  用于判断最终排序质量。

同时记录：

- `gold_in_candidate_not_final_count`
- `gold_promoted_by_rerank_count`
- `gold_demoted_by_rerank_count`

这些指标用于直接观察 rerank 的主要作用是不是“把已经召回的 gold chunk 推进 final top10”。

## Expanded Retrieval Results

### Dense No Rerank

- build
  - `Hit@10 = 0.407407`
  - `MRR = 0.288272`
  - `candidate Hit@50 = 0.740741`
- dev
  - `Hit@10 = 0.433333`
  - `MRR = 0.153651`
  - `candidate Hit@50 = 0.700000`
- test
  - `Hit@10 = 0.442308`
  - `MRR = 0.302564`
  - `candidate Hit@50 = 0.750000`

结论：dense baseline 在 candidate top50 上已经能召回相当一部分 gold chunk，但 final top10 排序明显不足。

### Dense Current Rerank

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

相对 `dense_no_rerank` 的提升：

- build
  - `delta_hit_at_1 = +0.296297`
  - `delta_hit_at_3 = +0.185186`
  - `delta_hit_at_5 = +0.185186`
  - `delta_hit_at_10 = +0.259260`
  - `delta_mrr = +0.256790`
- dev
  - `delta_hit_at_1 = +0.033333`
  - `delta_hit_at_3 = +0.066667`
  - `delta_hit_at_5 = +0.166666`
  - `delta_hit_at_10 = +0.033334`
  - `delta_mrr = +0.055238`
- test
  - `delta_hit_at_1 = +0.134616`
  - `delta_hit_at_3 = +0.134615`
  - `delta_hit_at_5 = +0.076923`
  - `delta_hit_at_10 = +0.153846`
  - `delta_mrr = +0.127320`

### Retrieval Finding

当前 expanded 数据集上的主要结论是：

- dense candidate set 的召回上界不低，build/dev/test 的 `candidate Hit@50` 分别达到 `0.740741 / 0.700000 / 0.750000`。
- 主要瓶颈仍是 final top10 排序，而不是 candidate recall 本身。
- `current rerank` 在 build/dev/test 三个 split 上都提升了 `Hit@10` 和 `MRR`。
- rerank 的主要作用是把 candidate top50 中已经存在的 gold chunk 继续推进到 top10/top5/top3/top1。

需要特别说明的是：当前 `current rerank` 不是“纯 Cohere rerank”，而是**当前系统策略：在线 rerank + 本地 fallback**。实验日志中已经观测到多次 Cohere `429`，但系统自动回退到本地 rerank，因此实验没有中断。

## Confidence Status

当前 confidence 只做了初步评估，不在本阶段继续调参。已有结论如下：

- `rank_and_margin` 是当前通用 baseline 中更稳的策略。
  - build: `high_precision = 1.0`, `low_capture = 0.571429`
  - dev: `high_precision = 1.0`, `low_capture = 0.666667`
- `system_top3_support` 更像强低置信度拦截策略。
  - build: `high_precision = 0.8`, `low_capture = 0.857143`
  - dev: `high_precision = 1.0`, `low_capture = 1.0`
- 但 `system_top3_support` 当前仍偏保守，`medium` 桶不足，最终阈值将在样本进一步扩展到 `150-170+ reviewed samples` 后重新确定。

## Current Limitations

- 当前正式 validated samples 为 `138`，虽然已明显超过 pilot，但总体规模仍有限。
- `definition_or_component_lookup` 仍偏少，仅 `4` 条。
- `cross_doc_multi` 和 `abstention_insufficient_evidence` 已经纳入，但数量仍偏小。
- expanded split 已有 test，但仍建议在最终 confidence 与 retrieval 策略冻结前谨慎使用 test 结果做结论。
- 当前 rerank 结果受到在线 rerank `429` fallback 影响，应在论文中注明。

## Recommended Next Steps

1. 继续补充 reviewed samples，把正式数据集稳定扩展到 `150-170` 条以上。
2. 对 `definition_or_component_lookup`、`cross_doc_multi`、`abstention_insufficient_evidence` 做定向补样。
3. 在 expanded build/dev 上重新确定最终 confidence 参数。
4. 补跑 `hybrid_no_rerank` 与 `hybrid_current_rerank`，与 dense baseline 做同维度对比。
5. 冻结最终策略后，再使用 expanded test 作为最终论文主结果。
