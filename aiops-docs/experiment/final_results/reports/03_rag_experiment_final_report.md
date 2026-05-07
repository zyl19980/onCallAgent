# RAG Experiment Final Report

## 1. 实验资产说明

本报告基于当前已冻结的 expanded RAG 实验资产、现有报告与结果表生成，不重新运行任何实验，不修改任何已有 JSON 结果，不重新生成 dataset、split、chunks 或 Milvus index。

输入报告：

- `aiops-docs/experiment/reports/expanded_experiment_stage_report.md`
- `aiops-docs/experiment/reports/final_retrieval_strategy_report.md`
- `aiops-docs/experiment/reports/query_decomposition_build_dev_report.md`
- `aiops-docs/experiment/reports/multi_query_retrieval_report.md`
- `aiops-docs/experiment/reports/confidence_tuning_build_dev_report.md`
- `aiops-docs/experiment/reports/confidence_final_report.md`

输入结果表：

- `aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded_v2.json`
- `aiops-docs/experiment/results/thesis_tables/retrieval/expanded/retrieval_experiment_comparison_expanded_v2.csv`
- `aiops-docs/experiment/results/retrieval/expanded/multi_query_retrieval_comparison.json`
- `aiops-docs/experiment/results/thesis_tables/retrieval/expanded/multi_query_retrieval_comparison.csv`
- `aiops-docs/experiment/results/confidence/final/confidence_final_dense_current_original_test.json`
- `aiops-docs/experiment/results/thesis_tables/confidence/confidence_final_dense_current_original_test.csv`

固定实验资产：

- expanded dataset: `aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`
- validated samples: `138`
- build/dev/test/reserve split: `aiops-docs/experiment/rag/splits/expanded/`
- split sizes: build `30`, dev `35`, test `60`, reserve `13`
- should_abstain distribution: build `3`, dev `5`, test `8`, reserve `1`
- chunk file: `aiops-docs/experiment/chunks/experiment_chunks.jsonl`
- chunk count: `4503`
- Milvus collection: `experiment_manuals_all`
- retrieval candidate_top_k: `50`
- retrieval final_top_k: `10`
- retrieval ks: `1,3,5,10`
- `should_abstain=true` 样本跳过 retrieval metrics

必须统一使用的 rerank 口径：

`current_rerank = online rerank + local fallback`

不能把 `current_rerank` 写成 pure Cohere rerank。

## 2. 实验阶段

本轮 RAG 实验按以下阶段完成：

| stage | objective | main output |
| --- | --- | --- |
| dataset / split | 构建并冻结 expanded RAG dataset 与 build/dev/test/reserve split | `experiment_rag_dataset_expanded.validated.jsonl`, `rag_build/dev/test/reserve.jsonl` |
| retrieval backbone comparison | 比较 dense / hybrid 与 no_rerank / current_rerank | `retrieval_experiment_comparison_expanded_v2.json/csv` |
| final retrieval strategy selection | 基于 build/dev 选择最终 retrieval backbone，并用 test 做确认 | `final_retrieval_strategy_report.md` |
| query decomposition ablation | 验证 Q0/Q1/Q2 是否缓解 query-side semantic gap | `multi_query_retrieval_report.md` |
| confidence / abstention | 在最终 retrieval/query 配置上选择低置信度策略并做 held-out test confirmation | `confidence_final_report.md` |

最终冻结主线：

- retrieval backbone: `dense_current_rerank`
- query mode: `original`
- confidence strategy: `score_threshold`
- confidence thresholds: `high=0.56`, `low=0.52`
- low confidence routing: `predicted low -> abstain / enter low-confidence governance`
- high/medium routing: `high/medium -> answer`

## 3. Retrieval Backbone 结果

### 3.1 四组 Retrieval 实验

已完成四组 expanded retrieval backbone 实验，均覆盖 build/dev/test：

- `dense_no_rerank`
- `dense_current_rerank`
- `hybrid_no_rerank`
- `hybrid_current_rerank`

核心结果如下：

| experiment | split | evaluated_samples | skipped_abstain | candidate Hit@50 | candidate Recall@50 | Hit@10 | MRR | gold_in_candidate_not_final_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_no_rerank | build | 27 | 3 | 0.740741 | 0.703704 | 0.407407 | 0.288272 | 9 |
| dense_current_rerank | build | 27 | 3 | 0.740741 | 0.703704 | 0.666667 | 0.545062 | 2 |
| hybrid_no_rerank | build | 27 | 3 | 0.703704 | 0.685185 | 0.666667 | 0.399691 | 1 |
| hybrid_current_rerank | build | 27 | 3 | 0.703704 | 0.685185 | 0.666667 | 0.501235 | 1 |
| dense_no_rerank | dev | 30 | 5 | 0.700000 | 0.700000 | 0.433333 | 0.153651 | 8 |
| dense_current_rerank | dev | 30 | 5 | 0.700000 | 0.700000 | 0.466667 | 0.208889 | 7 |
| hybrid_no_rerank | dev | 30 | 5 | 0.800000 | 0.800000 | 0.600000 | 0.216111 | 6 |
| hybrid_current_rerank | dev | 30 | 5 | 0.800000 | 0.800000 | 0.633333 | 0.344854 | 5 |
| dense_no_rerank | test | 52 | 8 | 0.750000 | 0.730769 | 0.442308 | 0.302564 | 16 |
| dense_current_rerank | test | 52 | 8 | 0.750000 | 0.730769 | 0.596154 | 0.429884 | 8 |
| hybrid_no_rerank | test | 52 | 8 | 0.653846 | 0.634615 | 0.634615 | 0.395406 | 1 |
| hybrid_current_rerank | test | 52 | 8 | 0.653846 | 0.634615 | 0.634615 | 0.400824 | 1 |

### 3.2 dense candidate recall 是否足够

结论：当前 dense candidate recall 已经足够作为主线 backbone 的候选层。

证据：

- dense candidate Hit@50 在 build/dev/test 上为 `0.740741 / 0.700000 / 0.750000`
- dense candidate Recall@50 在 build/dev/test 上为 `0.703704 / 0.700000 / 0.730769`
- `dense_no_rerank -> dense_current_rerank` 不改变 candidate metrics，但显著改善 final Hit@10 和 MRR

因此，当前主要瓶颈不在 dense top50 的候选覆盖上限，而在 final top10 排序质量。

### 3.3 rerank 是否是稳定收益来源

结论：是。rerank 是本轮 retrieval 实验中最稳定的收益来源。

`dense_no_rerank -> dense_current_rerank`：

| split | Hit@10 delta | MRR delta | gold_in_candidate_not_final |
| --- | ---: | ---: | --- |
| build | +0.259260 | +0.256790 | 9 -> 2 |
| dev | +0.033334 | +0.055238 | 8 -> 7 |
| test | +0.153846 | +0.127320 | 16 -> 8 |

rerank 的核心作用是把已经进入 candidate top50 的 gold evidence 推进 final top10。

### 3.4 hybrid 是否稳定提升 candidate recall

结论：不是。

`dense_no_rerank -> hybrid_no_rerank` 的 candidate Recall@50：

| split | dense | hybrid | conclusion |
| --- | ---: | ---: | --- |
| build | 0.703704 | 0.685185 | 下降 |
| dev | 0.700000 | 0.800000 | 上升 |
| test | 0.730769 | 0.634615 | 下降 |

hybrid 只在 dev 上提升 candidate recall，在 build/test 上下降。因此不能写成 hybrid 稳定提升 candidate recall。

### 3.5 为什么最终选择 dense_current_rerank

最终 retrieval backbone 选择：

`dense_current_rerank`

理由：

- dense candidate recall 已经可用，主问题是 final top10 排序
- current rerank 对 dense 路线的 Hit@10 和 MRR 提升最稳定
- `dense_current_rerank` 在 build/test 上 MRR 高于 `hybrid_current_rerank`
- dense 路线更简单，candidate generation 行为更稳定，便于后续 confidence / abstention

### 3.6 为什么 hybrid_current_rerank 只是备选

`hybrid_current_rerank` 的优势是 dev/test Hit@10 更高：

- dev: `0.466667 -> 0.633333`
- test: `0.596154 -> 0.634615`

但它不是默认主线，原因是：

- candidate recall 不稳定，build/test 低于 dense
- MRR 不稳定，build/test 低于 `dense_current_rerank`
- hybrid 链路更复杂
- current rerank 运行中存在 online rerank fallback，不能视作纯在线 rerank

因此，`hybrid_current_rerank` 适合作为更重视 top10 evidence coverage 的备选路线，而不是当前默认 backbone。

## 4. Query Decomposition 结果

### 4.1 Query Modes

本阶段固定 backbone 为 `dense_current_rerank`，比较三种 query mode：

- Q0: `original`
- Q1: `original_keyword`
- Q2: `original_keyword_expanded`

### 4.2 Build/Dev 结果

| query mode | split | evaluated_samples | skipped_abstain | candidate Hit@50 | Hit@10 | MRR | duplicate_candidate_ratio_avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q0 original | build | 27 | 3 | 0.740741 | 0.666667 | 0.545062 | 0.000000 |
| Q0 original | dev | 30 | 5 | 0.700000 | 0.466667 | 0.201481 | 0.000000 |
| Q1 original_keyword | build | 27 | 3 | 0.777778 | 0.703704 | 0.569136 | 0.220000 |
| Q1 original_keyword | dev | 30 | 5 | 0.700000 | 0.533333 | 0.280040 | 0.214000 |
| Q2 original_keyword_expanded | build | 27 | 3 | 0.777778 | 0.666667 | 0.500000 | 0.434074 |
| Q2 original_keyword_expanded | dev | 30 | 5 | 0.700000 | 0.533333 | 0.332593 | 0.436000 |

### 4.3 为什么 Q1 进入 test

Q1 在 build/dev 阶段显示出最清晰的正向信号：

- build 上 candidate Hit@50、Hit@10、MRR 都高于 Q0
- dev 上 Hit@10 和 MRR 高于 Q0
- keyword_query 在 build/dev 均出现 `keyword_only=1` 的新增 gold hit
- duplicate_candidate_ratio 约 `21%~22%`，仍可接受

因此，Q1 是唯一进入 held-out test 的 query decomposition 候选。

### 4.4 为什么 Q2 没有进入 test

Q2 没有进入 test，原因是：

- build 上 Hit@10 和 MRR 相比 Q1 回退
- dev 上虽然 MRR 继续提升，但 Hit@10 没有继续提升
- duplicate_candidate_ratio 升至约 `43%~44%`
- expanded_query 的独立新增命中极少，build `expanded_only=0`，dev `expanded_only=1`

Q2 当前更像引入了重复候选和噪声，没有证明自己稳定优于 Q1。

### 4.5 Q1 test 结果

| experiment | split | query mode | evaluated_samples | skipped_abstain | candidate Hit@50 | candidate Recall@50 | Hit@10 | MRR | gold_in_candidate_not_final_count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_current_rerank baseline | test | original | 52 | 8 | 0.750000 | 0.730769 | 0.596154 | 0.429884 | 8 |
| Q1 selected | test | original_keyword | 52 | 8 | 0.730769 | 0.711538 | 0.596154 | 0.415140 | 7 |

Q1 test 相比 original baseline：

- candidate Hit@50: `-0.019231`
- candidate Recall@50: `-0.019231`
- Hit@10: `+0.000000`
- MRR: `-0.014744`
- gold_in_candidate_not_final_count: `8 -> 7`

### 4.6 为什么最终 query mode 仍保留 original

Q1 在 build/dev 上有探索价值，但 held-out test 没有复现主要收益：

- Hit@10 只与 baseline 持平
- candidate Hit@50 和 MRR 低于 original baseline
- keyword_query 虽然带来少量新增 gold hit，但规模不足以拉动整体指标

因此，最终 query mode 保留：

`original`

不能宣称 multi-query retrieval 已经验证有效并默认接入。

更准确的结论是：

keyword-side decomposition 在 build/dev 上显示出一定潜力，但 held-out test 没有稳定复现。expanded_query 当前没有证明自己值得进入最终路径。

## 5. Confidence / Abstention 结果

### 5.1 Build/Dev 选择

confidence / abstention 阶段固定：

- retrieval backbone: `dense_current_rerank`
- query mode: `original`
- current_rerank: `online rerank + local fallback`

build/dev 冻结进入 test 的方案：

- strategy: `score_threshold`
- score_direction: `higher_is_better`
- high_threshold: `0.56`
- low_threshold: `0.52`

test 没有参与 threshold 选择，只用于 held-out final confirmation。

### 5.2 Test 最终指标

| metric | value |
| --- | ---: |
| total_samples | 60 |
| answerable_samples | 52 |
| should_abstain_samples | 8 |
| abstention_precision | 1.000000 |
| abstention_recall | 1.000000 |
| false_confident_count | 0 |
| low_confidence_capture_rate | 0.275862 |
| answerable_over_abstention_count | 0 |
| coverage | 0.866667 |
| accuracy_if_answered | 0.596154 |

Confidence bucket distribution:

| confidence | count |
| --- | ---: |
| high | 50 |
| medium | 2 |
| low | 8 |

对 `should_abstain=true` 样本：

- test 中共有 `8` 个 `should_abstain=true`
- `8` 个全部被判为 low confidence / abstain
- false confident abstain sample 为 `0`

对 answerable 样本：

- test 中共有 `52` 个 answerable 样本
- answerable over-abstention 为 `0`
- high/medium answered 样本为 `52`
- answered Hit@10 success 为 `31`
- answered Hit@10 miss 为 `21`

### 5.3 Confidence Caveat

必须明确保留以下 caveat：

- `should_abstain=true` 样本来自 split label 并入
- retrieval JSON 中这些样本对应空 retrieval row
- 因此 `score_threshold` 的效果是在当前 expanded merged 离线评估口径下成立
- 如果未来线上对 abstain query 也实际执行 retrieval，需要重新校准 threshold

这意味着 confidence 结果可以作为离线验证通过的保守 abstention baseline，但不能写成线上验证完成。

### 5.4 Rerank Provider / Fallback

confidence test 中 top1 provider 分布：

| provider | sample_count |
| --- | ---: |
| local | 43 |
| cohere | 9 |
| none | 8 |

final results provider 分布：

| provider | result_count |
| --- | ---: |
| local | 430 |
| cohere | 90 |

`none` 来自并入后的空 abstain rows。

这进一步说明：

`current_rerank = online rerank + local fallback`

不能写成 pure Cohere rerank。

## 6. 最终推荐系统配置

最终推荐配置：

| component | final setting |
| --- | --- |
| retrieval backbone | `dense_current_rerank` |
| retrieval strategy | `dense` |
| rerank | `current` |
| rerank interpretation | `online rerank + local fallback` |
| query mode | `original` |
| confidence strategy | `score_threshold` |
| score_direction | `higher_is_better` |
| high_threshold | `0.56` |
| low_threshold | `0.52` |
| low confidence routing | `predicted low -> abstain / enter low-confidence governance` |
| high/medium routing | `high/medium -> answer` |

推荐接入方式：

- retrieval 默认走 `dense_current_rerank`
- query 默认使用原始用户 query
- confidence score 低于 `0.52` 时进入 abstention 或低置信度治理
- confidence 为 high/medium 时继续回答

该配置是离线实验冻结结果，不应写成线上验证完成。

## 7. 论文可用结论

本研究基于扩展后的工业手册 RAG 数据集，对检索候选生成、重排、多级查询分解与低置信度拒答进行了分阶段评估。实验结果表明，在当前数据与评测口径下，dense retrieval 已能提供相对稳定的 candidate top50 覆盖，而主要性能瓶颈集中在 final top10 排序。引入 current rerank 后，dense 路线在 build/dev/test 上均提升了 Hit@10 或 MRR，并显著减少了 gold evidence 已进入候选集但未进入 final top10 的情况。因此，最终检索主线选择 `dense_current_rerank`。

hybrid retrieval 在部分 split 上提高了 final Hit@10，但并未稳定提升 candidate recall，且在 build/test 上的 MRR 低于 dense current rerank，因此仅作为更重视 top10 evidence coverage 场景下的备选方案。多级 query decomposition 中，`original_keyword` 在 build/dev 上显示出一定潜力，但 held-out test 未复现主要增益；`original_keyword_expanded` 则因重复候选比例较高且收益不稳定，未进入 test。因此最终 query mode 保留 `original`。

在 confidence / abstention 阶段，基于 build/dev 选择的 `score_threshold` 策略在 held-out test 上正确拦截了全部 `should_abstain=true` 样本，且没有对 answerable 样本产生过度 abstention。需要强调的是，`should_abstain=true` 样本在当前 retrieval JSON 中对应空 retrieval row，因此该 confidence 结论仅在当前 expanded merged 离线评估口径下成立，不能视为线上验证完成。所有 current rerank 结果均应表述为 `online rerank + local fallback`，而不是 pure Cohere rerank。

## 8. 局限性

本轮实验仍有以下局限：

1. Dataset size 仍有限。expanded dataset 为 `138` 条样本，其中 test 为 `60` 条，虽然已覆盖多类工业问答场景，但仍不足以支撑非常细粒度的分组结论。
2. `current_rerank` 包含 `online rerank + local fallback`，不是纯在线 rerank。其结果受外部服务可用性、限流与 fallback 策略共同影响。
3. Cohere `429` 导致 fallback 频繁。此前 hybrid current-rerank 日志中合计记录 `96` 次 fallback，confidence final test 中也有 `43` 个 sample 的 top1 provider 为 local。
4. confidence 中 `should_abstain=true` 样本来自 split label 并入，在 retrieval JSON 中对应空 retrieval row。因此 score threshold 对 abstain 的表现不能直接外推到“线上也对 abstain query 执行 retrieval”的场景。
5. Multi-query test 未复现 build/dev 增益。Q1 在 build/dev 上表现较好，但 test 上 Hit@10 持平、MRR 下降，因此不能默认接入 multi-query retrieval。
6. Answer-level generation quality 尚未完整自动评估。本轮主要评估 retrieval、rerank 与 retrieval-level confidence，还没有对最终答案忠实性、完整性、可读性和引用一致性做系统自动评估。
7. Reserve split 尚未用于最终确认。reserve 当前保留为后续扩展或额外验证资产，不参与本轮最终选择。

## 9. 最终结论

当前 RAG 实验主线已冻结为：

- retrieval backbone: `dense_current_rerank`
- query mode: `original`
- confidence strategy: `score_threshold`
- confidence thresholds: `high=0.56`, `low=0.52`
- low confidence routing: `predicted low -> abstain / enter low-confidence governance`
- high/medium routing: `high/medium -> answer`

这是一套经过 expanded build/dev/test 离线实验验证的保守配置。它优先保证 retrieval 主线稳定、query mode 不引入未复现的复杂度，并用低置信度阈值处理 insufficient-evidence 类样本。后续工作应在该配置上继续评估 answer-level generation quality 与线上真实流量下的 confidence calibration。
