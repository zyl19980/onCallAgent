# Multi-Query Retrieval Report

## 1. 实验目的

本轮实验的目标是验证 query-side semantic gap mitigation 是否有效。

更具体地说，本阶段希望回答：

1. 在固定 retrieval backbone 后，仅通过 query decomposition 是否还能带来 retrieval 提升
2. 收益是否主要来自 `keyword_query`
3. `expanded_query` 是否能进一步稳定提升
4. build/dev 上观察到的收益，是否能在 held-out test 上复现

本报告基于以下现有结果生成，不重新运行任何实验：

- [query_decomposition_build_dev_report.md](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/reports/query_decomposition_build_dev_report.md)
- [mq1_original_keyword_dense_current_rerank_expanded_test.json](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/retrieval/expanded/mq1_original_keyword_dense_current_rerank_expanded_test.json)
- [mq1_original_keyword_dense_current_rerank_expanded_test.csv](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/thesis_tables/retrieval/expanded/mq1_original_keyword_dense_current_rerank_expanded_test.csv)
- [dense_current_rerank_expanded_test.json](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_test.json)
- [dense_current_rerank_expanded_test.csv](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/thesis_tables/retrieval/expanded/dense_current_rerank_expanded_test.csv)

配套对比表：

- [multi_query_retrieval_comparison.json](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/retrieval/expanded/multi_query_retrieval_comparison.json)
- [multi_query_retrieval_comparison.csv](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/aiops-docs/experiment/results/thesis_tables/retrieval/expanded/multi_query_retrieval_comparison.csv)

## 2. 固定 Backbone

本轮 multi-query 实验固定 backbone 为：

- `dense_current_rerank`

其中：

- retrieval strategy: `dense`
- rerank: `current`

必须明确说明：

`current_rerank = online rerank + local fallback`

不能把它写成 pure Cohere rerank。

## 3. Query Modes

本轮比较的 query modes 为：

- Q0: `original`
- Q1: `original_keyword`
- Q2: `original_keyword_expanded`

语义如下：

- `original`
  - 只使用原始 query
- `original_keyword`
  - 使用原始 query + keyword query
- `original_keyword_expanded`
  - 使用原始 query + keyword query + expanded query

## 4. Build/Dev 决策过程

### 4.1 为什么 Q1 进入 test

Q1 是基于 build/dev 决策后进入 test 的唯一 query mode。

选择理由：

1. build 上，Q1 相比 Q0 同时提升了 `candidate Hit@50`、`Hit@10`、`MRR`
2. dev 上，Q1 相比 Q0 提升了 `Hit@10` 和 `MRR`
3. `keyword_query` 在 build/dev 都出现了 `keyword_only` gold hit
4. Q1 的 `duplicate_candidate_ratio` 约 `21%~22%`，仍在可接受范围

因此，Q1 是 build/dev 阶段最合理的 test 候选。

### 4.2 为什么 Q2 不进入 test

Q2 没有进入 test，原因如下：

1. build 上，Q2 相比 Q1 的 `Hit@10` 和 `MRR` 明显回退
2. dev 上，Q2 只在 `MRR` 上有提升，但 `Hit@10` 没有继续提升
3. Q2 的 `duplicate_candidate_ratio` 上升到约 `43%~44%`
4. `expanded_query` 的独立命中非常少，`expanded_only` 只在 dev 出现了 `1` 个

因此，Q2 没有证明自己能稳定优于 Q1，更像是引入了额外噪声。

### 4.3 test 没有用于调 query_mode

必须明确：

- query mode 的选择在 build/dev 阶段已经冻结
- test 只用于最终验证被冻结的 Q1
- Q0/Q2 没有进入 test，也没有使用 test 结果重新回调 query mode

## 5. Test 最终结果

### 5.1 Q1 original_keyword test 指标

Q1 test 结果如下：

- `evaluated_samples = 52`
- `skipped_abstain = 8`
- `candidate Hit@50 = 0.730769`
- `candidate Recall@50 = 0.711538`
- `Hit@10 = 0.596154`
- `MRR = 0.415140`
- `gold_in_candidate_not_final_count = 7`

诊断指标：

- `raw_candidate_count avg = 100.000`
- `union_candidate_count avg = 80.307692`
- `duplicate_candidate_ratio avg = 0.196923`

### 5.2 与 dense_current_rerank original baseline test 对比

baseline test 为原始 query 模式下的 `dense_current_rerank_expanded_test`：

- `candidate Hit@50 = 0.750000`
- `candidate Recall@50 = 0.730769`
- `Hit@10 = 0.596154`
- `MRR = 0.429884`
- `gold_in_candidate_not_final_count = 8`

Q1 test 相比 baseline 的 delta：

- `candidate Hit@50: -0.019231`
- `candidate Recall@50: -0.019231`
- `Hit@10: +0.000000`
- `MRR: -0.014744`
- `gold_in_candidate_not_final_count: -1`

结论：

- Q1 在 held-out test 上没有复现 build/dev 的主要收益
- `Hit@10` 与 baseline 持平
- `candidate Hit@50` 和 `MRR` 都低于 baseline
- 只有 `gold_in_candidate_not_final_count` 略有改善

## 6. 诊断分析

### 6.1 source_query_hit

Q1 test 的 `source_query_hit` 分布：

- `main_query hit = 39`
- `keyword_query hit = 27`
- `main_only = 13`
- `keyword_only = 1`
- `main+keyword = 26`
- `none = 12`

这说明：

- `keyword_query` 仍然能补到 `main_query` 没召回的 gold
- 但这个增益规模很小
- 在 held-out test 上，这种补充并没有转化为更好的整体主指标

### 6.2 duplicate_candidate_ratio

Q1 test：

- `duplicate_candidate_ratio avg = 0.196923`
- `duplicate_candidate_ratio max = 0.460000`

这和 build/dev 的 Q1 基本一致，说明两路 union 的重复度是稳定的，不是 test 异常导致的问题。

### 6.3 raw_candidate_count / union_candidate_count

Q1 test：

- `raw_candidate_count avg = 100.000`
- `union_candidate_count avg = 80.307692`

说明：

- 两路召回确实带来了额外候选
- 但额外候选并没有稳定提升 rerank 后的最终指标

### 6.4 keyword_query 是否带来新增 gold hit

结论：带来了，但规模偏小。

直接证据：

- build `keyword_only = 1`
- dev `keyword_only = 1`
- test `keyword_only = 1`

因此，`keyword_query` 的作用不是不存在，而是存在但偏弱，且不足以在 held-out test 上拉动整体 `Hit@10/MRR`。

### 6.5 expanded_query 为什么没有进入 test

原因不是“expanded query 完全没用”，而是：

1. build/dev 上的收益不稳定
2. `expanded_only` gold hit 极少
3. `duplicate_candidate_ratio` 明显偏高
4. Q2 build 出现了明显回退

所以，expanded query 当前没有达到进入 held-out test 的门槛。

## 7. 结论

### 7.1 query decomposition 是否有效

结论：在 build/dev 上部分有效，但 held-out test 没有复现主要收益。

更准确地说：

- build/dev 上，Q1 `original_keyword` 对 Q0 `original` 有正向信号
- 但 test 上，Q1 没有优于 original baseline

因此不能把这轮实验总结为：

- “multi-query retrieval 已经被验证有效并应默认接入”

更诚实的表述是：

- “keyword-side query decomposition 在 build/dev 上显示出一定潜力，但在 held-out test 上没有稳定复现”

### 7.2 收益主要来自 keyword_query 还是 expanded_query

结论：现阶段只有 `keyword_query` 显示出有限收益，`expanded_query` 没有证明自己值得进入最终路径。

具体表现：

- `keyword_query`
  - 在 build/dev/test 都存在 `keyword_only` gold 命中
  - 但总体收益有限
- `expanded_query`
  - build/dev 上独立增益很弱
  - 更容易引入重复候选和噪声

### 7.3 是否建议最终系统接入 original_keyword

结论：当前不建议直接把 `original_keyword` 作为最终系统默认 query mode。

原因：

1. held-out test 上没有优于 original baseline
2. `Hit@10` 只是持平
3. `candidate Hit@50` 和 `MRR` 反而下降

最终系统建议仍然保留：

- final retrieval backbone: `dense_current_rerank`
- final query mode: `original`

### 7.4 如果 Q1 test 没有复现 build/dev 增益，应如何表述

应明确写成：

- Q1 `original_keyword` 在 build/dev 上提供了有价值的探索信号
- 但 held-out test 没有复现该增益
- 因此本轮不建议用它替换最终系统的 `original` query mode

这是当前最诚实、也最符合实验纪律的结论。

### 7.5 rerank 的实验口径

本轮所有结论都建立在：

`current_rerank = online rerank + local fallback`

Q1 test 日志中仍然出现：

- Cohere `429`
- `36` 次 local fallback

因此，任何对外报告都不应把本轮结果写成 pure Cohere rerank 的结论。

## 8. 后续实验

下一步建议不要继续扩展 query decomposition 分支，而是在最终 retrieval backbone + final query mode 上做 confidence / abstention。

当前推荐的最终配置是：

- retrieval backbone: `dense_current_rerank`
- query mode: `original`

下一阶段建议：

1. 固定 `dense_current_rerank + original`
2. 在该配置上做 confidence / abstention
3. 评估 low-confidence routing、abstention precision/recall、以及 fallback handling

如果之后还要继续研究 multi-query 路线，建议作为独立后续课题，而不是当前主线默认方案。
