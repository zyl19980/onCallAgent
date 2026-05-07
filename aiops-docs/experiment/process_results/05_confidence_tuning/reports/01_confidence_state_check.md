# Confidence State Check

## 检查范围

本检查只读取当前 confidence 相关脚本、结果目录与 retrieval 结果，不运行 confidence 实验，不修改任何已有结果，也不改 dataset、split、chunks 或 Milvus 索引。

主要检查对象：

- `scripts/experiment/evaluate_retrieval_confidence.py`
- `aiops-docs/experiment/results/confidence/baseline/`
- `aiops-docs/experiment/results/confidence/tuning/`
- `aiops-docs/experiment/results/thesis_tables/confidence/`
- `aiops-docs/experiment/reports/confidence_results_summary.csv`
- 当前冻结 retrieval 主线结果：
  - `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_build.json`
  - `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_dev.json`
  - `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_test.json`

## 1. 当前已有 confidence baseline 有哪些？

### 1.1 当前脚本支持的 confidence strategies

`scripts/experiment/evaluate_retrieval_confidence.py` 当前支持 4 种策略：

1. `score_threshold`
2. `score_margin`
3. `rank_and_margin`
4. `system_top3_support`

其中：

- `rank_and_margin` 使用 `top1_score + top1/top2 margin`
- `system_top3_support` 使用：
  - top1 是否强相关
  - top2/top3 是否为 support candidate
  - top3 平均分

### 1.2 当前仓库里已有落盘 baseline 结果

当前实际存在的 baseline JSON 只有 4 个：

1. `confidence_eval_dense_current_rerank_build.json`
   - strategy=`rank_and_margin`
2. `confidence_eval_dense_current_rerank_dev.json`
   - strategy=`rank_and_margin`
3. `confidence_eval_system_top3_support_dense_current_rerank_build.json`
   - strategy=`system_top3_support`
4. `confidence_eval_system_top3_support_dense_current_rerank_dev.json`
   - strategy=`system_top3_support`

因此，当前“已有 baseline”可以明确列为：

- `rank_and_margin`
- `system_top3_support`

### 1.3 其他策略状态

`score_threshold` 和 `score_margin` 在脚本里是支持的，但当前结果目录中没有对应 JSON。

`confidence_results_summary.csv` 里也把它们标成了 `missing_file`。

因此当前状态应表述为：

- 脚本支持 `score_threshold / score_margin`
- 但仓库中没有现成 baseline 结果可复用

## 2. 当前已有 confidence tuning 结果有哪些？

当前 tuning 结果只看到一类：

- `system_top3_support`

具体文件：

1. `aiops-docs/experiment/results/confidence/tuning/confidence_tuning_system_top3_support_build.json`
2. `aiops-docs/experiment/results/confidence/tuning/confidence_tuning_system_top3_support_dev.json`

对应 CSV：

1. `aiops-docs/experiment/results/thesis_tables/confidence/tuning/confidence_tuning_system_top3_support_build.csv`
2. `aiops-docs/experiment/results/thesis_tables/confidence/tuning/confidence_tuning_system_top3_support_dev.csv`

当前 best config 为：

- build:
  - `score_direction=lower_is_better`
  - `strong_threshold=0.68`
  - `support_threshold=0.57`
  - `high_avg_threshold=0.6`
- dev:
  - `score_direction=higher_is_better`
  - `strong_threshold=0.6`
  - `support_threshold=0.27`
  - `high_avg_threshold=0.67`

这两个 best config 本身就不一致，说明旧 tuning 结果并没有收敛成一个稳定、可直接冻结的全局阈值方案。

## 3. 旧结果是基于什么 retrieval 配置？

### 3.1 旧 confidence 结果绑定的 retrieval 文件

旧 baseline/tuning JSON 的 `results` 字段明确指向：

- `aiops-docs/experiment/results/dense_current_rerank_build.json`
- `aiops-docs/experiment/results/dense_current_rerank_dev.json`

这不是当前 expanded 主线目录，而是旧的 pilot 结果路径。

### 3.2 样本规模也说明它们不是 expanded 结果

旧 confidence baseline 的 `evaluated_samples` 为：

- build=`30`
- dev=`20`

这与 pilot 阶段报告完全一致：

- `build=30`
- `dev=20`
- `test=0`

而当前 expanded 主线是：

- build `evaluated_samples=27`
- dev `evaluated_samples=30`
- test `evaluated_samples=52`

因此旧 confidence 结果不是基于 expanded dataset。

### 3.3 是否基于 dense_current_rerank + original

从 retrieval 文件名看，旧结果确实基于：

- `dense_current_rerank`

但它们属于旧 pilot 阶段，默认 query mode 也是旧单路 original，不是当前 expanded 主线冻结后的完整最终配置。

更准确的表述应是：

- 旧 confidence 结果基于“旧 pilot 阶段的 dense_current_rerank”
- 不是基于“当前 expanded dataset + expanded build/dev/test + 最终冻结 retrieval 主线”

### 3.4 是否和当前最终配置一致

不一致。

不一致点至少有 4 个：

1. 数据集不是 expanded dataset
2. split 不是当前 expanded build/dev/test
3. 没有 test split
4. 没有 `abstention_insufficient_evidence`

所以不能把旧 confidence 结果直接当成当前最终配置下的 confidence 结论。

## 4. 旧 threshold 是否可以直接复用？

结论：不建议直接复用。

理由如下：

### 4.1 旧结果绑定的是 pilot retrieval 结果，不是 expanded 主线

旧 threshold 是在：

- 30 build
- 20 dev

的小样本 pilot 数据上选出来的。

当前正式主线已经切换到 expanded：

- build 27
- dev 30
- test 52

数据分布、错误分布、question type 组成都已经变化。

### 4.2 旧实验本身明确写了“阈值尚未最终确定”

`experiment_progress_summary.md` 和 `experiment_findings.json` 已经明确记录：

- `system_top3_support thresholds are not final and should be re-tuned after dataset expansion`
- 当前没有 `abstention_insufficient_evidence`
- 应在 build/dev 上重新确定 confidence 参数

### 4.3 旧 tuning 的 build/dev best config 互相不稳定

旧 tuning 结果中：

- build 推荐 `lower_is_better`
- dev 推荐 `higher_is_better`

这已经说明旧阈值配置对数据分布相当敏感，不适合直接冻结复用。

### 4.4 当前最终 retrieval 配置包含 online rerank + local fallback

当前 retrieval 主线是：

- `dense_current_rerank`
- `current_rerank = online rerank + local fallback`

而 rerank provider 和 fallback 行为会影响最终 score 分布，因此旧 score threshold 更不应跨阶段硬复用。

### 4.5 结论

旧 threshold 不应直接复用。

更合理的做法是：

- 重新在当前 expanded build/dev 上选择 threshold
- test 只用于最终确认

## 5. 当前 confidence 最合理的输入文件应该是什么？

当前最终 retrieval 主线已经冻结为：

- retrieval backbone: `dense_current_rerank`
- query mode: `original`

因此当前 confidence 最合理的输入文件就是：

1. build:
   - `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_build.json`
2. dev:
   - `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_dev.json`
3. test:
   - `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_test.json`

这三个文件与当前最终配置一致，优先级高于所有 multi-query 或 hybrid 结果。

不建议把下面这些文件作为 confidence 主线输入：

- `mq1_original_keyword_*`
  - 因为 multi-query 最终没有进入冻结主线
- `hybrid_*`
  - 因为 hybrid 不是最终冻结 backbone

## 6. 是否需要重新生成 confidence feature？

结论：需要重新抽取当前 expanded 主线上的 confidence feature，但不需要重跑 retrieval。

更准确地说，是：

- 不需要重新生成 retrieval result
- 需要基于当前 expanded retrieval JSON 重新生成 confidence evaluation 所需 feature

### 6.1 当前 retrieval JSON 已经具备的 feature

`dense_current_rerank_expanded_build/dev/test.json` 的 `per_sample` 已经包含：

- `final_results[*].rank`
- `final_results[*].score`
- `final_results[*].rerank_score`
- `final_results[*].rerank_provider`
- `final_results[*].original_rank`
- `candidate_hit_at_k`
- `candidate_recall_at_k`
- `hit_at_k`
- `recall_at_k`
- `first_relevant_rank`
- `mrr`
- `gold_in_candidate_not_final`

因此下面这些 feature 可以直接从 retrieval JSON 里重新抽取：

1. `rerank score`
2. `rank`
3. `margin`
4. `top-k support`
5. `candidate/final hit 信息`

### 6.2 当前缺的关键标签：should_abstain

当前 `evaluate_rag_retrieval.py` 会直接跳过：

- `should_abstain=true`

所以 retrieval result JSON 的 `per_sample` 里只保留了非 abstain 样本。

这意味着：

- 如果下一阶段目标只是“retrieval-success confidence”，现有 per-sample feature 基本够用
- 如果目标是“confidence / abstention”联合策略，就必须把 `should_abstain` 标签重新引入 feature pipeline

因此：

- 对 `should_abstain` 标签，当前需要重新从 expanded build/dev/test dataset 或 split 文件中对齐补回

### 6.3 结论

需要重新生成的不是 retrieval 本身，而是：

- 面向当前 expanded 主线的 confidence feature 表

至少应包含：

1. `rerank_score`
2. `rank`
3. `score_margin`
4. `top-k support`
5. `candidate/final hit`
6. `should_abstain`

## 7. 推荐下一步如何执行

推荐执行顺序如下：

### 7.1 第一步：固定输入

固定当前最终 retrieval 主线输入：

- build:
  - `dense_current_rerank_expanded_build.json`
- dev:
  - `dense_current_rerank_expanded_dev.json`
- test:
  - `dense_current_rerank_expanded_test.json`

并明确：

- `current_rerank = online rerank + local fallback`

### 7.2 第二步：在 build/dev 上重新抽 feature 并选 threshold

在 build/dev 上重新做：

1. confidence feature 抽取
2. baseline strategy 复评：
   - `rank_and_margin`
   - `system_top3_support`
   - 如有必要再补 `score_threshold / score_margin`
3. threshold 选择

这一步应该以 build/dev 为准，不动 test。

### 7.3 第三步：把 abstention 标签纳入评估

由于当前 expanded dataset 已经有：

- `should_abstain=true`

下一轮 confidence 实验不应只停留在 retrieval-success confidence，而应面向：

- confidence + abstention

也就是说，要显式评估：

1. high confidence precision
2. low confidence error capture
3. abstention interception / abstention quality
4. overall confidence accuracy

### 7.4 第四步：冻结 threshold 后只在 test 上确认

推荐纪律：

1. 在 build/dev 上选 strategy 和 threshold
2. 冻结后只跑一次 test
3. test 只做最终确认，不再反向调参

## 8. 最终建议

当前 confidence 状态的保守结论如下：

1. 仓库中已有的 confidence baseline 主要是：
   - `rank_and_margin`
   - `system_top3_support`
2. 已有 tuning 只覆盖：
   - `system_top3_support`
3. 这些旧结果基于旧 pilot 阶段的 `dense_current_rerank`，不是当前 expanded 主线
4. 旧 threshold 不建议直接复用
5. 当前最合理的 confidence 输入文件应当是：
   - `dense_current_rerank_expanded_build.json`
   - `dense_current_rerank_expanded_dev.json`
   - `dense_current_rerank_expanded_test.json`
6. 下一步不需要重跑 retrieval，但需要重新抽取 confidence feature，并把 `should_abstain` 标签纳入 build/dev threshold 选择流程
