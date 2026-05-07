# Confidence Tuning Build/Dev Report

## 1. 实验范围

本阶段只做 build/dev tuning，不运行 test，不重跑 retrieval，不修改 dataset、split、chunks 或 Milvus index。

冻结 retrieval 主线：

- retrieval backbone: `dense_current_rerank`
- query mode: `original`
- `current_rerank = online rerank + local fallback`

输入文件：

- build retrieval: `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_build.json`
- dev retrieval: `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_dev.json`
- build split labels: `aiops-docs/experiment/rag/splits/expanded/rag_build.jsonl`
- dev split labels: `aiops-docs/experiment/rag/splits/expanded/rag_dev.jsonl`

输出文件：

- JSON: `aiops-docs/experiment/results/confidence/tuning/confidence_tuning_dense_current_original_build_dev.json`
- CSV: `aiops-docs/experiment/results/thesis_tables/confidence/confidence_tuning_dense_current_original_build_dev.csv`

## 2. should_abstain 并入方式

当前 expanded retrieval JSON 没有包含 `should_abstain=true` 样本，因为 retrieval evaluation 会跳过这些样本的 retrieval metrics。

本轮 tuning 的做法是：

1. 从 expanded split JSONL 读取完整 build/dev 样本。
2. 用 `sample_id` 对齐 retrieval `per_sample`。
3. 对于 retrieval JSON 中不存在、但 split 中 `should_abstain=true` 的样本，补成空 retrieval row。

这意味着本轮 confidence / abstention tuning 的离线口径是：

- answerable 样本：使用真实 retrieval 结果
- abstain 样本：使用空 retrieval row，并保留 `should_abstain=true`

这个口径满足当前 build/dev tuning 需要，但也带来一个重要限制：

- 某些策略会部分利用“空 retrieval row”本身来识别 abstain
- 因此 build/dev 最优阈值不能被误写成已经完全验证过的在线最终结论
- test 只应该做最终确认，且如果未来对 abstain 查询也实际执行 retrieval，则 threshold 需要重新校准

## 3. Confidence Feature 来源

本轮复用了 [evaluate_retrieval_confidence.py](/root/workspace/python_code/super_biz_agent_py-release-2026-03-21/scripts/experiment/evaluate_retrieval_confidence.py) 并做了最小扩展，抽取的关键 feature 包括：

- `sample_id`
- `split`
- `should_abstain`
- `top1_score`
- `top1_rerank_score`
- `top2_score`
- `top1_top2_margin`
- `first_relevant_rank`
- `hit_at_10`
- `candidate_hit_at_50`
- `gold_in_candidate_not_final`
- `top3_support_features`
- `rerank_provider_top1`
- `rerank_provider_counts`
- `rerank_used_local_fallback`

本轮 summary 指标包括：

- `abstention_precision`
- `abstention_recall`
- `false_confident_count`
- `low_confidence_capture_rate`
- `answerable_over_abstention_count`
- `coverage`
- `accuracy_if_answered`

这里的默认 abstention 动作定义为：

- `predicted_confidence == low` 时 abstain
- `predicted_confidence in {high, medium}` 时继续回答

## 4. Threshold 搜索范围

### 4.1 rank_and_margin

- `score_direction`: `higher_is_better`, `lower_is_better`
- `high_threshold`: `0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.70`
- `low_threshold`: `0.52, 0.54, 0.56, 0.58, 0.60`
- `margin_threshold`: `0.003, 0.005, 0.010, 0.015, 0.020, 0.030`

### 4.2 system_top3_support

- `score_direction`: `higher_is_better`, `lower_is_better`
- `strong_threshold`: `0.58, 0.62, 0.66, 0.70, 0.74, 0.78`
- `support_threshold`: `0.45, 0.50, 0.55, 0.60, 0.65`
- `high_avg_threshold`: `0.55, 0.58, 0.61, 0.64, 0.67`

### 4.3 附加输出策略

脚本也同时扫了：

- `score_threshold`
- `score_margin`

但本报告的主要比较仍放在：

- `rank_and_margin`
- `system_top3_support`

## 5. Build/Dev 样本规模

| split | total merged samples | answerable_samples | should_abstain_samples |
| --- | ---: | ---: | ---: |
| build | 30 | 27 | 3 |
| dev | 35 | 30 | 5 |

## 6. Build/Dev 最佳配置对比

### 6.1 主比较策略

| strategy | best global config | build abstention P/R | build over-abstain | build coverage | build accuracy_if_answered | dev abstention P/R | dev over-abstain | dev coverage | dev accuracy_if_answered |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `rank_and_margin` | `higher_is_better`, `high=0.56`, `low=0.52`, `margin=0.003` | `0.500 / 1.000` | 3 | 0.800000 | 0.625000 | `0.833333 / 1.000` | 1 | 0.828571 | 0.448276 |
| `system_top3_support` | `higher_is_better`, `strong=0.58`, `support=0.45`, `high_avg=0.55` | `0.300 / 1.000` | 7 | 0.666667 | 0.700000 | `0.312500 / 1.000` | 11 | 0.542857 | 0.421053 |

### 6.2 附加策略

| strategy | best global config | build abstention P/R | build over-abstain | build coverage | dev abstention P/R | dev over-abstain | dev coverage |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| `score_threshold` | `higher_is_better`, `high=0.56`, `low=0.52` | `1.000 / 1.000` | 0 | 0.900000 | `1.000 / 1.000` | 0 | 0.857143 |
| `score_margin` | `higher_is_better`, `margin=0.003` | `0.500 / 1.000` | 3 | 0.800000 | `0.833333 / 1.000` | 1 | 0.828571 |

## 7. 主要观察

### 7.1 rank_and_margin 是主比较策略里更稳的方案

`rank_and_margin` 在 build/dev 上给出了同一组 best threshold：

- `score_direction = higher_is_better`
- `high_threshold = 0.56`
- `low_threshold = 0.52`
- `margin_threshold = 0.003`

它相对 `system_top3_support` 的优势是：

- build/dev 都更高的 abstention precision
- 更少的 `answerable_over_abstention_count`
- 更高的 coverage
- 不存在 build/dev 方向相反的问题

### 7.2 system_top3_support 存在明显不稳定性

`system_top3_support` 的 split-level best config 不一致：

- build best: `higher_is_better`, `strong=0.58`, `support=0.45`, `high_avg=0.55`
- dev best: `lower_is_better`, `strong=0.66`, `support=0.65`, `high_avg=0.55`

这说明它对当前 expanded build/dev 的 score 分布仍然敏感，而且在 build/dev 上都出现较重的 over-abstention：

- build `answerable_over_abstention_count = 7`
- dev `answerable_over_abstention_count = 11`

因此它不适合作为当前最稳健的主推 threshold family。

### 7.3 score_threshold 在当前 merged 离线口径下是全局最优

按当前 build/dev merged setting，最优全局配置是：

- `strategy = score_threshold`
- `score_direction = higher_is_better`
- `high_threshold = 0.56`
- `low_threshold = 0.52`

它在 build/dev 上表现为：

- abstention precision = `1.0 / 1.0`
- abstention recall = `1.0 / 1.0`
- `false_confident_count = 0 / 0`
- `answerable_over_abstention_count = 0 / 0`
- coverage = `0.900000 / 0.857143`

从离线指标看，这一配置显著优于 `rank_and_margin` 和 `system_top3_support`。

### 7.4 但 score_threshold 的优势带有“空 retrieval row”偏差

必须明确说明：

- 本轮 `should_abstain=true` 样本是从 split 标签重新并入的
- 这些样本在 retrieval JSON 中原本被跳过，因此对应的是空 retrieval row
- `score_threshold(low=0.52)` 很容易把这些空 row 识别为 low confidence

所以这组阈值的 build/dev 优势，部分来自“空 retrieval row vs 非空 retrieval row”的结构性差异，而不仅仅是更强的 confidence calibration。

因此更准确的结论是：

- 在当前 expanded build/dev 的 merged 离线口径下，`score_threshold(0.56/0.52)` 是最优候选
- 但它的优势需要在 held-out test 上谨慎确认
- 如果未来对 `should_abstain` 查询也实际执行 retrieval，而不是跳过，则必须重新调 threshold

### 7.5 low_confidence_capture_rate 说明当前 feature 对 answerable retrieval miss 的拦截仍弱

推荐配置 `score_threshold` 的 `low_confidence_capture_rate` 为：

- build: `0.250000`
- dev: `0.238095`

这说明当前 low-confidence 主要在拦截 `should_abstain` 样本，并没有稳定覆盖 answerable 但 retrieval fail 的样本。

也就是说，当前 confidence / abstention 结果的主要收益是：

- 避免对明显的 insufficient-evidence 空样本过度回答

而不是：

- 稳定识别所有 retrieval miss

## 8. Rerank Provider / Fallback 观察

本轮 build/dev tuning 依赖的 retrieval 结果来自：

- `dense_current_rerank_expanded_build`
- `dense_current_rerank_expanded_dev`

因此必须保持如下表述：

`current_rerank = online rerank + local fallback`

不能写成 pure Cohere rerank。

在推荐配置下，sample-level fallback 统计为：

| split | rerank_fallback_sample_count | top1 provider local | top1 provider cohere | top1 provider none |
| --- | ---: | ---: | ---: | ---: |
| build | 17 | 17 | 10 | 3 |
| dev | 29 | 29 | 1 | 5 |

其中 `none` 对应的是并入后的空 abstain rows。

## 9. 推荐进入 Test 的方案

### 9.1 推荐方案

当前 build/dev tuning 推荐进入 test 的全局方案是：

- `strategy = score_threshold`
- `score_direction = higher_is_better`
- `high_threshold = 0.56`
- `low_threshold = 0.52`

推荐理由：

1. build/dev 一致，不存在 split 间阈值漂移。
2. 在当前 merged 口径下，`abstention_precision` 和 `abstention_recall` 都达到 `1.0`。
3. `false_confident_count = 0`，`answerable_over_abstention_count = 0`。
4. coverage 仍维持在 `0.857143+`，没有出现大规模过度 abstain。

### 9.2 为什么不选 rank_and_margin

`rank_and_margin` 是主比较策略里更稳的方案，但它仍然有明显 over-abstention：

- build over-abstain = `3`
- dev over-abstain = `1`

而且它对 answerable retrieval miss 的拦截收益并没有明显超过 `score_threshold`。

### 9.3 为什么不选 system_top3_support

不选择 `system_top3_support` 的原因有两点：

1. split-level best config 不稳定，build/dev 的方向与阈值都不一致。
2. global best config 仍然过度 abstain，build/dev 的 over-abstention 分别达到 `7` 和 `11`。

## 10. 结论

本轮 build/dev tuning 的正式结论是：

1. 旧 pilot confidence threshold 不可直接复用，expanded 主线需要重新标定。
2. 在当前 `dense_current_rerank + original` 的 expanded build/dev merged setting 下，推荐进入 test 的方案是：
   - `score_threshold`
   - `higher_is_better`
   - `high_threshold = 0.56`
   - `low_threshold = 0.52`
3. 如果只在既有主比较策略里二选一，则 `rank_and_margin` 明显优于 `system_top3_support`。
4. `system_top3_support` 当前不够稳定，不建议作为 test 主方案。
5. test 未参与 threshold 选择。后续 test 只用于最终确认，不用于继续调参。

需要保留的 caveat：

- 本轮 `score_threshold` 的优势，部分依赖于 `should_abstain` 样本在 retrieval JSON 中是空 row 的事实
- 因此 test 结论必须写成“在当前 expanded merged 离线评估口径下成立”
- 如果后续把 abstain 查询也送入真实 retrieval，再做 confidence / abstention，就应重新调 threshold
