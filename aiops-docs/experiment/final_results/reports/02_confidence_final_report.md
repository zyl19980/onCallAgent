# Confidence Final Report

## 1. Final Retrieval / Query Configuration

本报告只做 held-out test final confirmation，不重新选择 threshold，不运行其他 confidence strategy。

最终固定配置：

- retrieval backbone: `dense_current_rerank`
- query mode: `original`
- retrieval input: `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_test.json`
- test split labels: `aiops-docs/experiment/rag/splits/expanded/rag_test.jsonl`
- `current_rerank = online rerank + local fallback`

`current_rerank` 不能写成 pure Cohere rerank。

## 2. Frozen Build/Dev Decision

build/dev tuning 已经冻结进入 held-out test 的 confidence 方案：

- strategy: `score_threshold`
- score_direction: `higher_is_better`
- high_threshold: `0.56`
- low_threshold: `0.52`

build/dev 选择依据来自：

- `aiops-docs/experiment/reports/confidence_tuning_build_dev_report.md`
- `aiops-docs/experiment/results/confidence/tuning/confidence_tuning_dense_current_original_build_dev.json`

test 没有参与 threshold 选择。

test 只用于 held-out final confirmation。

## 3. Test Output Files

- JSON: `aiops-docs/experiment/results/confidence/final/confidence_final_dense_current_original_test.json`
- CSV: `aiops-docs/experiment/results/thesis_tables/confidence/confidence_final_dense_current_original_test.csv`

## 4. Test Final Metrics

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

## 5. should_abstain=true 样本表现

test 中共有 `8` 个 `should_abstain=true` 样本。

| outcome | count |
| --- | ---: |
| correctly predicted low confidence / abstain | 8 |
| false confident abstain sample | 0 |

结论：

- `abstention_recall = 1.000000`
- 所有 test abstain 标签样本都被拦截为 low confidence
- 没有 `should_abstain=true` 样本被错误放行为 high/medium confidence

## 6. Answerable 样本表现

test 中共有 `52` 个 answerable 样本。

| outcome | count |
| --- | ---: |
| answerable samples predicted low / over-abstained | 0 |
| answerable samples answered as high/medium | 52 |
| answered samples with Hit@10 success | 31 |
| answered samples with Hit@10 miss | 21 |

answerable 样本没有被过度 abstain，因此：

- `answerable_over_abstention_count = 0`
- `coverage = 52 / 60 = 0.866667`
- `accuracy_if_answered = 31 / 52 = 0.596154`

按 confidence bucket 看：

| confidence | answerable_count | Hit@10 success |
| --- | ---: | ---: |
| high | 50 | 31 |
| medium | 2 | 0 |
| low | 0 | 0 |

需要注意的是，当前 threshold 对 answerable retrieval miss 的拦截仍然有限。`low_confidence_capture_rate = 0.275862`，对应的是：

- low confidence 捕获了 `8` 个 should-abstain 样本
- 但没有捕获 `21` 个 answerable retrieval miss

因此该方案更像 conservative abstention baseline，而不是完整的 retrieval-miss detector。

## 7. Caveat

本轮 final confirmation 必须保留以下 caveat：

1. 当前 `should_abstain=true` 样本是从 split label 并入的。
2. 在 retrieval JSON 中，这些样本对应空 retrieval row，因为 retrieval evaluation 会跳过 `should_abstain=true` 样本的 retrieval metrics。
3. 因此 `score_threshold` 的效果是在当前 expanded merged 离线评估口径下成立。
4. 如果未来线上对 abstain query 也实际执行 retrieval，需要重新校准 threshold。

这个 caveat 不影响本轮 held-out test 的离线结论，但限制了结论的外推范围。

## 8. Rerank Provider / Fallback

test 结果中的 top1 provider 分布：

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

其中 `none` 来自并入后的空 abstain rows。

`rerank_fallback_sample_count = 43`。

因此报告中应继续表述为：

`current_rerank = online rerank + local fallback`

不能写成 pure Cohere rerank。

## 9. Final Recommendation

在当前 expanded merged 离线评估口径下，`score_threshold` 方案通过 held-out test confirmation：

- `should_abstain=true` 样本全部被 low confidence 拦截
- 没有 false confident abstain sample
- 没有 answerable over-abstention
- coverage 保持在 `0.866667`

建议将其作为离线验证通过的保守 abstention baseline，用于后续低置信度治理设计。

该结论不能写成线上验证完成。上线前仍需要在真实线上路径确认：

- abstain query 是否也会执行 retrieval
- online rerank 与 local fallback 的 provider 分布是否稳定
- threshold 是否需要按线上 score 分布重新校准
