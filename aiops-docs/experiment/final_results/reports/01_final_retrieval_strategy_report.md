# Final Retrieval Strategy Report

## 1. 实验资产说明

本报告基于 expanded RAG retrieval 实验的最终 comparison v2 结果生成，不重新运行任何实验，不修改任何已有实验产物。

固定实验资产如下：

- expanded dataset:
  - `aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`
- build/dev/test split:
  - `aiops-docs/experiment/rag/splits/expanded/`
- fixed chunk file:
  - `aiops-docs/experiment/chunks/experiment_chunks.jsonl`
- Milvus collection:
  - `experiment_manuals_all`
- comparison inputs:
  - `aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded_v2.json`
  - `aiops-docs/experiment/results/thesis_tables/retrieval/expanded/retrieval_experiment_comparison_expanded_v2.csv`

评测口径保持不变：

- `candidate_top_k=50`
- `final_top_k=10`
- `ks=1,3,5,10`
- `should_abstain=true` 样本跳过 retrieval metrics
- 保留 `candidate_results` 和 `final_results` 拆分结构

## 2. 已完成实验组

本轮 expanded retrieval 已完成四组实验，均覆盖 `build/dev/test`：

1. `dense_no_rerank`
2. `dense_current_rerank`
3. `hybrid_no_rerank`
4. `hybrid_current_rerank`

其中：

- `dense` 表示 dense retrieval candidate generation
- `hybrid` 表示 hybrid retrieval candidate generation
- `current_rerank` 必须理解为 `online rerank + local fallback`
- `current_rerank` 不能写成 pure Cohere rerank

## 3. 关键指标表

| experiment | split | evaluated_samples | skipped_abstain | candidate Hit@50 | Hit@10 | MRR | gold_in_candidate_not_final_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_no_rerank | build | 27 | 3 | 0.740741 | 0.407407 | 0.288272 | 9 |
| dense_current_rerank | build | 27 | 3 | 0.740741 | 0.666667 | 0.545062 | 2 |
| hybrid_no_rerank | build | 27 | 3 | 0.703704 | 0.666667 | 0.399691 | 1 |
| hybrid_current_rerank | build | 27 | 3 | 0.703704 | 0.666667 | 0.501235 | 1 |
| dense_no_rerank | dev | 30 | 5 | 0.700000 | 0.433333 | 0.153651 | 8 |
| dense_current_rerank | dev | 30 | 5 | 0.700000 | 0.466667 | 0.208889 | 7 |
| hybrid_no_rerank | dev | 30 | 5 | 0.800000 | 0.600000 | 0.216111 | 6 |
| hybrid_current_rerank | dev | 30 | 5 | 0.800000 | 0.633333 | 0.344854 | 5 |
| dense_no_rerank | test | 52 | 8 | 0.750000 | 0.442308 | 0.302564 | 16 |
| dense_current_rerank | test | 52 | 8 | 0.750000 | 0.596154 | 0.429884 | 8 |
| hybrid_no_rerank | test | 52 | 8 | 0.653846 | 0.634615 | 0.395406 | 1 |
| hybrid_current_rerank | test | 52 | 8 | 0.653846 | 0.634615 | 0.400824 | 1 |

补充 candidate recall@50：

| experiment | build | dev | test |
| --- | ---: | ---: | ---: |
| dense_no_rerank | 0.703704 | 0.700000 | 0.730769 |
| dense_current_rerank | 0.703704 | 0.700000 | 0.730769 |
| hybrid_no_rerank | 0.685185 | 0.800000 | 0.634615 |
| hybrid_current_rerank | 0.685185 | 0.800000 | 0.634615 |

## 4. 主要分析

### 4.1 dense candidate recall 是否足够？

结论：整体上足够。

依据：

- dense `candidate Hit@50` 在三组 split 上分别为 `0.740741 / 0.700000 / 0.750000`
- dense `candidate_recall_at_50` 在三组 split 上分别为 `0.703704 / 0.700000 / 0.730769`
- dense no-rerank 到 dense current-rerank 的 candidate metrics 完全不变，但 final `Hit@10 / MRR` 显著上升

这说明当前 dense 路线的主要瓶颈不在 candidate top50 上界，而在 final top10 排序。

### 4.2 rerank 是否改善 final top10 排序？

结论：是，且是当前最稳定的提升来源。

`dense_no_rerank -> dense_current_rerank`：

- build: `Hit@10 0.407407 -> 0.666667`, `MRR 0.288272 -> 0.545062`
- dev: `Hit@10 0.433333 -> 0.466667`, `MRR 0.153651 -> 0.208889`
- test: `Hit@10 0.442308 -> 0.596154`, `MRR 0.302564 -> 0.429884`

`hybrid_no_rerank -> hybrid_current_rerank`：

- build: `Hit@10 0.666667 -> 0.666667`, `MRR 0.399691 -> 0.501235`
- dev: `Hit@10 0.600000 -> 0.633333`, `MRR 0.216111 -> 0.344854`
- test: `Hit@10 0.634615 -> 0.634615`, `MRR 0.395406 -> 0.400824`

同时，dense 路线中 `gold_in_candidate_not_final_count` 从 `9 -> 2`、`8 -> 7`、`16 -> 8`，说明 rerank 的核心价值就是把已经进入 candidate 集的 gold 往 final top10 推进。

### 4.3 hybrid 是否稳定提升 candidate recall？

结论：不是。

`dense_no_rerank -> hybrid_no_rerank` 的 `candidate_recall_at_50`：

- build: `0.703704 -> 0.685185`，下降
- dev: `0.700000 -> 0.800000`，上升
- test: `0.730769 -> 0.634615`，下降

`candidate Hit@50` 的趋势一致：

- build: `0.740741 -> 0.703704`
- dev: `0.700000 -> 0.800000`
- test: `0.750000 -> 0.653846`

因此，当前 hybrid 不能表述为“稳定提升 candidate recall”。它只在 `dev` 明显更强，在 `build/test` 反而更弱。

### 4.4 hybrid_current_rerank 是否全面优于 dense_current_rerank？

结论：不是。

对比 `dense_current_rerank -> hybrid_current_rerank`：

- build:
  - `Hit@10: 0.666667 -> 0.666667`，持平
  - `MRR: 0.545062 -> 0.501235`，下降
- dev:
  - `Hit@10: 0.466667 -> 0.633333`，上升
  - `MRR: 0.208889 -> 0.344854`，上升
- test:
  - `Hit@10: 0.596154 -> 0.634615`，上升
  - `MRR: 0.429884 -> 0.400824`，下降

这意味着 `hybrid_current_rerank` 在 `Hit@10` 上对 `dev/test` 更好，但在 `MRR` 上并不稳定，不能说它全面优于 `dense_current_rerank`。

### 4.5 为什么 hybrid 可能提升 Hit@10 但降低 MRR？

这组结果符合一种常见现象：hybrid 扩大了候选覆盖面，但候选前几名的排序质量不一定更稳定。

在当前实验里，hybrid 的表现更像：

- 对部分 query，能把原本 dense 没召回到 top10 的 gold chunk 拉进 final top10，所以 `Hit@10` 上升
- 但 gold chunk 进入 top10 后，未必能稳定落在更靠前的位置，所以 `MRR` 可能不升反降

换句话说：

- `Hit@10` 更关注“有没有进 top10”
- `MRR` 更关注“第一个相关证据排在多前面”

当前 hybrid 的收益更偏向前者，而不是后者。

### 4.6 为什么 current_rerank 不能写成 pure Cohere rerank？

因为当前系统真实跑出来的路径不是纯在线 Cohere 重排，而是 `online rerank + local fallback`。

需要明确记录的事实：

- `hybrid_current_rerank` 日志中出现了 Cohere `429 Too Many Requests`
- 三个 `hybrid_current_rerank` 日志合计记录了 `96` 次 fallback
- 结果文件里也能看到同时存在 `rerank_provider=cohere` 和 `rerank_provider=local`

因此，报告里应写成：

`current_rerank = online rerank + local fallback`

而不是：

`current_rerank = pure Cohere rerank`

这不仅是表述准确性问题，也直接影响实验可复现性和线上解释。

## 5. 最终策略建议

策略选择原则：

- 以 `build/dev` 为主做选择
- `test` 只用于最终确认，不用于继续调参

### 5.1 主推荐：dense_current_rerank

本轮建议将 `dense_current_rerank` 作为最终 retrieval backbone。

理由如下：

1. 它是更稳健、更简单的强基线。
2. dense candidate recall 已经不低，当前主要问题是 final 排序，rerank 可以稳定解决这部分问题。
3. 在 `build` 上，`dense_current_rerank` 与 `hybrid_current_rerank` 的 `Hit@10` 持平，但 `MRR` 更高。
4. 在 `test` 上，`dense_current_rerank` 的 `MRR` 也高于 `hybrid_current_rerank`。
5. 相比 hybrid，dense 路线更简单，候选生成行为更稳定，也更便于后续继续叠加 query decomposition。

需要同时明确的代价：

- `dev/test` 上的 `Hit@10` 不如 `hybrid_current_rerank`
- 如果系统目标更偏向“只要 final top10 里覆盖到正确 evidence 即可”，那么 hybrid current 仍然是值得保留的备选路线

### 5.2 备选：hybrid_current_rerank

`hybrid_current_rerank` 不建议作为当前默认 backbone，但可以作为强调 top10 evidence coverage 的备选方案。

适用场景：

- 更重视 `Hit@10`
- 更关心 final top10 里是否尽量包含正确证据
- 愿意接受更复杂的候选生成链路

它的优势：

- `dev/test` 的 `Hit@10` 高于 `dense_current_rerank`
- `dev` 的 `MRR` 也明显更好

它的风险：

- `MRR` 不稳定，`build/test` 都低于 `dense_current_rerank`
- candidate recall 并不稳定提升
- `current_rerank` 依赖 `online rerank + local fallback`
- `hybrid_current_rerank` 日志中出现 Cohere `429`，三组日志合计有 `96` 次 fallback，说明线上质量会受到外部服务可用性影响

### 5.3 最终结论

当前最稳妥的最终 retrieval backbone 建议为：

`dense_current_rerank`

当前最值得保留的备选实验路线为：

`hybrid_current_rerank`

其中：

- 默认推荐 `dense_current_rerank`，因为它更稳、更简单，且在 `build/test` 的 `MRR` 更好
- 如果后续系统目标明确转向 top10 evidence coverage 优先，可以重新评估是否切到 `hybrid_current_rerank`

## 6. 后续实验建议

下一阶段建议不要继续在 dense vs hybrid 上反复微调，而是在最终 backbone 上做 query decomposition。

推荐顺序：

1. 固定 final backbone：
   - 主线建议：`dense_current_rerank`
2. 在该 backbone 上做 multi-level query decomposition：
   - `Q0`: original
   - `Q1`: original + keyword
   - `Q2`: original + keyword + expanded
3. 对 `Q0/Q1/Q2` 在 build/dev 上比较 candidate recall、Hit@10、MRR
4. 冻结 query mode 后，再做 confidence / abstention

建议原因：

- 当前 dense candidate recall 已经达到可用水平
- rerank 已证明是稳定收益项
- 下一步最可能带来新收益的变量，是 query formulation，而不是继续纠结同一层级的 candidate strategy

## 7. 报告口径建议

对外汇报时建议统一使用下面的口径：

- expanded dataset、expanded split、`experiment_chunks.jsonl`、Milvus collection `experiment_manuals_all` 已冻结
- 当前 retrieval 主结论是：dense candidate recall 基本够用，主要收益来自 rerank
- `current_rerank` 应写成 `online rerank + local fallback`
- hybrid 不是稳定提升 candidate recall 的结论
- `hybrid_current_rerank` 在 `Hit@10` 上有吸引力，但还不是全面优于 dense current 的默认方案
