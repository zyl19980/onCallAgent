# Query Decomposition Build/Dev Report

## 1. 实验设置

本报告只基于 build/dev 结果，不包含 test，不把本阶段结论写成最终 test 结论。

固定实验设置如下：

- fixed backbone:
  - `dense_current_rerank`
- retrieval strategy:
  - `dense`
- rerank:
  - `current`
  - 必须理解为 `online rerank + local fallback`
  - 不能写成 pure Cohere rerank
- query modes:
  - `original`
  - `original_keyword`
  - `original_keyword_expanded`
- fixed chunks:
  - `aiops-docs/experiment/chunks/experiment_chunks.jsonl`
- `candidate_top_k=50`
- `final_top_k=10`
- `ks=1,3,5,10`
- `should_abstain=true` 样本跳过 retrieval metrics

本轮输入结果文件：

- Q0
  - `mq0_original_dense_current_rerank_expanded_build.json`
  - `mq0_original_dense_current_rerank_expanded_dev.json`
- Q1
  - `mq1_original_keyword_dense_current_rerank_expanded_build.json`
  - `mq1_original_keyword_dense_current_rerank_expanded_dev.json`
- Q2
  - `mq2_original_keyword_expanded_dense_current_rerank_expanded_build.json`
  - `mq2_original_keyword_expanded_dense_current_rerank_expanded_dev.json`

## 2. Q0/Q1/Q2 指标表

| query mode | split | evaluated_samples | skipped_abstain | candidate Hit@50 | Hit@10 | MRR | gold_in_candidate_not_final_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q0 original | build | 27 | 3 | 0.740741 | 0.666667 | 0.545062 | 2 |
| Q0 original | dev | 30 | 5 | 0.700000 | 0.466667 | 0.201481 | 7 |
| Q1 original_keyword | build | 27 | 3 | 0.777778 | 0.703704 | 0.569136 | 2 |
| Q1 original_keyword | dev | 30 | 5 | 0.700000 | 0.533333 | 0.280040 | 5 |
| Q2 original_keyword_expanded | build | 27 | 3 | 0.777778 | 0.666667 | 0.500000 | 3 |
| Q2 original_keyword_expanded | dev | 30 | 5 | 0.700000 | 0.533333 | 0.332593 | 5 |

## 3. 多级查询诊断指标

### 3.1 candidate union 统计

| query mode | split | raw_candidate_count avg | union_candidate_count avg | duplicate_candidate_ratio avg |
| --- | --- | ---: | ---: | ---: |
| Q0 | build | 50.000 | 50.000 | 0.000000 |
| Q0 | dev | 50.000 | 50.000 | 0.000000 |
| Q1 | build | 100.000 | 78.000 | 0.220000 |
| Q1 | dev | 100.000 | 78.600 | 0.214000 |
| Q2 | build | 150.000 | 84.889 | 0.434074 |
| Q2 | dev | 150.000 | 84.600 | 0.436000 |

解释：

- Q0 是单路召回，所以无重复
- Q1 两路 union 后平均重复率约 `21%~22%`
- Q2 三路 union 后平均重复率上升到约 `43%~44%`

这说明：

- `keyword_query` 带来了一定新增候选，但仍有可接受重复
- `expanded_query` 进一步增加了大量重叠候选，边际新增候选较少

### 3.2 source_query_hit 分布

#### Q0

- build:
  - `main_query hit = 20`
  - `none = 7`
- dev:
  - `main_query hit = 21`
  - `none = 9`

#### Q1

- build:
  - `main_query hit = 20`
  - `keyword_query hit = 18`
  - `main_only = 3`
  - `keyword_only = 1`
  - `main+keyword = 17`
  - `none = 6`
- dev:
  - `main_query hit = 21`
  - `keyword_query hit = 13`
  - `main_only = 9`
  - `keyword_only = 1`
  - `main+keyword = 12`
  - `none = 8`

#### Q2

- build:
  - `main_query hit = 20`
  - `keyword_query hit = 18`
  - `expanded_query hit = 20`
  - `keyword_only = 1`
  - `main+expanded = 3`
  - `all3 = 17`
  - `none = 6`
- dev:
  - `main_query hit = 21`
  - `keyword_query hit = 13`
  - `expanded_query hit = 22`
  - `keyword_only = 1`
  - `expanded_only = 1`
  - `main+expanded = 9`
  - `all3 = 12`
  - `none = 7`

结论：

- `keyword_query` 的确能命中一部分 `main_query` 没命中的 gold
- `expanded_query` 大部分命中与 `main_query` 或 `all3` 重叠
- 真正的 `expanded_only` 很少，只在 dev 出现了 `1` 个

## 4. 消融分析

### 4.1 Q1 相比 Q0 是否提升 candidate Hit@50、Hit@10、MRR

结论：Q1 相比 Q0 有稳定正收益。

build：

- `candidate Hit@50: 0.740741 -> 0.777778`
- `Hit@10: 0.666667 -> 0.703704`
- `MRR: 0.545062 -> 0.569136`

dev：

- `candidate Hit@50: 0.700000 -> 0.700000`
- `Hit@10: 0.466667 -> 0.533333`
- `MRR: 0.201481 -> 0.280040`

样本级变化：

- build：`candidate new hits +1`，`final top10 new hits +1`
- dev：`candidate new hits +1`，同时 `candidate lost hits -1`；`final top10 new hits +3`，`final lost hits -1`

结论更准确地说：

- Q1 在 build/dev 的 final retrieval 效果都优于 Q0
- candidate recall 提升不算大，但 final rerank 后的收益是可见的

### 4.2 Q2 相比 Q1 是否进一步提升

结论：Q2 没有稳定进一步提升。

build：

- `candidate Hit@50: 0.777778 -> 0.777778`
- `Hit@10: 0.703704 -> 0.666667`
- `MRR: 0.569136 -> 0.500000`
- `gold_in_candidate_not_final_count: 2 -> 3`

dev：

- `candidate Hit@50: 0.700000 -> 0.700000`
- `Hit@10: 0.533333 -> 0.533333`
- `MRR: 0.280040 -> 0.332593`
- `gold_in_candidate_not_final_count: 5 -> 5`

样本级变化：

- build：`candidate new hits +0`，`final new hits +2`，但 `final lost hits -3`
- dev：`candidate new hits +1`，同时 `candidate lost hits -1`；`final new hits +1`，同时 `final lost hits -1`

这说明 Q2 的增益不稳定，build 甚至明显回退。

### 4.3 keyword_query 是否带来 main_query 没召回的 gold evidence

结论：是，但数量不大。

直接证据：

- build `keyword_only = 1`
- dev `keyword_only = 1`

这意味着 `keyword_query` 至少在 build/dev 都各自补到了 `main_query` 没召回的 gold。

同时，Q1 相比 Q0 在 build 增加了 `1` 个 candidate gold hit，在 dev 也有 `1` 个新增 candidate gold hit。

### 4.4 expanded_query 是否带来 keyword_query 没召回的 gold evidence

结论：有，但很弱，而且不稳定。

直接证据：

- build `expanded_only = 0`
- dev `expanded_only = 1`

也就是说：

- build 没有出现真正只能靠 expanded query 才打到的 gold
- dev 只有 `1` 个 expanded-only 样本

更重要的是：

- Q2 相比 Q1 没有带来 candidate Hit@50 的整体提升
- build 的 final `Hit@10/MRR` 反而下降

所以 expanded query 的边际收益目前不足以证明它值得进入默认路径。

### 4.5 是否存在 query expansion 引入噪声

结论：存在，而且主要发生在 Q2。

依据：

1. Q2 的 `duplicate_candidate_ratio` 明显偏高，约 `43%~44%`
2. Q2 的 `union_candidate_count` 只比 Q1 略增，但 raw candidate 数量从 `100` 增到 `150`
3. Q2 build 的 `Hit@10` 和 `MRR` 明显回退
4. Q2 build 的 `gold_in_candidate_not_final_count` 还从 `2` 上升到 `3`

这说明：

- expanded query 当前引入了大量重叠候选
- 但这些额外候选没有稳定转化为更好的 final 排序
- 相反，它增加了 rerank 的排序压力和噪声

## 5. 决策建议

### 5.1 推荐进入 test 的 query_mode

当前建议进入 test 的 query mode 是：

- `original_keyword`

即：

- Q1

### 5.2 为什么推荐 Q1

原因：

1. Q1 在 build/dev 上都优于 Q0 的 final 指标
2. Q1 在 build 上也提升了 `candidate Hit@50`
3. Q1 的重复率仍在可接受范围
4. `keyword_query` 的确补到了 `main_query` 没打到的 gold

因此 Q1 是当前最合理的下一步 test 候选。

### 5.3 为什么不推荐 Q2 直接进入 test

原因：

1. Q2 相比 Q1 没有稳定提升
2. build 上 `Hit@10/MRR` 明显回退
3. `expanded_query` 的独立增益很弱
4. `duplicate_candidate_ratio` 明显过高
5. Q2 更像是引入了额外噪声，而不是稳定缓解 semantic gap

### 5.4 是否需要保留 Q0

Q0 不建议作为下一步 test 主候选，但它仍然是重要基线。

原因：

- 如果 Q1 的 test 结果不能复现 build/dev 的增益，Q0 仍然是最保守的 fallback
- 当前实验不应为了增加创新点而强行选择更复杂但更差的 query_mode

### 5.5 当前 build/dev 阶段结论

本阶段 build/dev 的保守结论是：

- `original_keyword` 是当前最有希望进入 test 的 query decomposition 方案
- `original_keyword_expanded` 目前没有证明自己比 Q1 更好
- 当前 query decomposition 结果仍然建立在 `dense_current_rerank` 上，而 `current_rerank` 必须明确表述为 `online rerank + local fallback`

补充运行事实：

- 本轮 6 个 build/dev 实验没有运行失败，也没有结果文件级错误
- 但日志中仍然出现 Cohere `429`
- 6 个实验合计记录了 `123` 次 fallback

因此，后续对外报告中仍然必须写：

`current_rerank = online rerank + local fallback`

而不能写成 pure Cohere rerank
