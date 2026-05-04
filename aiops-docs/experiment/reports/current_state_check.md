# Current State Check

## 检查范围

本检查基于以下现有实验文档与产物，只做读取和归纳，不修改任何已有 dataset、split、结果 JSON 或代码：

- `aiops-docs/experiment/README.md`
- `aiops-docs/experiment/rag/README.md`
- `aiops-docs/experiment/results/README.md`
- `aiops-docs/experiment/reports/expanded_experiment_stage_report.md`
- `aiops-docs/experiment/reports/next_steps.md`
- `aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`
- `aiops-docs/experiment/rag/splits/expanded/rag_split_report.json`
- `aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded.json`

## 当前已经完成的实验

### 1. 扩展版 RAG 数据集流水线已经完成

已完成的阶段：

1. 原始文档整理
2. chunk 构建
3. annotation pool 构建
4. 两批候选题生成
5. 人工审核与 merged reviewed candidates 合并
6. 正式 expanded dataset 构建
7. dataset validate
8. expanded split

当前主实验资产已经切换到 expanded：

- 正式 validated dataset：`138` 条
- split：`build=30`，`dev=35`，`test=60`，`reserve=13`
- `should_abstain` 分布：
  - `build: false 27 / true 3`
  - `dev: false 30 / true 5`
  - `test: false 52 / true 8`
  - `reserve: false 12 / true 1`

补充状态：

- 当前数据集 warning 仍有 `duplicate_user_input_candidates: 19`
- split report 有 2 条 cross-split leakage warning：
  - `rockwell ... page_145: build,test`
  - `rockwell ... page_244: dev,test`

### 2. retrieval evaluation 已完成的实验

expanded 阶段已经跑完并沉淀结果的 retrieval 实验只有 2 组策略，各自覆盖 `build/dev/test`：

1. `dense_no_rerank`
2. `dense_current_rerank`

现成结果文件共有 6 个：

- `aiops-docs/experiment/results/retrieval/expanded/dense_no_rerank_expanded_build.json`
- `aiops-docs/experiment/results/retrieval/expanded/dense_no_rerank_expanded_dev.json`
- `aiops-docs/experiment/results/retrieval/expanded/dense_no_rerank_expanded_test.json`
- `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_build.json`
- `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_dev.json`
- `aiops-docs/experiment/results/retrieval/expanded/dense_current_rerank_expanded_test.json`

并已有汇总对比：

- `aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded.json`

### 3. 当前 retrieval 结论

在 expanded 数据集上，当前已经被文档和结果共同确认的结论是：

- dense candidate recall 不低，`candidate Hit@50` 为：
  - `build: 0.740741`
  - `dev: 0.700000`
  - `test: 0.750000`
- 主要瓶颈在 final top10 排序，不在 candidate top50 召回上界本身。
- `current rerank` 在 `build/dev/test` 都优于 `no_rerank`。

关键对比：

| split | dense_no_rerank Hit@10 / MRR | dense_current_rerank Hit@10 / MRR | delta Hit@10 | delta MRR |
| --- | --- | --- | --- | --- |
| build | `0.407407 / 0.288272` | `0.666667 / 0.545062` | `+0.259260` | `+0.256790` |
| dev | `0.433333 / 0.153651` | `0.466667 / 0.208889` | `+0.033334` | `+0.055238` |
| test | `0.442308 / 0.302564` | `0.596154 / 0.429884` | `+0.153846` | `+0.127320` |

rerank 行为侧结论：

- build：`gold_in_candidate_not_final_count 9 -> 2`
- dev：`8 -> 7`
- test：`16 -> 8`

这说明当前 rerank 的主要作用确实是把已经进入 candidate top50 的 gold chunk 推进到 final top10。

## 哪些结果可以直接复用

### 可直接复用的主实验资产

以下产物已经是当前推荐入口，可直接复用，不需要重建：

1. validated dataset
   - `aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`
2. expanded split
   - `aiops-docs/experiment/rag/splits/expanded/`
   - 当前已知 split report：`aiops-docs/experiment/rag/splits/expanded/rag_split_report.json`
3. expanded retrieval 主结果
   - `aiops-docs/experiment/results/retrieval/expanded/`
4. expanded retrieval 汇总对比
   - `aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded.json`
5. 论文表格入口
   - `aiops-docs/experiment/results/thesis_tables/retrieval/expanded/`
6. 阶段总结结论
   - `aiops-docs/experiment/reports/expanded_experiment_stage_report.md`

### 当前最适合直接引用的 retrieval 结果

如果现在要复用 retrieval 结果，优先复用下面这两组：

1. dense baseline
   - `dense_no_rerank_expanded_build/dev/test`
2. current system rerank
   - `dense_current_rerank_expanded_build/dev/test`

注意：

- `current_rerank` 不是“纯 Cohere rerank”。
- 它代表当前系统策略：在线 rerank + 本地 fallback。
- 阶段报告已经记录过 Cohere `429` fallback，因此该结果可复用，但在后续汇报中应保留这个实验说明。

## 哪些实验不能重复跑或不能覆盖

这里区分“实验纪律上不应重复使用/覆盖”和“代码层面暂不支持”。

### 1. 当前不应覆盖的实验产物

以下现有产物应视为冻结输入或已沉淀结果，不应直接覆盖：

1. validated dataset
   - `aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`
2. expanded split 及其报告
   - `aiops-docs/experiment/rag/splits/expanded/`
   - `aiops-docs/experiment/rag/splits/expanded/rag_split_report.json`
3. 已存在的 expanded retrieval 结果
   - `aiops-docs/experiment/results/retrieval/expanded/*.json`
4. 已存在的 retrieval comparison
   - `aiops-docs/experiment/results/retrieval/expanded/retrieval_experiment_comparison_expanded.json`

### 2. 当前不应重复用于调参的实验

以下约束来自阶段报告，应作为当前实验纪律：

1. `test` 不应用于继续调 retrieval 或 confidence 参数。
   - 当前建议只在 `build/dev` 上继续试验。
   - `test` 应保留给最终冻结策略后的主结果。
2. 当前阶段不重新验证 chunk 切片策略。
   - evidence base 固定为 `aiops-docs/experiment/chunks/experiment_chunks.jsonl`
3. 当前阶段不重新构造 evidence base。
4. 当前阶段不恢复旧评测系统，不再兼容旧 `context_catalog_v1.jsonl`。

### 3. 当前实际上还没跑出来的实验

根据 `next_steps.md` 和结果目录现状，以下实验被列为下一步，但当前没有现成结果文件，因此不能当作“已完成可复用结果”：

1. `hybrid_no_rerank`
2. `hybrid_current_rerank`

结果目录 `aiops-docs/experiment/results/retrieval/expanded/` 中目前只有 `dense_*`，没有任何 `hybrid_*` JSON。

## 当前 retrieval evaluation 的脚本入口

主入口脚本：

- `scripts/experiment/evaluate_rag_retrieval.py`

其职责：

- 读取 dataset JSONL
- 读取固定 chunks JSONL
- 运行 `mock` 或 `live` retrieval
- 产出结果 JSON 和 summary CSV

相关汇总脚本：

- `scripts/experiment/compare_retrieval_experiments.py`

它负责读取多个 retrieval result JSON，生成 comparison JSON 和 CSV。

## 是否已经支持 hybrid / no_rerank / current_rerank 开关

### 结论

- `no_rerank`：已支持
- `current_rerank`：已支持
- `hybrid`：底层服务有能力，但当前 retrieval evaluation 脚本入口未开放，不算“当前实验入口已支持”

### 依据

#### 1. 当前实验入口 `evaluate_rag_retrieval.py`

脚本参数限制是：

- `--retrieval-strategy` 只允许 `dense`
- `--rerank` 只允许 `none` 或 `current`

也就是说，当前评测入口支持的是：

1. `dense + no_rerank`
2. `dense + current_rerank`

不支持的是：

1. `hybrid + no_rerank`
2. `hybrid + current_rerank`

此外，`validate_retrieval_config()` 里也直接限制：

- `retrieval_strategy != "dense"` 会报错
- `rerank not in {"none", "current"}` 会报错

#### 2. 底层服务层

`app/services/hybrid_retrieval_service.py` 本身实现了：

- vector recall
- BM25 keyword recall
- fuse
- rerank

因此从服务能力看，系统内部存在 hybrid 检索逻辑。

但当前实验评测入口没有把它接成 `--retrieval-strategy hybrid`，而是：

- live retrieval 仍走 `vector_search_service.search_similar_documents(...)`
- `current_rerank` 只复用了 `hybrid_retrieval_service` 的 rerank 部分

因此当前状态应表述为：

- “系统里有 hybrid retrieval service”
- “但当前实验评测脚本还没有支持 hybrid retrieval strategy 开关”

## 当前建议的工作边界

在不改动现有实验产物的前提下，当前最安全的继续方式是：

1. 复用 expanded dataset、split 和 dense 系列 retrieval 结果
2. 不覆盖现有 JSON/CSV 结果
3. 后续如需补实验，优先新增输出文件，不覆盖已有 expanded 结果
4. 若继续做 retrieval 对比，保持 `test` 只用于最终冻结后的结果确认

## 一句话结论

当前 expanded RAG 实验已经完成 dataset/split 冻结和 `dense_no_rerank`、`dense_current_rerank` 的 `build/dev/test` retrieval 评测；这些结果可以直接复用。当前 retrieval evaluation 脚本入口是 `scripts/experiment/evaluate_rag_retrieval.py`，它只支持 `dense + {none,current}`，尚未在实验入口层支持 `hybrid` 开关，且现有 expanded dataset、split、retrieval 结果都不应被覆盖。
