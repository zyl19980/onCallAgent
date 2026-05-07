# Query Decomposition State Check

## 检查范围

本检查只读取当前代码与现有 expanded dataset/split，不修改任何 dataset、split、retrieval result，也不重新运行 build/dev/test。

主要检查对象：

- `app/services/hybrid_retrieval_service.py`
- `scripts/experiment/evaluate_rag_retrieval.py`
- `app/services/rag_agent_service.py`
- `app/services/conversation_memory_service.py`
- `app/services/query_fingerprint_service.py`
- `aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`

## 1. 当前代码已有能力有哪些

### 1.1 已有 query understanding 能力

`app/services/hybrid_retrieval_service.py` 已经实现了 `HybridRetrievalService._understand_query(...)`，输出结构为：

- `primary_query`
- `keyword_query`
- `expanded_queries`
- `keywords`

当前逻辑是：

1. 规范化当前 query
2. 从 `summary + recent_messages + query` 拼出的文本里做规则化关键词抽取
3. 生成 `keyword_query`
4. 生成最多 3 条 `expanded_queries`

这说明系统内部已经存在一个可复用的 query decomposition 雏形。

### 1.2 已有 keyword extraction 能力

`_understand_query(...)` 内部通过 `_extract_keywords(...)` 做规则化关键词抽取，当前 token 规则覆盖：

- 英文单词
- 数字
- 下划线/连字符
- 中文单字

并带有一组 stopwords。

这套逻辑已经能抽出一部分：

- fault code / alarm token
- part name / parameter token
- action word
- 中英文混合术语

虽然比较轻量，但已经足够作为离线实验的 `keyword_query` 起点。

### 1.3 已有 expanded query 拼接能力

`_understand_query(...)` 当前会构造：

- 原始规范化 query
- `keyword_query`
- `summary + query`
- `recent_context + query`

也就是说，服务层已经预留了：

- 单轮 query reformulation
- 历史摘要增强
- 最近 user turns 增强

### 1.4 已有多轮上下文输入位

线上 runtime 路径中，`app/services/rag_agent_service.py` 会在检索前取：

- `conversation_memory_service.get_summary(session_id)`
- `conversation_memory_service.get_history(session_id)`

然后调用：

- `hybrid_retrieval_service.retrieve(question, summary=summary, recent_messages=history)`

`app/services/conversation_memory_service.py` 也已经维护：

- `summary`
- `history`

因此系统运行态是支持“历史摘要 + 最近几轮消息”输入检索的。

### 1.5 已有实验入口对 `_understand_query` 的局部复用

`scripts/experiment/evaluate_rag_retrieval.py` 里当前已经有两处复用：

1. `HybridRetrievalAdapter.retrieve(...)`
   - 直接调用 `hybrid_retrieval_service._understand_query(query, "", [])`
   - 再用 `expanded_queries` 分别做 vector recall 和 BM25 recall

2. `CurrentRerankAdapter.rerank(...)`
   - 也调用 `hybrid_retrieval_service._understand_query(query, "", [])`
   - 之后将分析结果传给 `_rerank_candidates(...)`

因此，实验脚本并不是完全没有 query understanding，只是目前用法非常有限，而且没有开放 query-mode 控制。

### 1.6 已有 query fingerprint 会消费 keyword 信息

`app/services/query_fingerprint_service.py` 已经基于 `QueryUnderstandingResult` 使用：

- `primary_query`
- `keyword_query`
- `keywords`

这说明 query understanding 的返回结构已经被其他模块消费过，具备一定稳定性。

## 2. 缺失能力有哪些

### 2.1 当前离线实验入口没有 query-mode 概念

`scripts/experiment/evaluate_rag_retrieval.py` 当前没有：

- `--query-mode`
- `main_query / keyword_query / expanded_query` 切换
- `Q0 / Q1 / Q2` 对比入口

因此当前无法在固定 backbone 下直接比较：

- Q0: original
- Q1: original + keyword
- Q2: original + keyword + expanded

### 2.2 当前 dense retrieval 路径不会用 decomposition 结果

当前主推荐 backbone 是 `dense_current_rerank`，但 `LiveRetrievalAdapter` 现在只会：

- 对 `sample["user_input"]` 做单次 dense search

它不会：

- 调 `_understand_query`
- 跑多 query dense recall
- 合并多 query candidate

所以如果下一阶段目标是“固定 dense_current_rerank backbone 做 query decomposition 实验”，当前代码还不具备直接实验能力。

### 2.3 当前 expanded_query 生成能力偏弱

现有 `_understand_query(...)` 虽然有 `expanded_queries` 字段，但在离线实验场景里调用方式是：

- `summary=""`
- `recent_messages=[]`

此时 `expanded_queries` 基本只剩：

1. normalized original query
2. keyword_query

也就是说：

- 现在的离线实验里几乎没有真正的 domain-aware expanded query
- 更谈不上真实的 multi-turn expanded query

### 2.4 当前没有独立的 domain-aware expansion 组件

代码里没有看到独立的：

- query expansion service
- keyword rewrite service
- decomposition planner
- offline prompt-based query generator

现有能力主要集中在 `hybrid_retrieval_service._understand_query(...)` 里的轻量规则逻辑。

### 2.5 当前离线 dataset 不包含多轮上下文字段

expanded dataset 当前样本字段包括：

- `user_input`
- `reference_answer`
- `reference_chunk_ids`
- `question_type`
- `reasoning_hops`
- `should_abstain`
- 等标注字段

未发现：

- `conversation_summary`
- `recent_messages`
- `recent_turns`
- `dialog_history`
- `session_context`

因此当前离线 dataset 是单轮样本集，不支持真实多轮 expanded query 实验。

## 3. 是否建议复用已有 `_understand_query`

结论：建议复用，但不要直接把它当成最终版 decomposition 能力。

更准确地说：

- 对 `keyword_query`：建议直接复用现有 `_understand_query(...)`
- 对 `main_query`：直接复用 `primary_query`
- 对 `expanded_query`：建议在 `_understand_query(...)` 基础上补一层更明确的离线实验逻辑

原因如下：

1. 它已经稳定输出 `primary_query / keyword_query / expanded_queries / keywords`
2. 它已经被 hybrid retrieval 和 current rerank 使用，改动成本低
3. 作为 Phase 1 实验基线足够合适
4. 但它当前的 `expanded_queries` 在离线无上下文场景下太弱，无法充分代表你要验证的 “domain-aware expanded query”

因此最合理的做法是：

- 复用 `_understand_query(...)` 作为 decomposition 的底层分析入口
- 在实验脚本侧增加 query-mode 组装逻辑
- 不要强依赖 runtime 的 summary/history

## 4. `expanded_query` 在当前离线 dataset 中如何构造最合理

结论：应退化为“基于当前 query 的 domain-aware expanded query”，不能编造不存在的多轮历史。

因为当前 dataset 没有真实多轮上下文，所以 expanded query 不应使用：

- 假造的 recent turns
- 假造的 conversation summary
- 任何从 `reference_answer`、`reference_chunk_ids`、`source_ids` 倒灌的信息

否则会引入明显的数据泄漏。

当前最合理的离线构造方式是：

1. `main_query`
   - 直接使用 `user_input`

2. `keyword_query`
   - 使用 `_understand_query(...)` 产出的 `keyword_query`

3. `expanded_query`
   - 基于 `user_input + keywords` 做 domain-aware rewrite
   - 只允许使用当前 query 中可观察到的信息
   - 可以做的增强包括：
     - 保留原 query 主语义
     - 显式补出 troubleshooting / alarm / fault / parameter / procedure 等意图词
     - 对故障码、部件名、参数名、动作词做结构化重排
     - 对中英文混合术语做规范化

更保守的第一版建议：

- Q0 = `primary_query`
- Q1 = `primary_query + keyword_query`
- Q2 = `primary_query + keyword_query + domain_aware_expansion`

其中 `domain_aware_expansion` 应优先走规则模板，而不是先上生成式 LLM。

原因是：

- 可控
- 可复现
- 更适合离线 build/dev 比较

## 5. 是否需要新增参数 `--query-mode`

结论：需要。

建议新增 `--query-mode`，最少支持三档：

- `q0`
  - 只用 `main_query`
- `q1`
  - 用 `main_query + keyword_query`
- `q2`
  - 用 `main_query + keyword_query + expanded_query`

也可以用更直观的命名：

- `main`
- `main_keyword`
- `main_keyword_expanded`

但如果目标是和实验设计文档一致，`q0/q1/q2` 更清晰。

为什么需要这个参数：

1. 便于在同一脚本、同一评测口径下做严格 A/B
2. 便于冻结 backbone 后只切 query formulation
3. 便于后续生成统一 comparison JSON/CSV

## 6. 是否需要新增独立脚本，还是直接扩展 `evaluate_rag_retrieval.py`

结论：建议直接扩展 `scripts/experiment/evaluate_rag_retrieval.py`，不要新增独立脚本。

理由：

1. 当前 retrieval evaluation 的指标、输出结构、summary CSV、per-sample 记录已经稳定
2. 当前脚本已经支持：
   - dense/hybrid retrieval strategy
   - none/current rerank
   - candidate/final 双层指标
   - should_abstain 跳过逻辑
3. query decomposition 本质上只是“query formulation 变量”，不应另起一套评测框架

建议做法：

- 在 `evaluate_rag_retrieval.py` 内新增 `--query-mode`
- 在 adapter 层增加一个“query planner / query builder”
- dense 与 hybrid 都通过统一 query bundle 消费：
  - dense: 多 query dense recall + merge + current rerank
  - hybrid: 多 query hybrid recall + current rerank

这样可以保持：

- 输出 JSON 结构不变
- comparison 脚本基本不变
- 已有实验结果可并列对比

## 7. 最终建议

### 7.1 当前代码已有能力总结

已有：

- query understanding
- keyword extraction
- limited expanded query generation
- runtime conversation summary / recent turns support
- hybrid retrieval 对 `_understand_query` 的直接复用

### 7.2 当前缺口总结

缺口：

- 离线 dataset 没有多轮上下文
- dense retrieval 实验入口不支持 multi-query recall
- 没有 `--query-mode`
- 现有 `expanded_query` 在离线模式下过弱

### 7.3 Phase 2 实现建议

下一阶段建议按最小可验证路径推进：

1. 在 `evaluate_rag_retrieval.py` 增加 `--query-mode`
2. 固定 backbone 为 `dense_current_rerank`
3. 先支持：
   - `q0`: original
   - `q1`: original + keyword
   - `q2`: original + keyword + rule-based domain-aware expansion
4. 先在 `build/dev` 上跑，不动 `test`
5. 不要使用任何虚构的 multi-turn history

### 7.4 本阶段推荐结论

推荐结论如下：

- 建议复用已有 `_understand_query(...)`
- 但 `expanded_query` 需要在离线实验侧补强
- 建议新增 `--query-mode`
- 建议直接扩展 `evaluate_rag_retrieval.py`
- 当前 expanded dataset 只能做“单轮退化版 expanded query”实验，不能声称是真实多轮 query decomposition
