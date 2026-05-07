# Next Steps

## Priority 1：数据集扩展
- 生成第二批候选题。
- 完成人工审核。
- 合并 reviewed candidates。
- 构建扩展后的 experiment_rag_dataset。
- 对扩展数据集执行 validate。
- 重新执行 split。

## Priority 2：最终检索实验
- 运行 dense_no_rerank。
- 运行 dense_current_rerank。
- 运行 hybrid_no_rerank。
- 运行 hybrid_current_rerank。

## Priority 3：置信度策略复验
- 在扩展后的 build/dev 上重新 tune system_top3_support。
- 冻结 test 后只运行最终选定策略。

## Priority 4：工程集成
- 抽取 retrieval_confidence_service。
- 接入 low_confidence_events。
- 在前端治理模块展示 confidence_debug。
