# Experiment Results

本目录用于存放实验运行结果，现已按实验主题整理，避免继续把 pilot / expanded、json / csv 混放在同一层级。

## 目录结构

- `retrieval/pilot/`: Pilot 阶段 retrieval / rerank 结果 JSON。
- `retrieval/expanded/`: 扩展阶段 retrieval / rerank 结果 JSON。
- `retrieval/mock/`: mock 检索评测结果。
- `confidence/baseline/`: 检索级置信度基线实验结果 JSON。
- `confidence/tuning/`: `system_top3_support` 调参结果 JSON。
- `diagnostics/`: live retrieval 对齐诊断结果。
- `indexing/`: Milvus 实验 collection 建索引报告。
- `thesis_tables/retrieval/`: 面向论文表格的 retrieval CSV，按 `pilot / expanded / mock` 分层保存。
- `thesis_tables/confidence/`: 面向论文表格的 confidence CSV，按 `baseline / tuning` 分层保存。

## 当前推荐入口

- 当前 expanded retrieval 主结果：
  `retrieval/expanded/`
- 当前 expanded 论文表格：
  `thesis_tables/retrieval/expanded/`
- 当前 confidence 基线结果：
  `confidence/baseline/`
- 当前 confidence tuning 结果：
  `confidence/tuning/`
