# Experiment Process Results

本目录集中保存阶段性实验过程产物，按实验推进顺序编号整理。

## 目录结构

- `01_overview/`: 实验过程总览、状态检查、交接、阶段汇总与过程 summary 表。
- `02_pilot_retrieval/`: pilot retrieval / mock retrieval 的运行 JSON 与论文表格 CSV。
- `03_expanded_retrieval/`: expanded retrieval backbone 对比过程产物，包括 dense / hybrid、rerank / no-rerank 的 build/dev/test 结果与 comparison 表。
- `04_query_decomposition/`: query decomposition 与 multi-query retrieval 的状态报告、build/dev/test 结果与对比表。
- `05_confidence_tuning/`: confidence baseline 与 build/dev tuning 的报告、JSON 结果和 CSV 表格。
- `06_diagnostics_indexing/`: live retrieval 诊断与 Milvus index 报告。

## 说明

- 文件名前缀表示阅读顺序，不表示实验重新运行顺序。
- JSON/CSV 内容保持原样迁移，内部记录的历史路径可作为当时生成产物的 provenance。
- 最终冻结结论和 held-out final confirmation 产物单独保存在 `../final_results/`。
