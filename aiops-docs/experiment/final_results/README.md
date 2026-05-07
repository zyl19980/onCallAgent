# Experiment Final Results

本目录单独保存最终冻结结论、最终测试确认结果和论文最终汇总表。

## 目录结构

- `reports/`: 最终检索策略、最终 confidence confirmation 与 RAG 实验总报告。
- `retrieval/raw_json/`: 最终选定 retrieval 主线的 held-out test JSON。
- `retrieval/thesis_tables/`: 最终选定 retrieval 主线的 held-out test CSV。
- `confidence/raw_json/`: 最终 confidence held-out test JSON。
- `confidence/thesis_tables/`: 最终 confidence held-out test CSV。
- `summary/`: 面向论文最终汇总的总表。

## 最终冻结配置

- retrieval backbone: `dense_current_rerank`
- query mode: `original`
- confidence strategy: `score_threshold`
- confidence thresholds: `high=0.56`, `low=0.52`
