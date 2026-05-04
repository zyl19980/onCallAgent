# AIOps Thesis Experiment Pipeline

本目录用于毕业论文实验数据集构造与实验结果沉淀，和现有线上/主流程 RAG 资产隔离。

## 目标

- 为论文实验提供独立的数据集、切分产物、训练/验证/测试划分与结果表格。
- 不再兼容旧 `aiops-docs/testsets/context_catalog_v1.jsonl`。
- 不依赖旧评测系统；后续实验脚本统一放在 `scripts/experiment/`。
- 不修改现有 RAG 主流程，实验数据与正式运行链路分离维护。

## 目录说明

- `sources/`: 实验原始来源文档与人工整理的源材料。
- `chunks/`: 由实验脚本生成的切分结果、中间结构化产物。
- `rag/`: RAG 实验资产，现已按 `candidates / reviews / datasets / splits` 分层整理。
- `rag/candidates/`: 候选题生成结果，按 `batch1 / batch2` 分开保存。
- `rag/reviews/`: 人工审核导出表、审核回填结果与 merged reviewed candidates。
- `rag/datasets/`: 正式 RAG dataset，按 `pilot / expanded` 分开保存。
- `rag/splits/`: RAG dataset 划分结果，按 `pilot / expanded` 分开保存。
- `agent/`: Agent 实验案例、轨迹标注、金标准答案。
- `agent/splits/`: Agent 实验的 train/dev/test 或回放划分文件。
- `results/`: 实验运行输出、统计汇总、分析结果，现已按 `retrieval / confidence / diagnostics / indexing` 分层整理。
- `results/thesis_tables/`: 面向论文正文/附录的表格导出结果，按实验类型继续分层保存。
- `reports/`: 面向论文写作与实验复盘的阶段性汇总报告。

## 当前推荐入口

- 当前候选题批次入口：`rag/candidates/batch1/`、`rag/candidates/batch2/`
- 当前审核结果入口：`rag/reviews/batch1/`、`rag/reviews/batch2/`、`rag/reviews/merged/`
- 当前正式数据集入口：`rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`
- 当前正式 split 入口：`rag/splits/expanded/`
- 当前 expanded retrieval 对比入口：`results/retrieval/expanded/` 与 `results/thesis_tables/retrieval/expanded/`
- 当前阶段报告入口：`reports/expanded_experiment_stage_report.md`

## 配套脚本

`scripts/experiment/` 用于放置新的实验流水线脚本，例如：

- 原始资料整理与清单生成
- chunk 构造与数据导出
- RAG/Agent 样本划分
- 结果统计与论文表格生成

## 约束

- 旧 `context_catalog_v1.jsonl` 仅保留为历史资产，不作为本目录输入。
- 旧评测系统不在本实验目录下恢复或继续兼容。
- 若需要把实验数据接入现有服务，应通过新增桥接脚本完成，而不是改写现有主流程。
