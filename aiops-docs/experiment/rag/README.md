# RAG Experiment Assets

本目录用于存放 RAG 实验资产，已按“候选题 -> 人工审核 -> 正式数据集 -> split”整理，避免继续依赖相似文件名区分阶段。

## 目录结构

- `candidates/batch1/`: 第一批候选题及生成报告。
- `candidates/batch2/`: 第二批候选题及生成报告。
- `reviews/batch1/`: 第一批审核导出表、审核 CSV、导入报告、reviewed candidates。
- `reviews/batch2/`: 第二批审核导出表、审核 CSV、导入报告、reviewed candidates。
- `reviews/merged/`: 合并后的 reviewed candidates 与 merged report。
- `datasets/pilot/`: Pilot 阶段正式 RAG dataset 与校验报告。
- `datasets/expanded/`: 扩展阶段正式 RAG dataset 与校验报告。
- `splits/pilot/`: Pilot 阶段 split 结果。
- `splits/expanded/`: 扩展阶段 split 结果。

## 当前推荐使用

- 当前 merged reviewed candidates：
  `reviews/merged/rag_candidate_questions.merged.reviewed.jsonl`
- 当前正式 validated dataset：
  `datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl`
- 当前 build/dev/test/reserve 划分：
  `splits/expanded/`

## 命名约定

- `batch1` / `batch2`: 候选题生成与人工审核批次。
- `pilot`: 第一轮 64 条正式样本实验资产。
- `expanded`: 扩展到 138 条正式样本后的当前主实验资产。
