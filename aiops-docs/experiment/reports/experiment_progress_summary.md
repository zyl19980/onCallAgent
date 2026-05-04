# Experiment Progress Summary

## 1. Experiment Scope
- 当前实验固定 `experiment_chunks.jsonl` 作为统一 evidence base。
- 当前暂不验证切片策略效果。
- 当前重点验证 retrieval、rerank、confidence strategy。
- 当前实验属于 pilot/build-dev 阶段，不是最终 test 结果。

## 2. Dataset Construction Summary
- source 文档数量：`5`
- experiment_chunks 总数：`4503`
- 各 source chunk 数量：`{"abb_manual_for_induction_motors_and_generators_en": 504, "grundfos_nbe_nbse_nke_tpe_tped_installation_and_operating_instructions": 265, "haascnc_com_rotary_troubleshooting_guide_ngc": 124, "rockwell_powerflex_520_series_user_manual_520_um001_en_e": 971, "s71200_system_manual": 2639}`
- 各 chunk_type 数量：`{"alarm_fault_code": 287, "concept_and_component": 37, "front_matter": 468, "installation_or_wiring": 313, "maintenance_procedure": 52, "other": 669, "parameter_and_configuration": 1746, "safety_and_constraint": 566, "troubleshooting_procedure": 365}`
- annotation_pool 数量：`728`
- annotation priority 分布：`{"high": 491, "low": 19, "medium": 218}`
- candidate questions 数量：`80`
- reviewed/revised/rejected：`80` / `64` / `16`
- final RAG dataset 数量：`64`
- validation valid/invalid：`64` / `0`
- split 分布：`build=30, dev=20, test=0, reserve=14`

## 3. Current RAG Dataset Status
- 当前正式样本数：`64`
- `build=30, dev=20, test=0, reserve=14`
- 当前 `test` 为空的原因：样本量还不足，后续扩展后再冻结 `test`。
- 当前数据集主要用于流程验证、dev 调参与 pilot 实验。

## 4. Retrieval Experiment Summary
- build evaluated_samples：`dense_no_rerank=30`, `dense_current_rerank=30`
- dev evaluated_samples：`dense_no_rerank=20`, `dense_current_rerank=20`
- build dense_no_rerank：`Hit@1/3/5/10=0.333333, 0.5, 0.6, 0.633333`, `Recall@1/3/5/10=0.333333, 0.5, 0.6, 0.633333`, `MRR=0.434444`
- build dense_current_rerank：`Hit@1/3/5/10=0.466667, 0.566667, 0.666667, 0.766667`, `Recall@1/3/5/10=0.466667, 0.566667, 0.666667, 0.766667`, `MRR=0.537593`
- dev dense_no_rerank：`Hit@1/3/5/10=0.15, 0.2, 0.3, 0.4`, `Recall@1/3/5/10=0.15, 0.2, 0.3, 0.4`, `MRR=0.20256`
- dev dense_current_rerank：`Hit@1/3/5/10=0.35, 0.5, 0.55, 0.55`, `Recall@1/3/5/10=0.35, 0.5, 0.55, 0.55`, `MRR=0.418333`
- candidate Hit@10/20/50：`build no_rerank=0.633333, 0.833333, 0.9`, `build current=0.633333, 0.833333, 0.9`, `dev no_rerank=0.4, 0.5, 0.85`, `dev current=0.4, 0.5, 0.85`
- gold_in_candidate_not_final_count：`build no_rerank=8`, `build current=4`, `dev no_rerank=9`, `dev current=6`
- gold_promoted_by_rerank_count：`build=11`, `dev=12`
- gold_demoted_by_rerank_count：`build=3`, `dev=1`
- rerank 相对 no_rerank 的 delta：`build Hit@10=0.133334, MRR=0.103149; dev Hit@10=0.15, MRR=0.215773`

## 5. Retrieval Findings
- dense baseline can recall many gold chunks into candidate top50
- final top10 ordering remains the bottleneck before rerank
- current rerank improves Hit@K and MRR on both build and dev
- rerank mainly promotes gold chunks from candidate top50 into final top10/top5/top3/top1
- current rerank reflects the current system strategy: online rerank with local fallback

## 6. Confidence Experiment Summary
- `rank_and_margin / build`: `high_precision=1.0`, `low_capture=0.571429`, `confidence_accuracy=0.633333`, `count_high=7`, `count_medium=11`, `count_low=12`, `low_ratio=0.4`, `score_direction=missing`, `strong=`, `support=`, `high_avg=`, notes=`legacy output missing score_direction:aiops-docs/experiment/results/confidence/baseline/confidence_eval_dense_current_rerank_build.json`
- `score_margin / build`: `high_precision=`, `low_capture=`, `confidence_accuracy=`, `count_high=`, `count_medium=`, `count_low=`, `low_ratio=`, `score_direction=missing`, `strong=`, `support=`, `high_avg=`, notes=`missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_margin_dense_current_rerank_build.json`
- `score_threshold / build`: `high_precision=`, `low_capture=`, `confidence_accuracy=`, `count_high=`, `count_medium=`, `count_low=`, `low_ratio=`, `score_direction=missing`, `strong=`, `support=`, `high_avg=`, notes=`missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_threshold_dense_current_rerank_build.json`
- `system_top3_support / build`: `high_precision=0.8`, `low_capture=0.857143`, `confidence_accuracy=0.333333`, `count_high=5`, `count_medium=0`, `count_low=25`, `low_ratio=0.833333`, `score_direction=higher_is_better`, `strong=0.78`, `support=0.45`, `high_avg=0.55`
- `system_top3_support_tuned / build`: `high_precision=0.909091`, `low_capture=0.571429`, `confidence_accuracy=0.533333`, `count_high=11`, `count_medium=4`, `count_low=15`, `low_ratio=0.5`, `score_direction=lower_is_better`, `strong=0.68`, `support=0.57`, `high_avg=0.6`, notes=`best config from tuning grid search`
- `rank_and_margin / dev`: `high_precision=1.0`, `low_capture=0.666667`, `confidence_accuracy=0.6`, `count_high=4`, `count_medium=5`, `count_low=11`, `low_ratio=0.55`, `score_direction=missing`, `strong=`, `support=`, `high_avg=`, notes=`legacy output missing score_direction:aiops-docs/experiment/results/confidence/baseline/confidence_eval_dense_current_rerank_dev.json`
- `score_margin / dev`: `high_precision=`, `low_capture=`, `confidence_accuracy=`, `count_high=`, `count_medium=`, `count_low=`, `low_ratio=`, `score_direction=missing`, `strong=`, `support=`, `high_avg=`, notes=`missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_margin_dense_current_rerank_dev.json`
- `score_threshold / dev`: `high_precision=`, `low_capture=`, `confidence_accuracy=`, `count_high=`, `count_medium=`, `count_low=`, `low_ratio=`, `score_direction=missing`, `strong=`, `support=`, `high_avg=`, notes=`missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_threshold_dense_current_rerank_dev.json`
- `system_top3_support / dev`: `high_precision=1.0`, `low_capture=1.0`, `confidence_accuracy=0.7`, `count_high=5`, `count_medium=0`, `count_low=15`, `low_ratio=0.75`, `score_direction=higher_is_better`, `strong=0.78`, `support=0.45`, `high_avg=0.55`
- `system_top3_support_tuned / dev`: `high_precision=1.0`, `low_capture=0.777778`, `confidence_accuracy=0.65`, `count_high=5`, `count_medium=3`, `count_low=12`, `low_ratio=0.6`, `score_direction=higher_is_better`, `strong=0.6`, `support=0.27`, `high_avg=0.67`, notes=`best config from tuning grid search`

## 7. Confidence Findings
- rank_and_margin is the most stable general baseline across current build/dev runs
- system_top3_support behaves more like a strong low-confidence interception strategy
- system_top3_support thresholds are not final and should be re-tuned after dataset expansion
- confidence results should not be over-interpreted yet because build/dev sample counts remain small

## 8. Current Limitations
- 当前正式样本只有 64 条
- 当前没有 test split
- Grundfos 样本偏少
- symptom_cause 样本偏少
- 没有 cross_doc_multi
- 没有 abstention_insufficient_evidence
- rerank 结果受 Cohere 429 fallback 影响
- confidence 阈值尚未最终确定

## 9. Recommended Next Steps
- 扩展候选题到 200+，人工审核后得到 150–170 条 reviewed RAG samples
- 增加 cross_doc_multi 和 abstention_insufficient_evidence
- 重新 split: build/dev/test
- 在 build/dev 上重新确定 confidence 参数
- 冻结 test 后只跑最终策略
- 补充 hybrid retrieval 对比
- 将最终 confidence strategy 抽取到项目服务层

## Missing Files
- `aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_margin_dense_current_rerank_build.json`
- `aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_margin_dense_current_rerank_dev.json`
- `aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_threshold_dense_current_rerank_build.json`
- `aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_threshold_dense_current_rerank_dev.json`

## Warnings
- `missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_threshold_dense_current_rerank_build.json`
- `missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_threshold_dense_current_rerank_dev.json`
- `missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_margin_dense_current_rerank_build.json`
- `missing_file:aiops-docs/experiment/results/confidence/baseline/confidence_eval_score_margin_dense_current_rerank_dev.json`
- `missing_metric:score_direction:aiops-docs/experiment/results/confidence/baseline/confidence_eval_dense_current_rerank_build.json`
- `missing_metric:score_direction:aiops-docs/experiment/results/confidence/baseline/confidence_eval_dense_current_rerank_dev.json`
