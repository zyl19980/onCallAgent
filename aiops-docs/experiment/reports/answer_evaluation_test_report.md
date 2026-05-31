# Answer Evaluation Build Report

## 1. 数据规模与拆分
- experiment: `dense_current_original_test`
- split: `test`
- evaluated samples: `60`
- answerable samples used for answer-quality means: `52`
- should_abstain samples reported separately: `8`
- question_type distribution: `{'troubleshooting_step': 20, 'safety_or_constraint': 7, 'parameter_or_fault_code': 16, 'symptom_cause': 4, 'abstention_insufficient_evidence': 8, 'cross_doc_multi': 5}`

## 2. 总体指标表
| experiment_name | split | evaluated_samples | abstain_samples | faithfulness_supported_by_retrieved_mean | faithfulness_supported_by_retrieved_mean_ci_low_95 | faithfulness_supported_by_retrieved_mean_ci_high_95 | faithfulness_supported_by_cited_mean | faithfulness_supported_by_cited_mean_ci_low_95 | faithfulness_supported_by_cited_mean_ci_high_95 | correctness_strict_mean | correctness_strict_mean_ci_low_95 | correctness_strict_mean_ci_high_95 | correctness_lenient_mean | correctness_lenient_mean_ci_low_95 | correctness_lenient_mean_ci_high_95 | citation_precision_mean | citation_precision_mean_ci_low_95 | citation_precision_mean_ci_high_95 | citation_recall_mean | citation_recall_mean_ci_low_95 | citation_recall_mean_ci_high_95 | citation_f1_mean | citation_f1_mean_ci_low_95 | citation_f1_mean_ci_high_95 | hallucination_rate_mean | hallucination_rate_mean_ci_low_95 | hallucination_rate_mean_ci_high_95 | abstention_precision | abstention_recall | generator_model | judge_model | prompt_version | run_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_current_original_test | test | 60 | 8 | 0.857948 | 0.795933 | 0.915018 | 0.793357 | 0.731777 | 0.857036 | 0.269231 | 0.153846 | 0.403846 | 0.475962 | 0.365385 | 0.585577 | 0.770513 | 0.693269 | 0.851923 | 0.902244 | 0.820513 | 0.966346 | 0.815049 | 0.73898 | 0.888523 | 0.122821 | 0.075198 | 0.174209 | 1.0 | 1.0 | qwen3.6-flash | deepseek-v3.2 | v1.1 | 2026-05-11 |

## 3. 按 question_type 分组
| experiment_name | split | question_type | n_samples | faithfulness_supported_by_retrieved_mean | faithfulness_supported_by_retrieved_mean_ci_low_95 | faithfulness_supported_by_retrieved_mean_ci_high_95 | correctness_strict_mean | correctness_strict_mean_ci_low_95 | correctness_strict_mean_ci_high_95 | citation_f1_mean | citation_f1_mean_ci_low_95 | citation_f1_mean_ci_high_95 | hallucination_rate_mean | hallucination_rate_mean_ci_low_95 | hallucination_rate_mean_ci_high_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_current_original_test | test | cross_doc_multi | 5 | 0.966667 | 0.9 | 1.0 | 0.0 | 0.0 | 0.0 | 0.840635 | 0.811429 | 0.869841 | 0.033333 | 0.0 | 0.1 |
| dense_current_original_test | test | parameter_or_fault_code | 16 | 0.876563 | 0.742188 | 0.976562 | 0.25 | 0.0625 | 0.5 | 0.8 | 0.625 | 0.954167 | 0.060938 | 0.007812 | 0.125 |
| dense_current_original_test | test | safety_or_constraint | 7 | 0.775541 | 0.667749 | 0.888095 | 0.428571 | 0.0 | 0.857143 | 0.557823 | 0.306122 | 0.829932 | 0.224459 | 0.104762 | 0.331602 |
| dense_current_original_test | test | symptom_cause | 4 | 0.7125 | 0.425 | 1.0 | 0.0 | 0.0 | 0.0 | 0.642857 | 0.595238 | 0.666667 | 0.2875 | 0.0 | 0.575 |
| dense_current_original_test | test | troubleshooting_step | 20 | 0.87381 | 0.775 | 0.957143 | 0.35 | 0.15 | 0.55 | 0.945159 | 0.88119 | 0.987302 | 0.12619 | 0.042857 | 0.225 |
The highest strict correctness group is `safety_or_constraint` and the lowest is `cross_doc_multi`, with a gap of `0.428571`.

## 4. 置信区间说明
Mean metrics use bootstrap confidence intervals with N=1000. The bootstrap is computed over answerable samples only for answer-quality metrics so that should_abstain=true samples do not dilute answerable performance.

## 5. 抽检结果
Five build samples were exported to `aiops-docs/experiment/results/answer/expanded/human_review_samples_build.csv` for manual spot-checking. Cohen's kappa is intentionally left for the manual review step after human labels are filled.

## 6. 诚实性 Caveat
The LLM judge is an offline proxy evaluator rather than ground truth. Results depend on the generator model version, judge model version, prompt hashes, API behavior, and temperature=0 setting used in this run. The should_abstain=true subset is evaluated as a separate abstention subset and is not mixed into answerable faithfulness, correctness, citation, or hallucination means. Because this build run contains only 60 samples and uses bootstrap N=1000, small differences between metrics should not be described as statistically significant without additional validation.
