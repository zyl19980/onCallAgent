# Answer Evaluation Build Report

## 1. 数据规模与拆分
- experiment: `dense_current_original_build`
- split: `build`
- evaluated samples: `30`
- answerable samples used for answer-quality means: `27`
- should_abstain samples reported separately: `3`
- question_type distribution: `{'safety_or_constraint': 8, 'troubleshooting_step': 2, 'parameter_or_fault_code': 10, 'symptom_cause': 3, 'definition_or_component_lookup': 2, 'abstention_insufficient_evidence': 3, 'cross_doc_multi': 2}`

## 2. 总体指标表
| experiment_name | split | evaluated_samples | abstain_samples | faithfulness_supported_by_retrieved_mean | faithfulness_supported_by_retrieved_mean_ci_low_95 | faithfulness_supported_by_retrieved_mean_ci_high_95 | faithfulness_supported_by_cited_mean | faithfulness_supported_by_cited_mean_ci_low_95 | faithfulness_supported_by_cited_mean_ci_high_95 | correctness_strict_mean | correctness_strict_mean_ci_low_95 | correctness_strict_mean_ci_high_95 | correctness_lenient_mean | correctness_lenient_mean_ci_low_95 | correctness_lenient_mean_ci_high_95 | citation_precision_mean | citation_precision_mean_ci_low_95 | citation_precision_mean_ci_high_95 | citation_recall_mean | citation_recall_mean_ci_low_95 | citation_recall_mean_ci_high_95 | citation_f1_mean | citation_f1_mean_ci_low_95 | citation_f1_mean_ci_high_95 | hallucination_rate_mean | hallucination_rate_mean_ci_low_95 | hallucination_rate_mean_ci_high_95 | abstention_precision | abstention_recall | generator_model | judge_model | prompt_version | run_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_current_original_build | build | 30 | 3 | 0.763742 | 0.613022 | 0.891681 | 0.703557 | 0.542181 | 0.835391 | 0.296296 | 0.148148 | 0.481481 | 0.577778 | 0.431481 | 0.738889 | 0.619577 | 0.478924 | 0.75291 | 0.814815 | 0.666667 | 0.962963 | 0.681952 | 0.539095 | 0.810758 | 0.08811 | 0.033951 | 0.154909 | 0.428571 | 1.0 | qwen3.6-flash | deepseek-v3.2 | v1.0 | 2026-05-08 |

## 3. 按 question_type 分组
| experiment_name | split | question_type | n_samples | faithfulness_supported_by_retrieved_mean | faithfulness_supported_by_retrieved_mean_ci_low_95 | faithfulness_supported_by_retrieved_mean_ci_high_95 | correctness_strict_mean | correctness_strict_mean_ci_low_95 | correctness_strict_mean_ci_high_95 | citation_f1_mean | citation_f1_mean_ci_low_95 | citation_f1_mean_ci_high_95 | hallucination_rate_mean | hallucination_rate_mean_ci_low_95 | hallucination_rate_mean_ci_high_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_current_original_build | build | cross_doc_multi | 2 | 0.5 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 |
| dense_current_original_build | build | definition_or_component_lookup | 2 | 0.333333 | 0.0 | 0.666667 | 0.5 | 0.0 | 1.0 | 0.5 | 0.0 | 1.0 | 0.166667 | 0.0 | 0.333333 |
| dense_current_original_build | build | parameter_or_fault_code | 10 | 0.766667 | 0.5 | 0.966667 | 0.5 | 0.2 | 0.8 | 0.63 | 0.33 | 0.87 | 0.033333 | 0.0 | 0.1 |
| dense_current_original_build | build | safety_or_constraint | 8 | 0.799851 | 0.635417 | 0.927083 | 0.125 | 0.0 | 0.375 | 0.798611 | 0.673611 | 0.902778 | 0.200149 | 0.06994 | 0.364583 |
| dense_current_original_build | build | symptom_cause | 3 | 0.962963 | 0.888889 | 1.0 | 0.333333 | 0.0 | 1.0 | 0.752381 | 0.4 | 0.952381 | 0.037037 | 0.0 | 0.111111 |
| dense_current_original_build | build | troubleshooting_step | 2 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.833333 | 0.666667 | 1.0 | 0.0 | 0.0 | 0.0 |
The highest strict correctness group is `definition_or_component_lookup` and the lowest is `cross_doc_multi`, with a gap of `0.500000`.

## 4. 置信区间说明
Mean metrics use bootstrap confidence intervals with N=1000. The bootstrap is computed over answerable samples only for answer-quality metrics so that should_abstain=true samples do not dilute answerable performance.

## 5. 抽检结果
Five build samples were exported to `aiops-docs/experiment/results/answer/expanded/human_review_samples_build.csv` for manual spot-checking. Cohen's kappa is intentionally left for the manual review step after human labels are filled.

## 6. 诚实性 Caveat
The LLM judge is an offline proxy evaluator rather than ground truth. Results depend on the generator model version, judge model version, prompt hashes, API behavior, and temperature=0 setting used in this run. The should_abstain=true subset is evaluated as a separate abstention subset and is not mixed into answerable faithfulness, correctness, citation, or hallucination means. Because this build run contains only 30 samples and uses bootstrap N=1000, small differences between metrics should not be described as statistically significant without additional validation.
