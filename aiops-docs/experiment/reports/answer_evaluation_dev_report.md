# Answer Evaluation Build Report

## 1. 数据规模与拆分
- experiment: `dense_current_original_dev`
- split: `dev`
- evaluated samples: `35`
- answerable samples used for answer-quality means: `30`
- should_abstain samples reported separately: `5`
- question_type distribution: `{'safety_or_constraint': 9, 'troubleshooting_step': 3, 'parameter_or_fault_code': 9, 'symptom_cause': 5, 'definition_or_component_lookup': 2, 'abstention_insufficient_evidence': 5, 'cross_doc_multi': 2}`

## 2. 总体指标表
| experiment_name | split | evaluated_samples | abstain_samples | faithfulness_supported_by_retrieved_mean | faithfulness_supported_by_retrieved_mean_ci_low_95 | faithfulness_supported_by_retrieved_mean_ci_high_95 | faithfulness_supported_by_cited_mean | faithfulness_supported_by_cited_mean_ci_low_95 | faithfulness_supported_by_cited_mean_ci_high_95 | correctness_strict_mean | correctness_strict_mean_ci_low_95 | correctness_strict_mean_ci_high_95 | correctness_lenient_mean | correctness_lenient_mean_ci_low_95 | correctness_lenient_mean_ci_high_95 | citation_precision_mean | citation_precision_mean_ci_low_95 | citation_precision_mean_ci_high_95 | citation_recall_mean | citation_recall_mean_ci_low_95 | citation_recall_mean_ci_high_95 | citation_f1_mean | citation_f1_mean_ci_low_95 | citation_f1_mean_ci_high_95 | hallucination_rate_mean | hallucination_rate_mean_ci_low_95 | hallucination_rate_mean_ci_high_95 | abstention_precision | abstention_recall | generator_model | judge_model | prompt_version | run_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_current_original_dev | dev | 35 | 5 | 0.8225 | 0.693333 | 0.9325 | 0.718294 | 0.600238 | 0.832698 | 0.3 | 0.133333 | 0.466667 | 0.383333 | 0.226667 | 0.546667 | 0.768519 | 0.635185 | 0.888889 | 0.836667 | 0.7 | 0.953333 | 0.785261 | 0.656111 | 0.891928 | 0.144167 | 0.051667 | 0.256667 | 1.0 | 1.0 | qwen3.6-flash | deepseek-v3.2 | v1.1 | 2026-05-08 |

## 3. 按 question_type 分组
| experiment_name | split | question_type | n_samples | faithfulness_supported_by_retrieved_mean | faithfulness_supported_by_retrieved_mean_ci_low_95 | faithfulness_supported_by_retrieved_mean_ci_high_95 | correctness_strict_mean | correctness_strict_mean_ci_low_95 | correctness_strict_mean_ci_high_95 | citation_f1_mean | citation_f1_mean_ci_low_95 | citation_f1_mean_ci_high_95 | hallucination_rate_mean | hallucination_rate_mean_ci_low_95 | hallucination_rate_mean_ci_high_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_current_original_dev | dev | cross_doc_multi | 2 | 0.95 | 0.9 | 1.0 | 0.0 | 0.0 | 0.0 | 0.833333 | 0.666667 | 1.0 | 0.05 | 0.0 | 0.1 |
| dense_current_original_dev | dev | definition_or_component_lookup | 2 | 1.0 | 1.0 | 1.0 | 0.5 | 0.0 | 1.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| dense_current_original_dev | dev | parameter_or_fault_code | 9 | 0.866667 | 0.644444 | 1.0 | 0.111111 | 0.0 | 0.333333 | 0.955556 | 0.911111 | 1.0 | 0.022222 | 0.0 | 0.066667 |
| dense_current_original_dev | dev | safety_or_constraint | 9 | 0.711111 | 0.433333 | 0.966667 | 0.555556 | 0.222222 | 0.888889 | 0.684205 | 0.425926 | 0.912854 | 0.288889 | 0.033333 | 0.566667 |
| dense_current_original_dev | dev | symptom_cause | 5 | 0.815 | 0.52 | 1.0 | 0.0 | 0.0 | 0.0 | 0.693333 | 0.333333 | 0.96 | 0.185 | 0.0 | 0.48 |
| dense_current_original_dev | dev | troubleshooting_step | 3 | 0.833333 | 0.5 | 1.0 | 0.666667 | 0.0 | 1.0 | 0.888889 | 0.666667 | 1.0 | 0.166667 | 0.0 | 0.5 |
The highest strict correctness group is `troubleshooting_step` and the lowest is `cross_doc_multi`, with a gap of `0.666667`.

## 4. 置信区间说明
Mean metrics use bootstrap confidence intervals with N=1000. The bootstrap is computed over answerable samples only for answer-quality metrics so that should_abstain=true samples do not dilute answerable performance.

## 5. 抽检结果
Five build samples were exported to `aiops-docs/experiment/results/answer/expanded/human_review_samples_build.csv` for manual spot-checking. Cohen's kappa is intentionally left for the manual review step after human labels are filled.

## 6. 诚实性 Caveat
The LLM judge is an offline proxy evaluator rather than ground truth. Results depend on the generator model version, judge model version, prompt hashes, API behavior, and temperature=0 setting used in this run. The should_abstain=true subset is evaluated as a separate abstention subset and is not mixed into answerable faithfulness, correctness, citation, or hallucination means. Because this build run contains only 35 samples and uses bootstrap N=1000, small differences between metrics should not be described as statistically significant without additional validation.
