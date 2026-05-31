# Agent Evaluation Report

Generated at: 2026-05-15T15:28:27.215093+00:00

## Inputs

- `aiops-docs/experiment/results/agent/agent_eval_A0_judge.jsonl`
- `aiops-docs/experiment/results/agent/agent_eval_A1_judge.jsonl`
- `aiops-docs/experiment/results/agent/agent_eval_A2_judge.jsonl`
- `aiops-docs/experiment/results/agent/agent_eval_A3_judge.jsonl`

## Main Results

| mode | n_cases | root_cause_accuracy_correct | root_cause_accuracy_partial | root_cause_accuracy_incorrect | evidence_completeness_mean | recommendation_correct | recommendation_partial | recommendation_incorrect | tool_precision_mean | tool_recall_mean | tool_call_count_mean | executed_steps_mean | replan_count_mean | latency_ms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 35 | 2 | 29 | 4 | 0.135143 | 1 | 21 | 13 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15308.771429 |
| A1 | 35 | 21 | 12 | 2 | 0.658477 | 13 | 18 | 4 | 0.971429 | 0.371403 | 1.142857 | 1.0 | 0.0 | 37912.628571 |
| A2 | 35 | 32 | 3 | 0 | 0.853429 | 16 | 19 | 0 | 0.942857 | 1.0 | 4.342857 | 3.0 | 1.0 | 31226.6 |
| A3 | 35 | 33 | 2 | 0 | 0.893963 | 14 | 21 | 0 | 0.971429 | 1.0 | 3.257143 | 3.0 | 1.0 | 32198.171429 |

## By Fault Type

CSV: `aiops-docs/experiment/results/thesis_tables/agent/agent_eval_by_fault_type.csv`

| fault_type | mode | n_cases | root_cause_accuracy_correct | root_cause_accuracy_partial | root_cause_accuracy_incorrect | evidence_completeness_mean | recommendation_correct | recommendation_partial | recommendation_incorrect | tool_precision_mean | tool_recall_mean | tool_call_count_mean | executed_steps_mean | replan_count_mean | latency_ms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu_overload | A0 | 6 | 0 | 6 | 0 | 0.055 | 0 | 4 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 12954.0 |
| cpu_overload | A1 | 6 | 3 | 3 | 0 | 0.583333 | 0 | 6 | 0 | 1.0 | 0.388867 | 1.166667 | 1.0 | 0.0 | 40570.833333 |
| cpu_overload | A2 | 6 | 6 | 0 | 0 | 0.945 | 2 | 4 | 0 | 0.958333 | 1.0 | 4.333333 | 3.0 | 1.0 | 31030.333333 |
| cpu_overload | A3 | 6 | 6 | 0 | 0 | 0.88945 | 2 | 4 | 0 | 1.0 | 1.0 | 3.333333 | 3.0 | 1.0 | 29023.333333 |
| disk_io_abnormal | A0 | 5 | 0 | 4 | 1 | 0.066 | 0 | 3 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15603.0 |
| disk_io_abnormal | A1 | 5 | 4 | 1 | 0 | 0.79934 | 3 | 2 | 0 | 0.8 | 0.39998 | 1.4 | 1.0 | 0.0 | 56474.6 |
| disk_io_abnormal | A2 | 5 | 5 | 0 | 0 | 0.934 | 2 | 3 | 0 | 0.9 | 1.0 | 4.2 | 3.0 | 1.0 | 34329.4 |
| disk_io_abnormal | A3 | 5 | 5 | 0 | 0 | 0.8672 | 2 | 3 | 0 | 0.95 | 1.0 | 3.2 | 3.0 | 1.0 | 30661.4 |
| equipment_alarm | A0 | 5 | 1 | 3 | 1 | 0.214 | 1 | 2 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 14488.8 |
| equipment_alarm | A1 | 5 | 1 | 3 | 1 | 0.374 | 1 | 3 | 1 | 1.0 | 0.3333 | 1.0 | 1.0 | 0.0 | 34797.2 |
| equipment_alarm | A2 | 5 | 3 | 2 | 0 | 0.768 | 3 | 2 | 0 | 1.0 | 1.0 | 4.8 | 3.0 | 1.0 | 30927.0 |
| equipment_alarm | A3 | 5 | 4 | 1 | 0 | 0.894 | 3 | 2 | 0 | 1.0 | 1.0 | 3.2 | 3.0 | 1.0 | 33205.2 |
| memory_leak | A0 | 6 | 0 | 4 | 2 | 0.0 | 0 | 4 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15682.833333 |
| memory_leak | A1 | 6 | 2 | 4 | 0 | 0.728333 | 3 | 2 | 1 | 1.0 | 0.444433 | 1.333333 | 1.0 | 0.0 | 35271.833333 |
| memory_leak | A2 | 6 | 5 | 1 | 0 | 0.778333 | 2 | 4 | 0 | 0.958333 | 1.0 | 4.166667 | 3.0 | 1.0 | 33087.5 |
| memory_leak | A3 | 6 | 6 | 0 | 0 | 0.801 | 1 | 5 | 0 | 0.916667 | 1.0 | 3.333333 | 3.0 | 1.0 | 30721.333333 |
| response_latency | A0 | 6 | 1 | 5 | 0 | 0.166667 | 0 | 4 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 14514.666667 |
| response_latency | A1 | 6 | 4 | 1 | 1 | 0.528333 | 1 | 3 | 2 | 1.0 | 0.3333 | 1.0 | 1.0 | 0.0 | 38632.833333 |
| response_latency | A2 | 6 | 6 | 0 | 0 | 0.78 | 3 | 3 | 0 | 0.833333 | 1.0 | 4.333333 | 3.0 | 1.0 | 29812.166667 |
| response_latency | A3 | 6 | 5 | 1 | 0 | 0.945 | 3 | 3 | 0 | 0.958333 | 1.0 | 3.5 | 3.0 | 1.0 | 31307.666667 |
| service_unavailable | A0 | 7 | 0 | 7 | 0 | 0.285714 | 0 | 4 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 18062.714286 |
| service_unavailable | A1 | 7 | 7 | 0 | 0 | 0.877143 | 5 | 2 | 0 | 1.0 | 0.3333 | 1.0 | 1.0 | 0.0 | 26247.142857 |
| service_unavailable | A2 | 7 | 7 | 0 | 0 | 0.905714 | 4 | 3 | 0 | 1.0 | 1.0 | 4.285714 | 3.0 | 1.0 | 29009.857143 |
| service_unavailable | A3 | 7 | 7 | 0 | 0 | 0.952857 | 3 | 4 | 0 | 1.0 | 1.0 | 3.0 | 3.0 | 1.0 | 37327.0 |

## By RAG Relevance

CSV: `aiops-docs/experiment/results/thesis_tables/agent/agent_eval_by_rag_relevant.csv`

| rag_relevant | mode | n_cases | root_cause_accuracy_correct | root_cause_accuracy_partial | root_cause_accuracy_incorrect | evidence_completeness_mean | recommendation_correct | recommendation_partial | recommendation_incorrect | tool_precision_mean | tool_recall_mean | tool_call_count_mean | executed_steps_mean | replan_count_mean | latency_ms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true | A2 | 5 | 3 | 2 | 0 | 0.768 | 3 | 2 | 0 | 1.0 | 1.0 | 4.8 | 3.0 | 1.0 | 30927.0 |
| true | A3 | 5 | 4 | 1 | 0 | 0.894 | 3 | 2 | 0 | 1.0 | 1.0 | 3.2 | 3.0 | 1.0 | 33205.2 |
| false | A2 | 30 | 29 | 1 | 0 | 0.867667 | 13 | 17 | 0 | 0.933333 | 1.0 | 4.266667 | 3.0 | 1.0 | 31276.533333 |
| false | A3 | 30 | 29 | 1 | 0 | 0.893957 | 11 | 19 | 0 | 0.966667 | 1.0 | 3.266667 | 3.0 | 1.0 | 32030.333333 |
