# Agent 固定案例回放数据集 Schema v1

文件：`agent_eval_cases_v1.jsonl`

每行一条 JSON case，按 `case_id` 升序排列。该数据集用于 A0/A1/A2/A3 四种 Agent 模式的固定故障回放评测。

## 顶层字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `case_id` | string | 案例 ID，格式为 `agent_case_001`。 |
| `fault_type` | string | 故障类型：`cpu_overload`、`memory_leak`、`disk_io_abnormal`、`service_unavailable`、`response_latency`、`equipment_alarm`。 |
| `difficulty` | string | 难度：`single_cause`、`multi_cause`、`ambiguous`。 |
| `task_text` | string | 运维人员提交给 Agent 的自然语言任务描述。 |
| `service_name` | string | 故障目标服务名。 |
| `alert_payload` | object | 对齐 Monitor mock 指标返回体中的 `alert_info`。当前 mock 没有独立告警工具。 |
| `metrics_payload` | object | 对齐 `query_cpu_metrics` 或 `query_memory_metrics` 的返回体。 |
| `logs_payload` | object | 对齐 CLS `search_log` 的返回体。 |
| `historical_ticket_payload` | object | 当前 mock server 未提供历史工单工具，固定为空对象 `{}`。 |
| `gold_root_cause` | string | 标准根因结论，必须说明由哪些 payload 字段支撑。 |
| `gold_evidence` | string[] | 关键证据，使用 `payload.path=value` 形式，`value` 必须能在对应 payload 中找到。 |
| `gold_recommendation` | string | 可执行处置建议，操作对象和数值需来自 payload。 |
| `gold_expected_tools` | string[] | A2/A3 预期调用的核心工具名，用于 Tool Calls 指标评判。 |
| `case_notes` | string | 案例设计意图和难点，不参与实验评分。 |
| `mock_routing_key` | string | mock server replay mode 路由键，当前与 `case_id` 一致。 |
| `rag_relevant` | boolean | 是否与当前工业设备维修知识库直接相关。`agent_case_031` 到 `agent_case_035` 为 `true`，前 30 条软件服务案例为 `false`。 |

## `metrics_payload`

CPU 指标 payload 对齐 `mcp_servers/monitor_server.py::query_cpu_metrics`：

```json
{
  "service_name": "order-batch-service",
  "metric_name": "cpu_usage_percent",
  "interval": "1m",
  "data_points": [
    {
      "timestamp": "09:20",
      "value": 38.2,
      "process_id": "pid-91001"
    }
  ],
  "statistics": {
    "avg": 78.22,
    "max": 96.0,
    "min": 38.2,
    "p95": 96.0,
    "spike_detected": true
  },
  "alert_info": {
    "triggered": true,
    "threshold": 80.0,
    "message": "CPU 使用率持续超过 80% 阈值"
  }
}
```

内存指标 payload 对齐 `mcp_servers/monitor_server.py::query_memory_metrics`：

```json
{
  "service_name": "profile-api-service",
  "metric_name": "memory_usage_percent",
  "interval": "1m",
  "data_points": [
    {
      "timestamp": "09:35",
      "value": 45.0,
      "used_gb": 3.6,
      "total_gb": 8.0
    }
  ],
  "statistics": {
    "avg": 70.28,
    "max": 88.6,
    "min": 45.0,
    "p95": 88.6,
    "memory_pressure": true
  },
  "alert_info": {
    "triggered": true,
    "threshold": 70.0,
    "message": "内存使用率超过 70% 阈值，存在内存压力"
  }
}
```

## `alert_payload`

`alert_payload` 是 `metrics_payload.alert_info` 的扁平引用，字段固定为：

```json
{
  "triggered": true,
  "threshold": 80.0,
  "message": "CPU 使用率持续超过 80% 阈值"
}
```

## `logs_payload`

日志 payload 对齐 `mcp_servers/cls_server.py::search_log`：

```json
{
  "topic_id": "topic-001",
  "start_time": 1778808000000,
  "end_time": 1778808180000,
  "query": "level:ERROR OR level:WARN",
  "limit": 100,
  "total": 3,
  "logs": [
    {
      "timestamp": "2026-05-15 09:20:00",
      "level": "INFO",
      "message": "batch job started: job_id=sync-742 batch_size=5000"
    }
  ],
  "took_ms": 50,
  "message": "成功查询 3 条应用日志"
}
```

## Replay Mode

当工具参数包含 `replay_case_id` 时：

- `query_cpu_metrics` / `query_memory_metrics` 返回对应 case 的 `metrics_payload`。
- `search_log` 返回对应 case 的 `logs_payload`。
- `search_topic_by_service_name` 返回一个与对应 case `service_name` 和 `logs_payload.topic_id` 对齐的固定 topic 查询结果。

未传 `replay_case_id` 时，mock server 保持原有随机或默认行为。
