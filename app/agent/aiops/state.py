"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import List, TypedDict, Annotated
import operator


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""
    
    # 用户输入（任务描述）
    input: str
    
    # 执行计划（步骤列表）
    plan: List[str]
    
    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]
    
    # 最终响应/报告
    response: str

    # 是否启用 RAG 经验检索。默认由 AIOpsService.execute 设置为 True。
    rag_enabled: bool

    # 固定案例回放 ID，用于 MCP mock server replay mode。
    replay_case_id: str

    # 实验运行时模型配置。
    model_name: str
    temperature: float
