"""本地 JSONL 旁路日志：只记录运行摘要，写入失败不影响客服请求。"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock
from time import perf_counter
from uuid import uuid4

from src.deepseek_client import DeepSeekRequestError


LOG_PATH = Path(__file__).resolve().parents[1] / "logs/runtime.jsonl"
_WRITE_LOCK = Lock()


def get_api_attempts(client):
    """只读取现有计数器；缺少真实整数计数时返回 None，不伪造 0。"""
    try:
        value = client.api_attempts
        return value if type(value) is int and value >= 0 else None
    except Exception:
        return None


def _tool_summary(execution):
    # 当前两个工具仅需要数字订单号；不保存额外参数或完整工具返回值。
    order_id = execution.arguments.get("order_id")
    arguments = {}
    if isinstance(order_id, str) and re.fullmatch(r"[0-9]{6,20}", order_id):
        arguments["order_id"] = order_id
    return {
        "tool_name": execution.name if execution.name in {"get_order_status", "get_refund_status"} else None,
        "tool_arguments": arguments,
        "tool_found": execution.result.get("found") if type(execution.result.get("found")) is bool else None,
        "is_mock": execution.result.get("is_mock") if type(execution.result.get("is_mock")) is bool else None,
    }


def log_agent_request(interface, started, *, result=None, error=None, client=None, api_attempts_before=None):
    """每个实际提交写一行；不调用 Agent、不改变输入或结果，返回写入是否成功。"""
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": uuid4().hex,
            "interface": interface,
            "path": result.path if result is not None else None,
            "success": result is not None and error is None,
            # 请求结束时计时，包含初始化（如有）和 Agent 处理，不包含日志写盘。
            "elapsed_ms": round(max(0, perf_counter() - started) * 1000, 2),
        }
        if result is not None and result.path == "TOOL":
            tools = [_tool_summary(execution) for execution in result.tool_executions]
            if len(tools) == 1:
                record.update(tools[0])
            elif tools:
                record["tools"] = tools  # 同轮多个工具仍只写一条请求记录。
        if error is not None:
            record["error_type"] = type(error).__name__[:64]
            # 不复制异常原文，防止 SDK/配置异常携带密钥、路径或用户文本。
            if isinstance(error, DeepSeekRequestError):
                record["error_message"] = "DeepSeek 请求失败。"
            elif isinstance(error, ValueError):
                record["error_message"] = "配置或输入校验失败。"
            elif isinstance(error, OSError):
                record["error_message"] = "本地资源访问失败。"
            else:
                record["error_message"] = "Agent 请求未完成。"
        after = get_api_attempts(client)
        if type(api_attempts_before) is int and after is not None and after >= api_attempts_before:
            record["api_attempts"] = after - api_attempts_before
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _WRITE_LOCK:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as file:
                file.write(line)
        return True
    except Exception:
        # 磁盘、权限、序列化等日志问题都在旁路内结束，不覆盖正常回答或原始错误。
        return False
