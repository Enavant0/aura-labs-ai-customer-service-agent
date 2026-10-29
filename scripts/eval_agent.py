"""Agent Eval V1：实际调用现有 Agent，确定性评分并逐题保存结果。"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter, sleep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_rag import csv_row, select_cases
from src.agent_eval_rules import PATHS, SCORING_VERSION, TOOL_NAMES, parse_arguments, score_result
from src.agent_service import AgentService
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, MAX_API_ATTEMPTS, RETRYABLE_STATUS_CODES
from src.embeddings import MODEL_NAME, load_model
from src.eval_rules import parse_concept_groups
from src.rag_service import RagService
from src.retriever import open_existing_collection


INPUT_FIELDS = ["case_id", "category", "user_question", "expected_path", "expected_tool",
                "expected_arguments", "expected_answer_points", "must_not_include"]
OUTPUT_FIELDS = INPUT_FIELDS + [
    "actual_path", "actual_tools", "actual_arguments", "tool_results", "tool_executions",
    "retrieved_sources", "answer", "latency", "api_attempts", "api_successes", "api_calls",
    "api_successful_calls", "api_retries", "api_success", "error", "error_kind",
    "path_pass", "tool_pass", "argument_pass", "no_unexpected_tool_pass", "handoff_pass",
    "final_answer_pass", "result", "failure_reason", "rule_details",
]


def load_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Agent 测试集不存在：{path}")
    cases, seen = [], set()
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            required = set(INPUT_FIELDS) - {"must_not_include"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("CSV 表头必须包含：" + ", ".join(sorted(required)))
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("CSV 存在重复表头。")
            for row in reader:
                if None in row or any(row.get(key) is None for key in required):
                    raise ValueError(f"CSV 第 {reader.line_num} 行列数不正确，请检查逗号和引号。")
                if not any(value and value.strip() for value in row.values()):
                    continue
                case = {key: (row.get(key) or "").strip() for key in INPUT_FIELDS}
                for key in ("case_id", "category", "user_question", "expected_path", "expected_answer_points"):
                    if not case[key]:
                        raise ValueError(f"CSV 第 {reader.line_num} 行的 {key} 不能为空。")
                if case["case_id"] in seen:
                    raise ValueError("case_id 重复：" + case["case_id"])
                if case["expected_path"] not in PATHS:
                    raise ValueError(case["case_id"] + " 的 expected_path 无效。")
                if case["expected_path"] == "TOOL":
                    if case["expected_tool"] not in TOOL_NAMES:
                        raise ValueError(case["case_id"] + " 的 expected_tool 无效。")
                    arguments = parse_arguments(case["expected_arguments"])
                    if arguments["order_id"] not in case["user_question"]:
                        raise ValueError(case["case_id"] + " 的预期订单号没有出现在用户问题中。")
                elif case["expected_tool"] or case["expected_arguments"]:
                    raise ValueError(case["case_id"] + " 的非工具题必须留空工具和参数。")
                parse_concept_groups(case["expected_answer_points"])
                seen.add(case["case_id"])
                cases.append(case)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("请检查测试集是否为正确的 UTF-8 CSV 文件。") from exc
    if not cases:
        raise ValueError("Agent 测试集没有有效案例。")
    return cases


class RecordingClient:
    """原样转发请求，只保留轨迹；最终 API 失败时仍能看到已经选择的工具。"""

    def __init__(self, client):
        self.client = client
        self.reset()

    def reset(self):
        self.path = ""
        self.executions = []
        self.sources = []
        self.api_attempts = 0
        self.api_successes = 0
        self.api_calls = 0
        self.api_successful_calls = 0
        self.decision_api_failed = False

    def _call_api(self, function, *args, **kwargs):
        # 客户端计数的是实际 HTTP 尝试；代理计数的是决策/生成等逻辑调用。
        before_attempts = getattr(self.client, "api_attempts", None)
        before_successes = getattr(self.client, "api_successes", None)
        self.api_calls += 1
        succeeded = False
        try:
            result = function(*args, **kwargs)
            succeeded = True
            self.api_successful_calls += 1
            return result
        finally:
            after_attempts = getattr(self.client, "api_attempts", None)
            after_successes = getattr(self.client, "api_successes", None)
            # 离线假客户端没有底层计数器，按单次调用记录，方便独立检查。
            self.api_attempts += (after_attempts - before_attempts
                                  if type(before_attempts) is int and type(after_attempts) is int else 1)
            self.api_successes += (after_successes - before_successes
                                   if type(before_successes) is int and type(after_successes) is int else int(succeeded))

    def chat_with_tools(self, messages, tools, tool_choice="auto"):
        if tool_choice == "none":
            results = {item["tool_call_id"]: json.loads(item["content"])
                       for item in messages if item["role"] == "tool"}
            for item in self.executions:
                if item["call_id"] in results:
                    item["result"] = results[item["call_id"]]
                    item["executed"] = True
        try:
            response = self._call_api(self.client.chat_with_tools, messages, tools, tool_choice=tool_choice)
        except DeepSeekRequestError:
            if tool_choice == "auto":
                self.decision_api_failed = True
            raise
        if tool_choice == "auto":
            if response.tool_calls:
                self.path = "TOOL"
                for call in response.tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"_invalid_json": call.function.arguments}
                    self.executions.append({"call_id": call.id, "name": call.function.name,
                                            "arguments": arguments, "result": {}, "executed": False})
            else:
                try:
                    route = json.loads(response.content)
                    self.path = route.get("route", "") if isinstance(route, dict) else ""
                except (json.JSONDecodeError, TypeError):
                    pass
        return response

    def generate_answer(self, system_prompt, user_content):
        try:
            context = json.loads(user_content)
            self.sources = [{key: chunk[key] for key in ("source_file", "heading", "chunk_id")}
                            for chunk in context.get("knowledge_chunks", [])]
        except (ValueError, TypeError, KeyError):
            pass
        return self._call_api(self.client.generate_answer, system_prompt, user_content)


def evaluate_one(case: dict, service, recorder: RecordingClient) -> dict:
    recorder.reset()
    started = perf_counter()
    answer, path, executions, sources, error, error_kind = "", "", [], [], "", ""
    try:
        result = service.answer_question(case["user_question"])
        answer, path = result.answer, result.path
        executions = [{**asdict(item), "executed": True} for item in result.tool_executions]
        sources = list(result.sources)
        if not answer.strip() or recorder.api_calls != recorder.api_successful_calls:
            error = "Agent 没有取得完整有效的回答。"
        if path == "RAG" and recorder.api_calls < 2:
            error = "RAG 未调用最终生成模型，未完成端到端链路。"
    except Exception as exc:
        error = str(exc)
        error_kind = "API" if isinstance(exc, DeepSeekRequestError) else "CHAIN"
        path, executions, sources = recorder.path, recorder.executions, recorder.sources
    if error and not error_kind:
        error_kind = "CHAIN"
    score = score_result(case, path, executions, answer, error, decision_api_failed=recorder.decision_api_failed)
    return {**case, "actual_path": path, "actual_tools": [item["name"] for item in executions],
            "actual_arguments": [item["arguments"] for item in executions],
            "tool_results": [item["result"] for item in executions], "tool_executions": executions,
            "retrieved_sources": sources, "answer": answer, "latency": round(perf_counter() - started, 4),
            "api_attempts": recorder.api_attempts, "api_successes": recorder.api_successes,
            "api_calls": recorder.api_calls, "api_successful_calls": recorder.api_successful_calls,
            "api_retries": recorder.api_attempts - recorder.api_calls,
            "api_success": not error and recorder.api_calls > 0, "error": error, "error_kind": error_kind, **score}


def metric(rows: list[dict], key: str) -> dict:
    applicable = [row for row in rows if row[key] is not None]
    passed = sum(bool(row[key]) for row in applicable)
    return {"passed": passed, "total": len(applicable), "accuracy": passed / len(applicable) if applicable else None}


def build_summary(rows: list[dict], planned: int, dataset_total: int) -> dict:
    tool_cases = [row for row in rows if row["expected_path"] == "TOOL"]
    return {
        "total_cases": planned, "dataset_total": dataset_total, "completed_cases": len(rows),
        "complete": len(rows) == planned, "full_dataset_evaluated": len(rows) == planned == dataset_total,
        "successful_cases": sum(row["api_success"] for row in rows),
        "api_requests": sum(row["api_attempts"] for row in rows),
        "api_successful_requests": sum(row["api_successes"] for row in rows),
        "api_logical_calls": sum(row.get("api_calls", row["api_attempts"]) for row in rows),
        "api_retries": sum(row.get("api_retries", 0) for row in rows),
        "api_error_cases": sum(row.get("error_kind") == "API" for row in rows),
        "decision_api_unavailable": sum(row["path_pass"] is None for row in rows),
        "path_error_case_ids": [row["case_id"] for row in rows if row["path_pass"] is False],
        "metrics": {
            "path_accuracy": metric(rows, "path_pass"),
            "tool_selection_accuracy": metric(tool_cases, "tool_pass"),
            "argument_accuracy": metric(tool_cases, "argument_pass"),
            "handoff_accuracy": metric(rows, "handoff_pass"),
            "final_answer_pass_rate": metric(rows, "final_answer_pass"),
            "no_unexpected_tool_accuracy": metric(rows, "no_unexpected_tool_pass"),
        },
        "overall": {"passed": sum(row["result"] == "pass" for row in rows), "total": len(rows),
                    "pass_rate": sum(row["result"] == "pass" for row in rows) / len(rows) if rows else None},
        "by_category": {
            category: {"total": len(group), "passed": sum(row["result"] == "pass" for row in group),
                       "pass_rate": sum(row["result"] == "pass" for row in group) / len(group)}
            for category in sorted({row["category"] for row in rows})
            for group in [[row for row in rows if row["category"] == category]]
        },
        "failure_case_ids": [row["case_id"] for row in rows if row["result"] == "fail"],
    }


def print_report(summary: dict, rows: list[dict], output_dir: Path):
    print(f"\nAgent Eval V1：总案例 {summary['total_cases']}，已完成 {summary['completed_cases']}，成功链路 {summary['successful_cases']}")
    labels = {"path_accuracy": "Path Accuracy / 路径选择",
              "tool_selection_accuracy": "Tool Selection Accuracy / 工具选择",
              "argument_accuracy": "Argument Accuracy / 参数提取", "handoff_accuracy": "Handoff Accuracy / 人工路径",
              "final_answer_pass_rate": "Final Answer Pass Rate / 最终回答",
              "no_unexpected_tool_accuracy": "No Unexpected Tool / 非工具题不乱调用"}
    for key, label in labels.items():
        stat = summary["metrics"][key]
        rate = f"{stat['accuracy']:.2%}" if stat["accuracy"] is not None else "不适用"
        print(f"{label}：{stat['passed']}/{stat['total']} ({rate})")
    print(f"整题全部通过：{summary['overall']['passed']}/{summary['overall']['total']}")
    print(f"DeepSeek 实际请求次数：{summary['api_requests']}，其中额外重试：{summary['api_retries']}")
    print(f"API 失败案例：{summary['api_error_cases']}；决策 API 失败、路径未评分：{summary['decision_api_unavailable']}")
    print("\n所有失败案例：")
    for row in rows:
        if row["result"] != "fail":
            continue
        print("------------------------------")
        for label, key in (("编号", "case_id"), ("问题", "user_question"), ("预期路径", "expected_path"),
                           ("实际路径", "actual_path"), ("预期工具", "expected_tool"),
                           ("实际工具", "actual_tools"), ("实际参数", "actual_arguments"),
                           ("工具返回", "tool_results"), ("预期回答要点", "expected_answer_points"),
                           ("最终回答", "answer"), ("失败原因", "failure_reason")):
            value = row[key]
            print(f"{label}：{json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value}")
    if not summary["failure_case_ids"]:
        print("无。")
    if not summary["full_dataset_evaluated"]:
        print("注意：当前是部分结果，不能作为完整 40 条基线。")
    print("最终回答为固定规则通过率，需要人工复核，不代表完整语义正确率。")
    print(f"\n结果：{output_dir / 'agent_eval_results.csv'}\n汇总：{output_dir / 'agent_eval_summary.json'}")


def run_evaluation(cases, service, recorder, output_dir, interval, dataset_total, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, interrupted = [], False
    try:
        with (output_dir / "agent_eval_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            file.flush()
            for index, case in enumerate(cases):
                if index and interval:
                    sleep(interval)
                row = evaluate_one(case, service, recorder)
                writer.writerow(csv_row(row))
                file.flush()  # 每题保存；后续请求失败不丢掉前面结果。
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == len(cases):
                    print(f"已完成：{len(rows)}/{len(cases)}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n已中断，保留已完成案例和部分统计。", flush=True)
    finally:
        summary = {**metadata, **build_summary(rows, len(cases), dataset_total), "interrupted": interrupted}
        (output_dir / "agent_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary, rows, output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent Eval V1：不改 Agent，只记录实际运行并规则评分。")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/agent_eval_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "eval")
    parser.add_argument("--interval", type=float, default=1.0, help="题目间隔秒数，默认 1 秒")
    parser.add_argument("--limit", type=int, help="仅运行前 N 条")
    parser.add_argument("--case-ids", nargs="+", help="仅运行指定编号，例如 AG-009 AG-034")
    args = parser.parse_args()
    client = None
    try:
        if not math.isfinite(args.interval) or args.interval < 0:
            raise ValueError("--interval 必须是大于等于 0 的有限秒数。")
        dataset = load_dataset(args.dataset)
        cases = select_cases(dataset, args.case_ids, args.limit)
        # 先校验密钥，再连接原有库；不建库、不更换向量化或检索方式。
        client = DeepSeekClient()
        collection = open_existing_collection(PROJECT_ROOT / "chroma_db")
        recorder = RecordingClient(client)
        service = AgentService(recorder, RagService(collection, recorder))
        print(f"Agent Eval V1：本次 {len(cases)}/{len(dataset)} 条，题目间隔 {args.interval} 秒。", flush=True)
        print("每题独立输入，不向 Agent 提供标准路径、标准工具、参数或答案。", flush=True)
        print(f"正在加载原有向量模型：{MODEL_NAME}", flush=True)
        load_model()
        tracked = ["src/agent_service.py", "src/tools.py", "src/deepseek_client.py", "src/rag_service.py",
                   "src/retriever.py", "src/embeddings.py", "src/vector_store.py", "src/agent_eval_rules.py",
                   "src/eval_rules.py", "scripts/eval_agent.py"]
        metadata = {
            "agent_version": "1.1", "dataset_version": "agent-eval-v1", "scoring_version": SCORING_VERSION,
            "answer_rules_version": "agent-rules-v1（必要概念及禁含逻辑保持不变）",
            "retry_policy": {"max_attempts_per_request": MAX_API_ATTEMPTS, "wait_seconds": [1, 2],
                             "http_status_codes": sorted(RETRYABLE_STATUS_CODES), "timeout": True},
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_case_ids": [case["case_id"] for case in cases],
            "dataset_path": str(args.dataset.resolve()),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "file_sha256": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in tracked},
            "knowledge_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                 for file in sorted((PROJECT_ROOT / "knowledge_base").glob("*.md"))},
            "deepseek_model": client.model, "embedding_model": MODEL_NAME, "top_k": 5,
            "collection_name": collection.name, "chunk_count": collection.count(), "interval_seconds": args.interval,
            "metric_definitions": {
                "path_accuracy": "路径严格相等 / 可取得决策的案例；决策 API 失败记 null 并单独计数，模型无效决策仍失败",
                "tool_selection_accuracy": "调用列表恰好只有预期工具 / 可评分的 TOOL 案例；决策 API 失败除外，漏调、多调均失败",
                "argument_accuracy": "选对唯一工具且参数字典完全相等 / 可评分的 TOOL 案例；决策 API 失败除外，检查类型和额外字段",
                "handoff_accuracy": "实际 HANDOFF 且无工具调用 / 可评分的预期 HANDOFF 案例；包括补充信息，决策 API 失败除外",
                "final_answer_pass_rate": "回答正文满足必要概念、禁含和有限工具事实校验且链路成功 / 全部已完成案例",
                "no_unexpected_tool_accuracy": "无工具调用 / 可评分的预期 RAG 或 HANDOFF 案例；决策 API 失败除外",
                "overall": "所有适用检查均通过 / 全部已完成案例；技术失败也记失败",
            },
            "limitations": "回答仍为 V1 固定关键词及有限事实校验；不放宽 AG-006/AG-037。V1.1 仅加强工具优先级、重试及技术失败计数。",
        }
        summary = run_evaluation(cases, service, recorder, args.output_dir, args.interval, len(dataset), metadata)
        return 130 if summary["interrupted"] else (0 if summary["successful_cases"] else 1)
    except KeyboardInterrupt:
        print("\n启动已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nAgent 评测无法运行：{exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
