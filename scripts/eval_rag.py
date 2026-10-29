"""端到端 RAG Eval V1：调用现有 RagService，逐题保存，规则评分。"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter, sleep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deepseek_client import DeepSeekClient
from src.embeddings import MODEL_NAME, load_model
from src.eval_rules import ACTIONS, SCORING_VERSION, parse_concept_groups, score_answer
from src.rag_service import RagService, SYSTEM_PROMPT
from src.retriever import open_existing_collection


INPUT_FIELDS = ["case_id", "category", "user_question", "expected_action", "expected_sources",
                "reference_answer", "must_include", "must_not_include"]
OUTPUT_FIELDS = INPUT_FIELDS + [
    "retrieved_sources", "answer", "latency", "api_called", "api_success", "error",
    "must_include_pass", "must_not_include_pass", "missing_keywords", "forbidden_hits",
    "action_pass", "action_evidence", "rule_details", "result", "failure_reason",
]


def load_dataset(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"测试集不存在：{path}。请先放入 eval/eval_dataset.csv。")
    cases = []
    seen = set()
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            if not reader.fieldnames or not set(INPUT_FIELDS).issubset(reader.fieldnames):
                raise ValueError("CSV 表头必须包含：" + ", ".join(INPUT_FIELDS))
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("CSV 表头存在重复列名。")
            for row in reader:
                if None in row:
                    raise ValueError(f"CSV 第 {reader.line_num} 行列数过多，请检查字段中的逗号是否正确加引号。")
                if not any(value and value.strip() for value in row.values()):
                    continue
                case = {key: (row.get(key) or "").strip() for key in INPUT_FIELDS}
                if any(not case[key] for key in ("case_id", "category", "user_question", "expected_action")):
                    raise ValueError(f"CSV 第 {reader.line_num} 行的编号、分类、问题或预期行动为空。")
                if case["case_id"] in seen:
                    raise ValueError(f"case_id 重复：{case['case_id']}")
                if case["expected_action"] not in ACTIONS:
                    raise ValueError(f"{case['case_id']} 的 expected_action 不受支持：{case['expected_action']}")
                try:
                    parse_concept_groups(case["must_include"])
                except ValueError as exc:
                    raise ValueError(f"{case['case_id']} 的 must_include 格式错误：{exc}") from exc
                cases.append(case)
                seen.add(case["case_id"])
    except UnicodeError as exc:
        raise ValueError("测试集必须为 UTF-8 编码，请重新保存后运行。") from exc
    except csv.Error as exc:
        raise ValueError(f"CSV 格式错误：{exc}") from exc
    if not cases:
        raise ValueError("测试集没有有效案例。")
    return cases


def select_cases(dataset: list[dict], case_ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """先按编号筛选（保留原 CSV 顺序），再取前 limit 条。参数错误时不调用 API。"""
    if limit is not None and limit < 1:
        raise ValueError("--limit 必须大于等于 1。")
    cases = dataset
    if case_ids is not None:
        if not case_ids:
            raise ValueError("--case-ids 至少需要一个编号。")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("--case-ids 存在重复编号，请删除重复项。")
        known = {case["case_id"] for case in dataset}
        unknown = [case_id for case_id in case_ids if case_id not in known]
        if unknown:
            raise ValueError("测试集中不存在这些 case_id：" + ", ".join(unknown))
        wanted = set(case_ids)
        cases = [case for case in dataset if case["case_id"] in wanted]
    return cases[:limit] if limit is not None else cases


class RecordingClient:
    """只记录调用边界，不修改 Prompt、上下文或现有客户端行为。"""

    def __init__(self, client):
        self.client = client
        self.reset()

    def reset(self) -> None:
        self.api_called = False
        self.api_success = False
        self.retrieved_sources = []

    def generate_answer(self, system_prompt: str, user_content: str) -> str:
        # 在请求前保留实际来源，因此 API 失败时也能分析检索结果。
        context = json.loads(user_content)
        self.retrieved_sources = [{key: chunk[key] for key in ("source_file", "heading", "chunk_id")}
                                  for chunk in context["knowledge_chunks"]]
        self.api_called = True
        answer = self.client.generate_answer(system_prompt, user_content)
        self.api_success = True
        return answer


def evaluate_one(case: dict[str, str], service, recorder: RecordingClient) -> dict:
    recorder.reset()
    started = perf_counter()
    answer = ""
    sources = []
    error = ""
    try:
        result = service.answer_question(case["user_question"])
        answer = result.answer
        sources = list(result.sources)
        if not recorder.api_called:
            error = "未调用 DeepSeek：RAG 未取得可用检索结果，本题不算完整端到端通过。"
    except Exception as exc:
        error = str(exc)
        sources = recorder.retrieved_sources
    latency = perf_counter() - started
    if not error:
        score = score_answer(case, answer)
    else:
        # 技术失败不做词面评分，避免把缺少答案误记成单纯的内容问题。
        score = {"must_include_pass": None, "must_not_include_pass": None, "missing_keywords": [],
                 "forbidden_hits": [], "action_pass": None, "action_evidence": {}, "rule_pass": False,
                 "rule_details": {"scoring_version": SCORING_VERSION, "evaluated": False, "reason": error,
                                  "must_include": [], "must_not_include": [],
                                  "expected_action": {"expected": case["expected_action"], "passed": None,
                                                      "mode": "not_evaluated"}},
                 "failure_reasons": ["调用链路失败：" + error]}
    return {
        **case, "retrieved_sources": sources, "answer": answer, "latency": round(latency, 4),
        "api_called": recorder.api_called, "api_success": recorder.api_success, "error": error,
        **{key: score[key] for key in ("must_include_pass", "must_not_include_pass", "missing_keywords",
                                     "forbidden_hits", "action_pass", "action_evidence", "rule_details")},
        "result": "pass" if score["rule_pass"] and recorder.api_success else "fail",
        "failure_reason": " | ".join(score["failure_reasons"]),
    }


def group_stats(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(row["result"] == "pass" for row in rows)
    return {"total": total, "passed": passed, "failed": total - passed,
            "api_success": sum(row["api_success"] for row in rows),
            "pass_rate": passed / total if total else None}


def build_summary(rows: list[dict], planned: int, dataset_total: int) -> dict:
    successful = [row for row in rows if row["api_success"]]
    categories = sorted({row["category"] for row in rows})
    return {
        "total_tests": planned,
        "dataset_total": dataset_total,
        "completed_tests": len(rows),
        "complete": len(rows) == planned,
        "full_dataset_evaluated": len(rows) == planned == dataset_total,
        "api_attempted": sum(row["api_called"] for row in rows),
        "successful_calls": len(successful),
        "api_failures": sum(row["api_called"] and not row["api_success"] for row in rows),
        "api_not_called": sum(not row["api_called"] for row in rows),
        "passed": sum(row["result"] == "pass" for row in rows),
        "overall_pass_rate": group_stats(rows)["pass_rate"],
        "pass_rate_among_successful_calls": group_stats(successful)["pass_rate"],
        "answer_quality_measured": bool(successful),
        "by_category": {category: group_stats([row for row in rows if row["category"] == category]) for category in categories},
        "by_group": {
            "正常回答": group_stats([row for row in rows if row["expected_action"] in {"ANSWER", "ANSWER_WITH_CAVEAT"}]),
            "拒答类": group_stats([row for row in rows if row["expected_action"] == "REFUSE"]),
            "转人工或业务系统类": group_stats([row for row in rows if row["expected_action"] in {"HANDOFF", "HANDOFF_OR_TOOL"}]),
            "安全边界类": group_stats([row for row in rows if row["category"] == "安全边界"]),
        },
        "failure_case_ids": [row["case_id"] for row in rows if row["result"] == "fail"],
    }


def csv_row(row: dict) -> dict:
    # 列表/字典保存为 JSON 字符串，Excel 可看，程序也能无损读回。
    return {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            for key, value in row.items()}


def run_evaluation(cases: list[dict], service, recorder: RecordingClient, output_dir: Path,
                   interval: float, dataset_total: int, metadata: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rag_eval_results.csv"
    summary_path = output_dir / "rag_eval_summary.json"
    rows = []
    interrupted = False
    try:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            file.flush()
            for index, case in enumerate(cases):
                if index and interval:
                    sleep(interval)
                row = evaluate_one(case, service, recorder)
                writer.writerow(csv_row(row))
                file.flush()  # 每题落盘，API 后续失败不会丢失已完成结果。
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == len(cases):
                    print(f"已完成：{len(rows)}/{len(cases)}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n评测已中断，正在保存已完成的部分结果。", flush=True)
    finally:
        summary = {**metadata, **build_summary(rows, len(cases), dataset_total), "interrupted": interrupted}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================================\nRAG Eval V1 汇总\n================================")
    print(f"总测试数：{len(cases)}，已完成：{len(rows)}")
    print(f"成功调用数：{summary['successful_calls']}，API 失败：{summary['api_failures']}")
    print(f"总体通过率：{format_rate(summary['overall_pass_rate'])}")
    print(f"仅成功调用中的规则通过率：{format_rate(summary['pass_rate_among_successful_calls'])}")
    for title, stats in (("按 category", summary["by_category"]), ("按测试类型", summary["by_group"])):
        print(f"\n{title}：")
        for label, stat in stats.items():
            print(f"- {label}：{stat['passed']}/{stat['total']}，通过率 {format_rate(stat['pass_rate'])}")
    print("\n所有失败案例：")
    for row in rows:
        if row["result"] == "fail":
            print("------------------------------")
            for label, key in (("case_id", "case_id"), ("问题", "user_question"), ("标准答案", "reference_answer"),
                               ("实际回答", "answer"), ("失败原因", "failure_reason")):
                print(f"{label}：{row[key] or '（无）'}")
            print("检索 Top 5 来源：" + json.dumps(row["retrieved_sources"], ensure_ascii=False))
    if not summary["failure_case_ids"]:
        print("无。")
    if not summary["full_dataset_evaluated"]:
        print("\n注意：本次没有完成整个测试集，不能作为完整 80 条基线。")
    if not summary["answer_quality_measured"]:
        print("\n注意：没有收到有效 DeepSeek 回答，尚未测得模型回答质量。")
    print(f"\n逐题结果：{csv_path}\n统计结果：{summary_path}")
    return summary


def format_rate(value) -> str:
    return f"{value:.2%}" if value is not None else "不适用（0 条）"


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端 RAG Eval V1，不修改现有 RAG 链路。")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/eval_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "eval/results")
    parser.add_argument("--interval", type=float, default=1.0, help="题目间隔秒数，默认 1 秒")
    parser.add_argument("--limit", type=int, help="仅运行前 N 条；结果明确标为部分测试")
    parser.add_argument("--case-ids", nargs="+", help="仅测试指定编号，例如 TC-029 TC-068；可配合 --limit")
    args = parser.parse_args()
    client = None
    try:
        if not math.isfinite(args.interval) or args.interval < 0:
            raise ValueError("--interval 必须是大于等于 0 的有限秒数。")
        dataset = load_dataset(args.dataset)
        cases = select_cases(dataset, args.case_ids, args.limit)
        client = DeepSeekClient()
        collection = open_existing_collection(PROJECT_ROOT / "chroma_db")
        recorder = RecordingClient(client)
        service = RagService(collection, recorder)
        print(f"RAG Eval V1：本次 {len(cases)}/{len(dataset)} 条，题目间隔 {args.interval} 秒。", flush=True)
        print(f"正在加载本地向量模型：{MODEL_NAME}", flush=True)
        load_model()
        metadata = {
            "scoring_version": SCORING_VERSION,
            "selected_case_ids": [case["case_id"] for case in cases],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(args.dataset.resolve()),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "deepseek_model": client.model, "embedding_model": MODEL_NAME,
            "top_k": 5, "interval_seconds": args.interval,
            "pass_rate_definition": "规则通过题数 / 已完成题数；API 或链路失败也计入失败",
            "safety_group_definition": "category 精确等于 安全边界；与行动类型组存在重叠",
        }
        summary = run_evaluation(cases, service, recorder, args.output_dir, args.interval, len(dataset), metadata)
        return 130 if summary["interrupted"] else (0 if summary["successful_calls"] else 1)
    except KeyboardInterrupt:
        print("\n启动已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n评测失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
