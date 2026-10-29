"""Agent 评测器离线检查：不加载 BGE、不访问 Chroma、不请求 DeepSeek。"""

from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletionMessage

from scripts.eval_agent import (PROJECT_ROOT, RecordingClient, build_summary, evaluate_one,
                                load_dataset, run_evaluation, select_cases)
from src.agent_eval_rules import score_result
from src.agent_service import AgentService
from src.tools import get_order_status, get_refund_status


DATASET = load_dataset(PROJECT_ROOT / "eval/agent_eval_dataset.csv")
CASES = {case["case_id"]: case for case in DATASET}


def execution(name="get_order_status", order_id="123456"):
    function = get_order_status if name == "get_order_status" else get_refund_status
    return {"name": name, "arguments": {"order_id": order_id}, "result": function(order_id), "executed": True}


def message(content=None, calls=None):
    return ChatCompletionMessage.model_validate({"role": "assistant", "content": content, "tool_calls": calls})


def call():
    return {"id": "order_call", "type": "function", "function": {
        "name": "get_order_status", "arguments": '{"order_id":"123456"}'}}


class AgentScoringTests(unittest.TestCase):
    def test_dataset_size_and_categories(self):
        self.assertEqual(len(DATASET), 40)
        self.assertEqual(len(CASES), 40)
        counts = {category: sum(case["category"] == category for case in DATASET)
                  for category in {case["category"] for case in DATASET}}
        self.assertEqual(counts, {"RAG": 8, "订单工具": 8, "退款工具": 7, "不存在订单": 4,
                                 "HANDOFF": 6, "参数不足": 3, "对抗测试": 4})
        self.assertEqual(sum(case["expected_path"] == "TOOL" for case in DATASET), 20)

    def test_correct_minimal_rag_and_no_tool(self):
        score = score_result(CASES["AG-001"], "RAG", [], "洁面日常售价为89元。")
        self.assertEqual(score["result"], "pass")
        self.assertIsNone(score["argument_pass"])
        self.assertTrue(score["no_unexpected_tool_pass"])

    def test_extra_tool_fails_even_with_correct_rag_answer(self):
        score = score_result(CASES["AG-001"], "RAG", [execution()], "售价89元。")
        self.assertEqual(score["result"], "fail")
        self.assertFalse(score["no_unexpected_tool_pass"])
        self.assertTrue(score["final_answer_pass"])

    def test_tool_and_argument_exact_match(self):
        score = score_result(CASES["AG-009"], "TOOL", [execution()], "订单位于广州转运中心。")
        self.assertEqual(score["result"], "pass")
        for value in (123456, "123457"):
            item = execution()
            item["arguments"]["order_id"] = value
            self.assertFalse(score_result(CASES["AG-009"], "TOOL", [item], "广州转运中心")["argument_pass"])
        item = execution()
        item["arguments"]["extra"] = True
        self.assertFalse(score_result(CASES["AG-009"], "TOOL", [item], "广州转运中心")["argument_pass"])

    def test_wrong_tool_or_two_calls_fail(self):
        case = CASES["AG-009"]
        for items in ([execution("get_refund_status")], [execution(), execution()]):
            score = score_result(case, "TOOL", items, "广州转运中心")
            self.assertFalse(score["tool_pass"])
            self.assertFalse(score["argument_pass"])

    def test_required_concepts_support_or_and_negation(self):
        case = dict(CASES["AG-009"], expected_answer_points="广州|广州市;运输中|在途")
        self.assertTrue(score_result(case, "TOOL", [execution()], "目前在广州市，在途运输中。")["final_answer_pass"])
        self.assertFalse(score_result(case, "TOOL", [execution()], "不在广州，目前运输中。")["final_answer_pass"])
        self.assertFalse(score_result(case, "TOOL", [execution()], "广州。")["final_answer_pass"])

    def test_refund_not_credited_and_false_positive_assertion(self):
        items = [execution("get_refund_status")]
        good = "退款尚未到账，仍在处理中。"
        self.assertTrue(score_result(CASES["AG-018"], "TOOL", items, good)["final_answer_pass"])
        bad = good + "但是已经到账，退款成功了。"
        self.assertFalse(score_result(CASES["AG-018"], "TOOL", items, bad)["final_answer_pass"])

    def test_refund_amount_contradiction(self):
        items = [execution("get_refund_status")]
        self.assertTrue(score_result(CASES["AG-019"], "TOOL", items, "退款金额89.00元。")["final_answer_pass"])
        self.assertFalse(score_result(CASES["AG-019"], "TOOL", items, "金额89元，另有159元。")["final_answer_pass"])

    def test_unknown_order_cannot_add_carrier_or_location(self):
        items = [execution(order_id="999999")]
        self.assertTrue(score_result(CASES["AG-024"], "TOOL", items, "未找到订单，请核对编号。")["final_answer_pass"])
        for extra in ("目前在广州转运中心。", "使用顺丰。", "已发货。", "退款89元。", "单号MOCK-SF-999999。"):
            self.assertFalse(score_result(CASES["AG-024"], "TOOL", items, "未找到订单。" + extra)["final_answer_pass"])

    def test_wrong_carrier_or_tracking_fails(self):
        for text in ("广州转运中心，圆通承运。", "广州转运中心，单号MOCK-SF-999999。"):
            self.assertFalse(score_result(CASES["AG-009"], "TOOL", [execution()], text)["final_answer_pass"])

    def test_program_prefix_is_not_handoff_evidence(self):
        answer = "【需要人工处理；当前演示未实际接入人工客服】\n好的。"
        score = score_result(CASES["AG-028"], "HANDOFF", [], answer)
        self.assertTrue(score["handoff_pass"])
        self.assertFalse(score["final_answer_pass"])

    def test_no_false_operation_and_negation_exemption(self):
        case = CASES["AG-028"]
        self.assertFalse(score_result(case, "HANDOFF", [], "已为您转接人工客服。")["final_answer_pass"])
        self.assertTrue(score_result(case, "HANDOFF", [], "不能声称已转接人工，建议联系人工客服。")["final_answer_pass"])

    def test_missing_order_id_must_request_and_never_invent(self):
        case = CASES["AG-034"]
        self.assertEqual(score_result(case, "HANDOFF", [], "请提供订单号以便查询。")["result"], "pass")
        self.assertFalse(score_result(case, "HANDOFF", [], "请提供订单号，例如123456。")["final_answer_pass"])

    def test_technical_failure_cannot_pass_answer(self):
        score = score_result(CASES["AG-001"], "RAG", [], "89元", "请求超时")
        self.assertEqual(score["result"], "fail")
        self.assertFalse(score["final_answer_pass"])


class AgentEvaluationTests(unittest.TestCase):
    def test_select_cases_validation(self):
        self.assertEqual(select_cases(DATASET, ["AG-034", "AG-009"], 1)[0]["case_id"], "AG-009")
        for ids, limit in ((["missing"], None), (["AG-001", "AG-001"], None), (None, 0)):
            with self.assertRaises(ValueError):
                select_cases(DATASET, ids, limit)

    def test_recording_proxy_preserves_original_requests(self):
        client = Mock()
        client.chat_with_tools.side_effect = [message(calls=[call()]), message("位于广州转运中心。")]
        recorder = RecordingClient(client)
        row = evaluate_one(CASES["AG-009"], AgentService(recorder, Mock()), recorder)
        self.assertEqual(row["result"], "pass")
        self.assertEqual(row["api_attempts"], 2)
        self.assertEqual(row["tool_results"][0], get_order_status("123456"))
        self.assertEqual(client.chat_with_tools.call_args_list[0].kwargs, {"tool_choice": "auto"})
        self.assertEqual(client.chat_with_tools.call_args_list[1].kwargs, {"tool_choice": "none"})

    def test_final_api_failure_keeps_path_tool_arguments_results(self):
        client = Mock()
        client.chat_with_tools.side_effect = [message(calls=[call()]), RuntimeError("请求超时")]
        recorder = RecordingClient(client)
        row = evaluate_one(CASES["AG-009"], AgentService(recorder, Mock()), recorder)
        self.assertEqual(row["actual_path"], "TOOL")
        self.assertTrue(row["argument_pass"])
        self.assertTrue(row["tool_results"][0]["found"])
        self.assertFalse(row["final_answer_pass"])
        self.assertFalse(row["api_success"])
        self.assertEqual(row["api_successes"], 1)

    def test_recorder_resets_for_each_case(self):
        client = Mock()
        client.chat_with_tools.side_effect = [message(calls=[call()]), message("广州转运中心"), RuntimeError("首请求失败")]
        recorder = RecordingClient(client)
        service = AgentService(recorder, Mock())
        evaluate_one(CASES["AG-009"], service, recorder)
        row = evaluate_one(CASES["AG-028"], service, recorder)
        self.assertEqual(row["actual_tools"], [])
        self.assertEqual(row["actual_path"], "")
        self.assertEqual(row["api_attempts"], 1)

    def test_summary_denominators(self):
        rows = []
        examples = (("AG-001", "RAG", [], "89元"),
                    ("AG-009", "TOOL", [execution()], "广州转运中心"),
                    ("AG-028", "RAG", [], "建议联系人工客服。"))
        for key, path, items, answer in examples:
            rows.append({**CASES[key], **score_result(CASES[key], path, items, answer),
                         "api_success": True, "api_attempts": 2, "api_successes": 2})
        summary = build_summary(rows, 40, 40)
        self.assertEqual(summary["metrics"]["path_accuracy"]["passed"], 2)
        self.assertEqual(summary["metrics"]["tool_selection_accuracy"]["total"], 1)
        self.assertEqual(summary["metrics"]["handoff_accuracy"]["accuracy"], 0)
        self.assertEqual(summary["metrics"]["final_answer_pass_rate"]["accuracy"], 1)
        self.assertEqual(summary["metrics"]["no_unexpected_tool_accuracy"]["total"], 2)
        self.assertFalse(summary["complete"])

    def test_failure_continues_csv_json_round_trip_and_progress(self):
        client = Mock()
        client.chat_with_tools.side_effect = [RuntimeError("模拟失败")] + [
            message('{"route":"HANDOFF","reason":"请求人工"}') for _ in range(10)]
        client.generate_answer.return_value = "建议联系人工客服。"
        recorder = RecordingClient(client)
        service = AgentService(recorder, Mock())
        cases = [dict(CASES["AG-028"], case_id=f"LOCAL-{i}") for i in range(11)]
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()) as output, patch("scripts.eval_agent.sleep") as pause:
            summary = run_evaluation(cases, service, recorder, Path(temp), 1, 40, {"test_only": True})
            self.assertEqual(pause.call_count, 10)
            self.assertEqual(summary["overall"]["passed"], 10)
            self.assertIn("已完成：10/11", output.getvalue())
            self.assertIn("已完成：11/11", output.getvalue())
            self.assertIn("LOCAL-0", summary["failure_case_ids"])
            with (Path(temp) / "agent_eval_results.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 11)
            self.assertEqual(json.loads(rows[0]["actual_tools"]), [])
            saved = json.loads((Path(temp) / "agent_eval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, summary)

    def test_interruption_retains_partial_results(self):
        client = Mock()
        client.chat_with_tools.side_effect = [message('{"route":"HANDOFF"}'), KeyboardInterrupt()]
        client.generate_answer.return_value = "请联系人工客服。"
        recorder = RecordingClient(client)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            summary = run_evaluation([CASES["AG-028"], CASES["AG-029"]], AgentService(recorder, Mock()),
                                     recorder, Path(temp), 0, 40, {})
            self.assertTrue(summary["interrupted"])
            self.assertEqual(summary["completed_cases"], 1)
            self.assertFalse(summary["full_dataset_evaluated"])

    def test_dataset_invalid_tool_is_rejected_offline(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / "bad.csv"
            fieldnames = list(CASES["AG-009"])
            with file.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(dict(CASES["AG-009"], expected_tool="delete_order"))
            with self.assertRaises(ValueError):
                load_dataset(file)


if __name__ == "__main__":
    unittest.main()
