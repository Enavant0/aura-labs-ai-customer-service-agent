"""使用真实 SDK + 本地 MockTransport 检查重试，不发送网络请求。"""

from contextlib import contextmanager
import json
import unittest
from unittest.mock import Mock, call, patch

import httpx
from openai import OpenAI as RealOpenAI

from scripts.eval_agent import PROJECT_ROOT, RecordingClient, build_summary, evaluate_one, load_dataset
from src.agent_eval_rules import score_result
from src.agent_service import AgentService
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError


CASES = {case["case_id"]: case for case in load_dataset(PROJECT_ROOT / "eval/agent_eval_dataset.csv")}


def completion(content="本地测试回答", tool=False):
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [{"id": "call_refund", "type": "function", "function": {
            "name": "get_refund_status", "arguments": '{"order_id":"123456"}'}}]
    return {"id": "local", "object": "chat.completion", "created": 0, "model": "configured-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message}]}


@contextmanager
def fake_client(responses):
    requests = []
    pending = iter(responses)

    def handler(request):
        requests.append(json.loads(request.content))
        response = next(pending)
        if response == "timeout":
            raise httpx.ReadTimeout("local timeout", request=request)
        if response == "connection":
            raise httpx.ConnectError("local connection failure", request=request)
        if isinstance(response, int):
            return httpx.Response(response, json={"error": {"message": "local status error", "type": "test"}})
        return httpx.Response(200, json=response)

    def factory(**kwargs):
        if kwargs.get("max_retries") != 0:
            raise AssertionError("必须关闭 SDK 重试，避免叠加")
        return RealOpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    env = {"DEEPSEEK_API_KEY": "local-test-only", "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
           "DEEPSEEK_MODEL": "configured-model"}
    with patch.dict("os.environ", env, clear=True), patch("src.deepseek_client.load_dotenv"), \
            patch("src.deepseek_client.OpenAI", side_effect=factory), patch("src.deepseek_client.sleep") as pause, \
            patch("src.deepseek_client.logging.getLogger"):
        client = DeepSeekClient()
        try:
            yield client, requests, pause
        finally:
            client.close()


class RetryTests(unittest.TestCase):
    def test_timeout_then_429_then_success_same_request(self):
        with fake_client(["timeout", 429, completion()]) as (client, requests, pause):
            self.assertEqual(client.generate_answer("原始规则", "原始问题"), "本地测试回答")
            self.assertEqual(pause.call_args_list, [call(1), call(2)])
            self.assertEqual(client.api_attempts, 3)
            self.assertEqual(client.api_successes, 1)
            self.assertTrue(all(request == requests[0] for request in requests))

    def test_retryable_http_status_codes(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status), fake_client([status, completion()]) as (client, requests, pause):
                client.generate_answer("规则", "问题")
                self.assertEqual(len(requests), 2)
                pause.assert_called_once_with(1)

    def test_three_timeouts_stop_after_two_extra_attempts(self):
        with fake_client(["timeout"] * 3) as (client, requests, pause):
            with self.assertRaises(DeepSeekRequestError) as caught:
                client.generate_answer("规则", "问题")
            self.assertEqual(len(requests), 3)
            self.assertEqual(caught.exception.api_attempts, 3)
            self.assertIn("已尝试 3 次", str(caught.exception))
            self.assertEqual(pause.call_args_list, [call(1), call(2)])
            self.assertEqual(client.api_successes, 0)

    def test_three_503_errors_also_stop(self):
        with fake_client([503] * 3) as (client, requests, pause):
            with self.assertRaises(DeepSeekRequestError):
                client.generate_answer("规则", "问题")
            self.assertEqual(len(requests), 3)
            self.assertEqual(pause.call_count, 2)

    def test_configuration_errors_do_not_retry(self):
        for status in (400, 401, 402, 403, 404, 422, 501):
            with self.subTest(status=status), fake_client([status]) as (client, requests, pause):
                with self.assertRaises(DeepSeekRequestError):
                    client.generate_answer("规则", "问题")
                self.assertEqual(len(requests), 1)
                pause.assert_not_called()

    def test_connection_failure_is_not_blindly_retried(self):
        with fake_client(["connection"]) as (client, requests, pause):
            with self.assertRaises(DeepSeekRequestError):
                client.generate_answer("规则", "问题")
            self.assertEqual(len(requests), 1)
            pause.assert_not_called()

    def test_empty_model_response_does_not_trigger_request_retry(self):
        with fake_client([completion("")]) as (client, requests, pause):
            with self.assertRaises(RuntimeError) as caught:
                client.generate_answer("规则", "问题")
            self.assertNotIsInstance(caught.exception, DeepSeekRequestError)
            self.assertEqual(len(requests), 1)
            pause.assert_not_called()


class AgentRetryAccountingTests(unittest.TestCase):
    def test_recovered_retries_do_not_fail_valid_tool_answer(self):
        responses = ["timeout", completion(None, tool=True), 503, "timeout", completion("退款仍在处理中，尚未到账。")]
        with fake_client(responses) as (client, requests, pause):
            recorder = RecordingClient(client)
            row = evaluate_one(CASES["AG-040"], AgentService(recorder, Mock()), recorder)
            self.assertEqual(row["result"], "pass")
            self.assertEqual(row["actual_path"], "TOOL")
            self.assertEqual(row["actual_arguments"], [{"order_id": "123456"}])
            self.assertEqual(row["api_attempts"], 5)
            self.assertEqual(row["api_calls"], 2)
            self.assertEqual(row["api_retries"], 3)
            self.assertEqual(row["api_successes"], 2)
            self.assertEqual(row["api_successful_calls"], 2)
            self.assertTrue(row["api_success"])
            self.assertEqual(pause.call_args_list, [call(1), call(1), call(2)])
            self.assertEqual(requests[0]["tool_choice"], "auto")
            self.assertEqual(requests[-1]["tool_choice"], "none")
            self.assertEqual(requests[-1]["messages"][-1]["role"], "tool")

    def test_decision_api_failure_is_not_route_error(self):
        with fake_client(["timeout"] * 3) as (client, requests, pause):
            recorder = RecordingClient(client)
            row = evaluate_one(CASES["AG-040"], AgentService(recorder, Mock()), recorder)
            for key in ("path_pass", "tool_pass", "argument_pass"):
                self.assertIsNone(row[key])
            self.assertEqual(row["error_kind"], "API")
            self.assertEqual(row["api_attempts"], 3)
            self.assertEqual(row["result"], "fail")
            self.assertFalse(row["final_answer_pass"])
            self.assertNotIn("路径错误", row["failure_reason"])
            self.assertNotIn("工具错误", row["failure_reason"])
            summary = build_summary([row], 1, 40)
            self.assertEqual(summary["decision_api_unavailable"], 1)
            self.assertEqual(summary["metrics"]["path_accuracy"]["total"], 0)
            self.assertEqual(summary["path_error_case_ids"], [])
            self.assertEqual(summary["api_retries"], 2)

    def test_final_api_failure_keeps_successful_handoff_decision(self):
        responses = [completion('{"route":"HANDOFF","reason":"需要人工修改地址"}')] + ["timeout"] * 3
        with fake_client(responses) as (client, requests, pause):
            recorder = RecordingClient(client)
            row = evaluate_one(CASES["AG-032"], AgentService(recorder, Mock()), recorder)
            self.assertTrue(row["path_pass"])
            self.assertTrue(row["handoff_pass"])
            self.assertFalse(row["final_answer_pass"])
            self.assertEqual(row["api_attempts"], 4)
            self.assertEqual(row["api_retries"], 2)

    def test_invalid_model_route_still_counts_as_wrong_decision(self):
        with fake_client([completion('{"route":"INVALID"}')]) as (client, requests, pause):
            recorder = RecordingClient(client)
            row = evaluate_one(CASES["AG-001"], AgentService(recorder, Mock()), recorder)
            self.assertFalse(row["path_pass"])
            self.assertEqual(row["error_kind"], "CHAIN")
            self.assertIn("路径错误", row["failure_reason"])
            self.assertEqual(build_summary([row], 1, 40)["decision_api_unavailable"], 0)

    def test_ag006_and_ag037_original_rule_failures_are_preserved(self):
        examples = [
            ("AG-006", "RAG", "当前公开资料中没有可确认的烟酰胺具体添加比例，因此我不能给出准确浓度。"),
            ("AG-037", "HANDOFF", "您好，查询订单物流需要订单号。请提供订单号，我再帮您核对。"),
        ]
        for case_id, path, answer in examples:
            with self.subTest(case_id=case_id):
                score = score_result(CASES[case_id], path, [], answer)
                self.assertTrue(score["path_pass"])
                self.assertFalse(score["final_answer_pass"])
                self.assertEqual(score["result"], "fail")


if __name__ == "__main__":
    unittest.main()
