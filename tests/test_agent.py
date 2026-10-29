"""Agent 离线检查：模拟 DeepSeek 返回的 Tool Calls，不执行网络请求。"""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from openai import OpenAI as RealOpenAI
from openai.types.chat import ChatCompletionMessage
from src.agent_service import AgentService, execute_tool_call
from src.deepseek_client import DeepSeekClient
from src.tools import TOOL_DEFINITIONS, get_order_status, get_refund_status
from scripts import test_agent


def message(content=None, calls=()):
    return ChatCompletionMessage.model_validate({"role": "assistant", "content": content, "tool_calls": list(calls) or None})


def tool_call(name="get_order_status", arguments=None, call_id="call_order"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(
        {"order_id": "123456"} if arguments is None else arguments, ensure_ascii=False)}}


class LocalToolTests(unittest.TestCase):
    def test_mock_order_and_refund_fields(self):
        order = get_order_status("123456")
        self.assertTrue(order["found"])
        self.assertTrue(order["is_mock"])
        self.assertTrue({"order_id", "order_status", "logistics_company", "tracking_number", "latest_location"} <= order.keys())
        refund = get_refund_status("123456")
        self.assertTrue({"order_id", "refund_status", "refund_amount", "refund_requested_at"} <= refund.keys())
        self.assertIn("尚未到账", refund["refund_status"])
        self.assertEqual(get_refund_status("123457")["refund_status"], "退款已到账")
        self.assertIsNone(get_order_status("123458")["tracking_number"])

    def test_missing_order_in_both_tools(self):
        for function in (get_order_status, get_refund_status):
            result = function("999999")
            self.assertFalse(result["found"])
            self.assertEqual(result["message"], "未找到订单")

    def test_return_values_do_not_mutate_fixtures(self):
        first = get_order_status("123456")
        original = first["order_status"]
        first["order_status"] = "changed"
        self.assertEqual(get_order_status("123456")["order_status"], original)

    def test_invalid_ids_and_model_arguments_are_not_executed(self):
        for order_id in (123456, "", "abc", "12345", "123456; print('x')"):
            with self.assertRaises(ValueError):
                get_order_status(order_id)
        for call in (tool_call(name="delete_order"), tool_call(arguments={"order_id": 123456}),
                     tool_call(arguments={"order_id": "123456", "refund": True}), tool_call(arguments=["123456"])):
            result = execute_tool_call(message(calls=[call]).tool_calls[0], "订单123456到哪了")
            self.assertFalse(result.result["ok"])
        malformed = tool_call()
        malformed["function"]["arguments"] = "{invalid json"
        self.assertFalse(execute_tool_call(message(calls=[malformed]).tool_calls[0], "订单123456到哪了").result["ok"])

    def test_model_cannot_guess_order_number(self):
        call = message(calls=[tool_call()]).tool_calls[0]
        result = execute_tool_call(call, "我的订单到哪里了？")
        self.assertIn("未在本次问题中", result.result["error"])
        result = execute_tool_call(call, "订单1234567到哪里了？")
        self.assertFalse(result.result["ok"])


class AgentFlowTests(unittest.TestCase):
    def test_rag_decision_reuses_original_question_and_sources(self):
        client = Mock()
        client.chat_with_tools.return_value = message('{"route":"RAG","reason":"知识问题"}')
        rag = Mock()
        rag.answer_question.return_value = SimpleNamespace(answer="RAG 原回答", sources=({"source_file": "products.md"},))
        result = AgentService(client, rag).answer_question("洁面多少钱？")
        self.assertEqual(result.path, "RAG")
        self.assertEqual(result.answer, "RAG 原回答")
        self.assertEqual(result.sources[0]["source_file"], "products.md")
        rag.answer_question.assert_called_once_with("洁面多少钱？")
        client.generate_answer.assert_not_called()

    def test_tool_round_trip_and_tool_call_id(self):
        client = Mock()
        client.chat_with_tools.side_effect = [message(calls=[tool_call()]), message("模拟订单正在广州转运中心。")]
        rag = Mock()
        result = AgentService(client, rag).answer_question("订单123456到哪里了？")
        self.assertEqual(result.path, "TOOL")
        self.assertEqual(result.tool_executions[0].name, "get_order_status")
        self.assertIn("本地模拟数据", result.answer)
        requests = client.chat_with_tools.call_args_list
        self.assertEqual(requests[0].kwargs["tool_choice"], "auto")
        self.assertEqual(requests[1].kwargs["tool_choice"], "none")
        history = requests[1].args[0]
        self.assertEqual(history[-1]["role"], "tool")
        self.assertEqual(history[-1]["tool_call_id"], "call_order")
        self.assertEqual(history[-2]["tool_calls"][0]["function"]["arguments"], '{"order_id": "123456"}')
        self.assertEqual(json.loads(history[-1]["content"])["order_id"], "123456")
        rag.answer_question.assert_not_called()

    def test_order_and_refund_can_be_requested_in_same_round(self):
        client = Mock()
        client.chat_with_tools.side_effect = [message(calls=[tool_call(), tool_call("get_refund_status", call_id="call_refund")]), message("模拟查询完成。")]
        result = AgentService(client, Mock()).answer_question("订单123456到哪了，退款呢？")
        self.assertEqual([execution.name for execution in result.tool_executions], ["get_order_status", "get_refund_status"])
        self.assertEqual(client.chat_with_tools.call_count, 2)

    def test_unknown_order_is_returned_to_model_without_fake_status(self):
        client = Mock()
        client.chat_with_tools.side_effect = [message(calls=[tool_call(arguments={"order_id": "999999"})]), message("未找到订单，请核对编号。")]
        result = AgentService(client, Mock()).answer_question("订单999999到哪里了？")
        self.assertFalse(result.tool_executions[0].result["found"])
        self.assertIn("未找到订单", result.answer)

    def test_handoff_is_model_decision_and_not_actual_transfer(self):
        client = Mock()
        client.chat_with_tools.return_value = message('{"route":"HANDOFF","reason":"用户要求人工"}')
        client.generate_answer.return_value = "此问题需要人工客服处理。"
        rag = Mock()
        result = AgentService(client, rag).answer_question("我要人工客服。")
        self.assertEqual(result.path, "HANDOFF")
        self.assertIn("未实际接入人工", result.answer)
        rag.answer_question.assert_not_called()

    def test_no_keyword_fallback_for_invalid_model_decision(self):
        client = Mock()
        rag = Mock()
        for content in ("洁面是89元", '{"route":"UNKNOWN"}'):
            client.chat_with_tools.return_value = message(content)
            with self.assertRaises(ValueError):
                AgentService(client, rag).answer_question("订单123456退款了吗？")
        rag.answer_question.assert_not_called()

    def test_limits_and_duplicate_call_ids_stop_execution(self):
        client = Mock()
        for calls in ([tool_call(), tool_call()], [tool_call(call_id=f"id_{i}") for i in range(3)]):
            client.chat_with_tools.return_value = message(calls=calls)
            with self.assertRaises(ValueError):
                AgentService(client, Mock()).answer_question("订单123456")

    def test_cli_repeats_and_prints_trace(self):
        client = Mock()
        result = SimpleNamespace(question="订单123456", path="TOOL", answer="模拟最终回答", sources=(), elapsed_seconds=0.1,
                                 tool_executions=(SimpleNamespace(name="get_order_status", arguments={"order_id": "123456"}, result={"is_mock": True}),))
        service = Mock()
        service.answer_question.side_effect = [RuntimeError("模拟超时"), result]
        output = StringIO()
        with patch.object(test_agent, "DeepSeekClient", return_value=client), \
             patch.object(test_agent, "log_agent_request", return_value=True), \
             patch.object(test_agent, "open_existing_collection", return_value=Mock()), \
             patch.object(test_agent, "AgentService", return_value=service), \
             patch("builtins.input", side_effect=["", "问题一", "订单123456", "quit"]), redirect_stdout(output):
            self.assertEqual(test_agent.main(), 0)
        for label in ("执行路径：TOOL", "工具名称：get_order_status", "工具参数", "工具返回结果", "最终回答", "响应时间", "模拟超时"):
            self.assertIn(label, output.getvalue())
        client.close.assert_called_once()


class ToolCallsSdkTests(unittest.TestCase):
    def test_actual_sdk_serializes_tool_definitions_and_tool_reply(self):
        captured = []
        def handler(request):
            body = json.loads(request.content)
            captured.append(body)
            self.assertEqual(request.url.host, "api.deepseek.com")
            first = len(captured) == 1
            reply = {"role": "assistant", "content": None, "tool_calls": [tool_call()]} if first else {"role": "assistant", "content": "模拟已查询。"}
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
                                           "choices": [{"index": 0, "finish_reason": "tool_calls" if first else "stop", "message": reply}]})
        def factory(**kwargs):
            return RealOpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        env = {"DEEPSEEK_API_KEY": "test-only", "DEEPSEEK_BASE_URL": "https://api.deepseek.com", "DEEPSEEK_MODEL": "configured-model"}
        with patch.dict("os.environ", env, clear=True), patch("src.deepseek_client.load_dotenv"), patch("src.deepseek_client.OpenAI", side_effect=factory):
            client = DeepSeekClient()
            try:
                result = AgentService(client, Mock()).answer_question("订单123456到哪里了？")
            finally:
                client.close()
        self.assertEqual(result.path, "TOOL")
        self.assertEqual(captured[0]["tools"], TOOL_DEFINITIONS)
        self.assertEqual(captured[0]["tool_choice"], "auto")
        self.assertEqual(captured[1]["tool_choice"], "none")
        self.assertEqual(captured[1]["messages"][-1]["tool_call_id"], "call_order")
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})

    def test_missing_key_stops_cli(self):
        with patch.dict("os.environ", {}, clear=True), patch("src.deepseek_client.load_dotenv"), \
             patch.object(test_agent, "open_existing_collection") as database, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(test_agent.main(), 1)
            database.assert_not_called()


if __name__ == "__main__":
    unittest.main()
