"""运行日志离线检查：使用临时文件，不写真实日志，不调用模型或 Chroma。"""

from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from streamlit.testing.v1 import AppTest

from scripts import test_agent
from src.agent_service import AgentResult, ToolExecution
from src.deepseek_client import DeepSeekRequestError
from src import runtime_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def result(path="RAG", tools=()):
    return AgentResult("PRIVATE_QUESTION", path, "PRIVATE_ANSWER", tuple(tools), (), 0.01)


class RuntimeLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / "logs/runtime.jsonl"
        self.override = patch.object(runtime_logger, "LOG_PATH", self.path)
        self.override.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.override.stop)

    def rows(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_rag_record_and_append_do_not_store_content(self):
        for _ in range(2):
            self.assertTrue(runtime_logger.log_agent_request("cli", perf_counter(), result=result()))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["request_id"], rows[1]["request_id"])
        self.assertEqual(UUID(rows[0]["request_id"]).version, 4)
        self.assertIsNotNone(datetime.fromisoformat(rows[0]["timestamp"]).tzinfo)
        self.assertEqual(rows[0]["path"], "RAG")
        self.assertTrue(rows[0]["success"])
        self.assertGreaterEqual(rows[0]["elapsed_ms"], 0)
        self.assertNotIn("tool_name", rows[0])
        self.assertNotIn("api_attempts", rows[0])
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_QUESTION", text)
        self.assertNotIn("PRIVATE_ANSWER", text)

    def test_both_tools_preserve_real_found_and_mock_flags(self):
        for name, found in (("get_order_status", True), ("get_refund_status", False)):
            execution = ToolExecution(name, {"order_id": "123456"}, {
                "found": found, "is_mock": True, "private_payload": "SHOULD_NOT_PERSIST"})
            runtime_logger.log_agent_request("streamlit", perf_counter(), result=result("TOOL", [execution]))
        rows = self.rows()
        self.assertEqual(rows[0]["tool_name"], "get_order_status")
        self.assertEqual(rows[1]["tool_name"], "get_refund_status")
        self.assertEqual(rows[0]["tool_arguments"], {"order_id": "123456"})
        self.assertTrue(rows[0]["is_mock"])
        self.assertFalse(rows[1]["tool_found"])
        self.assertTrue(rows[1]["success"])  # 未找到是查询结果，不代表请求异常。
        self.assertNotIn("SHOULD_NOT_PERSIST", self.path.read_text(encoding="utf-8"))

    def test_multiple_tools_are_one_request_record(self):
        executions = [ToolExecution(name, {"order_id": "123456"}, {"found": True, "is_mock": True})
                      for name in ("get_order_status", "get_refund_status")]
        runtime_logger.log_agent_request("cli", perf_counter(), result=result("TOOL", executions))
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["tools"]), 2)

    def test_unknown_tool_flags_are_null_not_false(self):
        execution = ToolExecution("get_order_status", {"order_id": "123456"}, {"ok": False})
        runtime_logger.log_agent_request("cli", perf_counter(), result=result("TOOL", [execution]))
        row = self.rows()[0]
        self.assertIsNone(row["tool_found"])
        self.assertIsNone(row["is_mock"])

    def test_only_valid_order_id_is_persisted_in_arguments(self):
        execution = ToolExecution("get_order_status", {"order_id": "SECRET_NOT_AN_ORDER", "Authorization": "PRIVATE_HEADER"}, {})
        runtime_logger.log_agent_request("cli", perf_counter(), result=result("TOOL", [execution]))
        self.assertEqual(self.rows()[0]["tool_arguments"], {})
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET_NOT_AN_ORDER", text)
        self.assertNotIn("PRIVATE_HEADER", text)

    def test_handoff_is_a_successful_request(self):
        runtime_logger.log_agent_request("streamlit", perf_counter(), result=result("HANDOFF"))
        row = self.rows()[0]
        self.assertEqual(row["path"], "HANDOFF")
        self.assertTrue(row["success"])

    def test_error_never_persists_raw_message_or_traceback(self):
        error = DeepSeekRequestError("DEEPSEEK_API_KEY=PRIVATE_SECRET Authorization: PRIVATE_HEADER\ntraceback", 3)
        runtime_logger.log_agent_request("cli", perf_counter(), error=error)
        row = self.rows()[0]
        self.assertFalse(row["success"])
        self.assertIsNone(row["path"])
        self.assertEqual(row["error_type"], "DeepSeekRequestError")
        self.assertEqual(row["error_message"], "DeepSeek 请求失败。")
        text = self.path.read_text(encoding="utf-8")
        for forbidden in ("DEEPSEEK_API_KEY", "PRIVATE_SECRET", "Authorization", "PRIVATE_HEADER", "traceback"):
            self.assertNotIn(forbidden, text)

    def test_api_attempt_delta_includes_retries_without_inventing_retry_count(self):
        client = SimpleNamespace(api_attempts=7)
        before = runtime_logger.get_api_attempts(client)
        client.api_attempts = 12
        runtime_logger.log_agent_request("cli", perf_counter(), result=result(), client=client, api_attempts_before=before)
        row = self.rows()[0]
        self.assertEqual(row["api_attempts"], 5)
        self.assertEqual(client.api_attempts, 12)
        self.assertNotIn("retry_count", row)
        self.assertNotIn("api_call_count", row)

    def test_unavailable_or_reset_counter_is_omitted(self):
        for client, before in ((None, None), (Mock(), None), (SimpleNamespace(api_attempts=1), 3)):
            runtime_logger.log_agent_request("cli", perf_counter(), result=result(), client=client, api_attempts_before=before)
        for row in self.rows():
            self.assertNotIn("api_attempts", row)

    def test_write_and_directory_failure_are_safely_ignored(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("PRIVATE_PATH")):
            self.assertFalse(runtime_logger.log_agent_request("cli", perf_counter(), result=result()))
        with patch.object(Path, "open", side_effect=OSError("disk failure")):
            self.assertFalse(runtime_logger.log_agent_request("cli", perf_counter(), result=result()))

    def test_streamlit_one_record_per_submit_not_rerun_and_logging_failure_keeps_answer(self):
        service = Mock()
        service.answer_question.return_value = result()
        with patch("src.deepseek_client.DeepSeekClient", return_value=Mock()), \
                patch("src.retriever.open_existing_collection", return_value=Mock()), \
                patch("src.agent_service.AgentService", return_value=service):
            app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=15).run()
            self.assertFalse(self.path.exists())
            app.chat_input[0].set_value("PRIVATE_QUESTION").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(self.rows()), 1)
            app.run()
            self.assertEqual(len(self.rows()), 1)
            self.assertEqual(service.answer_question.call_count, 1)
            with patch.object(Path, "open", side_effect=PermissionError("log unavailable")):
                app.chat_input[0].set_value("第二个问题").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["messages"][-1]["result"].answer, "PRIVATE_ANSWER")
            self.assertEqual(len(self.rows()), 1)
            self.assertEqual(service.answer_question.call_count, 2)

    def test_streamlit_agent_failure_writes_one_sanitized_record(self):
        service = Mock()
        service.answer_question.side_effect = RuntimeError("PRIVATE_SECRET")
        with patch("src.deepseek_client.DeepSeekClient", return_value=Mock()), \
                patch("src.retriever.open_existing_collection", return_value=Mock()), \
                patch("src.agent_service.AgentService", return_value=service):
            app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=15).run()
            app.chat_input[0].set_value("PRIVATE_QUESTION").run()
            app.run()
            self.assertFalse(app.exception)
            rows = self.rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["interface"], "streamlit")
            self.assertFalse(rows[0]["success"])
            self.assertNotIn("PRIVATE_SECRET", self.path.read_text(encoding="utf-8"))

    def test_cli_logs_success_failure_but_not_empty_or_exit(self):
        service = Mock()
        service.answer_question.side_effect = [result("HANDOFF"), RuntimeError("PRIVATE_SECRET")]
        with patch.object(test_agent, "DeepSeekClient", return_value=Mock()), \
                patch.object(test_agent, "open_existing_collection", return_value=Mock()), \
                patch.object(test_agent, "AgentService", return_value=service), \
                patch("builtins.input", side_effect=["", "问题一", "问题二", "quit"]), redirect_stdout(StringIO()):
            self.assertEqual(test_agent.main(), 0)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["success"])
        self.assertFalse(rows[1]["success"])
        self.assertTrue(all(row["interface"] == "cli" for row in rows))
        self.assertNotIn("PRIVATE_SECRET", self.path.read_text(encoding="utf-8"))

    def test_cli_write_failure_does_not_change_answer_or_exit(self):
        service = Mock()
        service.answer_question.return_value = result("HANDOFF")
        output = StringIO()
        with patch.object(test_agent, "DeepSeekClient", return_value=Mock()), \
                patch.object(test_agent, "open_existing_collection", return_value=Mock()), \
                patch.object(test_agent, "AgentService", return_value=service), \
                patch("builtins.input", side_effect=["问题", "quit"]), \
                patch.object(Path, "open", side_effect=PermissionError("logs denied")), redirect_stdout(output):
            self.assertEqual(test_agent.main(), 0)
        self.assertIn("PRIVATE_ANSWER", output.getvalue())
        self.assertIn("执行路径：HANDOFF", output.getvalue())


if __name__ == "__main__":
    unittest.main()
