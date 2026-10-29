"""通过 OpenAI 兼容 SDK 请求 DeepSeek；不调用 OpenAI 模型。"""

import logging
import os
from pathlib import Path
from time import sleep
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_API_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class DeepSeekRequestError(RuntimeError):
    """标识请求层故障，评测时与模型决策错误分开。"""

    def __init__(self, message: str, api_attempts: int):
        super().__init__(message)
        self.api_attempts = api_attempts


class DeepSeekClient:
    def __init__(self):
        # 明确读取本项目 .env，保留终端已设置的环境变量优先级。
        load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8-sig")
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY。请在项目 .env 中填写 DeepSeek API 密钥后重新运行。")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", "").strip()
        if not self.model:
            raise ValueError("未配置 DEEPSEEK_MODEL。请在 .env 中填写模型名称，参考 .env.example。")
        url = urlsplit(base_url)
        if (url.scheme != "https" or url.netloc != "api.deepseek.com"
                or url.path not in ("", "/v1") or url.query or url.fragment):
            raise ValueError("DEEPSEEK_BASE_URL 必须为 https://api.deepseek.com 或 https://api.deepseek.com/v1。")
        self.base_url = base_url
        # 关闭 SDK 自带重试，避免与下面的两次额外重试叠加。
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0)
        self.api_attempts = 0  # 客户端生命周期内实际发送请求的总次数，包含重试。
        self.api_successes = 0

    def generate_answer(self, system_prompt: str, user_content: str) -> str:
        response = self._request_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])
        return self._answer_text(response)

    def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto"):
        """保留模型的 tool_calls，让 Agent 执行；本方法不自行调用本地工具。"""
        if tool_choice not in {"auto", "none"}:
            raise ValueError("当前只支持 auto 或 none 的工具选择方式。")
        response = self._request_completion(messages, tools=tools, tool_choice=tool_choice)
        if not response.choices:
            raise RuntimeError("DeepSeek 未返回 Agent 决策，请稍后重试。")
        choice = response.choices[0]
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise RuntimeError("DeepSeek Agent 响应不完整或被中断，请重试。")
        if choice.finish_reason == "tool_calls" and not choice.message.tool_calls:
            raise RuntimeError("DeepSeek 声明了工具调用但未提供调用参数，请重试。")
        if tool_choice == "none" and choice.message.tool_calls:
            raise RuntimeError("工具结果返回后模型仍请求工具，已停止本轮以避免重复调用。")
        if not choice.message.tool_calls:
            self._answer_text(response)
        return choice.message

    def _request_completion(self, messages: list[dict], **options):
        # 原有 RAG 的请求参数保持一致，Agent 只额外提供 tools/tool_choice。
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            self.api_attempts += 1
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    temperature=0.2,
                    max_tokens=800,
                    extra_body={"thinking": {"type": "disabled"}},
                    **options,
                )
            except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
                if isinstance(exc, APITimeoutError):
                    message = "DeepSeek 请求超时，请检查网络或稍后重试。"
                    retryable = True
                elif isinstance(exc, APIStatusError):
                    explanations = {
                        400: "请求参数不被接受，请检查模型名称及参数是否适用于该模型。",
                        401: "密钥无效或认证失败，请检查 DEEPSEEK_API_KEY。",
                        402: "账户余额不足，请检查 DeepSeek 账户余额。",
                        403: "请求被拒绝，请检查账户权限。",
                        404: "接口或模型不存在，请检查基础地址和 DEEPSEEK_MODEL。",
                        422: "请求参数校验失败，请检查模型和请求参数。",
                        429: "请求过于频繁或账户限流，请稍后重试。",
                    }
                    detail = explanations.get(exc.status_code, "服务暂时异常，请稍后重试。")
                    message = f"DeepSeek API 请求失败（HTTP {exc.status_code}）：{detail}"
                    retryable = exc.status_code in RETRYABLE_STATUS_CODES
                else:
                    message = "无法连接 DeepSeek API，请检查网络是否可以访问 api.deepseek.com，以及代理或防火墙设置。"
                    retryable = False
                if retryable and attempt < MAX_API_ATTEMPTS:
                    logging.getLogger(__name__).warning(
                        "%s %s 秒后进行第 %s/%s 次请求。", message, attempt, attempt + 1, MAX_API_ATTEMPTS,
                    )
                    sleep(attempt)  # 第一次失败等 1 秒，第二次失败等 2 秒。
                    continue
                # 不打印 SDK 原始异常正文，避免暴露凭据或请求内容。
                raise DeepSeekRequestError(f"{message}（本次请求已尝试 {attempt} 次）", attempt) from exc
            self.api_successes += 1
            return response

    @staticmethod
    def _answer_text(response) -> str:
        if not response.choices:
            raise RuntimeError("DeepSeek 未返回回答，请稍后重试。")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("DeepSeek 回答超过本次输出长度限制，未展示被截断的答案。请缩短或拆分问题后重试。")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek 返回了空回答，请稍后重试。")
        return content.strip()

    def close(self) -> None:
        self._client.close()
