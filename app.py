"""Streamlit 展示层：复用冻结的 Agent V1.1，不改变其决策与上下文。"""

from pathlib import Path
from time import perf_counter

import streamlit as st

from src.agent_service import AgentService
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError
from src.rag_service import RagService
from src.retriever import open_existing_collection
from src.runtime_logger import get_api_attempts, log_agent_request


PROJECT_ROOT = Path(__file__).resolve().parent


def get_agent_service():
    """与命令行相同的初始化顺序；只在本 Session 首次使用时创建。"""
    if "agent_service" not in st.session_state:
        client = DeepSeekClient()
        try:
            collection = open_existing_collection(PROJECT_ROOT / "chroma_db")
            service = AgentService(client, RagService(collection, client))
        except Exception:
            client.close()
            raise
        st.session_state.agent_service = service
    return st.session_state.agent_service


def friendly_error(exc):
    # 只显示固定提示，不把异常原文、堆栈或本机配置直接送到网页。
    if isinstance(exc, DeepSeekRequestError):
        return "DeepSeek 请求失败，请检查网络、账户或本机配置后重试。"
    if isinstance(exc, ValueError) and str(exc).startswith("未配置 DEEPSEEK_API_KEY"):
        return "尚未配置 DeepSeek API 密钥，请在本机 .env 中填写后重试。"
    return "暂时无法完成处理，请检查本机配置、网络和现有知识库后重试。"


def render_details(result):
    """以下分支只决定如何展示实际返回值，不参与业务路由。"""
    with st.expander("查看 Agent 执行详情", expanded=False):
        st.write(f"Path：{result.path}")
        if result.path == "RAG":
            st.markdown("**检索参考来源**")
            if result.sources:
                st.table(list(result.sources))
            else:
                st.caption("本次没有返回可展示的检索来源。")
        elif result.path == "TOOL":
            st.info("本地模拟数据，不连接真实订单或退款系统。")
            for execution in result.tool_executions:
                st.write(f"Tool Name：{execution.name}")
                st.markdown("**Tool Arguments**")
                st.json(execution.arguments)
                st.markdown("**Tool Result · 本地模拟数据**")
                st.json(execution.result)
        elif result.path == "HANDOFF":
            st.caption("人工处理建议；当前演示未接入真实人工客服队列。")
            st.markdown(result.answer)
        st.caption(f"Agent 处理耗时：{result.elapsed_seconds:.2f} 秒")


def render_message(message):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        elif "error" in message:
            st.error(message["error"])
        else:
            result = message["result"]
            st.markdown(result.answer)
            render_details(result)


def main():
    st.set_page_config(page_title="AURA LABS · AI Customer Service", page_icon="💬", layout="centered")
    st.title("AURA LABS AI Customer Service Agent")
    st.caption("AI 电商客服作品集 Demo · UI MVP V0.1 · Agent V1.1")
    st.write("支持企业知识问答、模拟订单状态查询、模拟退款状态查询和人工处理提示。")
    st.info("订单和退款工具使用本地模拟数据，不连接真实淘宝、抖音、ERP、物流、支付或客服系统。")
    st.caption("当前版本每次问题独立处理，不使用多轮上下文记忆。")

    with st.sidebar:
        st.header("示例问题")
        for example in ("洁面多少钱？", "订单 123456 到哪里了？", "123456 退款到账了吗？", "我要人工客服。"):
            st.write(f"• {example}")
        st.caption("查询具体订单时，请在本次问题中提供订单号。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 这里只重绘 UI 记录，不重新调用 Agent，也不把历史放进本次问题。
    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("请输入问题，例如：洁面多少钱？")
    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_message(user_message)
        with st.spinner("Agent 正在处理……"):
            started = perf_counter()
            client, api_before = None, None
            try:
                service = get_agent_service()
                client = service.client
                api_before = get_api_attempts(client)
                result = service.answer_question(question)
                log_agent_request("streamlit", started, result=result, client=client, api_attempts_before=api_before)
                assistant_message = {"role": "assistant", "result": result}
            except Exception as exc:
                log_agent_request("streamlit", started, error=exc, client=client, api_attempts_before=api_before)
                assistant_message = {"role": "assistant", "error": friendly_error(exc)}
        st.session_state.messages.append(assistant_message)
        render_message(assistant_message)


if __name__ == "__main__":
    main()
