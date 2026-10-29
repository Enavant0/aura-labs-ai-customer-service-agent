"""命令行 Agent：RAG、两个本地模拟只读工具、人工处理提示。"""

from pathlib import Path
import json
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_service import AgentService
from src.deepseek_client import DeepSeekClient
from src.rag_service import RagService
from src.retriever import open_existing_collection
from src.runtime_logger import get_api_attempts, log_agent_request


def print_result(result) -> None:
    print(f"\n用户问题：{result.question}\n执行路径：{result.path}")
    for execution in result.tool_executions:
        print(f"\n工具名称：{execution.name}")
        print("工具参数：" + json.dumps(execution.arguments, ensure_ascii=False))
        print("工具返回结果：" + json.dumps(execution.result, ensure_ascii=False, indent=2))
    print(f"\n最终回答：\n{result.answer}")
    if result.sources:
        print("\nRAG 检索参考来源：")
        for source in result.sources:
            print(f"- {source['source_file']} / {source['heading']} [{source['chunk_id']}]")
    print(f"\n响应时间：{result.elapsed_seconds:.2f} 秒\n------------------------------")


def main() -> int:
    client = None
    print("AURA LABS AI 客服 Agent（本地模拟工具版）", flush=True)
    try:
        client = DeepSeekClient()
        collection = open_existing_collection(PROJECT_ROOT / "chroma_db")
        service = AgentService(client, RagService(collection, client))
        print("演示订单：123456、123457、123458、123459。工具数据均为虚构。")
        print("每次提问独立处理，不记忆上一次订单号。输入 exit 或 quit 退出。")
        # BGE 沿用原模块的延迟加载：只有模型选择 RAG 后才需要本地向量模型。
        while True:
            try:
                question = input("\n请输入问题：").strip()
                if question.lower() in {"exit", "quit"}:
                    print("已退出 Agent 测试。")
                    return 0
                if not question:
                    print("问题不能为空，请重新输入。")
                    continue
                started = perf_counter()
                api_before = get_api_attempts(client)
                print("正在由 DeepSeek 决定执行路径……", flush=True)
                try:
                    result = service.answer_question(question)
                except Exception as exc:
                    log_agent_request("cli", started, error=exc, client=client, api_attempts_before=api_before)
                    print(f"\n用户问题：{question}\n本次处理失败：{exc}")
                    print(f"响应时间：{perf_counter() - started:.2f} 秒")
                else:
                    log_agent_request("cli", started, result=result, client=client, api_attempts_before=api_before)
                    print_result(result)
            except (EOFError, KeyboardInterrupt):
                print("\n已退出 Agent 测试。")
                return 0
    except KeyboardInterrupt:
        print("\n已停止 Agent 启动。")
        return 130
    except Exception as exc:
        print(f"\n启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
