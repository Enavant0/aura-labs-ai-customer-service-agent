"""命令行 RAG 客服测试：每次提问独立检索和回答，不保存会话历史。"""

from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deepseek_client import DeepSeekClient
from src.embeddings import MODEL_NAME, load_model
from src.rag_service import RagService
from src.retriever import open_existing_collection


def print_result(result) -> None:
    print(f"\n用户问题：{result.question}")
    print(f"\n检索到的 Top 5 来源（本次实际 {len(result.chunks)} 个片段）：")
    for rank, chunk in enumerate(result.chunks, start=1):
        print(f"{rank}. {chunk.source_file} / {chunk.heading} [{chunk.chunk_id}]")
    if not result.chunks:
        print("无。")
    print(f"\nDeepSeek 最终回答{'（无结果时由程序直接拒答）' if not result.chunks else ''}：\n{result.answer}")
    print("\n引用来源（程序生成的本次检索参考）：")
    for source in result.sources:
        print(f"- {source['source_file']} / {source['heading']} [{source['chunk_id']}]")
    if not result.sources:
        print("无。")
    print(f"\n响应时间：{result.elapsed_seconds:.2f} 秒")
    print("------------------------------")


def main() -> int:
    client = None
    print("================================\nAURA LABS 命令行 RAG 客服\n================================", flush=True)
    try:
        # 先验证密钥配置，缺失时立即停止，不加载本地模型。
        client = DeepSeekClient()
        collection = open_existing_collection(PROJECT_ROOT / "chroma_db")
        print(f"Collection：{collection.name}，现有片段：{collection.count()}")
        print(f"DeepSeek 地址：{client.base_url}，模型：{client.model}")
        print(f"正在加载本地向量模型：{MODEL_NAME}", flush=True)
        load_model()
        service = RagService(collection, client)
        print("准备完成。每次问题独立处理；输入 exit 或 quit 退出。")
        while True:
            try:
                question = input("\n请输入问题：").strip()
                if question.lower() in {"exit", "quit"}:
                    print("已退出 RAG 测试。")
                    return 0
                if not question:
                    print("问题不能为空，请重新输入。")
                    continue
                started = perf_counter()
                print("正在检索知识库并生成回答……", flush=True)
                try:
                    print_result(service.answer_question(question))
                except Exception as exc:
                    print(f"\n用户问题：{question}\n本次回答失败：{exc}")
                    print(f"响应时间：{perf_counter() - started:.2f} 秒")
            except (EOFError, KeyboardInterrupt):
                print("\n已退出 RAG 测试。")
                return 0
    except KeyboardInterrupt:
        print("\n已退出 RAG 测试。")
        return 130
    except Exception as exc:
        print(f"\n启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
