"""第三阶段：在终端反复提问，只检索现有知识片段，不生成客服回答。"""

from pathlib import Path
import sys


# 无论从哪个目录运行，都读取这个项目自己的数据库。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import MODEL_NAME, load_model
from src.retriever import open_existing_collection, retrieve


def print_results(results: dict) -> None:
    ids = (results.get("ids") or [[]])[0]
    if not ids:
        print("没有检索到知识片段，请尝试换一种问法。")
        return
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    print(f"\n找到 {len(ids)} 个相关片段（距离越小越接近，不代表知识库一定能回答）：")
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1,
    ):
        metadata = metadata or {}
        distance_text = f"{distance:.6f}" if distance is not None else "未提供"
        print("------------------------------")
        print(f"第 {rank} 名")
        print(f"来源：{metadata.get('source_file', '未提供')}")
        print(f"标题：{metadata.get('heading', '未提供')}")
        print(f"Chunk：{metadata.get('chunk_id', chunk_id)}")
        print(f"距离：{distance_text}")
        print(f"\n正文：\n{document or '（正文为空）'}")
    print("------------------------------")


def main() -> int:
    print("================================\nAURA LABS 本地向量检索测试\n================================", flush=True)
    try:
        collection = open_existing_collection(PROJECT_ROOT / "chroma_db")
        print(f"Collection：{collection.name}")
        print(f"现有知识片段：{collection.count()} 个")
        print(f"正在加载向量模型：{MODEL_NAME}", flush=True)
        load_model()
    except KeyboardInterrupt:
        print("\n已退出检索测试。")
        return 130
    except Exception as exc:
        print(f"\n启动失败：{exc}", file=sys.stderr)
        return 1

    print("模型加载完成。默认返回前 5 个片段。输入 exit 或 quit 退出。")
    while True:
        try:
            question = input("\n请输入问题：").strip()
            if question.lower() in {"exit", "quit"}:
                print("已退出检索测试。")
                return 0
            if not question:
                print("问题不能为空，请重新输入。")
                continue
            print_results(retrieve(collection, question))
        except (EOFError, KeyboardInterrupt):
            print("\n已退出检索测试。")
            return 0
        except Exception as exc:
            # 单次失败后仍能继续提问，终端不打印复杂堆栈。
            print(f"\n检索未完成：{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
