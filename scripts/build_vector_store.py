"""第二阶段入口：知识库 → 切分 → BGE → 本地 Chroma。"""

import argparse
from pathlib import Path
import sys


# 使用脚本所在位置确定项目根目录，不依赖终端当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import find_markdown_files, load_documents, split_documents
from src.embeddings import MODEL_NAME, embed_documents, load_model
from src.vector_store import COLLECTION_NAME, check_build_allowed, connect_store, write_chunks


def build(rebuild: bool) -> None:
    print("================================\nAURA LABS 知识库建立程序\n================================", flush=True)
    files = find_markdown_files(PROJECT_ROOT / "knowledge_base")
    print(f"发现知识库文件：{len(files)} 个")
    for path in files:
        print(f"- {path.name}")
    print("\n正在读取知识库……", flush=True)
    documents = load_documents(files)
    print("读取完成。\n\n正在切分文档……", flush=True)
    chunks = split_documents(documents)
    print(f"生成知识片段：{len(chunks)} 个")
    for chunk in chunks:
        if len(chunk.document) > 600:
            print(f"提示：{chunk.chunk_id} 的完整句子超过 600 字，已保留完整句子，请检查原文。")

    client = connect_store(PROJECT_ROOT / "chroma_db")
    check_build_allowed(client, rebuild)
    print(f"\n正在加载向量模型：\n{MODEL_NAME}")
    print("第一次运行时模型可能需要下载，请耐心等待。", flush=True)
    load_model()
    print("模型加载完成。\n\n正在生成知识向量……", flush=True)
    embeddings = embed_documents([chunk.embedding_text for chunk in chunks])
    print("向量生成完成。\n\n正在写入 Chroma……", flush=True)
    collection = write_chunks(client, chunks, embeddings, rebuild)
    print("写入完成。")
    print(f"\nCollection：{COLLECTION_NAME}")
    print(f"知识片段数量：{len(chunks)}")
    print(f"Chroma 实际记录：{collection.count()}")
    print(f"数据库位置：{PROJECT_ROOT / 'chroma_db'}")
    print("================================\n向量知识库建立成功\n================================")


def main() -> int:
    parser = argparse.ArgumentParser(description="建立 AURA LABS 本地向量知识库，不调用 DeepSeek。")
    parser.add_argument("--rebuild", action="store_true", help="只删除并重建 aura_customer_service Collection")
    args = parser.parse_args()
    try:
        build(args.rebuild)
    except KeyboardInterrupt:
        print("\n已停止建库。如果中断发生在写入阶段，下次请使用 --rebuild。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n建库失败：{exc}", file=sys.stderr)
        print("请根据提示修复后重试；写入中断或记录不完整时请使用 --rebuild。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
