"""共用的本地检索逻辑：只打开已有 Collection，显式传入 BGE 查询向量。"""

from pathlib import Path

from src.embeddings import embed_query
from src.vector_store import COLLECTION_NAME, connect_store, get_collection


def open_existing_collection(database_path: Path):
    # PersistentClient 会自动初始化目录，必须先检查，避免把错误路径变成空库。
    if not database_path.is_dir():
        raise ValueError(f"chroma_db 不存在：{database_path}。请确认已有数据库位于项目根目录。")
    if not (database_path / "chroma.sqlite3").is_file():
        raise ValueError(f"目录中没有找到已有 Chroma 数据库：{database_path}。请确认数据库路径。")
    try:
        client = connect_store(database_path)
        # 此处只调用 get_collection，绝不新建 Collection；禁用默认向量模型。
        collection = get_collection(client)
        if collection is None:
            raise ValueError(f"Collection 不存在：{COLLECTION_NAME}。请确认打开的是已建库的项目目录。")
        if collection.count() == 0:
            raise ValueError(f"Collection 为空：{COLLECTION_NAME}，目前没有可检索的知识片段。")
        return collection
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"无法打开现有 Chroma 数据库，请检查路径、目录权限和依赖。原因：{exc}") from exc


def retrieve(collection, question: str, top_k: int = 5) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空，请重新输入。")
    if top_k < 1:
        raise ValueError("返回片段数量必须大于 0。")
    try:
        # 复用建库阶段的 BGE：问题前缀和向量归一化均由 embed_query 统一处理。
        query_embedding = embed_query(question)
    except ValueError as exc:
        raise ValueError("问题无法向量化，可能超过模型长度上限，请缩短问题后重试。") from exc
    except Exception as exc:
        raise RuntimeError(f"问题向量生成失败：{exc}") from exc
    try:
        count = collection.count()
        if count == 0:
            raise ValueError("Collection 为空，目前没有可检索的知识片段。")
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(f"查询失败：请检查数据库是否可访问、向量维度是否匹配。原因：{exc}") from exc


