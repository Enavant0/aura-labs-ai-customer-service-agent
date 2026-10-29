"""Chroma 只负责持久化：向量全部由 BGE 显式提供。"""

from pathlib import Path

from src.document_loader import KnowledgeChunk
from src.embeddings import MODEL_NAME


COLLECTION_NAME = "aura_customer_service"


def connect_store(database_path: Path):
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError("缺少 Chroma 依赖，请先运行 python -m pip install -r requirements.txt。") from exc
    return chromadb.PersistentClient(
        path=str(database_path), settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client):
    from chromadb.errors import NotFoundError

    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        return None


def check_build_allowed(client, rebuild: bool) -> None:
    collection = get_collection(client)
    if collection is not None and collection.count() > 0 and not rebuild:
        raise ValueError(
            "知识库已经存在，如需重新建立，请使用：\n"
            "python scripts/build_vector_store.py --rebuild"
        )


def write_chunks(client, chunks: list[KnowledgeChunk], embeddings: list[list[float]], rebuild: bool):
    if not chunks or len(chunks) != len(embeddings):
        raise ValueError("片段数量与向量数量不一致，未修改 Collection。")
    # 再检查一次；模型和向量准备成功后才删除旧 Collection。
    check_build_allowed(client, rebuild)
    existing = get_collection(client)
    if existing is not None and rebuild:
        client.delete_collection(COLLECTION_NAME)
        existing = None
    collection = existing if existing is not None else client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"embedding_model": MODEL_NAME, "normalized": True},
        configuration={"hnsw": {"space": "cosine"}},
    )
    # 同时遵守 Chroma 单批写入数量限制，小知识库通常只需一批。
    batch_size = min(256, client.get_max_batch_size())
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.document for chunk in batch],
            embeddings=embeddings[start:start + batch_size],
            metadatas=[chunk.metadata for chunk in batch],
        )
    actual = collection.count()
    if actual != len(chunks):
        raise RuntimeError(
            f"建库验证失败：生成 {len(chunks)} 个片段，但 Chroma 实际保存 {actual} 条。"
            "请检查磁盘空间和写入权限，修复后使用 --rebuild 重新建立。"
        )
    return collection
