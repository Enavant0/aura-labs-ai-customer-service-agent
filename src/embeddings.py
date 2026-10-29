"""统一文档和问题的 BGE 向量规则，不调用任何模型 API。"""

from functools import lru_cache
import logging
from pathlib import Path


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MODEL_CACHE = Path(__file__).resolve().parents[1] / ".model_cache"


@lru_cache(maxsize=1)
def load_model():
    """进程内只加载一次；缓存留在项目目录，后续运行可以复用。"""
    # 保留下载进度，但将依赖库反复打印的网络警告收敛为下面的中文错误。
    for logger_name in ("huggingface_hub", "transformers", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("缺少向量模型依赖，请先运行 python -m pip install -r requirements.txt。") from exc
    try:
        return SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE), device="cpu")
    except Exception as exc:
        raise RuntimeError(
            "BGE 模型加载失败。首次运行需要联网下载，请检查网络是否可以访问 "
            "huggingface.co，以及磁盘空间和项目目录写入权限；修复后重新运行。"
            "程序不会切换到其他模型。"
        ) from exc


def _encode(texts: list[str], show_progress: bool) -> list[list[float]]:
    model = load_model()
    # 模型有 token 上限；提前检查，避免正文尾部被静默截断。
    tokenized = model.tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
    for index, token_ids in enumerate(tokenized["input_ids"]):
        if len(token_ids) > model.max_seq_length:
            raise ValueError(
                f"第 {index + 1} 条向量化文本超过模型的 {model.max_seq_length} token 上限。"
                "请将对应章节增加段落或子标题后重新建库，程序没有截断文本。"
            )
    try:
        return model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=show_progress, convert_to_numpy=True,
        ).tolist()
    except Exception as exc:
        raise RuntimeError("生成 BGE 向量失败，请检查可用内存和本地模型文件。") from exc


def embed_documents(texts: list[str]) -> list[list[float]]:
    """调用方传入“标题 + 正文”；文档不加检索提示词。"""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("待向量化的知识片段不能为空。")
    return _encode(texts, show_progress=True)


def embed_query(question: str) -> list[float]:
    """预留问题向量化能力；仅问题添加 BGE 官方检索提示。"""
    if not question.strip():
        raise ValueError("问题不能为空。")
    return _encode([QUERY_INSTRUCTION + question.strip()], show_progress=False)[0]
