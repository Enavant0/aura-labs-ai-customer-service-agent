"""读取 UTF-8 Markdown，按标题、段落和完整句子切分知识。"""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class MarkdownDocument:
    source_file: str
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    document: str
    source_file: str
    heading: str
    chunk_id: str

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "source_file": self.source_file,
            "heading": self.heading,
            "chunk_id": self.chunk_id,
        }

    @property
    def embedding_text(self) -> str:
        # 标题参与向量化，但 Chroma 的 document 仍只保存正文。
        return f"{self.heading}\n\n{self.document}"


def find_markdown_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"知识库目录不存在：{directory}。请创建目录并放入 Markdown 文件。")
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".md"),
        key=lambda path: path.name,
    )
    if not files:
        raise ValueError(f"知识库中没有找到 .md 文件：{directory}。请先放入知识库资料。")
    # 防止同名不同大小写的扩展名造成重复 ID。
    if len({path.stem.casefold() for path in files}) != len(files):
        raise ValueError("知识库存在同名 Markdown 文件，请调整文件名，避免片段编号冲突。")
    return files


def load_documents(files: list[Path]) -> list[MarkdownDocument]:
    documents = []
    for path in files:
        try:
            # UTF-8-SIG 同时兼容带 BOM 和不带 BOM 的 UTF-8 文件。
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeError as exc:
            raise ValueError(f"{path.name} 不是有效 UTF-8 文本，请用 UTF-8 编码重新保存。") from exc
        except OSError as exc:
            raise ValueError(f"无法读取 {path.name}，请检查文件权限及文件是否仍然存在。") from exc
        if not text.strip():
            raise ValueError(f"{path.name} 是空文件，请补充知识内容后再建立知识库。")
        documents.append(MarkdownDocument(path.name, text))
    return documents


def _sections(text: str, fallback_heading: str) -> list[tuple[str, str]]:
    sections = []
    headings: list[tuple[int, str]] = []
    lines: list[str] = []
    fence_char = ""
    fence_length = 0

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            # 完整标题路径避免“基础信息”等同名小节失去所属产品上下文。
            heading = " / ".join(title for _, title in headings) or fallback_heading
            sections.append((heading, body))
        lines.clear()

    for line in text.splitlines():
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = ""
            lines.append(line)
            continue
        if fence_char:
            lines.append(line)
            continue
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            title = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
            headings = [(depth, name) for depth, name in headings if depth < level]
            headings.append((level, title))
        elif re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", line):
            lines.append("")  # 分隔线不是业务知识。
        else:
            lines.append(line)
    flush()
    return sections


def _natural_units(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    # 长列表先在列表项之间切分，保留每条规则。普通长段落再按完整句子拆分。
    items = re.split(r"(?m)(?=^[ \t]*(?:[-+*]|\d+[.)、])[ \t]+)", paragraph)
    units = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if len(item) <= max_chars:
            units.append(item)
        else:
            # 句号、问号、感叹号是句子边界，分号不强行当作一条规则的结束。
            sentences = re.findall(r'.+?(?:[。！？!?]+[”’」』）)]*|\n|$)', item, flags=re.S)
            units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units


def split_documents(documents: list[MarkdownDocument], max_chars: int = 600) -> list[KnowledgeChunk]:
    if max_chars <= 0:
        raise ValueError("片段目标上限必须大于 0。")
    chunks = []
    for source in documents:
        number = 0
        for heading, body in _sections(source.text, Path(source.source_file).stem):
            paragraphs = re.split(r"\n[ \t]*\n", body)
            units = [unit for paragraph in paragraphs if paragraph.strip()
                     for unit in _natural_units(paragraph.strip(), max_chars)]
            current = ""
            bodies = []
            for unit in units:
                candidate = f"{current}\n\n{unit}" if current else unit
                if current and len(candidate) > max_chars:
                    bodies.append(current)
                    current = unit
                else:
                    current = candidate
            if current:
                bodies.append(current)
            for document in bodies:
                number += 1
                chunks.append(KnowledgeChunk(
                    document=document,
                    source_file=source.source_file,
                    heading=heading,
                    chunk_id=f"{Path(source.source_file).stem}_{number:03d}",
                ))
    if not chunks:
        raise ValueError("没有可用的知识正文，请检查文件是否只有标题或分隔线。")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("发现重复的片段编号，请检查知识库文件名。")
    return chunks
