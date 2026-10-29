"""仅评测来源文件的 Top 1/3/5 命中率，不调用 DeepSeek，不改写知识库。"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import MODEL_NAME, load_model
# 复用已验证的检索链路：embed_query → 显式 query_embeddings → 现有 Collection。
# 打开数据库的函数内部复用 src/vector_store.py，且不会创建 Collection。
from scripts.test_retrieval import open_existing_collection, retrieve


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    user_question: str
    expected_sources: tuple[str, ...]


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    sources: tuple[str, ...]
    headings: tuple[str, ...]
    first_hit_rank: int | None


def parse_expected_sources(value: str) -> tuple[str, ...]:
    value = value.strip()
    if value.startswith("["):
        try:
            sources = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("expected_sources 的 JSON 数组格式错误，请使用双引号，或改用 | 分隔文件名。") from exc
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise ValueError("expected_sources 必须是文件名字符串组成的数组。")
    else:
        # 兼容常见 CSV 标签写法；含逗号的字段仍需按 CSV 规则加引号。
        sources = re.split(r"[|;,，；、\n]+", value)
    sources = tuple(dict.fromkeys(source.strip() for source in sources if source.strip()))
    if not sources:
        raise ValueError("expected_sources 不能为空；没有预期来源的案例无法计算来源命中率。")
    return sources


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise ValueError(f"测试集不存在：{path}。请先将 eval_dataset.csv 放入项目的 eval 目录。")
    cases = []
    seen_ids = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            required = {"case_id", "user_question", "expected_sources"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("CSV 表头必须包含：case_id、user_question、expected_sources。")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("CSV 存在重复列名，请检查表头。")
            for row in reader:
                if None in row:
                    raise ValueError(f"CSV 第 {reader.line_num} 行列数过多；字段含逗号时请用英文双引号包围。")
                if not any(value and value.strip() for value in row.values()):
                    continue  # 完全空白的行不算测试案例。
                case_id = (row.get("case_id") or "").strip()
                question = (row.get("user_question") or "").strip()
                if not case_id or not question:
                    raise ValueError(f"CSV 第 {reader.line_num} 行的 case_id 或 user_question 为空。")
                if case_id in seen_ids:
                    raise ValueError(f"CSV 存在重复 case_id：{case_id}。")
                try:
                    expected = parse_expected_sources(row.get("expected_sources") or "")
                except ValueError as exc:
                    raise ValueError(f"CSV 第 {reader.line_num} 行（{case_id}）：{exc}") from exc
                cases.append(EvalCase(case_id, question, expected))
                seen_ids.add(case_id)
    except UnicodeError as exc:
        raise ValueError("测试集不是有效 UTF-8 文本，请使用 UTF-8 编码保存 CSV。") from exc
    except csv.Error as exc:
        raise ValueError(f"CSV 格式错误：{exc}。请检查字段引号和分隔符。") from exc
    except OSError as exc:
        raise ValueError(f"无法读取测试集：{path}，请检查文件权限。") from exc
    if not cases:
        raise ValueError("测试集没有有效案例，请检查 CSV 是否只有表头或空白行。")
    return cases


def evaluate_cases(collection, cases: list[EvalCase]) -> list[EvalResult]:
    results = []
    for index, case in enumerate(cases, start=1):
        try:
            retrieved = retrieve(collection, case.user_question, top_k=5)
            ids = (retrieved.get("ids") or [[]])[0]
            metadatas = (retrieved.get("metadatas") or [[]])[0]
            if len(ids) != len(metadatas) or len(ids) > 5:
                raise ValueError("检索结果的记录数与 metadata 不一致，或返回数量超过 5。")
            # 保留每个片段的排名；同一文件出现多次时不能去重后再计算排名。
            sources = tuple((metadata or {}).get("source_file") or "（缺少来源）" for metadata in metadatas)
            headings = tuple((metadata or {}).get("heading") or "（缺少标题）" for metadata in metadatas)
            first_hit = next((rank for rank, source in enumerate(sources, start=1)
                              if source in case.expected_sources), None)
        except Exception as exc:
            # 查询报错和“检索成功但未命中”分开；中断时不发布不完整的总体命中率。
            raise RuntimeError(f"案例 {case.case_id} 评测失败：{exc}。本次评测停止，不输出完整命中率。") from exc
        results.append(EvalResult(case, sources, headings, first_hit))
        status = f"首个正确来源在第 {first_hit} 名" if first_hit is not None else "Top 5 未命中"
        print(f"[{index}/{len(cases)}] {case.case_id}：{status}", flush=True)
    return results


def summarize(results: list[EvalResult]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("没有评测结果，无法计算命中率。")
    return {
        "total": total,
        "hits": {k: sum(result.first_hit_rank is not None and result.first_hit_rank <= k
                        for result in results) for k in (1, 3, 5)},
        "failures": [result for result in results if result.first_hit_rank is None],
        "edge_cases": [result for result in results if result.first_hit_rank in (4, 5)],
    }


def print_case(result: EvalResult) -> None:
    print("------------------------------")
    print(f"case_id：{result.case.case_id}")
    print(f"用户问题：{result.case.user_question}")
    print(f"expected_sources：{' | '.join(result.case.expected_sources)}")
    if result.first_hit_rank is not None:
        print(f"首个正确来源排名：{result.first_hit_rank}")
    print(f"实际 Top 5 来源：{list(result.sources)}")
    print("Top 5 标题：")
    for rank, (source, heading) in enumerate(zip(result.sources, result.headings), start=1):
        print(f"  {rank}. {source} / {heading}")
    if not result.sources:
        print("  （没有返回结果）")


def print_report(results: list[EvalResult]) -> None:
    summary = summarize(results)
    total = summary["total"]
    print("\n================================\nRetrieval Eval 汇总\n================================")
    print(f"总测试数：{total}")
    for k, hits in summary["hits"].items():
        print(f"Top {k} 命中：{hits}/{total}，命中率：{hits / total:.2%}")
    print(f"\nTop 5 未命中案例：{len(summary['failures'])} 个")
    for result in summary["failures"]:
        print_case(result)
    if not summary["failures"]:
        print("无。")
    print(f"\n边缘案例（首个正确来源在第 4 或第 5 名）：{len(summary['edge_cases'])} 个")
    for result in summary["edge_cases"]:
        print_case(result)
    if not summary["edge_cases"]:
        print("无。")
    print("\n评测完成。以上为来源文件命中率，不代表答案正确率或具体规则命中率。")


def main() -> int:
    parser = argparse.ArgumentParser(description="使用现有 BGE 和 Chroma 评测 Top 1/3/5 来源命中率。")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/eval_dataset.csv", help="CSV 路径，默认项目 eval/eval_dataset.csv")
    args = parser.parse_args()
    print("================================\nAURA LABS 批量检索评测\n================================", flush=True)
    try:
        cases = load_cases(args.dataset)
        print(f"测试集：{args.dataset.resolve()}")
        print(f"读取测试案例：{len(cases)} 个")
        print("判定规则：文件名精确匹配；多个 expected_sources 中任意一个命中即可，每题每档只计一次。")
        collection = open_existing_collection(PROJECT_ROOT / "chroma_db")
        print(f"Collection：{collection.name}，现有记录：{collection.count()}")
        print(f"正在加载向量模型：{MODEL_NAME}", flush=True)
        load_model()
        print_report(evaluate_cases(collection, cases))
    except KeyboardInterrupt:
        print("\n评测已中断，未输出完整评测结果。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n评测失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
