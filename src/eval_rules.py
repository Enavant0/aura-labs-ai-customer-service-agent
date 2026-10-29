"""确定性评分 V1.2：必含概念的显式同义表达、禁含否定语境与行动特征。"""

import re
import unicodedata


SCORING_VERSION = "rag-rules-v1.2"
ACTIONS = {"ANSWER", "ANSWER_WITH_CAVEAT", "REFUSE", "HANDOFF", "HANDOFF_OR_TOOL"}

# 这些规则在运行前固定，不能根据本次模型输出临时调整以提高成绩。
REFUSAL_WORDS = (
    "无法", "不能", "不允许", "不得", "不应", "不是", "不属于", "不支持", "不可以", "不建议", "未公开", "没有可确认",
    "无法确认", "不能确认", "没有公开信息", "没有公开", "暂无公开信息", "无法提供", "不能判断",
)
CAVEAT_WORDS = (
    "通常", "一般", "预计", "可能", "原则上", "建议", "不建议", "不能", "无法", "需", "取决于", "审核", "核实", "视情况", "以实际", "为准",
)
HANDOFF_PATTERNS = (
    r"转人工|人工客服|人工核实|售后审核",
    r"(?:转接|转交|转至|转|联系|咨询|提交).{0,8}(?:人工|客服|售后|财务)",
    r"(?:需要|需|请|建议|应|可|由).{0,12}(?:人工|客服|售后|财务).{0,10}(?:核查|核实|核对|处理|审核|确认)",
    r"(?:建议|请|需要|应|及时|尽快|必要时).{0,16}(?:就医|看医生|咨询医生|咨询专业医生|寻求专业医疗|寻求医疗|医疗帮助|专业医疗意见)",
)
TOOL_PATTERNS = (
    r"查询(?:业务|订单|物流|售后)系统",
    r"(?:需要|需|请|建议|应|可|必须|要|才能).{0,24}(?:查询|核查|核对|查阅).{0,16}(?:订单|物流|退款|流水|系统|业务|售后)",
    r"(?:需要|需|请|建议|应|可|必须|要).{0,20}(?:订单系统|物流系统|售后系统|业务系统).{0,12}(?:查询|核查|核对)",
    r"(?:请|建议|可|需要|需|应).{0,12}联系.{0,8}(?:快递|承运|物流公司)",
)

# 只接受同一短句、前方最多 12 字的明确否定，以及有限的修饰词。
# 限制连接词可避免“不舒服，建议继续使用”等表达因出现一个“不”而被豁免。
NEGATION_PATTERN = re.compile(
    r"(?P<negation>不能保证|不能承诺|不要承诺|不建议|不支持|不应当|不应该|不属于|不是|没有|不会|不应|无法|不得|不要|禁止|不可|不能|不)"
    r"(?:(?:您|你|再|仍然|仍|直接|随意|自行|轻易|擅自|向您|为您|给您|给你|对您|建议|推荐|承诺|保证|绝对|提供|进行|允许|应该|应当|会|能|可以|认定|声称|说))*$"
)
CLAUSE_BOUNDARY = re.compile(r"[，,。；;！？!?：:\n]|但是|不过|然而|可是|而是|但|却")


def negation_context(prefix: str) -> str | None:
    """返回直接约束后续短语的否定表达；不跨标点、转折或较远上下文。"""
    nearby = CLAUSE_BOUNDARY.split(prefix)[-1][-12:]
    nearby = re.sub(r"[“”‘’「」『』\"']", "", nearby)
    # 明显双重否定不自动豁免，避免“不能不继续使用”被当作禁止继续使用。
    if re.search(r"不是不|并非不|不能不|不得不|不可不|没有不", nearby):
        return None
    match = NEGATION_PATTERN.search(nearby)
    return match.group() if match else None


def inspect_forbidden(answer: str, keywords: list[str]) -> list[dict]:
    # 换行也作为短句边界，避免上一行的否定词错误作用于下一行。
    text = normalize_text(re.sub(r"[\r\n]+", "。", answer))
    details = []
    for keyword in keywords:
        normalized = normalize_text(keyword)
        occurrences = []
        for match in re.finditer(re.escape(normalized), text):
            negation = negation_context(text[:match.start()])
            occurrences.append({
                "position": match.start(),
                "context": text[max(0, match.start() - 16):match.end() + 8],
                "exempted": negation is not None,
                "negation": negation,
            })
        # 同一短语出现多次时，任何一次肯定表述都仍然算违规。
        violation = any(not occurrence["exempted"] for occurrence in occurrences)
        details.append({"keyword": keyword, "detected": bool(occurrences),
                        "exempted": bool(occurrences) and not violation,
                        "violation": violation, "occurrences": occurrences})
    return details


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    # 仅忽略空格、换行、大小写、全半角、常见 Markdown 标记和横线差异。
    return re.sub(r"[\s*`#_]+", "", text)


def split_keywords(value: str) -> list[str]:
    return [word.strip() for word in re.split(r"[;；]", value) if word.strip()]


def parse_concept_groups(value: str) -> list[tuple[str, list[str]]]:
    """分号之间全部必需；同一组的竖线候选满足任意一个即可。"""
    groups = []
    for group in split_keywords(value):
        alternatives = [expression.strip() for expression in re.split(r"[|｜]", group)]
        if any(not normalize_text(expression) for expression in alternatives):
            raise ValueError(f"必含概念组存在空候选表达：{group}，请检查竖线两侧。")
        groups.append((group, list(dict.fromkeys(alternatives))))
    return groups


def _positive_matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            prefix = text[:match.start()]
            # 否定的转接/查询不是正向行动；额外处理“不需要查询”“无需查询”。
            if negation_context(prefix) or re.search(r"(?:不需要|无需|不必|不会|暂不)[^，,。；;]{0,4}$|[不无别]$", prefix[-12:]):
                continue
            matches.append(match.group())
    return list(dict.fromkeys(matches))


def action_signals(answer: str) -> dict:
    text = normalize_text(re.sub(r"[\r\n]+", "。", answer))
    refusal = [word for word in REFUSAL_WORDS if word in text]
    caveats = [word for word in CAVEAT_WORDS if word in text]
    handoff = _positive_matches(text, HANDOFF_PATTERNS)
    tool = _positive_matches(text, TOOL_PATTERNS)
    return {"refusal": refusal, "caveat": caveats, "handoff": handoff, "tool": tool}


def score_answer(case: dict[str, str], answer: str) -> dict:
    normalized = normalize_text(answer)
    required = parse_concept_groups(case["must_include"])
    forbidden = split_keywords(case["must_not_include"])
    include_details = []
    for group, alternatives in required:
        matched = [expression for expression in alternatives if normalize_text(expression) in normalized]
        include_details.append({
            "concept_group": group, "alternatives": alternatives, "matched": bool(matched),
            "matched_expression": matched[0] if matched else None, "matched_expressions": matched,
        })
    missing = [item["concept_group"] for item in include_details if not item["matched"]]
    forbidden_details = inspect_forbidden(answer, forbidden)
    found_forbidden = [item["keyword"] for item in forbidden_details if item["violation"]]
    signals = action_signals(answer)
    action = case["expected_action"]
    if action not in ACTIONS:
        raise ValueError(f"不支持的 expected_action：{action}")
    action_pass = {
        # 两类正常回答不再强求固定句式或限制词，主要由必含/禁含信息判断。
        "ANSWER": bool(answer.strip()),
        "ANSWER_WITH_CAVEAT": bool(answer.strip()),
        "REFUSE": bool(signals["refusal"]),
        "HANDOFF": bool(signals["handoff"]),
        "HANDOFF_OR_TOOL": bool(signals["handoff"] or signals["tool"]),
    }[action]
    reasons = []
    if not answer.strip():
        reasons.append("实际回答为空")
    if missing:
        reasons.append("缺少必要概念组：" + "；".join(missing))
    if found_forbidden:
        reasons.append("出现未被否定语境豁免的禁止内容：" + "；".join(found_forbidden))
    if not action_pass:
        reasons.append(f"行动规则未通过：{action}（详见 action_evidence）")
    return {
        "must_include_pass": not missing,
        "must_not_include_pass": not found_forbidden,
        "missing_keywords": missing,
        "forbidden_hits": found_forbidden,
        "action_pass": action_pass,
        "action_evidence": signals,
        "rule_details": {
            "scoring_version": SCORING_VERSION,
            "evaluated": True,
            "must_include_logic": "组间全部满足（;），组内任意表达命中（|）",
            "must_include": include_details,
            "must_not_include": forbidden_details,
            "expected_action": {"expected": action, "passed": action_pass,
                                "mode": "keywords_only" if action in {"ANSWER", "ANSWER_WITH_CAVEAT"} else "action_signals",
                                "evidence": signals},
        },
        "rule_pass": not reasons,
        "failure_reasons": reasons,
    }
