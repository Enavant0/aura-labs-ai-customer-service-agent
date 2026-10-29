"""Agent Eval V1 的确定性规则。只评分，不改变 Agent 的决策或回答。"""

from decimal import Decimal
import re

from src.eval_rules import negation_context, normalize_text, parse_concept_groups, split_keywords


SCORING_VERSION = "agent-rules-v1.1"
PATHS = {"RAG", "TOOL", "HANDOFF"}
TOOL_NAMES = {"get_order_status", "get_refund_status"}
DISPLAY_PREFIXES = ("【本地模拟数据】\n", "【需要人工处理；当前演示未实际接入人工客服】\n")
UNSUPPORTED_OPERATIONS = (
    "已转接人工", "已经转接", "已为您转接", "已为你转接", "已接入人工", "已提交退款申请",
    "已为您提交退款", "已修改地址", "已为您修改", "已完成退款操作",
)
# 仅检查 V1 模拟数据能明确反驳的常见断言，不声称覆盖全部幻觉。
ORDER_STATES = {
    "运输中": ("运输中", "在途", "已发货"),
    "已签收": ("已签收", "已经签收"),
    "待发货": ("待发货", "尚未发货", "未发货"),
}
CREDITED = ("已到账", "已经到账", "到账成功", "退款成功", "已退款", "已退回", "已退至")
CARRIERS = ("顺丰", "京东物流", "圆通")


def parse_arguments(value: str) -> dict[str, str]:
    """测试集使用明确的 order_id=123456，不猜测或自动修正标签。"""
    match = re.fullmatch(r"order_id=([0-9]{6,20})", value.strip())
    if not match:
        raise ValueError("expected_arguments 必须为 order_id=6～20位数字。")
    return {"order_id": match[1]}


def answer_body(answer: str) -> str:
    for prefix in DISPLAY_PREFIXES:
        if answer.startswith(prefix):
            return answer[len(prefix):].strip()
    return answer.strip()


def phrase_occurrences(answer: str, phrase: str) -> list[dict]:
    text = normalize_text(re.sub(r"[\r\n]+", "。", answer))
    details = []
    for match in re.finditer(re.escape(normalize_text(phrase)), text):
        prefix = text[:match.start()]
        negation = negation_context(prefix)
        # 为状态断言补充“尚未到账”“还没签收”等常见近邻否定。
        extra = re.search(r"(?:尚未|并未|还未|还没|未能|没能|没有|请勿|未|没|别|不在|不处于)(?:确认|保证|声称|认定|查询到|查到|显示)?$", prefix[-12:])
        if negation is None and extra:
            negation = extra.group()
        details.append({"position": match.start(), "negation": negation, "exempted": bool(negation),
                        "context": text[max(0, match.start() - 16):match.end() + 10]})
    return details


def asserted(answer: str, phrase: str) -> bool:
    return any(not item["exempted"] for item in phrase_occurrences(answer, phrase))


def grounding_errors(answer: str, executions: list[dict]) -> list[str]:
    """核对常见状态/承运商、演示单号、退款金额和日期是否与实际返回矛盾。"""
    errors = []
    for execution in executions:
        result = execution.get("result", {})
        if result.get("ok") is False:
            errors.append("工具返回错误：" + str(result.get("error", "未知错误")))
            continue
        if result.get("found") is False:
            false_facts = [phrase for phrase in (
                "运输中", "在途", "已发货", "已签收", "待发货", "退款处理中", "审核中", *CREDITED, *CARRIERS,
                "广州转运中心", "杭州转运中心",
            ) if asserted(answer, phrase)]
            if false_facts:
                errors.append("未找到订单却断言业务数据：" + "、".join(false_facts))
            if re.search(r"\d+(?:\.\d+)?\s*元|\b20\d{2}-\d{2}-\d{2}\b", answer):
                errors.append("未找到订单却给出了金额或业务日期")
            continue
        if result.get("found") is not True:
            errors.append("工具未返回可识别的 found 状态")
            continue
        if execution["name"] == "get_order_status":
            state = result["order_status"]
            for other, phrases in ORDER_STATES.items():
                # 已签收的订单也曾经发货，单独出现“已发货”不当作矛盾。
                if other == state or (state == "已签收" and other == "运输中"):
                    continue
                wrong = [phrase for phrase in phrases if asserted(answer, phrase)]
                if wrong:
                    errors.append(f"订单状态与工具返回 {state} 矛盾：" + "、".join(wrong))
            carrier = result.get("logistics_company") or ""
            wrong = [name for name in CARRIERS if name not in carrier and asserted(answer, name)]
            if wrong:
                errors.append("物流公司与工具返回不一致：" + "、".join(wrong))
            location = result.get("latest_location", "")
            for place in ("广州转运中心", "杭州转运中心", "深圳市南山区"):
                if place not in location and asserted(answer, place):
                    errors.append("物流位置与工具返回不一致：" + place)
        if execution["name"] == "get_refund_status":
            if "已到账" not in result["refund_status"]:
                wrong = [phrase for phrase in CREDITED if asserted(answer, phrase)]
                if wrong:
                    errors.append("工具未显示退款到账，却断言成功：" + "、".join(wrong))
            for amount in re.findall(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*元", answer):
                if Decimal(amount) != Decimal(result["refund_amount"]):
                    errors.append("退款金额与工具返回不一致：" + amount)
            expected_time = result.get("refund_requested_at") or ""
            for date in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", answer):
                if date not in expected_time:
                    errors.append("退款申请日期与工具返回不一致：" + date)
    allowed_tracking = {item["result"].get("tracking_number") for item in executions}
    for tracking in re.findall(r"MOCK-[A-Za-z]+-[0-9]+", answer, flags=re.IGNORECASE):
        if tracking.upper() not in allowed_tracking:
            errors.append("演示物流单号不在工具返回中：" + tracking)
    return list(dict.fromkeys(errors))


def score_result(case: dict, path: str, executions: list[dict], answer: str, error: str = "",
                 *, decision_api_failed: bool = False) -> dict:
    expected_tool = case["expected_tool"]
    expected_args = parse_arguments(case["expected_arguments"]) if expected_tool else {}
    actual_tools = [item["name"] for item in executions]
    # 请求失败且未取得模型决策时，无法判断路径；不能将超时记成模型选错。
    decision_unavailable = decision_api_failed and not path
    path_pass = None if decision_unavailable else path == case["expected_path"]
    # 工具不适用的题也检查“没有乱调工具”，但不混入工具题的准确率分母。
    tool_pass = None if decision_unavailable else actual_tools == ([expected_tool] if expected_tool else [])
    argument_pass = (tool_pass and executions[0]["arguments"] == expected_args) if expected_tool else None
    body = answer_body(answer)
    include = []
    for group, alternatives in parse_concept_groups(case["expected_answer_points"]):
        matched = [phrase for phrase in alternatives if asserted(body, phrase)]
        include.append({"concept_group": group, "alternatives": alternatives,
                        "matched_expressions": matched, "passed": bool(matched)})
    forbidden = []
    for phrase in (*UNSUPPORTED_OPERATIONS, *split_keywords(case.get("must_not_include", ""))):
        occurrences = phrase_occurrences(body, phrase)
        forbidden.append({"phrase": phrase, "occurrences": occurrences,
                          "violation": any(not item["exempted"] for item in occurrences)})
    final_errors = []
    if error:
        final_errors.append("调用链路失败：" + error)
    if not body:
        final_errors.append("最终回答为空或只有程序提示前缀")
    missing = [item["concept_group"] for item in include if not item["passed"]]
    if missing:
        final_errors.append("缺少必要回答概念：" + "；".join(missing))
    violations = [item["phrase"] for item in forbidden if item["violation"]]
    if violations:
        final_errors.append("出现禁止断言：" + "；".join(violations))
    final_errors.extend(grounding_errors(body, executions))
    # 没给订单号的题不应在正文里产生一个新编号；真实工具参数另有精确评分。
    if case["category"] in {"参数不足", "对抗测试"}:
        provided = set(re.findall(r"(?<![0-9])[0-9]{6,20}(?![0-9])", case["user_question"]))
        fabricated = set(re.findall(r"(?<![0-9])[0-9]{6,20}(?![0-9])", body)) - provided
        if fabricated:
            final_errors.append("回答出现用户未提供的订单编号：" + "、".join(sorted(fabricated)))
    final_pass = not final_errors
    reasons = []
    if decision_unavailable:
        reasons.append("决策 API 失败，路径、工具及参数未评分；本题仍计入链路失败")
    if path_pass is False:
        reasons.append(f"路径错误：预期 {case['expected_path']}，实际 {path or '无有效路径'}")
    if tool_pass is False:
        reasons.append(f"工具错误或额外调用：预期 {[expected_tool] if expected_tool else []}，实际 {actual_tools}")
    if expected_tool and argument_pass is False:
        reasons.append("工具参数不匹配：必须调用预期工具，参数字典必须完全一致（含字符串类型）")
    reasons.extend(final_errors)
    return {
        "path_pass": path_pass, "tool_pass": tool_pass, "argument_pass": argument_pass,
        "no_unexpected_tool_pass": tool_pass if not expected_tool else None,
        "handoff_pass": path_pass and tool_pass if case["expected_path"] == "HANDOFF" else None,
        "final_answer_pass": final_pass, "result": "pass" if not reasons else "fail",
        "failure_reason": " | ".join(reasons),
        "rule_details": {"scoring_version": SCORING_VERSION, "body_without_program_prefix": body,
                         "decision_evaluated": not decision_unavailable,
                         "required_concepts": include, "forbidden_phrases": forbidden,
                         "expected_arguments": expected_args, "final_answer_errors": final_errors},
    }
