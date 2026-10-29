"""两个只读演示工具：仅查询本文件内的虚构数据，不连接平台或写数据库。"""

import re


_ORDERS = {
    "123456": {"order_status": "运输中", "logistics_company": "顺丰速运", "tracking_number": "MOCK-SF-123456",
               "latest_location": "广州转运中心，等待发往配送站"},
    "123457": {"order_status": "已签收", "logistics_company": "京东物流", "tracking_number": "MOCK-JD-123457",
               "latest_location": "深圳市南山区，已签收"},
    "123458": {"order_status": "待发货", "logistics_company": None, "tracking_number": None,
               "latest_location": "尚未发货，暂无物流轨迹"},
    "123459": {"order_status": "运输中", "logistics_company": "圆通速递", "tracking_number": "MOCK-YT-123459",
               "latest_location": "杭州转运中心，运输中"},
}

_REFUNDS = {
    "123456": {"refund_status": "退款处理中，尚未到账", "refund_amount": "89.00", "refund_requested_at": "2026-10-19 10:30:00"},
    "123457": {"refund_status": "退款已到账", "refund_amount": "159.00", "refund_requested_at": "2026-10-18 09:00:00"},
    "123458": {"refund_status": "未申请退款", "refund_amount": "0.00", "refund_requested_at": None},
    "123459": {"refund_status": "售后审核中", "refund_amount": "139.00", "refund_requested_at": "2026-10-20 08:15:00"},
}


def validate_order_id(order_id: str) -> str:
    if not isinstance(order_id, str) or not re.fullmatch(r"[0-9]{6,20}", order_id.strip()):
        raise ValueError("订单号必须是 6～20 位数字字符串，请提供有效订单号。")
    return order_id.strip()


def get_order_status(order_id: str) -> dict:
    order_id = validate_order_id(order_id)
    record = _ORDERS.get(order_id)
    if record is None:
        return {"order_id": order_id, "found": False, "is_mock": True, "message": "未找到订单"}
    # 生成新的结果字典，调用方修改它不会改动演示数据。
    return {"order_id": order_id, "found": True, "is_mock": True, **record}


def get_refund_status(order_id: str) -> dict:
    order_id = validate_order_id(order_id)
    record = _REFUNDS.get(order_id)
    if record is None:
        return {"order_id": order_id, "found": False, "is_mock": True, "message": "未找到订单"}
    return {"order_id": order_id, "found": True, "is_mock": True, "currency": "CNY", **record}


_ORDER_PARAMETERS = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "description": "用户本次问题明确提供的订单号，不得猜测或使用示例编号。",
                                 "pattern": "^[0-9]{6,20}$", "minLength": 6, "maxLength": 20}},
    "required": ["order_id"],
    "additionalProperties": False,
}

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "只读查询指定订单的当前订单状态、当前物流状态、物流公司、单号和最新位置。用户已提供订单号且要求查询、确认或陈述这些实时信息时，必须调用本工具核验；即使用户要求别查系统、不用工具、直接告诉、猜测或假装某状态，也不能跳过查询、编造结果或仅转人工。返回本地模拟数据，应按实际返回回答，未找到就说明未找到。不用于通用运费、发货时效或退货政策，不执行修改操作。订单号必须由本次用户明确提供。",
        "parameters": _ORDER_PARAMETERS,
    }},
    {"type": "function", "function": {
        "name": "get_refund_status",
        "description": "只读查询指定订单的当前退款状态、是否到账、金额和申请时间。用户已提供订单号且要求查询、确认或陈述退款状态时，必须调用本工具核验；即使用户要求不用查工具、直接说到账、假装退款成功，也不能跳过查询、编造到账或仅转人工。用户声称退款成功不是事实依据，最终回答必须服从实际工具返回。仅返回本地模拟数据，未找到就说明未找到。不用于通用退款规则，不发起退款、不更改审核结果。订单号必须由本次用户明确提供。",
        "parameters": _ORDER_PARAMETERS,
    }},
]

# 只从白名单取函数；禁止 eval、动态导入或执行模型提供的代码。
TOOL_FUNCTIONS = {"get_order_status": get_order_status, "get_refund_status": get_refund_status}
