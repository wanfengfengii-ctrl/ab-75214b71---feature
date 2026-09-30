"""复原 API 入参校验。"""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    FRAME_COUNT_MAX,
    FRAME_COUNT_MIN,
    PAYLOAD_MAX_LEN,
    PAYLOAD_MIN_LEN,
    SLIPPAGE_MAX_LIMIT,
    SYNC_MAX_LEN,
    SYNC_MIN_LEN,
)

RECEIVED_MAX_LEN = 4096  # 远大于 8 帧 * 68 位 + 滑移余量，防止异常巨量输入


class ValidationError(Exception):
    def __init__(self, fields: dict[str, str]):
        self.fields = fields
        super().__init__("; ".join(f"{k}: {v}" for k, v in fields.items()))


@dataclass(frozen=True)
class RecoverRequest:
    received: str
    frame_count: int
    sync: str
    payload_len: int
    max_slippage: int


def _is_int(value) -> bool:
    # 拒绝 bool（Python 中 bool 是 int 子类）
    return isinstance(value, int) and not isinstance(value, bool)


def validate(data: object) -> RecoverRequest:
    """校验请求 JSON；错误聚合为 ``{字段: 中文说明}`` 一次性返回。"""
    errors: dict[str, str] = {}

    if not isinstance(data, dict):
        raise ValidationError({"_body": "请求体必须是 JSON 对象"})

    # received
    received = data.get("received")
    if "received" not in data or received is None:
        errors["received"] = "必填：接收比特串（仅含 0/1 的字符串）"
    elif not isinstance(received, str):
        errors["received"] = "必须是 0/1 组成的字符串"
    elif len(received) == 0:
        errors["received"] = "不得为空串"
    elif not all(c in "01" for c in received):
        errors["received"] = "只能包含字符 0 和 1"
    elif len(received) > RECEIVED_MAX_LEN:
        errors["received"] = f"长度不得超过 {RECEIVED_MAX_LEN} 位"

    # frame_count
    fc = data.get("frame_count")
    if "frame_count" not in data or fc is None:
        errors["frame_count"] = f"必填：帧数（{FRAME_COUNT_MIN}..{FRAME_COUNT_MAX}）"
    elif not _is_int(fc):
        errors["frame_count"] = "必须是整数"
    elif not (FRAME_COUNT_MIN <= fc <= FRAME_COUNT_MAX):
        errors["frame_count"] = (
            f"必须在 {FRAME_COUNT_MIN}..{FRAME_COUNT_MAX} 之间")

    # sync
    sync = data.get("sync")
    if "sync" not in data or sync is None:
        errors["sync"] = (f"必填：{SYNC_MIN_LEN}..{SYNC_MAX_LEN} 位同步字"
                          "（0/1 字符串）")
    elif not isinstance(sync, str):
        errors["sync"] = "必须是 0/1 组成的字符串"
    elif not all(c in "01" for c in sync):
        errors["sync"] = "只能包含字符 0 和 1"
    elif not (SYNC_MIN_LEN <= len(sync) <= SYNC_MAX_LEN):
        errors["sync"] = f"长度必须在 {SYNC_MIN_LEN}..{SYNC_MAX_LEN} 位之间"

    # payload_len
    pl = data.get("payload_len")
    if "payload_len" not in data or pl is None:
        errors["payload_len"] = (
            f"必填：载荷长度（{PAYLOAD_MIN_LEN}..{PAYLOAD_MAX_LEN} 位）")
    elif not _is_int(pl):
        errors["payload_len"] = "必须是整数"
    elif not (PAYLOAD_MIN_LEN <= pl <= PAYLOAD_MAX_LEN):
        errors["payload_len"] = (
            f"必须在 {PAYLOAD_MIN_LEN}..{PAYLOAD_MAX_LEN} 之间")

    # max_slippage
    ms = data.get("max_slippage")
    if "max_slippage" not in data or ms is None:
        errors["max_slippage"] = (
            f"必填：滑移预算（0..{SLIPPAGE_MAX_LIMIT}）")
    elif not _is_int(ms):
        errors["max_slippage"] = "必须是整数"
    elif not (0 <= ms <= SLIPPAGE_MAX_LIMIT):
        errors["max_slippage"] = f"必须在 0..{SLIPPAGE_MAX_LIMIT} 之间"

    if errors:
        raise ValidationError(errors)

    return RecoverRequest(received, fc, sync, pl, ms)
