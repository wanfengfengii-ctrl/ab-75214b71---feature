"""帧模型、CRC-8 校验与联合插入/漏失复原求解器。

信道模型
--------
原始发送序列为若干等长帧 ``同步字 | 载荷 | CRC8`` 的拼接。接收序列相对
校正（发送）序列只可能偶发地：

* 插入一个比特（接收流中多出一个噪声比特）；
* 漏失一个比特（发送的某一位未被收到）。

二者各计一次滑移。复原在整条接收比特串上联合解释所有插入与漏失，先最小化
滑移次数，再在同代价解中取校正串字典序最小者，并判定最优解是否唯一。

算法
----
在整条接收流上做分层动态规划，帧内状态为
``(接收游标 i, 已用滑移 cost, CRC 寄存器 reg)``，帧内发送偏移 j 按层推进。
转移只允许：

* match     收发各消耗一比特且必须相等（同步字区域还须等于同步字位）；
* deletion  发送消耗一比特、接收不动（漏失），该位取 0/1 两个候选，代价 +1；
* insertion 接收消耗一比特、发送不动（插入），代价 +1，校正串不变。

到帧末时再移入 8 个零，余数为零则帧校验通过，状态折叠到下一帧帧首。

每个状态只保留一条字典序最小的校正前缀，并以标志位记录到达该状态的
**不同校正串**是 1 个还是多个（不同脚本可能产生同一串，合并时显式去重）。
帧边界同一接收位置只保留最小代价：此后未来可行集只取决于该位置与剩余
预算，高代价路径不可能进入全局最优解。滑移预算 <= 6，同一层接收游标满足
``|i-j| <= cost``，状态空间有界。

业务约束
--------
固定长度模式下所有帧等长；启用**帧内长度**（``intra_frame_length``）后，
每帧结构为 ``同步字 | 6 位长度码 | 载荷余部 | CRC8``，长度码按无符号大端
编码"总载荷长度 - 16"，仅 0..32 有效（载荷 16..48 位）。帧的真实长度须
在收到长度码后才确定，求解器在同一条接收流上联合恢复长度码、帧边界、
载荷、CRC 与滑移；长度码越界（33..63）的候选不得成帧。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# 业务约束
FRAME_COUNT_MIN = 3
FRAME_COUNT_MAX = 8
SYNC_MIN_LEN = 6
SYNC_MAX_LEN = 12
PAYLOAD_MIN_LEN = 16
PAYLOAD_MAX_LEN = 48
SLIPPAGE_MAX_LIMIT = 6
CRC_LEN = 8
CRC_POLY = 0x07  # x^8 + x^2 + x + 1，省略最高项 x^8
# 帧内长度模式：每帧载荷前 6 位为无符号大端长度码 = 总载荷长度 - 16，
# 即合法码 0..32 对应载荷 16..48 位；33..63 越界，不得成帧。
LENGTH_FIELD_LEN = 6
LENGTH_CODE_MAX = PAYLOAD_MAX_LEN - PAYLOAD_MIN_LEN  # 32

# 不同校正串数量的截断表示：1 = 唯一，2 = 多个（唯一性判定只需布尔）。
MULTIPLE = 2


def crc8(bits: str) -> int:
    """对 0/1 比特串计算 CRC-8（MSB 优先，初值零）。

    多项式 x^8+x^2+x+1（0x07）。8 位移位寄存器处理消息位的结果，代数上
    正是"消息多项式补八个零后对生成多项式取余"（CRC-8/SMBus，目录校验
    值 "123456789" -> 0xF4）。
    """
    crc = 0
    for bit in bits:
        crc ^= (ord(bit) - ord("0")) << 7
        if crc & 0x80:
            crc = ((crc << 1) ^ CRC_POLY) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
    return crc


def _crc_step(reg: int, bit: int) -> int:
    v = reg ^ (bit << 7)
    if v & 0x80:
        return ((v << 1) ^ CRC_POLY) & 0xFF
    return (v << 1) & 0xFF


def frame_is_valid(frame: str, sync: str, payload_len: int) -> bool:
    """校验一帧：长度/同步字匹配且 CRC 正确。"""
    if len(frame) != len(sync) + payload_len + CRC_LEN:
        return False
    if not frame.startswith(sync):
        return False
    body = frame[:-CRC_LEN]
    # crc8(body) 即 body 补八个零后对生成多项式的余数
    return crc8(body) == int(frame[-CRC_LEN:], 2)


@dataclass(frozen=True)
class FrameResult:
    index: int
    payload: str
    crc: str
    raw: str
    payload_length: int | None = None
    length_field: str | None = None

    def to_dict(self) -> dict:
        out = {"index": self.index, "payload": self.payload,
               "crc": self.crc, "raw": self.raw}
        if self.length_field is not None:
            out["payload_length"] = self.payload_length
            out["length_field"] = self.length_field
        return out


@dataclass(frozen=True)
class SlipEvent:
    kind: str  # "insertion" | "deletion"
    position: int  # 基于校正串（发送侧）的 0 计位
    frame_index: int | None
    offset: int | None
    bit: str | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "position": self.position,
            "frame_index": self.frame_index,
            "offset": self.offset,
            "bit": self.bit,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReconstructionResult:
    recoverable: bool
    corrected: str | None
    frames: tuple[FrameResult, ...]
    slippage_count: int
    events: tuple[SlipEvent, ...]
    unique: bool
    alternatives: int  # 0 或 >=1（不同同代价校正串数量的截断标志）
    minimum_slippage_lower_bound: int | None
    budget: int

    def to_dict(self) -> dict:
        if self.recoverable:
            return {
                "recoverable": True,
                "corrected": self.corrected,
                "frames": [f.to_dict() for f in self.frames],
                "slippage_count": self.slippage_count,
                "events": [e.to_dict() for e in self.events],
                "unique": self.unique,
                "alternatives": self.alternatives,
                "budget": self.budget,
            }
        return {
            "recoverable": False,
            "reason": "滑移预算内不存在通过同步字与 CRC 校验的完整帧流",
            "minimum_slippage_lower_bound": self.minimum_slippage_lower_bound,
            "verified_up_to": self.budget,
            "budget": self.budget,
        }


def reconstruct(received: str, frame_count: int, sync: str, payload_len: int | None,
                max_slippage: int,
                intra_frame_length: bool = False) -> ReconstructionResult:
    """在整条接收流上联合复原 ``frame_count`` 个帧。

    ``intra_frame_length`` 为假时所有帧等长（``payload_len`` 必填）；为真时
    每帧载荷长度由该帧前 6 位长度码决定（``payload_len`` 应为 None）。

    滑移预算从 0 逐档放宽：第一档存在完整帧流时，该档代价即为全局最小
    滑移次数（无滑移档只有唯一的匹配路径，求解极快）。
    """
    n = len(received)
    rb = [ord(c) - ord("0") for c in received]
    sb = [ord(c) - ord("0") for c in sync]

    if intra_frame_length:
        runner = lambda budget: _run_budget_variable(
            rb, sb, frame_count, n, budget)
        split_args = (True, None, None)
        # 变长模式无先验总长度，delta 剪枝不适用；下界只反映预算+1。
        lower_bound = max_slippage + 1
    else:
        frame_len = len(sync) + payload_len + CRC_LEN
        total_len = frame_count * frame_len
        delta = n - total_len  # 全局 插入数 - 漏失数
        runner = lambda budget: _run_budget(
            rb, sb, frame_count, frame_len, n, delta, budget)
        split_args = (False, payload_len, frame_len)
        lower_bound = max(max_slippage + 1, abs(delta))

    for budget in range(0, max_slippage + 1):
        answer = runner(budget)
        if answer is not None:
            best_cost, finals = answer
            tied = [(rep, count) for i, (c, rep, count) in finals.items()
                    if i == n and c == best_cost]
            corrected = min(rep for rep, _ in tied)
            unique = len(tied) == 1 and tied[0][1] == 1
            frames, events = _split_frames(
                corrected, received, sync, *split_args)
            return ReconstructionResult(
                recoverable=True, corrected=corrected, frames=tuple(frames),
                slippage_count=best_cost, events=tuple(events),
                unique=unique, alternatives=0 if unique else 1,
                minimum_slippage_lower_bound=None, budget=max_slippage,
            )

    return ReconstructionResult(
        recoverable=False, corrected=None, frames=(),
        slippage_count=0, events=(), unique=False, alternatives=0,
        minimum_slippage_lower_bound=lower_bound, budget=max_slippage,
    )


def _run_budget(rb, sb, frame_count, frame_len, n, delta, budget):
    """在固定滑移预算下做分层帧 DP；成功返回 (best_cost, finals)。"""
    sync_len = len(sb)
    # 帧边界状态：{接收游标 i: (代价, 字典序最小校正前缀, 不同串标志)}
    boundary: dict[int, tuple[int, str, int]] = {0: (0, "", 1)}
    finals: dict[int, tuple[int, str, int]] = {}

    for k in range(frame_count):
        if not boundary:
            return None
        # 帧内状态：(i, cost, reg) -> (不同串标志, 字典序最小前缀)
        cur: dict[tuple[int, int, int], tuple[int, str]] = {
            (i, cost, 0): (count, rep)
            for i, (cost, rep, count) in boundary.items()
        }
        sent_base = k * frame_len

        for j in range(frame_len):
            cur = _insertion_closure(cur, n, budget, sent_base + j, delta)
            nxt: dict[tuple[int, int, int], tuple[int, str]] = {}
            expected = sb[j] if j < sync_len else None
            sent_here = sent_base + j
            for (i, cost, reg), (count, rep) in cur.items():
                # match：收发同位且必须相等
                if i < n:
                    b = rb[i]
                    if expected is None or b == expected:
                        ni, nc, nr = i + 1, cost, _crc_step(reg, b)
                        if _feasible(ni, nc, sent_here + 1, delta, budget):
                            _merge(nxt, (ni, nc, nr), count, rep + str(b))
                # deletion：漏失的发送位（同步字区域值唯一）
                if cost < budget:
                    candidates = ((expected,) if expected is not None
                                  else (0, 1))
                    for b in candidates:
                        ni, nc = i, cost + 1
                        if _feasible(ni, nc, sent_here + 1, delta, budget):
                            _merge(nxt, (ni, nc, _crc_step(reg, b)),
                                   count, rep + str(b))
            cur = nxt

        cur = _insertion_closure(cur, n, budget, sent_base + frame_len, delta)
        folded: dict[int, tuple[int, str, int]] = {}
        for (i, cost, reg), (count, rep) in cur.items():
            if reg != 0:
                continue
            _fold(folded, i, cost, rep, count)

        if k == frame_count - 1:
            for i, v in folded.items():
                _fold(finals, i, *v)
        boundary = folded

    best_cost = min((c for i, (c, _, _) in finals.items() if i == n),
                    default=None)
    if best_cost is None:
        return None
    return best_cost, finals


def _run_budget_variable(rb, sb, frame_count, n, budget):
    """帧内长度模式下的分层帧 DP。

    每帧发送侧依次为：同步字（定长）→ 6 位无符号大端长度码 →
    载荷余部（位数由码决定）→ 8 位 CRC。帧内状态键为
    ``(i, cost, reg, code)``：code 在同步字阶段为 None，长度码移入期间
    累积为 0..63，码越界（>32）的状态直接丢弃；不同长度码的状态绝不合并，
    只有帧边界（reg 归零）处才按接收游标折叠。
    """
    sync_len = len(sb)
    frame_min = sync_len + PAYLOAD_MIN_LEN + CRC_LEN       # S + 24
    frame_max = sync_len + PAYLOAD_MAX_LEN + CRC_LEN       # S + 56
    # 帧边界状态：{i: (代价, 代表校正串, 不同串标志)}
    boundary: dict[int, tuple[int, str, int]] = {0: (0, "", 1)}
    finals: dict[int, tuple[int, str, int]] = {}

    def prune(cur, rem_lo, rem_hi, rem_after):
        """后缀可行性剪枝。

        本帧剩余发送位数 ∈ [rem_lo, rem_hi]（消费后口径），其后还有
        rem_after 个整帧（每个 Fmin..Fmax 位）。令 d = 剩余接收位数、
        S = 剩余发送位数，则后缀至少需要 |d-S| 次滑移；超过剩余预算的
        状态直接丢弃。
        """
        lo = rem_lo + rem_after * frame_min
        hi = rem_hi + rem_after * frame_max
        out = {}
        for key, val in cur.items():
            i, cost = key[0], key[1]
            d = n - i
            need = lo - d if d < lo else (d - hi if d > hi else 0)
            if need <= budget - cost:
                out[key] = val
        return out

    def advance(cur, expected, rem_lo, rem_hi, rem_after, code_bit=False):
        """推进一个发送位；rem_* 为消费前的剩余位数，消费后减一。"""
        cur = _insertion_closure_any(cur, n, budget)
        nxt: dict[tuple, tuple[int, str]] = {}
        for key, (count, rep) in cur.items():
            i, cost, reg, code = key
            if i < n:
                b = rb[i]
                if expected is None or b == expected:
                    nc_code = ((code or 0) << 1 | b) if code_bit else code
                    nkey = (i + 1, cost, _crc_step(reg, b), nc_code)
                    _merge(nxt, nkey, count, rep + str(b))
            if cost < budget:
                candidates = ((expected,) if expected is not None
                              else (0, 1))
                for b in candidates:
                    nc_code = ((code or 0) << 1 | b) if code_bit else code
                    nkey = (i, cost + 1, _crc_step(reg, b), nc_code)
                    _merge(nxt, nkey, count, rep + str(b))
        return prune(nxt, rem_lo - 1, rem_hi - 1, rem_after)

    for k in range(frame_count):
        if not boundary:
            return None
        rem_after = frame_count - 1 - k
        cur: dict[tuple, tuple[int, str]] = {
            (i, cost, 0, None): (count, rep)
            for i, (cost, rep, count) in boundary.items()
        }

        # 1) 同步字：位值固定
        for j in range(sync_len):
            cur = advance(cur, sb[j],
                          rem_lo=frame_min - j, rem_hi=frame_max - j,
                          rem_after=rem_after)

        # 2) 6 位长度码（大端，位值自由，码值累积）
        for c in range(LENGTH_FIELD_LEN):
            consumed = sync_len + c
            cur = advance(cur, None,
                          rem_lo=frame_min - consumed,
                          rem_hi=frame_max - consumed,
                          rem_after=rem_after, code_bit=True)

        # 3) 按码分组：越界码不得成帧；合法码继续推进载荷余部与 CRC
        completed: dict[tuple, tuple[int, str]] = {}
        groups: dict[int, dict[tuple, tuple[int, str]]] = {}
        for key, val in cur.items():
            code = key[3]
            if code is not None and code <= LENGTH_CODE_MAX:
                groups.setdefault(code, {})[key] = val
        for code, states in groups.items():
            payload_len = PAYLOAD_MIN_LEN + code
            frame_len_code = sync_len + payload_len + CRC_LEN
            rest = (payload_len - LENGTH_FIELD_LEN) + CRC_LEN  # 载荷余部+CRC
            g = states
            for q in range(rest):
                consumed = sync_len + LENGTH_FIELD_LEN + q
                # 码已定，本帧发送总长固定：剩余位数用真实帧长
                g = advance(g, None,
                            rem_lo=frame_len_code - consumed,
                            rem_hi=frame_len_code - consumed,
                            rem_after=rem_after)
            completed.update(g)

        # 4) 插入闭包 + CRC 余数归零，折叠到帧边界
        completed = _insertion_closure_any(completed, n, budget)
        folded: dict[int, tuple[int, str, int]] = {}
        for key, (count, rep) in completed.items():
            i, cost, reg, _code = key
            if reg != 0:
                continue
            _fold(folded, i, cost, rep, count)

        if k == frame_count - 1:
            for i, v in folded.items():
                _fold(finals, i, *v)
        boundary = folded

    best_cost = min((c for i, (c, _, _) in finals.items() if i == n),
                    default=None)
    if best_cost is None:
        return None
    return best_cost, finals


def _insertion_closure_any(cur, n, budget):
    """变长 DP 的插入闭包：状态键前两位固定为 (i, cost)，其余透传。"""
    out = dict(cur)
    queue = deque(out.keys())
    while queue:
        key = queue.popleft()
        i, cost = key[0], key[1]
        if i >= n or cost >= budget:
            continue
        count, rep = out[key]
        nkey = (i + 1, cost + 1) + tuple(key[2:])
        if _merge(out, nkey, count, rep):
            queue.append(nkey)
    return out


def _feasible(i, cost, sent_done, delta, budget):
    """后缀可行性剪枝：剩余 插入-漏失 差必须能被剩余预算吸收。

    已发生 插入-漏失 = i(已消耗接收位) - sent_done(已消耗发送位)；
    后缀必须满足 (插入-漏失) = delta - q，其最小代价为 |delta-q|。
    """
    q = i - sent_done
    return abs(delta - q) <= budget - cost


def _insertion_closure(cur, n, budget, sent_done, delta):
    """固定帧内偏移 j 上沿接收方向传播插入边 (i,c,r)->(i+1,c+1,r)。

    校正串不变；用队列做有界 BFS，串标志沿用来源状态。
    """
    out = dict(cur)
    queue = deque(out.keys())
    while queue:
        i, cost, reg = queue.popleft()
        if i >= n or cost >= budget:
            continue
        if not _feasible(i + 1, cost + 1, sent_done, delta, budget):
            continue
        count, rep = out[(i, cost, reg)]
        if _merge(out, (i + 1, cost + 1, reg), count, rep):
            queue.append((i + 1, cost + 1, reg))
    return out


def _merge(table, key, count, rep) -> bool:
    """把 (count, rep) 并入状态。

    返回是否发生了"新串/新多解标志"变化（供插入闭包继续传播）。
    代表串相同只算同一个不同校正串；代表串不同则计数升级为多个。
    """
    old = table.get(key)
    if old is None:
        table[key] = (count, rep)
        return True
    oc, ore = old
    if rep == ore:
        nc = max(oc, count)  # 同一代表串：多解标志取并，不重复计数
    else:
        nc = MULTIPLE
    nrep = rep if rep < ore else ore
    if nc != oc or nrep != ore:
        table[key] = (nc, nrep)
        return True
    return False


def _fold(table, i, cost, rep, count):
    """帧边界/终点折叠：同一接收位置只保留最小代价。"""
    old = table.get(i)
    if old is None:
        table[i] = (cost, rep, count)
        return
    oc, ore, ocount = old
    if cost < oc:
        table[i] = (cost, rep, count)
    elif cost == oc:
        if rep == ore:
            ncount = max(ocount, count)
        else:
            ncount = MULTIPLE
        table[i] = (cost, min(rep, ore), ncount)


def _split_frames(corrected: str, received: str, sync: str,
                  intra_frame_length: bool, payload_len: int | None = None,
                  frame_len: int | None = None):
    """切分校正串为逐帧结果，并用最小编辑对齐求插入/漏失事件位置。

    ``intra_frame_length`` 为假时帧等长（``frame_len``/``payload_len``
    指定）；为真时每帧长度由该帧 6 位长度码解码得到，帧长可逐帧不同。

    位置基于校正串（发送侧）0 计位：

    * insertion：噪声位位于校正串该位置之前（0=流首，串长=流尾）；
    * deletion：漏失的发送比特位于校正串该位置，值取自校正串。

    相邻相同比特产生等价脚本时（例如在全 1 游程中插入一个 1，插入位
    置本质不可区分），采用正向贪心：能匹配就匹配，使事件位置尽量靠后，
    结果确定且每种报告都是对接收串的合法解释。
    """
    sync_len = len(sync)
    frames = []
    # 逐帧切分边界（变长模式由长度码决定帧长），同时建立 发送位 -> (帧,帧内偏移)
    frame_of_pos: list[tuple[int | None, int | None]] = [
        (None, None)] * len(corrected)
    starts: list[int] = []
    pos = 0
    idx = 0
    while pos < len(corrected):
        starts.append(pos)
        if intra_frame_length:
            length_field = corrected[pos + sync_len:
                                     pos + sync_len + LENGTH_FIELD_LEN]
            plen = PAYLOAD_MIN_LEN + int(length_field, 2)
            flen = sync_len + plen + CRC_LEN
        else:
            plen = payload_len
            flen = frame_len
            length_field = None
        raw = corrected[pos:pos + flen]
        frames.append(FrameResult(
            index=idx,
            payload=raw[sync_len:sync_len + plen],
            crc=raw[-CRC_LEN:],
            raw=raw,
            payload_length=plen if intra_frame_length else None,
            length_field=length_field,
        ))
        for off in range(flen):
            frame_of_pos[pos + off] = (idx, off)
        pos += flen
        idx += 1

    n, m = len(received), len(corrected)
    INF = 10 ** 9
    # dp[i][j]：后缀 (received[i:], corrected[j:]) 的最小 ins/del 代价，
    # 供正向贪心在分歧点判断哪条边仍在最优脚本上。
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[n][m] = 0
    for j in range(m - 1, -1, -1):
        dp[n][j] = dp[n][j + 1] + 1
    for i in range(n - 1, -1, -1):
        dp[i][m] = dp[i + 1][m] + 1
        for j in range(m - 1, -1, -1):
            v = min(dp[i + 1][j] + 1, dp[i][j + 1] + 1)
            if received[i] == corrected[j]:
                v = min(v, dp[i + 1][j + 1])
            dp[i][j] = v

    def frame_of(p):
        if 0 <= p < len(corrected):
            return frame_of_pos[p]
        return None, None

    events: list[SlipEvent] = []
    i = j = 0
    while i < n or j < m:
        if (i < n and j < m and received[i] == corrected[j]
                and dp[i][j] == dp[i + 1][j + 1]):
            i += 1
            j += 1
        elif j < m and dp[i][j] == dp[i][j + 1] + 1:
            bit = corrected[j]
            fi, off = frame_of(j)
            events.append(SlipEvent(
                kind="deletion", position=j, frame_index=fi,
                offset=off, bit=bit,
                detail=(f"帧 {fi} 内偏移 {off}（校正串位置 {j}）"
                        f"的发送比特 {bit} 在接收流中漏失"),
            ))
            j += 1
        else:
            bit = received[i]
            fi, off = frame_of(j)
            if j == 0:
                where = "流首"
            elif j == m:
                where = "流尾"
            else:
                where = f"位置 {j}"
            events.append(SlipEvent(
                kind="insertion", position=j, frame_index=fi,
                offset=off, bit=bit,
                detail=f"噪声比特 {bit} 插入于校正串{where}之前",
            ))
            i += 1
    return frames, events
