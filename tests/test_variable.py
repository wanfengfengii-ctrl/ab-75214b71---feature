"""变长帧（帧内长度）模式：求解器、朴素穷举对拍与事件回放测试。"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import (  # noqa: E402
    LENGTH_FIELD_LEN,
    PAYLOAD_MIN_LEN,
    crc8,
    reconstruct_variable,
    variable_frame_is_valid,
)


def make_var_frame(sync: str, payload_len: int, rng) -> str:
    """构造一帧变长帧：载荷前六位为长度字段（总载荷长度减 16）。"""
    code = payload_len - PAYLOAD_MIN_LEN
    payload = format(code, "06b") + "".join(
        rng.choice("01") for _ in range(payload_len - LENGTH_FIELD_LEN))
    body = sync + payload
    return body + format(crc8(body), "08b")


def edit_distance_ins_del(a: str, b: str) -> int:
    """仅允许插入/删除的编辑距离。"""
    n, m = len(a), len(b)
    inf = 10 ** 9
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        nxt = [i] + [inf] * m
        for j in range(1, m + 1):
            nxt[j] = min(
                dp[j] + 1,
                nxt[j - 1] + 1,
                dp[j - 1] if a[i - 1] == b[j - 1] else inf,
            )
        dp = nxt
    return dp[m]


def replay_events(corrected: str, events) -> str:
    """按事件回放：删除位去掉、插入位补上，应得到接收串。"""
    s = corrected
    for ev in sorted(events, key=lambda e: e.position, reverse=True):
        if ev.kind == "deletion":
            s = s[:ev.position] + s[ev.position + 1:]
        else:
            s = s[:ev.position] + ev.bit + s[ev.position:]
    return s


class VariableBasicTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20260930)
        self.sync = "11001010"
        self.plens = [18, 26, 34]
        self.frames = [make_var_frame(self.sync, p, self.rng)
                       for p in self.plens]
        # 在帧 1 载荷中部嵌入伪同步字（不在任何帧首）
        f1 = self.frames[1]
        at = len(self.sync) + 10
        f1 = f1[:at] + self.sync + f1[at + len(self.sync):]
        body1 = f1[:-8]
        self.frames[1] = body1 + format(crc8(body1), "08b")
        self.stream = "".join(self.frames)

    def test_clean_stream(self):
        r = reconstruct_variable(self.stream, 3, self.sync, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 0)
        self.assertEqual(r.corrected, self.stream)
        self.assertTrue(r.unique)
        self.assertEqual(r.alternatives, 0)
        self.assertEqual(r.events, ())
        for i, f in enumerate(r.frames):
            self.assertEqual(f.index, i)
            self.assertEqual(f.raw, self.frames[i])
            self.assertEqual(f.payload_len, self.plens[i])
            self.assertEqual(f.length_field,
                             format(self.plens[i] - 16, "06b"))
            self.assertEqual(f.length_code, self.plens[i] - 16)
            self.assertEqual(f.payload, self.frames[i][len(self.sync):-8])
            self.assertEqual(f.crc, self.frames[i][-8:])
            self.assertTrue(variable_frame_is_valid(f.raw, self.sync))

    def test_pseudo_sync_word_in_payload(self):
        # 载荷中的伪同步字不得误导帧界：无滑移时也必须恢复真实边界
        self.assertGreaterEqual(self.stream.count(self.sync), 4)
        r = reconstruct_variable(self.stream, 3, self.sync, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual([f.raw for f in r.frames], self.frames)

    def test_deletion_inside_length_field(self):
        # 帧 1 长度字段中挑一个两侧比特均不同的位漏失：事件位置唯一
        lf_start = len(self.frames[0]) + len(self.sync)
        pos = next(
            p for p in range(lf_start, lf_start + LENGTH_FIELD_LEN)
            if (self.stream[p - 1] != self.stream[p]
                and self.stream[p + 1] != self.stream[p]))
        damaged = self.stream[:pos] + self.stream[pos + 1:]
        r = reconstruct_variable(damaged, 3, self.sync, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 1)
        self.assertEqual(r.corrected, self.stream)
        self.assertEqual([f.raw for f in r.frames], self.frames)
        self.assertEqual(len(r.events), 1)
        ev = r.events[0]
        self.assertEqual(ev.kind, "deletion")
        self.assertEqual(ev.position, pos)
        self.assertEqual(ev.frame_index, 1)
        self.assertGreaterEqual(ev.offset, len(self.sync))
        self.assertLess(ev.offset, len(self.sync) + LENGTH_FIELD_LEN)
        self.assertEqual(ev.bit, self.stream[pos])
        self.assertEqual(replay_events(r.corrected, r.events), damaged)

    def test_insertion_inside_length_field(self):
        # 在帧 2 长度字段内插入一个与两侧都不同的比特：位置唯一
        lf2 = len(self.frames[0]) + len(self.frames[1]) + len(self.sync)
        pos = next(
            p for p in range(lf2 + 1, lf2 + LENGTH_FIELD_LEN)
            if self.stream[p - 1] == self.stream[p])
        bit = "1" if self.stream[pos] == "0" else "0"
        damaged = self.stream[:pos] + bit + self.stream[pos:]
        r = reconstruct_variable(damaged, 3, self.sync, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 1)
        self.assertEqual(r.corrected, self.stream)
        self.assertEqual([f.raw for f in r.frames], self.frames)
        self.assertEqual(len(r.events), 1)
        ev = r.events[0]
        self.assertEqual(ev.kind, "insertion")
        self.assertEqual(ev.bit, bit)
        self.assertEqual(ev.frame_index, 2)
        self.assertGreaterEqual(ev.offset, len(self.sync))
        self.assertLess(ev.offset, len(self.sync) + LENGTH_FIELD_LEN)
        self.assertEqual(replay_events(r.corrected, r.events), damaged)

    def test_length_code_out_of_range_cannot_frame(self):
        # 长度码 40（>32）的"帧"即使 CRC 自洽也不得成帧；其帧长超出合法
        # 总帧长区间，长度差下界直接给出不可复原结论
        bad = []
        for _ in range(3):
            payload = format(40, "06b") + "".join(
                self.rng.choice("01") for _ in range(50))
            body = self.sync + payload
            bad.append(body + format(crc8(body), "08b"))
        damaged = "".join(bad)
        r = reconstruct_variable(damaged, 3, self.sync, 6)
        self.assertFalse(r.recoverable)
        self.assertIsNone(r.corrected)
        self.assertEqual(r.frames, ())
        # 3*(8+56) = 192 位接收 vs 合法总长区间 [3*32, 3*56] = [96, 168]
        self.assertEqual(r.minimum_slippage_lower_bound, 192 - 168)

    def test_over_budget_returns_lower_bound(self):
        damaged = self.stream
        for p in (100, 90, 80, 70, 60, 40, 20):
            damaged = damaged[:p] + damaged[p + 1:]
        r = reconstruct_variable(damaged, 3, self.sync, 6)
        self.assertFalse(r.recoverable)
        self.assertIsNone(r.corrected)
        self.assertEqual(r.frames, ())
        self.assertEqual(r.events, ())
        self.assertGreaterEqual(r.minimum_slippage_lower_bound, 7)

    def test_events_replay_consistency_fuzz(self):
        rng = random.Random(4242)
        for _ in range(10):
            frames = [make_var_frame(self.sync, rng.randint(16, 48), rng)
                      for _ in range(4)]
            stream = "".join(frames)
            damaged = stream
            for _ in range(rng.randint(1, 4)):
                p = rng.randrange(len(damaged))
                if rng.random() < 0.5:
                    damaged = damaged[:p] + damaged[p + 1:]
                else:
                    damaged = damaged[:p] + rng.choice("01") + damaged[p:]
            r = reconstruct_variable(damaged, 4, self.sync, 6)
            self.assertTrue(r.recoverable)
            self.assertEqual(len(r.events), r.slippage_count)
            self.assertEqual(
                edit_distance_ins_del(damaged, r.corrected),
                r.slippage_count,
            )
            self.assertEqual(replay_events(r.corrected, r.events), damaged)
            for f in r.frames:
                self.assertTrue(variable_frame_is_valid(f.raw, self.sync))
                self.assertEqual(
                    f.payload_len,
                    PAYLOAD_MIN_LEN + int(f.length_field, 2))
                self.assertEqual(f.length_field,
                                 f.raw[len(self.sync):
                                       len(self.sync) + LENGTH_FIELD_LEN])


class VariableOptimalityTests(unittest.TestCase):
    """与朴素穷举对拍：最小滑移、字典序最小、唯一性。"""

    @staticmethod
    def brute(recv, nf, sync, budget):
        s = len(sync)
        found = set()

        def step(reg, b):
            v = reg ^ (b << 7)
            return (((v << 1) ^ 0x07) & 0xFF
                    if v & 0x80 else ((v << 1) & 0xFF))

        def rec(ri, k, j, flen, reg, corr, cost):
            if cost > budget:
                return
            if k == nf:
                if ri == len(recv):
                    found.add(corr)
                return
            if flen and j == flen:
                if reg == 0:
                    rec(ri, k + 1, 0, 0, 0, corr, cost)
                return
            cands = (int(sync[j]),) if j < s else (0, 1)
            if ri < len(recv):
                rec(ri + 1, k, j, flen, reg, corr, cost + 1)
            for b in cands:
                nreg = step(reg, b)
                nflen = flen
                if j == s + LENGTH_FIELD_LEN - 1:
                    code = int((corr + str(b))[-LENGTH_FIELD_LEN:], 2)
                    if code > 32:
                        continue  # 长度码越界：候选不得成帧
                    nflen = s + PAYLOAD_MIN_LEN + code + 8
                ncorr = corr + str(b)
                if ri < len(recv) and int(recv[ri]) == b:
                    rec(ri + 1, k, j + 1, nflen, nreg, ncorr, cost)
                rec(ri, k, j + 1, nflen, nreg, ncorr, cost + 1)

        rec(0, 0, 0, 0, 0, "", 0)
        return found

    def test_matches_brute_force(self):
        rng = random.Random(777)
        checked = 0
        for trial in range(40):
            slen = rng.randint(6, 7)
            plens = [rng.randint(16, 18) for _ in range(3)]
            sync = "".join(rng.choice("01") for _ in range(slen))
            stream = "".join(make_var_frame(sync, p, rng) for p in plens)
            damaged = stream
            for _ in range(rng.randint(0, 2)):
                p = rng.randrange(len(damaged))
                if rng.random() < 0.5:
                    damaged = damaged[:p] + damaged[p + 1:]
                else:
                    damaged = damaged[:p] + rng.choice("01") + damaged[p:]
            budget = rng.randint(1, 2)
            r = reconstruct_variable(damaged, 3, sync, budget)
            opt = self.brute(damaged, 3, sync, budget)
            if not opt:
                self.assertFalse(r.recoverable, trial)
                continue
            costs = {x: edit_distance_ins_del(damaged, x) for x in opt}
            best_cost = min(costs.values())
            best = {x for x, c in costs.items() if c == best_cost}
            self.assertTrue(r.recoverable)
            self.assertEqual(r.slippage_count, best_cost)
            self.assertEqual(r.corrected, min(best))
            self.assertEqual(r.unique, len(best) == 1)
            checked += 1
        self.assertGreater(checked, 15)


class VariableBoundaryTests(unittest.TestCase):
    def test_min_max_lengths_and_frames(self):
        rng = random.Random(7)
        sync = "111000101011"
        plens = [16, 48, 20, 33, 24, 40, 17, 30]
        frames = [make_var_frame(sync, p, rng) for p in plens]
        stream = "".join(frames)
        r = reconstruct_variable(stream, 8, sync, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.corrected, stream)
        self.assertEqual([f.payload_len for f in r.frames], plens)

    def test_insertion_at_stream_edges(self):
        rng = random.Random(3)
        sync = "101011"
        frames = [make_var_frame(sync, 16, rng) for _ in range(3)]
        stream = "".join(frames)
        head_bit = "0" if stream[0] == "1" else "1"
        tail_bit = "0" if stream[-1] == "1" else "1"
        for damaged, pos in ((head_bit + stream, 0),
                             (stream + tail_bit, len(stream))):
            r = reconstruct_variable(damaged, 3, sync, 6)
            self.assertTrue(r.recoverable)
            self.assertEqual(r.corrected, stream)
            self.assertEqual(r.events[0].kind, "insertion")
            self.assertEqual(r.events[0].position, pos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
