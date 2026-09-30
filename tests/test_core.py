"""CRC 与复原求解器测试。"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import (  # noqa: E402
    CRC_POLY,
    crc8,
    frame_is_valid,
    reconstruct,
)


def make_frame(sync: str, payload: str) -> str:
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


class Crc8Tests(unittest.TestCase):
    def test_catalog_check_value(self):
        # CRC-8/SMBus（poly 0x07, init 0, MSB first）目录校验值
        bits = "".join(format(c, "08b") for c in b"123456789")
        self.assertEqual(crc8(bits), 0xF4)

    def test_polynomial(self):
        self.assertEqual(CRC_POLY, 0x07)

    def test_full_frame_residue_zero(self):
        rng = random.Random(1)
        sync = "11001011"
        for _ in range(20):
            body = sync + "".join(rng.choice("01") for _ in range(20))
            frame = body + format(crc8(body), "08b")
            self.assertEqual(crc8(frame), 0)

    def test_append_eight_zeros_interpretation(self):
        # crc8(M) 等价于 M 补八个零后用九位寄存器长除的余数
        bits = "1101011" * 5

        def long_division(msg: str) -> int:
            reg = 0
            for ch in msg:
                reg = (reg << 1) | (ord(ch) - 48)
                if reg & 0x100:
                    reg ^= 0x107
            return reg

        self.assertEqual(crc8(bits), long_division(bits + "0" * 8))


class ReconstructionBasicTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        self.sync = "111000101"
        self.plen = 16
        self.nf = 4
        self.frames = [
            make_frame(self.sync,
                       "".join(self.rng.choice("01")
                               for _ in range(self.plen)))
            for _ in range(self.nf)
        ]
        self.stream = "".join(self.frames)

    def test_clean_stream(self):
        r = reconstruct(self.stream, self.nf, self.sync, self.plen, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 0)
        self.assertEqual(r.corrected, self.stream)
        self.assertTrue(r.unique)
        self.assertEqual(r.alternatives, 0)
        self.assertEqual(r.events, ())
        for i, f in enumerate(r.frames):
            self.assertEqual(f.index, i)
            self.assertEqual(f.raw, self.frames[i])
            self.assertEqual(f.payload, self.frames[i][9:25])
            self.assertEqual(f.crc, self.frames[i][-8:])

    def test_single_insertion(self):
        # 选一个两侧比特均与插入位不同的位置，事件位置才可唯一确定
        pos = next(p for p in range(1, len(self.stream) - 1)
                   if self.stream[p - 1] == "0" and self.stream[p] == "0")
        bit = "1"
        recv = self.stream[:pos] + bit + self.stream[pos:]
        r = reconstruct(recv, self.nf, self.sync, self.plen, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 1)
        self.assertEqual(r.corrected, self.stream)
        kinds = [e.kind for e in r.events]
        self.assertEqual(kinds, ["insertion"])
        self.assertEqual(r.events[0].bit, bit)
        self.assertEqual(r.events[0].position, pos)
        for f in r.frames:
            self.assertTrue(frame_is_valid(f.raw, self.sync, self.plen))

    def test_single_insertion_inside_run_is_ambiguous_but_valid(self):
        # 在同值游程中插入：位置不可区分，但报告的脚本必须能回放
        pos = 30
        recv = self.stream[:pos] + "1" + self.stream[pos:]
        r = reconstruct(recv, self.nf, self.sync, self.plen, 6)
        self.assertEqual(r.slippage_count, 1)
        self.assertEqual(r.corrected, self.stream)
        ev = r.events[0]
        self.assertEqual(ev.kind, "insertion")
        rebuilt = (r.corrected[:ev.position] + ev.bit
                   + r.corrected[ev.position:])
        self.assertEqual(rebuilt, recv)

    def test_single_deletion(self):
        # 选删除位与其前后位都不同的位置，位置才可唯一确定
        pos = next(p for p in range(1, len(self.stream) - 1)
                   if (self.stream[p] == "1"
                       and self.stream[p - 1] == "0"
                       and self.stream[p + 1] == "0"))
        recv = self.stream[:pos] + self.stream[pos + 1:]
        r = reconstruct(recv, self.nf, self.sync, self.plen, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 1)
        self.assertEqual(r.corrected, self.stream)
        self.assertEqual([e.kind for e in r.events], ["deletion"])
        ev = r.events[0]
        self.assertEqual(ev.position, pos)
        self.assertEqual(ev.bit, self.stream[pos])

    def test_insertion_and_deletion_combined(self):
        recv = self.stream[:10] + "0" + self.stream[10:40] + self.stream[41:]
        r = reconstruct(recv, self.nf, self.sync, self.plen, 6)
        self.assertTrue(r.recoverable)
        self.assertEqual(r.slippage_count, 2)
        self.assertEqual(r.corrected, self.stream)
        self.assertEqual(sorted(e.kind for e in r.events),
                         ["deletion", "insertion"])

    def test_events_are_consistent_with_corrected(self):
        # 任意破坏后：事件数 == 编辑距离，且按事件回放可由校正串得到接收串
        rng = random.Random(99)
        for _ in range(10):
            stream = self.stream
            for _ in range(rng.randint(1, 4)):
                p = rng.randrange(len(stream))
                if rng.random() < 0.5:
                    stream = stream[:p] + stream[p + 1:]
                else:
                    stream = stream[:p] + rng.choice("01") + stream[p:]
            r = reconstruct(stream, self.nf, self.sync, self.plen, 6)
            self.assertTrue(r.recoverable)
            self.assertEqual(len(r.events), r.slippage_count)
            self.assertEqual(
                edit_distance_ins_del(stream, r.corrected),
                r.slippage_count,
            )
            # 回放：删除位补上、插入位去掉
            s = r.corrected
            # 逆序处理插入/漏失（按校正位置）
            for ev in sorted(r.events, key=lambda e: e.position, reverse=True):
                if ev.kind == "deletion":
                    s = s[:ev.position] + s[ev.position + 1:]
                else:
                    s = s[:ev.position] + ev.bit + s[ev.position:]
            self.assertEqual(s, stream)

    def test_over_budget_returns_lower_bound(self):
        recv = self.stream
        for p in (90, 77, 60, 44, 20, 12, 5):
            recv = recv[:p] + recv[p + 1:]
        r = reconstruct(recv, self.nf, self.sync, self.plen, 6)
        self.assertFalse(r.recoverable)
        self.assertIsNone(r.corrected)
        self.assertEqual(r.frames, ())
        self.assertEqual(r.events, ())
        self.assertGreaterEqual(r.minimum_slippage_lower_bound, 7)

    def test_length_gap_lower_bound(self):
        # 长度差本身超出预算 => 下界至少是长度差
        recv = self.stream + "1010101"
        r = reconstruct(recv, self.nf, self.sync, self.plen, 6)
        self.assertFalse(r.recoverable)
        self.assertEqual(r.minimum_slippage_lower_bound, 7)

    def test_no_partial_frames_on_failure(self):
        # 预算为 0 且有任何错位都不得返回局部帧
        recv = self.stream[:5] + self.stream[6:]
        r = reconstruct(recv, self.nf, self.sync, self.plen, 0)
        self.assertFalse(r.recoverable)
        self.assertEqual(r.frames, ())


class OptimalityTests(unittest.TestCase):
    """与朴素穷举对拍：最小滑移、字典序最小、唯一性。"""

    @staticmethod
    def brute(recv, nf, sync, plen, budget):
        sl = len(sync)
        fl = sl + plen + 8

        def step(reg, b):
            v = reg ^ (b << 7)
            return (((v << 1) ^ 0x07) & 0xFF
                    if v & 0x80 else ((v << 1) & 0xFF))

        found = set()

        def rec(ri, k, j, reg, corr, cost):
            if cost > budget:
                return
            if k == nf:
                if ri == len(recv):
                    found.add(corr)
                return
            if j == fl:
                if reg == 0:
                    rec(ri, k + 1, 0, 0, corr, cost)
                return
            cands = (int(sync[j]),) if j < sl else (0, 1)
            if ri < len(recv):
                rec(ri + 1, k, j, reg, corr, cost + 1)
            for b in cands:
                if ri < len(recv) and int(recv[ri]) == b:
                    rec(ri + 1, k, j + 1, step(reg, b), corr + str(b), cost)
                rec(ri, k, j + 1, step(reg, b), corr + str(b), cost + 1)

        rec(0, 0, 0, 0, "", 0)
        return found

    def test_matches_brute_force(self):
        rng = random.Random(321)
        checked = 0
        for trial in range(60):
            slen = rng.randint(6, 7)
            plen = rng.randint(16, 18)
            nf = 3
            sync = "".join(rng.choice("01") for _ in range(slen))
            stream = ""
            for _ in range(nf):
                body = sync + "".join(rng.choice("01") for _ in range(plen))
                stream += body + format(crc8(body), "08b")
            damaged = stream
            for _ in range(rng.randint(0, 2)):
                p = rng.randrange(len(damaged))
                if rng.random() < 0.5:
                    damaged = damaged[:p] + damaged[p + 1:]
                else:
                    damaged = damaged[:p] + rng.choice("01") + damaged[p:]
            budget = rng.randint(1, 2)
            r = reconstruct(damaged, nf, sync, plen, budget)
            opt = self.brute(damaged, nf, sync, plen, budget)
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
        self.assertGreater(checked, 20)


class BoundaryRangeTests(unittest.TestCase):
    def test_min_and_max_params(self):
        rng = random.Random(7)
        for nf, slen, plen in [(3, 6, 16), (8, 12, 48)]:
            sync = "".join(rng.choice("01") for _ in range(slen))
            stream = ""
            for _ in range(nf):
                body = sync + "".join(rng.choice("01") for _ in range(plen))
                stream += body + format(crc8(body), "08b")
            r = reconstruct(stream, nf, sync, plen, 6)
            self.assertTrue(r.recoverable)
            self.assertEqual(r.corrected, stream)
            self.assertEqual(len(r.frames), nf)

    def test_insertion_at_stream_edges(self):
        rng = random.Random(3)
        sync = "101011"
        body = sync + "".join(rng.choice("01") for _ in range(16))
        stream = make_frame(sync, body[len(sync):]) * 3
        # 流首插入与首比特不同的位；流尾插入与末比特不同的位：位置唯一
        head_bit = "0" if stream[0] == "1" else "1"
        tail_bit = "0" if stream[-1] == "1" else "1"
        cases = [(head_bit + stream, 0),
                 (stream + tail_bit, len(stream))]
        for damaged, pos in cases:
            r = reconstruct(damaged, 3, sync, 16, 6)
            self.assertTrue(r.recoverable)
            self.assertEqual(r.corrected, stream)
            self.assertEqual(r.events[0].kind, "insertion")
            self.assertEqual(r.events[0].position, pos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
