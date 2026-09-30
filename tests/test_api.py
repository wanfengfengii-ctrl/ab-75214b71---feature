"""HTTP API 端到端测试（标准库 http 服务 + urllib 客户端）。"""

import json
import os
import random
import sys
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import crc8  # noqa: E402
from app.server import create_server  # noqa: E402


def make_frame(sync, payload):
    body = sync + payload
    return body + format(crc8(body), "08b")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(self, path, payload=None, method="POST"):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health(self):
        for path in ("/healthz", "/ready"):
            status, body = self._request(path, method="GET")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")

    def test_recover_clean(self):
        rng = random.Random(5)
        sync = "1101001101"
        frames = [make_frame(sync, "".join(rng.choice("01") for _ in range(24)))
                  for _ in range(3)]
        stream = "".join(frames)
        status, body = self._request("/api/v1/recover", {
            "received": stream, "frame_count": 3, "sync": sync,
            "payload_len": 24, "max_slippage": 6,
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["recoverable"])
        self.assertEqual(body["corrected"], stream)
        self.assertEqual(len(body["frames"]), 3)
        self.assertEqual(body["slippage_count"], 0)
        self.assertTrue(body["unique"])
        for i, f in enumerate(body["frames"]):
            self.assertEqual(f["index"], i)
            self.assertEqual(f["raw"], frames[i])
            self.assertEqual(f["crc"], frames[i][-8:])

    def test_recover_with_insertion_and_deletion(self):
        rng = random.Random(6)
        sync = "1001110001"
        frames = [make_frame(sync, "".join(rng.choice("01") for _ in range(20)))
                  for _ in range(4)]
        stream = "".join(frames)
        damaged = stream[:15] + "1" + stream[15:50] + stream[51:]
        status, body = self._request("/api/v1/recover", {
            "received": damaged, "frame_count": 4, "sync": sync,
            "payload_len": 20, "max_slippage": 6,
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["recoverable"])
        self.assertEqual(body["corrected"], stream)
        self.assertEqual(body["slippage_count"], 2)
        kinds = sorted(e["kind"] for e in body["events"])
        self.assertEqual(kinds, ["deletion", "insertion"])

    def test_pseudo_sync_word_trap(self):
        """接收流中混入伪同步字时不得误判帧首。

        在真实帧之间插入若干噪声位，其中包含同步字本身的重复；逐段按同步字
        截取会被伪同步字误导，而联合 CRC 求解仍能恢复完整帧流。
        """
        rng = random.Random(11)
        sync = "10101011"
        frames = [make_frame(sync, "".join(rng.choice("01") for _ in range(18)))
                  for _ in range(3)]
        stream = "".join(frames)
        frame_len = len(sync) + 18 + 8
        # 在非帧界位置，借助接收流中既有的同步字末两位 "11"，在其前面插入
        # 同步字前 6 位，从而凭空拼出一个完整伪同步字（共 6 次插入）。
        trap_pos = next(
            p for p in range(6, len(stream) - 1)
            if (stream[p:p + 2] == sync[-2:]
                and p % frame_len != 0
                and stream[p - 6:p] != sync[:-2])  # 避开真实同步字的末两位
        )
        damaged = stream[:trap_pos] + sync[:-2] + stream[trap_pos:]
        self.assertEqual(damaged[trap_pos:trap_pos + len(sync)], sync)
        self.assertNotEqual(trap_pos % frame_len, 0)  # 伪同步字不在真帧界
        status, body = self._request("/api/v1/recover", {
            "received": damaged, "frame_count": 3, "sync": sync,
            "payload_len": 18, "max_slippage": 6,
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["recoverable"], body)
        self.assertEqual(body["slippage_count"], 6)
        for f in body["frames"]:
            self.assertTrue(f["raw"].startswith(sync))
            body_bits = f["raw"][len(sync):len(sync) + 18]
            expect_crc = format(crc8(sync + body_bits), "08b")
            self.assertEqual(f["crc"], expect_crc)
        self.assertIn(body["unique"], (True, False))

        # 朴素做法：在接收流里按同步字出现位置逐段定长截取，必被伪同步字
        # 误导（至少一帧同步字或 CRC 不成立），证明不能靠同步字硬切。
        positions = []
        start = 0
        while len(positions) < 3:
            p = damaged.find(sync, start)
            if p < 0:
                break
            positions.append(p)
            start = p + 1
        naive_bad = False
        if len(positions) >= 3:
            for p in positions[:3]:
                seg = damaged[p:p + frame_len]
                if len(seg) < frame_len:
                    naive_bad = True
                    break
                from app.core import frame_is_valid
                if not frame_is_valid(seg, sync, 18):
                    naive_bad = True
                    break
        self.assertTrue(naive_bad, "伪同步字应使朴素定长截取产生非法帧")

    def test_unrecoverable_returns_lower_bound_no_partials(self):
        rng = random.Random(8)
        sync = "11001100"
        frames = [make_frame(sync, "".join(rng.choice("01") for _ in range(16)))
                  for _ in range(3)]
        stream = "".join(frames)
        damaged = stream + "1010101"  # 长度差 7 > 预算 6
        status, body = self._request("/api/v1/recover", {
            "received": damaged, "frame_count": 3, "sync": sync,
            "payload_len": 16, "max_slippage": 6,
        })
        self.assertEqual(status, 200)
        self.assertFalse(body["recoverable"])
        self.assertNotIn("frames", body)
        self.assertNotIn("corrected", body)
        self.assertGreaterEqual(body["minimum_slippage_lower_bound"], 7)

    def test_validation_errors_are_field_specific(self):
        status, body = self._request("/api/v1/recover", {
            "received": "012", "frame_count": 9, "sync": "10",
            "payload_len": 8, "max_slippage": 9,
        })
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_failed")
        for name in ("received", "frame_count", "sync",
                     "payload_len", "max_slippage"):
            self.assertIn(name, body["fields"])

    def test_bad_json(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/v1/recover",
            data=b"{not json", headers={"Content-Type": "application/json"},
            method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            body = json.loads(exc.read().decode("utf-8"))
            self.assertIn("_body", body["fields"])
        else:
            self.fail("应返回 400")

    def test_unknown_route(self):
        status, _ = self._request("/nope", method="GET")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
