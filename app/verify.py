"""一次性自检服务：编译检查 + 单元测试 + 含伪同步字的复原冒烟。

作为 Docker Compose 中名为 ``verify`` 的一次性服务运行：全部通过则
进程以 0 退出，任一步失败以非零码退出并在汇总中标明失败环节。
"""

from __future__ import annotations

import json
import py_compile
import random
import sys
import threading
import traceback
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(label: str):
    print(f"\n===== verify: {label} =====", flush=True)


def compile_check() -> bool:
    step("构建检查：字节码编译全部源文件")
    ok = True
    for path in list(ROOT.glob("app/*.py")) + list(ROOT.glob("tests/*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"  ok  {path.relative_to(ROOT)}")
        except py_compile.PyCompileError as exc:
            ok = False
            print(f"  FAIL {path}: {exc}")
    return ok


def unit_tests() -> bool:
    step("代码测试：unittest 全套")
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


def _req_url(url: str, payload=None, method="POST"):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def smoke() -> bool:
    step("复原冒烟：调用 API（含伪同步字陷阱）")
    import time
    from app.core import crc8, frame_is_valid

    # Compose 中通过 TELEMETRY_BASE_URL 指向常驻 api 服务做端到端冒烟；
    # 本地直接运行时进程内临时起服，自启服务在用完后关闭。
    base_url = __import__("os").environ.get("TELEMETRY_BASE_URL")
    owned_server = None
    thread = None
    if base_url:
        # 等待目标服务就绪（compose 的 healthcheck 已把关，这里再兜底）
        last_err = None
        for _ in range(30):
            try:
                status, _ = _req_url(f"{base_url}/healthz", method="GET")
                if status == 200:
                    break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(1)
        else:
            print(f"  FAIL：等待 API 就绪超时：{last_err}")
            return False
        print(f"  目标 API：{base_url}（外部服务）")
    else:
        from app.server import create_server
        owned_server = create_server(0)  # 内核分配临时端口
        port = owned_server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=owned_server.serve_forever,
                                  daemon=True)
        thread.start()
        print(f"  目标 API：{base_url}（进程内临时服务）")

    def post(path, payload=None, method="POST"):
        return _req_url(f"{base_url}{path}", payload, method)

    ok = True
    try:
        # 1) 健康检查
        status, body = post("/healthz", method="GET")
        print(f"  GET /healthz -> {status}")
        if status != 200 or body.get("status") != "ok":
            ok = False

        # 2) 构造 3 帧，破坏流：插入 6 位凭空制造一个伪同步字
        rng = random.Random(2026)
        sync = "10101011"
        plen = 18
        nf = 3
        frames = []
        for _ in range(nf):
            body_bits = sync + "".join(rng.choice("01") for _ in range(plen))
            frames.append(body_bits + format(crc8(body_bits), "08b"))
        stream = "".join(frames)
        frame_len = len(sync) + plen + 8
        trap_pos = next(
            p for p in range(6, len(stream) - 1)
            if stream[p:p + 2] == sync[-2:]
            and p % frame_len != 0
            and stream[p - 6:p] != sync[:-2]
        )
        damaged = stream[:trap_pos] + sync[:-2] + stream[trap_pos:]

        status, body = post("/api/v1/recover", {
            "received": damaged, "frame_count": nf, "sync": sync,
            "payload_len": plen, "max_slippage": 6,
        })
        print(f"  POST /api/v1/recover (伪同步字) -> {status}, "
              f"slippage={body.get('slippage_count')}")
        if status != 200 or not body.get("recoverable"):
            ok = False
            print("  FAIL：预期在预算内可复原")
        else:
            if body["slippage_count"] != 6:
                ok = False
                print("  FAIL：滑移次数应为 6")
            if len(body["frames"]) != nf:
                ok = False
                print("  FAIL：应返回恰好 3 帧")
            for f in body["frames"]:
                if not frame_is_valid(f["raw"], sync, plen):
                    ok = False
                    print(f"  FAIL：帧 {f['index']} 同步字/CRC 校验不通过")
                if f["payload"] != f["raw"][len(sync):len(sync) + plen]:
                    ok = False
                    print(f"  FAIL：帧 {f['index']} 载荷切分不一致")
                if f["crc"] != f["raw"][-8:]:
                    ok = False
            print(f"  unique={body['unique']}, "
                  f"events={len(body['events'])}, "
                  f"corrected_len={len(body['corrected'])}")

        # 3) 朴素按同步字逐段截取必然至少产生一帧非法帧
        positions, start = [], 0
        while True:
            p = damaged.find(sync, start)
            if p < 0:
                break
            positions.append(p)
            start = p + 1
        naive_ok = all(
            len(damaged[p:p + frame_len]) == frame_len
            and frame_is_valid(damaged[p:p + frame_len], sync, plen)
            for p in positions[:nf]
        ) if len(positions) >= nf else False
        print(f"  朴素同步字截取找到 {len(positions)} 个候选位置，"
              f"是否全部成帧: {naive_ok}")
        if naive_ok:
            ok = False
            print("  FAIL：陷阱未生效，测试构造有问题")

        # 4) 超预算：必须返回不可复原与下界，且不携带局部帧/猜测载荷
        status, body = post("/api/v1/recover", {
            "received": stream + "1010101", "frame_count": nf,
            "sync": sync, "payload_len": plen, "max_slippage": 6,
        })
        print(f"  POST /api/v1/recover (超预算) -> {status}, "
              f"lower_bound={body.get('minimum_slippage_lower_bound')}")
        if body.get("recoverable") or "frames" in body or "corrected" in body:
            ok = False
            print("  FAIL：不得返回局部帧或猜测载荷")
        if body.get("minimum_slippage_lower_bound", 0) < 7:
            ok = False
            print("  FAIL：已验证最小滑移下界应 >= 7")

        # 5) 非法输入：逐字段错误
        status, body = post("/api/v1/recover", {
            "received": "02", "frame_count": 2, "sync": "10",
            "payload_len": 8, "max_slippage": 9,
        })
        print(f"  POST 非法输入 -> {status}，字段错误: "
              f"{sorted(body.get('fields', {}))}")
        if status != 422 or set(
                ("received", "frame_count", "sync",
                 "payload_len", "max_slippage")) - set(
                body.get("fields", {})):
            ok = False
            print("  FAIL：应给出全部问题字段的明确错误")
    except Exception:  # noqa: BLE001
        ok = False
        traceback.print_exc()
    finally:
        if owned_server is not None:
            owned_server.shutdown()
            owned_server.server_close()
            thread.join(timeout=5)
    return ok


def main() -> int:
    results = {
        "build": compile_check(),
        "tests": unit_tests(),
        "smoke": smoke(),
    }
    step("汇总")
    for name, passed in results.items():
        print(f"  {name:6s}: {'PASS' if passed else 'FAIL'}")
    code = 0 if all(results.values()) else 1
    print(f"\nverify {'ALL PASS' if code == 0 else 'HAS FAILURES'} "
          f"(exit {code})", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
