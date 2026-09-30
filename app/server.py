"""遥测帧复原 HTTP API（仅依赖 Python 标准库）。

路由：
* ``GET  /healthz``           存活检查
* ``GET  /ready``             就绪检查
* ``GET  /``                  服务信息
* ``POST /api/v1/recover``    提交接收比特串进行联合复原
"""

from __future__ import annotations

import json
import logging
import os
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import reconstruct
from .validation import ValidationError, validate

LOG = logging.getLogger("telemetry")

READY = True


class _Handler(BaseHTTPRequestHandler):
    server_version = "TelemetryRecovery/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/healthz", "/ready"):
            if READY:
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(503, {"status": "not ready"})
        elif path == "/":
            self._send_json(200, {
                "service": "telemetry-frame-recovery",
                "usage": "POST /api/v1/recover",
            })
        else:
            self._send_json(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/v1/recover":
            self._send_json(404, {"error": "not found", "path": path})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "bad Content-Length"})
            return
        if length <= 0:
            self._send_json(400, {
                "error": "validation_failed",
                "fields": {"_body": "缺少 JSON 请求体"},
            })
            return
        if length > 1_048_576:
            self._send_json(413, {"error": "request body too large"})
            return
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {
                "error": "validation_failed",
                "fields": {"_body": f"请求体不是合法 JSON：{exc}"},
            })
            return

        try:
            req = validate(data)
        except ValidationError as exc:
            self._send_json(422, {
                "error": "validation_failed",
                "fields": exc.fields,
            })
            return

        try:
            result = reconstruct(
                req.received, req.frame_count, req.sync,
                req.payload_len, req.max_slippage,
            )
        except Exception as exc:  # 防御：服务不因单个请求崩溃
            LOG.exception("reconstruction failed: %s", exc)
            self._send_json(500, {"error": "internal_error",
                                  "message": str(exc)})
            return

        # 输入合法但预算内无解属于正常计算结论（200 + recoverable=false），
        # 不应当作协议错误；字段非法才返回 4xx。
        self._send_json(200, result.to_dict())

    def log_message(self, fmt: str, *args) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)


def create_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    server.daemon_threads = True
    return server


def serve(port: int | None = None) -> None:
    port = port or int(os.environ.get("TELEMETRY_PORT", "8080"))
    logging.basicConfig(
        level=os.environ.get("TELEMETRY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = create_server(port)

    def _graceful(_sig, _frame):
        LOG.info("shutting down")
        threading_shutdown(server)

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)
    LOG.info("telemetry recovery API listening on 0.0.0.0:%s", port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def threading_shutdown(server: ThreadingHTTPServer) -> None:
    import threading
    threading.Thread(target=server.shutdown, daemon=True).start()


if __name__ == "__main__":
    serve()
