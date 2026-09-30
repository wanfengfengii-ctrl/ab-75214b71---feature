# 卫星遥测帧复原服务：纯 Python 标准库实现，无第三方运行时依赖。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TELEMETRY_PORT=8080

WORKDIR /app

COPY app/ ./app/
COPY tests/ ./tests/

EXPOSE 8080

# 容器健康检查：命中进程内 HTTP 健康端点；端口可由 TELEMETRY_PORT 配置。
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:'+os.environ['TELEMETRY_PORT']+'/healthz', timeout=2).status==200 else sys.exit(1)"

# 默认启动常驻 API；一次性自检服务以 `python -m app.verify` 覆盖入口。
ENTRYPOINT ["python", "-m"]
CMD ["app.server"]
