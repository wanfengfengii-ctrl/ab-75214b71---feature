# 卫星遥测连续帧联合复原服务

卫星地面站接收连续遥测时，链路会偶发**插入**或**漏掉单个比特**。本服务在
**整条接收比特流**上联合解释全部插入与漏失，恢复完整帧流，避免仅按同步字
逐段截取时把**伪同步字**误判为帧首。

- 纯 Python 3.11 标准库实现，**零第三方运行时依赖**。
- 常驻 HTTP API，端口由 `TELEMETRY_PORT` 配置，带健康检查。
- 一次性 `verify` 服务：构建检查 + 单元测试 + 含伪同步字的端到端冒烟，
  以退出码汇总结果。

## 帧与 CRC

每帧固定为：

```
| 同步字 (6..12 位) | 载荷 (16..48 位) | CRC (8 位) |
```

- 帧数 3..8，所有帧共用同一同步字、同一载荷长度。
- CRC-8，多项式 `x^8+x^2+x+1`（0x07），初值 0，**最高位优先**；
  CRC 字段为「同步字+载荷」**补八个零后**对生成多项式取余
  （标准 CRC-8/SMBus 移位实现，目录校验值 `"123456789" → 0xF4`）。
  合法整帧（含 CRC）经过同一移位寄存器后余数为 0。

信道模型只允许两类差错，各计 **1 次滑移**：

- `insertion`：接收流多出一个噪声比特（发送侧不消耗位）；
- `deletion`：发送的一位在接收流中缺失（接收侧不消耗位）。

比特翻转等价于「同位置一次漏失 + 一次插入」（代价 2）。

## 复原准则

在整条接收流上做带 CRC 寄存器状态的分层动态规划（状态含接收游标、帧内
偏移、已用滑移、CRC 余数），转移只允许匹配 / 漏失 / 插入：

1. **先最小化滑移次数**（预算 0..`max_slippage` ≤ 6，逐档求解）；
2. 滑移次数相同的解中，取**校正串字典序最小**者；
3. 枚举同代价的不同校正串，报告最优解 `unique`（唯一）或存在其他
   同代价校正串（`alternatives ≥ 1`）。

预算内无解时，只返回**不可复原结论**与**已验证的最小所需滑移下界**
（`minimum_slippage_lower_bound`），绝不返回局部帧或猜测载荷。

> 关于位置：当插入/漏失发生在连续相同比特游程中时，事件位置在信息论上
> 不可区分（在 `111` 游程中任意位置插入一个 `1` 得到同一接收串）。
> 服务以确定的正向对齐（能匹配先匹配，事件尽量靠后）报告一组合法位置，
> 该组位置回放后必与接收串逐位一致。

## HTTP API

### `POST /api/v1/recover`

请求：

```json
{
  "received": "10101011…01",
  "frame_count": 3,
  "sync": "10101011",
  "payload_len": 18,
  "max_slippage": 6
}
```

| 字段 | 类型 | 范围 |
| --- | --- | --- |
| `received` | string | 非空，仅含 `0`/`1` |
| `frame_count` | int | 3..8 |
| `sync` | string | 6..12 位 `0`/`1` |
| `payload_len` | int | 16..48 |
| `max_slippage` | int | 0..6 |

成功（HTTP 200，`recoverable=true`）：

```json
{
  "recoverable": true,
  "corrected": "1010…（重建的完整发送比特串）",
  "slippage_count": 2,
  "unique": true,
  "alternatives": 0,
  "budget": 6,
  "frames": [
    {"index": 0, "payload": "…18 位…", "crc": "11001010",
     "raw": "同步字+载荷+CRC 整帧"}
  ],
  "events": [
    {"kind": "insertion", "position": 40, "frame_index": 1,
     "offset": 6, "bit": "1", "detail": "…"},
    {"kind": "deletion",  "position": 51, "frame_index": 1,
     "offset": 17, "bit": "0", "detail": "…"}
  ]
}
```

- `events[].position` 基于校正串（发送侧）0 计位；插入位在该位置之前
  （0 = 流首，串长 = 流尾），漏失位即该位置。
- 不可复原（HTTP 200，`recoverable=false`）：

```json
{
  "recoverable": false,
  "reason": "滑移预算内不存在通过同步字与 CRC 校验的完整帧流",
  "minimum_slippage_lower_bound": 7,
  "verified_up_to": 6,
  "budget": 6
}
```

- 输入非法返回 **422**，逐字段给出中文错误：

```json
{"error": "validation_failed",
 "fields": {"sync": "长度必须在 6..12 位之间", "…": "…"}}
```

非法 JSON 为 **400**（`fields._body`），未知路由 **404**。

健康检查：`GET /healthz`、`GET /ready` → `200 {"status":"ok"}`。

## 运行

```bash
# 构建并启动常驻 API（默认端口 8080）
docker compose up --build api

# 自定义端口
TELEMETRY_PORT=9000 docker compose up --build api

# 一次性自检（先等 api 健康，再对其做端到端冒烟；退出码即结论）
docker compose build && docker compose run --rm verify
# 或一步到位：
docker compose up --build --abort-on-container-exit --exit-code-from verify verify
```

`verify` 汇总三项，全过退出码为 0：

- `build`：全部源文件字节码编译检查；
- `tests`：`tests/` 全套单元 / 对拍 / HTTP 测试；
- `smoke`：健康检查、含**伪同步字陷阱**的复原、超预算下界、非法字段错误。

## 本地开发（无需 Docker）

```bash
python3 -m unittest discover -s tests -v   # 测试（含朴素穷举对拍）
python3 -m app.verify                      # 一次性自检
TELEMETRY_PORT=8080 python3 -m app.server  # 启动 API
```

目录：

```
app/core.py        # CRC-8、联合复原 DP、事件定位
app/validation.py  # 入参校验（逐字段错误）
app/server.py      # 标准库 HTTP API + 健康检查
app/verify.py      # 一次性自检（构建/测试/冒烟 + 退出码）
tests/             # CRC、求解器对拍、校验、HTTP 端到端测试
Dockerfile, docker-compose.yml
```
