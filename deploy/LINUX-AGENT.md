# 把 AINOVEL 部署成 Linux 上的 Agent 工具服务

目标：在 Linux 服务器上常驻一个只有本机/内网能访问的 HTTP 服务，供 Hermes / Codex 等 Agent 调用。

## 1. 安装
```bash
cd /opt && git clone <你的仓库地址> ainovel && cd ainovel
bash install.sh
# 生成 config.json 后填入 LLM 的 api_key（勿提交 git）
```

## 2. 定义端口与令牌
```bash
cp deploy/ainovel.env.example deploy/ainovel.env
vim deploy/ainovel.env
# AINOVEL_HOST=127.0.0.1   <- Agent 与工具都在同一台设备时保持默认
# AINOVEL_PORT=8123        <- 端口就由这里定义，改这里即全局生效
# AINOVEL_TOKEN=一段随机串 <- 所有 /api/* 请求都要带 X-AINOVEL-Token
```

端口选择建议：避开 80/443/22/8000 等常用口；挑一个自己记得住的（8123 就是默认值）。
只要 `AINOVEL_PORT` 在 .env 里，服务端和客户端统一走 `AINOVEL_URL` 环境变量，不需要到处硬编码。

## 3. 注册 systemd 服务
```bash
sudo cp deploy/ainovel.service /etc/systemd/system/ainovel.service
# 根据实际安装路径修改 service 文件里的 /opt/ainovel
sudo systemctl daemon-reload
sudo systemctl enable --now ainovel
systemctl status ainovel
```

## 4. 验证
```bash
export AINOVEL_URL=http://127.0.0.1:8123
export AINOVEL_TOKEN=<.env 里的令牌>
python3 agent_tool/ainovel_client.py status --project /tmp/test
```

## 5. 给 Agent 的入口
- 人用：浏览器打开 `http://127.0.0.1:8123`（没有 token 限制，静态页面直接可看）。
- Agent 用：`agent_tool/SKILL.md` + `agent_tool/ainovel_client.py`（只有 /api/* 需要 token）。

## 跨机器 / 远程访问
- Agent 在另一台机器时：`AINOVEL_HOST=0.0.0.0`，并
  1. 设置强随机 `AINOVEL_TOKEN`；
  2. 防火墙只放行 Agent 所在 IP（如 `ufw allow from <agent-ip> to any port 8123`）；
  3. 可选：套一层 nginx + TLS。

## 常见问题
- 端口被占：`lsof -i :8123` 查看占用进程，改 .env 换端口后 `systemctl restart ainovel`。
- CLI 依然可用：`cd /opt/ainovel && ./novel from-docs ...`，与 web 服务共用同一套数据和 config.json。
- 多个项目并行生成：不同 `project` 目录各自有独立 job，互不干扰。
