#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AINOVEL Agent 工具客户端（仅用标准库，Windows / macOS / Linux 通用）。

用法示例：
  export AINOVEL_URL=http://127.0.0.1:8123        # 默认就是它
  export AINOVEL_TOKEN=你的令牌                    # 服务端设置过才需要

  # 一键全流程（阻塞等待，最多 --timeout 秒）：
  python3 ainovel_client.py run \
    --project /opt/ainovel-projects/我的小说 \
    --guidance "都市悬疑，双男主，年下，破镜重圆" \
    --docs /home/user/docs/参考文档.txt \
    --docs /home/user/docs/设定集.epub \
    --words 90000 --chapters 36 --per-chapter 2500 \
    --workers 4 --timeout 1800

  # 只启动不等待，之后用 status 轮询：
  python3 ainovel_client.py run --project ... --guidance ... --no-wait
  python3 ainovel_client.py status --project ...
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("AINOVEL_URL", "http://127.0.0.1:8123").rstrip("/")
TOKEN = os.environ.get("AINOVEL_TOKEN", "").strip()


def request(method: str, path: str, payload=None, timeout: int = 60) -> dict:
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["X-AINOVEL-Token"] = TOKEN
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = None
        message = (detail or {}).get("detail") or str(exc)
        return {"ok": False, "error": message, "http_status": exc.code}
    except Exception as exc:
        return {"ok": False, "error": f"无法连接 {BASE}：{exc}"}


def cmd_run(args):
    payload = {
        "project": args.project,
        "guidance": args.guidance,
        "docs": args.docs or [],
        "wait": not args.no_wait,
        "timeout": args.timeout,
        "poll_interval": args.poll_interval,
    }
    optional = {
        "target_words": args.words,
        "target_chapters": args.chapters,
        "words_per_chapter": args.per_chapter,
        "workers": args.workers,
        "rounds": args.rounds,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    print(f"→ {BASE}/api/agent/run  project={args.project}", file=sys.stderr)
    if not args.no_wait:
        print(f"→ 全流程约需数分钟（生成三套方案 → 辩论 → 采用 → 拆解 → 合并 → 写正文），最多等待 {args.timeout} 秒", file=sys.stderr)
    result = request("POST", "/api/agent/run", payload, timeout=args.timeout + 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("ok") else 1)


def cmd_status(args):
    query = urllib.parse.urlencode({"path": args.project})
    result = request("GET", "/api/agent/status?" + query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("ok") else 1)


def main():
    parser = argparse.ArgumentParser(prog="ainovel_client.py", description="AINOVEL 全流程小说生成工具（给 Agent 调用）")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="入参考文档并跑完整 LangGraph 全流程（默认阻塞等待）")
    run.add_argument("--project", required=True, help="项目目录（服务器上的路径）")
    run.add_argument("--guidance", required=True, help="一句话说明想写什么：题材/标签/关系走向")
    run.add_argument("--docs", action="append", default=[], help="服务器上的参考文档路径（.txt/.md/.epub），可传多个")
    run.add_argument("--words", type=int, default=None, help="大概总字数（必填或项目里已保存）")
    run.add_argument("--chapters", type=int, default=None, help="大概章节数")
    run.add_argument("--per-chapter", type=int, default=None, help="每章大概字数")
    run.add_argument("--workers", type=int, default=None, help="并行度（默认服务端取 4）")
    run.add_argument("--rounds", type=int, default=None, help="攻辩轮数（1-3）")
    run.add_argument("--no-wait", action="store_true", help="只启动流水线不等待，稍后用 status 查询")
    run.add_argument("--timeout", type=int, default=1800, help="等待上限秒数（默认 1800）")
    run.add_argument("--poll-interval", type=int, default=2, help="服务端轮询间隔秒数（默认 2）")
    run.set_defaults(func=cmd_run)
    status = sub.add_parser("status", help="查询流水线进度")
    status.add_argument("--project", required=True, help="项目目录")
    status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
