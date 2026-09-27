# -*- coding: utf-8 -*-
"""AI Novel Generator 命令行工具。

面向 Hermes 等外部智能体调用：所有成功与失败结果都以 JSON 形式写到 stdout，
配合进程退出码（0 成功 / 1 失败）供调用方判断。文档生成链路复用 Web 版同一套核心逻辑。
"""

import argparse
import json
import os
import sys
from pathlib import Path

from config_manager import load_config
from novel_generator import book_store
from novel_generator import outline_tree
from novel_generator import unit_generation


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
SUPPORTED_DOC_SUFFIXES = {".txt", ".md", ".markdown"}


def _out(payload, code=0):
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.exit(code)


def _fail(message):
    _out({"ok": False, "error": str(message)}, code=1)


def _llm(task_key="chapter_outline_llm", name=None):
    config = load_config(str(CONFIG_FILE))
    configs = config.get("llm_configs") or {}
    if not configs:
        raise RuntimeError("config.json 里没有可用的 llm_configs，请先配置大模型")
    chosen = name or (config.get("choose_configs") or {}).get(task_key) or config.get("last_llm_config_name")
    if name and name not in configs:
        raise RuntimeError(f"未找到名为 {name!r} 的模型配置，可用：{', '.join(sorted(configs))}")
    if chosen not in configs:
        chosen = next(iter(configs))
    return {"name": chosen, **configs[chosen]}


def _book_mainline(book):
    parts = []
    if str(book.get("title") or "").strip():
        parts.append(f"书名：{book['title']}")
    if str(book.get("mainline") or "").strip():
        parts.append(f"单主线：{book['mainline']}")
    return "\n".join(parts)


def _read_document(path):
    doc = Path(path)
    if not doc.is_file():
        raise FileNotFoundError(f"文档不存在：{doc}")
    if doc.suffix.lower() not in SUPPORTED_DOC_SUFFIXES:
        raise ValueError(f"暂不支持 {doc.suffix or '无扩展名'} 格式，请提供 .txt / .md 文档")
    text = doc.read_text(encoding="utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return doc.name, text


def _expand_progress(stage, detail, done, total):
    if total:
        print(f"[{stage}] {detail} ({done}/{total})", file=sys.stderr, flush=True)
    else:
        print(f"[{stage}] {detail}", file=sys.stderr, flush=True)


def _outline_stats(tree):
    stats = []
    for branch in tree.get("branches", []):
        chapter_plan_count = 0
        chapter_count = 0
        for event in branch.get("events", []):
            chapter_branches = [item for item in event.get("chapter_branches", []) if item.get("chapters")]
            chapter_plan_count += len(chapter_branches)
            chapter_count += sum(len(item.get("chapters", [])) for item in chapter_branches)
        stats.append({
            "plan_key": branch.get("plan_key"),
            "title": (branch.get("plan") or {}).get("title"),
            "event_count": len(branch.get("events", [])),
            "chapter_plan_count": chapter_plan_count,
            "chapter_count": chapter_count,
        })
    return stats


def _parse_plan_keys(value):
    keys = [key.strip() for key in str(value or "").split(",") if key.strip()]
    return keys or None


def _plan_without_steps(plan):
    if not isinstance(plan, dict):
        return plan
    return {key: value for key, value in plan.items() if key != "steps"}


def _clean_branches(branches):
    cleaned = []
    for branch in branches or []:
        events = []
        for event in branch.get("events", []):
            chapter_branches = []
            for chapter_branch in event.get("chapter_branches", []):
                if not chapter_branch.get("chapters"):
                    continue
                chapter_branches.append({
                    "plan_key": chapter_branch.get("plan_key"),
                    "plan": _plan_without_steps(chapter_branch.get("plan")),
                    "chapters": chapter_branch.get("chapters", []),
                })
            cleaned_event = dict(event)
            cleaned_event["chapter_branches"] = chapter_branches
            events.append(cleaned_event)
        cleaned.append({
            "plan_key": branch.get("plan_key"),
            "plan": _plan_without_steps(branch.get("plan")),
            "events": events,
        })
    return cleaned


def _emit_result(project, unit_id):
    book = book_store.load_book(project)
    unit = book_store.get_unit(book, unit_id) or {}
    tree = unit.get("outline_tree") or {}
    routes = tree.get("directory_options") or unit.get("directory_options") or []
    _out({
        "ok": True,
        "project": project,
        "title": book.get("title", ""),
        "mainline": book.get("mainline", ""),
        "world": book.get("world", {}),
        "characters": book.get("characters", []),
        "unit_id": unit_id,
        "routes": routes,
        "branches": _clean_branches(tree.get("branches", [])),
        "stats": _outline_stats(tree),
    })


def run_from_docs(args):
    """根据上传文档，生成世界观、人物设定与多路线单元。"""
    project = str(Path(args.project).expanduser().resolve())
    os.makedirs(project, exist_ok=True)

    if not getattr(args, "resume", False) and not getattr(args, "docs", None):
        raise ValueError("请用 --docs 提供至少一份参考文档；续跑时用 --resume 可省略")

    model = _llm("chapter_outline_llm", getattr(args, "llm", None))
    guidance = str(args.guidance or "").strip()

    if getattr(args, "resume", False):
        book = book_store.load_book(project)
        if not book.get("units") or not book.get("current"):
            raise RuntimeError("项目里没有可继续的单元，请去掉 --resume 重新运行")
        book = outline_tree.expand_outline_tree(
            project=project,
            unit_id=book["current"],
            model=model,
            guidance=guidance,
            progress=_expand_progress,
            plan_keys=_parse_plan_keys(getattr(args, "plan_keys", "")),
            include_design=bool(getattr(args, "with_design", False)),
            workers=int(getattr(args, "workers", 4) or 4),
        )
        _emit_result(project, book["current"])

    for doc_path in args.docs:
        name, content = _read_document(doc_path)
        book_store.add_reference_file(project, name, content)

    reference_text = book_store.render_reference_files(project)
    if not reference_text.strip():
        raise RuntimeError("参考文本为空，请检查上传的文档内容")

    print("[setup] 正在根据文档生成世界观与人物设定（第 1 次模型请求，请稍候）", file=sys.stderr, flush=True)
    brief = unit_generation.generate_reference_brief(
        reference_text=reference_text,
        guidance=guidance,
        model=model,
    ).get("brief") or {}

    book_store.save_book_meta(project, {
        "title": str(brief.get("title") or "").strip(),
        "mainline": str(brief.get("mainline") or "").strip(),
    })
    book_store.save_world(project, brief.get("world") or {})
    characters = book_store.save_characters(project, brief.get("characters") or [])
    cast = [item["id"] for item in characters if item.get("name")]

    book = book_store.create_unit(project, {
        "title": str(brief.get("title") or "").strip(),
        "goal": str(brief.get("mainline") or "").strip(),
        "cast": cast,
    })
    unit_id = book.get("current")
    unit = book_store.get_unit(book, unit_id)
    if not unit:
        raise RuntimeError("创建单元失败")

    if getattr(args, "routes_only", False):
        routes = unit_generation.generate_directory_options(
            world_text=book_store.render_world(book.get("world") or {}),
            cast_text=unit_generation.render_cast(book.get("characters") or [], unit.get("cast") or []),
            unit_goal=str(unit.get("goal") or ""),
            lead_in=str(unit.get("lead_in") or ""),
            guidance=guidance,
            carry_over="",
            graph_text="",
            mainline_text=_book_mainline(book),
            reference_text=reference_text,
            history=[],
            model=model,
        ).get("directory_options") or []
        working = book_store.load_unit_working(project, unit_id)
        working["directory_options"] = routes
        working["chosen_plan"] = ""
        working["segments"] = []
        book_store.save_unit_working(project, unit_id, working)
    else:
        book = outline_tree.expand_outline_tree(
            project=project,
            unit_id=unit_id,
            model=model,
            guidance=guidance,
            progress=_expand_progress,
            plan_keys=_parse_plan_keys(getattr(args, "plan_keys", "")),
            include_design=bool(getattr(args, "with_design", False)),
            workers=int(getattr(args, "workers", 4) or 4),
        )

    _emit_result(project, unit_id)


def run_show(args):
    """读取一个已生成项目的摘要，方便调用方核对结果。"""
    project = str(Path(args.project).expanduser().resolve())
    if not Path(project).is_dir():
        raise FileNotFoundError(f"项目目录不存在：{project}")
    book = book_store.load_book(project)
    units = []
    for unit in book.get("units", []):
        routes = unit.get("directory_options") or []
        units.append({
            "id": unit.get("id"),
            "seq": unit.get("seq"),
            "title": unit.get("title"),
            "goal": unit.get("goal"),
            "route_count": len(routes),
            "routes": [
                {"key": r.get("key"), "title": r.get("title"), "step_count": len(r.get("steps", []))}
                for r in routes
            ],
            "stats": _outline_stats(unit.get("outline_tree") or {}),
        })
    _out({
        "ok": True,
        "project": project,
        "title": book.get("title", ""),
        "mainline": book.get("mainline", ""),
        "world": book.get("world", {}),
        "characters": book.get("characters", []),
        "units": units,
    })


def run_config(args):
    """列出可用的大模型配置（不输出完整密钥），方便调用方确认环境就绪。"""
    config = load_config(str(CONFIG_FILE))
    llms = []
    for name, cfg in (config.get("llm_configs") or {}).items():
        llms.append({
            "name": name,
            "model_name": cfg.get("model_name"),
            "base_url": cfg.get("base_url"),
            "has_key": bool(str(cfg.get("api_key") or "").strip()),
        })
    _out({
        "ok": True,
        "llm_configs": llms,
        "choose_configs": config.get("choose_configs", {}),
    })


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="AI Novel Generator 命令行工具：面向 Hermes 等外部智能体，stdout 输出 JSON。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    docs = sub.add_parser("from-docs", help="根据上传文档生成世界观、人物与多路线单元")
    docs.add_argument("--project", required=True, help="项目输出目录")
    docs.add_argument("--docs", nargs="+", help="参考文档路径（.txt / .md，可传多个；--resume 时省略）")
    docs.add_argument("--guidance", default="", help="想要的题材 / 标签 / 关系走向等方向说明")
    docs.add_argument("--llm", default=None, help="指定 llm_configs 中的模型配置名，默认走 choose_configs")
    docs.add_argument("--routes-only", action="store_true", help="只生成多套目录路线，不做章节拆分与内容设计")
    docs.add_argument("--with-design", action="store_true", help="继续给每一章做内容设计（默认只展开到章节方案）")
    docs.add_argument("--plan-keys", default="", help="只展开指定路线（逗号分隔，如 A,C；默认全部展开）")
    docs.add_argument("--workers", type=int, default=4, help="并行调用数（默认 4）")
    docs.add_argument("--resume", action="store_true", help="跳过设定与文档导入，直接继续展开项目当前单元")
    docs.set_defaults(func=run_from_docs)

    show = sub.add_parser("show", help="读取已生成项目的摘要")
    show.add_argument("--project", required=True, help="项目目录")
    show.set_defaults(func=run_show)

    cfg = sub.add_parser("config", help="列出可用的大模型配置")
    cfg.set_defaults(func=run_config)

    web_cmd = sub.add_parser("web", help="启动本地 Web 创作台（默认 http://127.0.0.1:8000）")
    web_cmd.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    web_cmd.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    web_cmd.set_defaults(func=run_web)

    return parser


def run_web(args):
    import threading
    import webbrowser

    import uvicorn

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    print(f"Web 创作台已启动：http://127.0.0.1:{args.port} （Ctrl+C 退出）")
    uvicorn.run("web_api:app", host="127.0.0.1", port=args.port, log_level="warning")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        _out({"ok": False, "error": "已中断；重新运行并加 --resume 可以继续未完成的部分"}, code=130)
    except Exception as exc:  # noqa: BLE001 - 统一转成 JSON 错误输出
        _fail(exc)


if __name__ == "__main__":
    main()
