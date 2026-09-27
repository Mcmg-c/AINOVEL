# -*- coding: utf-8 -*-
"""AI Novel Generator 的本地 Web API（单主线 + 章节 + 图谱创作流程）。"""

import json
import os
import threading
import time
import concurrent.futures
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from langchain_core.utils.function_calling import convert_to_openai_tool

from config_manager import load_config, save_config
import epub_utils
from llm_adapters import check_base_url
from novel_generator import book_store
from novel_generator import debate_flow
from novel_generator import full_flow
from novel_generator import unit_generation
from novel_generator.common import LLM_INTERACTION_LOG, add_llm_trace_listener

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
_LLM_TAIL_BYTES = 128 * 1024

AINOVEL_HOST = os.environ.get("AINOVEL_HOST", "127.0.0.1").strip() or "127.0.0.1"
try:
    AINOVEL_PORT = int(os.environ.get("AINOVEL_PORT", "8123") or 8123)
except ValueError:
    AINOVEL_PORT = 8123
AINOVEL_TOKEN = os.environ.get("AINOVEL_TOKEN", "").strip()

app = FastAPI(title="AI Novel Generator · 小说创作台", version="3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def agent_token_guard(request, call_next):
    """设置 AINOVEL_TOKEN 后，所有 /api/* 请求都必须带 X-AINOVEL-Token 头。"""
    if AINOVEL_TOKEN and str(request.url.path).startswith("/api"):
        supplied = request.headers.get("x-ainovel-token", "")
        if supplied != AINOVEL_TOKEN:
            return JSONResponse({"ok": False, "error": "缺少或无效的 X-AINOVEL-Token"}, status_code=401)
    return await call_next(request)


def _project_path(value: str, create: bool = False) -> Path:
    if not value or not value.strip():
        raise HTTPException(400, "请先填写项目保存路径")
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.is_dir():
        raise HTTPException(400, f"项目路径不存在：{path}")
    return path


def _config() -> dict:
    return load_config(str(CONFIG_FILE)) or {}


def _llm(task_key: str) -> dict:
    config = _config()
    configs = config.get("llm_configs", {})
    name = config.get("choose_configs", {}).get(task_key) or config.get("last_llm_config_name")
    if name not in configs:
        name = next(iter(configs), None)
    if not name:
        raise HTTPException(400, "尚未配置大模型")
    return {"name": name, **configs[name]}


def _ok(project: Path) -> dict:
    return {"ok": True, "book": book_store.load_book(str(project))}


def _book_mainline(book: dict) -> str:
    """把书名 + 单主线渲染成提示词片段。"""
    title = str(book.get("title", "") or "").strip()
    mainline = str(book.get("mainline", "") or "").strip()
    parts = []
    if title:
        parts.append(f"书名：{title}")
    if mainline:
        parts.append(f"单主线：{mainline}")
    return "\n".join(parts)


def _py_repr(value) -> str:
    return repr(str(value))


def _message_class(role: str) -> str:
    return {"system": "SystemMessage", "assistant": "AIMessage"}.get(role, "HumanMessage")


def build_langchain_preview(model: dict, prompt: str = "", messages: list | None = None, tools: list | None = None) -> str:
    """把一次 LLM 调用显示成 LangChain 风格，和真实调用路径一致。"""
    if messages:
        msg_list = unit_generation.render_message_list(messages)
    else:
        msg_list = [{"role": "user", "content": prompt}]

    fmt = str(model.get("interface_format", "OpenAI")).strip().lower()
    model_name = str(model.get("model_name", "") or "").strip() or "MODEL"
    base_url = check_base_url(str(model.get("base_url", "") or "").strip())
    temperature = float(model.get("temperature", 0.7))
    max_tokens = int(model.get("max_tokens", 8192))
    timeout = int(model.get("timeout", 600))

    if fmt in {"openai", "deepseek", "ollama", "ml studio"}:
        kwargs = [
            f"model={_py_repr(model_name)}",
            f"base_url={_py_repr(base_url)}",
            f"max_tokens={max_tokens!r}",
            f"temperature={temperature!r}",
            f"timeout={timeout!r}",
        ]
        used_classes = sorted({_message_class(item["role"]) for item in msg_list})
        message_imports = f"from langchain_core.messages import {', '.join(used_classes)}"
        message_lines = [
            f"    {_message_class(item['role'])}(content={_py_repr(item['content'])}),"
            for item in msg_list
        ]
        bind_chain = ""
        if tools:
            tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
            bind_chain = f".bind_tools([{', '.join(tool_names)}])"
        tool_import = ""
        if tools:
            tool_import = "from novel_generator.unit_generation import " + ", ".join(sorted(set(tool_names))) + "\n"
        return (
            f"{message_imports}\n"
            f"{tool_import}"
            f"from langchain_openai import ChatOpenAI\n\n"
            f"ChatOpenAI(\n"
            f"    {',\n    '.join(kwargs)},\n"
            f"){bind_chain}.invoke([\n"
            f"{chr(10).join(message_lines)}\n"
            f"])"
        )

    return json.dumps(msg_list, ensure_ascii=False, indent=2)


def _openai_tool_schema(tool) -> dict:
    try:
        return convert_to_openai_tool(tool)
    except Exception:
        name = getattr(tool, "name", str(tool))
        description = getattr(tool, "description", "") or ""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


def build_raw_request_preview(model: dict, prompt: str = "", messages: list | None = None, tools: list | None = None) -> str:
    """把一次 LLM 调用还原成最终发给 OpenAI 兼容接口的原始 HTTP 请求体。"""
    if messages:
        msg_list = unit_generation.render_message_list(messages)
    else:
        msg_list = [{"role": "user", "content": prompt}]

    fmt = str(model.get("interface_format", "OpenAI")).strip().lower()
    model_name = str(model.get("model_name", "") or "").strip() or "MODEL"
    base_url = check_base_url(str(model.get("base_url", "") or "").strip()) or "https://api.openai.com/v1"
    temperature = float(model.get("temperature", 0.7))
    max_tokens = int(model.get("max_tokens", 8192))

    if fmt in {"openai", "deepseek", "ollama", "ml studio"}:
        url = base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": model_name,
            "messages": msg_list,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [_openai_tool_schema(tool) for tool in tools]
            body["tool_choice"] = "auto"
        return f"POST {url}\n{json.dumps(body, ensure_ascii=False, indent=2)}"

    return build_langchain_preview(model, prompt=prompt, messages=messages, tools=tools)


def _require_unit(book: dict, unit_id: str) -> dict:
    unit = book_store.get_unit(book, unit_id)
    if not unit:
        raise HTTPException(404, "全书工作区不存在")
    return unit


def _require_ideation(book: dict, unit: dict) -> None:
    if not book_store.is_world_filled(book.get("world", {})):
        raise HTTPException(400, "请先在「设定」里填写世界观")
    if int(book.get("target_words") or 0) <= 0:
        raise HTTPException(400, "请先在「设定」页填写大概总字数")


def _event_by_seq(unit: dict, seq: int) -> dict:
    for event in unit.get("segments", []):
        if int(event.get("seq", 0)) == seq:
            return event
    raise HTTPException(404, "大事件不存在")


def _chapter_by_seqs(unit: dict, event_seq: int, chapter_seq: int) -> dict:
    event = _event_by_seq(unit, event_seq)
    for chapter in event.get("chapters", []):
        if int(chapter.get("seq", 0)) == chapter_seq:
            return chapter
    raise HTTPException(404, "章节不存在")


def _ordered_chapters(unit: dict) -> list:
    """按阅读顺序把全书章节展平，便于跨事件承接上一章。"""
    items = []
    for event in unit.get("segments", []):
        for chapter in event.get("chapters", []):
            items.append({"event": event, "chapter": chapter})
    return items


def _previous_chapter(unit: dict, event_seq: int, chapter_seq: int):
    items = _ordered_chapters(unit)
    for index, item in enumerate(items):
        if int(item["event"].get("seq", 0)) == event_seq and int(item["chapter"].get("seq", 0)) == chapter_seq:
            return items[index - 1]["chapter"] if index > 0 else None
    return None


def _previous_summary(unit: dict, event_seq: int, chapter_seq: int) -> str:
    previous = _previous_chapter(unit, event_seq, chapter_seq)
    if not previous:
        return ""
    draft = str(previous.get("draft", "") or "").strip()
    if draft:
        return draft[-600:]
    chosen = str(previous.get("chosen_design", "") or "")
    if chosen:
        option = next(
            (item for item in previous.get("design_options", []) if str(item.get("key", "")) == chosen),
            None,
        )
        if option:
            text = f"{option.get('title', '')}\n{option.get('summary', '')}\n{option.get('content', '')}".strip()
            return text[-600:]
    return ""


def _previous_tail(unit: dict, event_seq: int, chapter_seq: int) -> str:
    previous = _previous_chapter(unit, event_seq, chapter_seq)
    if not previous:
        return str(unit.get("lead_in", "") or "").strip()
    return str(previous.get("draft", "") or "").strip()[-800:]


def _global_chapter_seq(unit: dict, event_seq: int, chapter_seq: int) -> int:
    """返回某章节全书连续的阅读序号，用于提示词中的「第 N 章」。"""
    for index, item in enumerate(_ordered_chapters(unit), start=1):
        if int(item["event"].get("seq", 0)) == event_seq and int(item["chapter"].get("seq", 0)) == chapter_seq:
            return index
    return chapter_seq


def _carry_units(book: dict, unit: dict) -> list:
    """返回本章应带上的历史章节：显式勾选优先，否则默认带最近一个已完成的章节。"""
    others = [item for item in book.get("units", []) if item.get("id") != unit.get("id")]
    by_id = {item.get("id"): item for item in others}
    chosen = [str(item) for item in unit.get("carry_units") or []]
    result = [by_id[uid] for uid in chosen if uid in by_id]
    if result:
        return result
    prev_seq = int(unit.get("seq", 1)) - 1
    if prev_seq < 1:
        return []
    previous = next((item for item in others if int(item.get("seq", 0)) == prev_seq), None)
    return [previous] if previous else []


def _unit_drafts_text(unit: dict) -> str:
    parts = []
    for event in unit.get("segments", []):
        for chapter in event.get("chapters", []):
            draft = str(chapter.get("draft", "") or "").strip()
            if draft:
                parts.append(f"事件{event.get('seq', 1)} · 第{chapter.get('seq', 1)}章：\n{draft}")
    return "\n\n".join(parts)


def _cast_names(book: dict, unit: dict) -> list:
    by_id = {item.get("id"): str(item.get("name", "")).strip() for item in book.get("characters", [])}
    return [by_id[cast_id] for cast_id in unit.get("cast", []) if by_id.get(cast_id)]


def _graph_working_set(book: dict, unit: dict) -> str:
    carried = _carry_units(book, unit)
    carried_seqs = [int(item.get("seq", 0)) for item in carried]
    return unit_generation.render_graph_working_set(
        book.get("graph", {}), _cast_names(book, unit), carried_seqs, unit.get("carry_facts") or []
    )


def _finished_units_text(book: dict) -> str:
    blocks = []
    for unit in book.get("units", [])[-20:]:
        snapshot = unit.get("snapshot") or {}
        has_content = bool(
            unit.get("chosen_plan")
            or str(unit.get("unit_summary", "") or "").strip()
            or snapshot.get("beats")
            or snapshot.get("character_states")
            or snapshot.get("hooks")
        )
        if has_content:
            blocks.append(unit_generation.render_unit_snapshot(unit))
    return "\n\n".join(blocks).strip() or "（还没有写完的章节）"


def _render_graph(graph: dict) -> str:
    graph = graph if isinstance(graph, dict) else {}
    nodes = {item.get("id"): item for item in graph.get("nodes", []) if isinstance(item, dict) and item.get("id")}
    lines = []
    node_lines = []
    for node in nodes.values():
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        kind = str(node.get("kind", "")).strip()
        note = str(node.get("note", "")).strip()
        status = str(node.get("status", "")).strip()
        label = f"{name}（{kind}）" if kind else name
        node_lines.append("- " + label + (f"：{note}" if note else "") + (f" · {status}" if status else ""))
    if node_lines:
        lines.append("节点：")
        lines.extend(node_lines)
    edge_lines = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str((nodes.get(edge.get("source")) or {}).get("name", edge.get("source", "")) or "").strip()
        target = str((nodes.get(edge.get("target")) or {}).get("name", edge.get("target", "")) or "").strip()
        relation_type = str(edge.get("type", "")).strip() or "有关联"
        edge_lines.append(f"- {source} —{relation_type}→ {target}")
    if edge_lines:
        lines.append("关系：")
        lines.extend(edge_lines)
    event_lines = []
    for event in graph.get("events", []):
        if not isinstance(event, dict):
            continue
        title = str(event.get("title", "")).strip()
        if title:
            event_lines.append(f"- 第{event.get('unit_seq', '?')}章 · {title}")
    if event_lines:
        lines.append("时间线：")
        lines.extend(event_lines)
    fact_lines = []
    for fact in graph.get("facts", []):
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("text", "")).strip()
        if text:
            entities = [str(entity).strip() for entity in (fact.get("entities") or []) if str(entity).strip()]
            line = f"- 第{fact.get('unit_seq', '?')}章 · {text}" + (("（涉及：" + "、".join(entities) + "）") if entities else "")
            fact_lines.append(line)
    if fact_lines:
        lines.append("原子小事件：")
        lines.extend(fact_lines)
    return "\n".join(lines).strip() or "（空）"


# ---------------------------------------------------------------- 基础与配置

@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    return _config()


@app.put("/api/config")
def put_config(payload: dict = Body(...)):
    if not isinstance(payload.get("llm_configs"), dict) or not payload["llm_configs"]:
        raise HTTPException(400, "至少需要一个大模型配置")
    if not save_config(payload, str(CONFIG_FILE)):
        raise HTTPException(500, "配置保存失败")
    return {"ok": True}


# ---------------------------------------------------------------- 项目快照

@app.get("/api/book")
def get_book(path: str):
    project = _project_path(path, create=True)
    return _ok(project)


@app.put("/api/world")
def put_world(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book_store.save_world(str(project), payload.get("world", {}))
    return _ok(project)


@app.put("/api/characters")
def put_characters(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book_store.save_characters(str(project), payload.get("characters", []))
    return _ok(project)


@app.put("/api/book/meta")
def put_book_meta(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else payload
    try:
        words = int(float(str(meta.get("target_words", 0) or 0)))
    except (TypeError, ValueError):
        words = 0
    if words <= 0:
        raise HTTPException(400, "请填写大概总字数（必须大于 0）")
    book_store.save_book_meta(str(project), meta)
    return _ok(project)


# ---------------------------------------------------------------- 参考文本

@app.get("/api/references")
def get_references(path: str):
    project = _project_path(path, create=True)
    return {"ok": True, "references": book_store.load_reference_files(str(project))}


@app.post("/api/references")
def add_reference(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    name = str(payload.get("name", "") or "").strip()
    content = payload.get("content", "")
    if not name:
        raise HTTPException(400, "请提供文件名")
    references = book_store.add_reference_file(str(project), name, content)
    return {"ok": True, "references": references}


@app.post("/api/references/upload")
async def upload_reference(path: str = Form(...), file: UploadFile = File(...)):
    """上传 .txt / .md / .epub 文件；epub 会在服务端转成纯文本后入库。"""
    project = _project_path(path, create=True)
    original = str(file.filename or "").strip()
    if not original:
        raise HTTPException(400, "请选择要上传的文件")
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(400, "文件过大（限制 50MB）")
    lower = original.lower()
    if lower.endswith(".epub"):
        try:
            text = epub_utils.epub_to_text(data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    elif lower.endswith((".txt", ".md")):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("gb18030", errors="ignore")
    else:
        raise HTTPException(400, "仅支持 .txt / .md / .epub 文件")
    stem = os.path.splitext(os.path.basename(original))[0].strip() or "reference"
    references = book_store.add_reference_file(str(project), f"{stem}.txt", text)
    return {"ok": True, "references": references}


@app.put("/api/references/{name}")
def update_reference(name: str, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    references = book_store.toggle_reference_file(str(project), name, bool(payload.get("enabled", True)))
    return {"ok": True, "references": references}


@app.delete("/api/references/{name}")
def delete_reference(name: str, path: str):
    project = _project_path(path, create=True)
    references = book_store.delete_reference_file(str(project), name)
    return {"ok": True, "references": references}


@app.put("/api/reference-limit")
def update_reference_limit(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    limit = book_store.save_reference_limit(str(project), payload.get("limit", 0))
    return {"ok": True, "reference_limit": limit}


@app.post("/api/references/brief")
def reference_brief(payload: dict = Body(...)):
    """参考已上传的 txt，让 AI 起草新书的单主线、世界观与人物预设。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    guidance = str(payload.get("guidance", "") or "").strip()
    if not guidance:
        raise HTTPException(400, "请说明你想写什么（题材 / 标签 / 关系走向）")
    model = _llm("chapter_outline_llm")
    try:
        target_words = int(float(str(payload.get("target_words", 0) or 0)))
    except (TypeError, ValueError):
        target_words = 0
    args = {
        "reference_text": book_store.render_reference_files(str(project)),
        "guidance": guidance,
        "target_words": target_words,
    }
    if payload.get("preview"):
        prompt = unit_generation.build_reference_brief_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "reference-brief", "label": "参考文本：生成设定建议",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        result = unit_generation.generate_reference_brief(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "brief": result.get("brief", {})}


@app.post("/api/proposals")
def generate_proposals(payload: dict = Body(...)):
    """根据参考文本一键生成三套完整方案：主线 + 世界观 + 人物 + 大事件目录。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    guidance = str(payload.get("guidance", "") or "").strip()
    if not guidance:
        raise HTTPException(400, "请先写一句想写什么（题材 / 标签 / 关系走向）")
    model = _llm("chapter_outline_llm")
    try:
        target_words = int(float(str(payload.get("target_words", 0) or 0)))
    except (TypeError, ValueError):
        target_words = 0
    try:
        result = unit_generation.generate_proposals(
            reference_text=book_store.render_reference_files(str(project)),
            guidance=guidance,
            target_words=target_words,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    proposals = result.get("proposals", [])
    book_store.save_proposals(str(project), proposals, debate=None)
    return {"ok": True, "proposals": proposals}


@app.post("/api/proposals/apply")
def apply_proposal(payload: dict = Body(...)):
    """采用其中一套方案：书名/主线/世界观/人物/大事件目录一次性写入。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    proposal = payload.get("proposal") or {}
    if not isinstance(proposal, dict):
        raise HTTPException(400, "方案数据无效")
    mainline = str(proposal.get("mainline", "") or "").strip()
    if not mainline:
        raise HTTPException(400, "这套方案缺少单主线，无法采用")
    try:
        target_words = int(float(str(payload.get("target_words", 0) or 0)))
    except (TypeError, ValueError):
        target_words = 0
    book_store.save_book_meta(str(project), { 
        "title": str(proposal.get("title", "") or "").strip(),
        "mainline": mainline,
        "target_words": target_words,
    })
    world = proposal.get("world") if isinstance(proposal.get("world"), dict) else {}
    book_store.save_world(str(project), world)
    characters = book_store.save_characters(str(project), proposal.get("characters") or [])
    directory = proposal.get("directory") if isinstance(proposal.get("directory"), list) else []
    steps = []
    for index, step in enumerate(directory, start=1):
        if not isinstance(step, dict):
            continue
        steps.append({
            "seq": int(step.get("seq", index) or index),
            "who": str(step.get("who", "") or "").strip(),
            "what": str(step.get("what", "") or "").strip(),
        })
    if not steps:
        raise HTTPException(400, "这套方案缺少大事件目录，请重新生成")
    book = book_store.load_book(str(project))
    units = book.get("units") or []
    unit_id = book.get("current") or ""
    if not unit_id and units:
        unit_id = units[0].get("id") or ""
    if not unit_id:
        created = book_store.create_unit(str(project), {})
        unit_id = created.get("current") or ((created.get("units") or [{}])[0].get("id") or "")
    if not unit_id:
        raise HTTPException(500, "无法创建全书工作区")
    book_store.update_unit(str(project), unit_id, {
        "title": str(proposal.get("title", "") or "").strip(),
        "goal": mainline,
        "cast": [character.get("id") for character in characters if character.get("id")],
    })
    working = book_store.load_unit_working(str(project), unit_id)
    working["directory_options"] = [{
        "key": "A",
        "title": str(proposal.get("title", "") or "").strip() or "方案 A",
        "thesis": str(proposal.get("thesis", "") or "").strip(),
        "angle": "",
        "steps": steps,
    }]
    working["chosen_plan"] = "A"
    working["segments"] = steps
    working["merged_route"] = None
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/proposals/debate")
def debate_proposals(payload: dict = Body(...)):
    """三套方案的三个 Agent 互相辩论，主持人评分裁决并可能融合出新方案。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    proposals = payload.get("proposals") or []
    if not isinstance(proposals, list) or not proposals:
        raise HTTPException(400, "请先生成三套方案，再发起辩论")
    try:
        target_words = int(float(str(payload.get("target_words", 0) or 0)))
    except (TypeError, ValueError):
        target_words = 0
    model = _llm("chapter_outline_llm")
    try:
        result = debate_flow.run_debate(
            reference_text=book_store.render_reference_files(str(project)),
            guidance=str(payload.get("guidance", "") or "").strip(),
            target_words=target_words,
            proposals=proposals,
            model=model,
            rebuttal_rounds=max(1, min(3, int(payload.get("rounds", 1) or 1))),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    book_store.save_proposals(str(project), proposals, debate=result)
    return {"ok": True, "debate": result}


_proposal_pipeline_jobs = {}
_proposal_pipeline_lock = threading.Lock()


def _pipeline_job_key(project: str) -> str:
    return str(_project_path(project, create=True))


def _set_pipeline_state(key: str, **fields) -> dict:
    with _proposal_pipeline_lock:
        state = dict(_proposal_pipeline_jobs.get(key, {}))
        state.update(fields)
        _proposal_pipeline_jobs[key] = state
    return state


def _get_pipeline_state(key: str) -> dict:
    with _proposal_pipeline_lock:
        return dict(_proposal_pipeline_jobs.get(key, {}))


def _run_proposal_pipeline(project: str, guidance: str, target_words: int, rounds: int, workers: int, target_chapters: int, words_per_chapter: int) -> None:
    """一次跑完 LangGraph 全流程：三套方案 → 三方辩论 → 主持人裁决 → 自动采用 → 逐事件拆解评分 → 校验合并 → 串行撰写各章正文，进度写入 job 状态。"""
    key = _pipeline_job_key(project)
    try:
        current_meta = book_store.load_book_meta(project)
        book_store.save_book_meta(project, {**current_meta, "target_chapters": target_chapters, "words_per_chapter": words_per_chapter})
        model = _llm("chapter_outline_llm")
        draft_model = _llm("prompt_draft_llm")

        def progress(stage, label, detail, done, total):
            _set_pipeline_state(key, status="running", stage=stage, stage_label=label, detail=detail, done=done, total=total)

        result = full_flow.run_full_pipeline(
            project=project,
            reference_text=book_store.render_reference_files(project),
            guidance=guidance,
            target_words=target_words,
            model=model,
            rebuttal_rounds=rounds,
            workers=workers,
            target_chapters=target_chapters,
            words_per_chapter=words_per_chapter,
            draft_model=draft_model,
            progress=progress,
        )
        chapters = (result.get("merged_route") or {}).get("chapters") or []
        written = int(result.get("written_count") or 0)
        _set_pipeline_state(key, status="done", stage="done", stage_label="完成", detail=f"LangGraph 全流程完成：三套方案经辩论裁决后自动采用，拆分并合并出一条 {len(chapters)} 章的完整路线，已串行写完 {written} 章正文", done=1, total=1, result=book_store.load_book(project))
    except Exception as exc:
        _set_pipeline_state(key, status="error", stage="error", stage_label="失败", message=str(exc))


@app.post("/api/proposals/pipeline")
def start_proposal_pipeline(payload: dict = Body(...)):
    """启动 LangGraph 全流水线；用 GET /api/proposals/pipeline 轮询进度。"""
    project = str(_project_path(str(payload.get("path", "")), create=True))
    guidance = str(payload.get("guidance", "") or "").strip()
    if not guidance:
        raise HTTPException(400, "请先写一句想写什么（题材 / 标签 / 关系走向）")
    try:
        target_words = int(float(str(payload.get("target_words", 0) or 0)))
    except (TypeError, ValueError):
        target_words = 0
    rounds = max(1, min(3, int(payload.get("rounds", 1) or 1)))
    workers = max(1, min(8, int(payload.get("workers", 4) or 4)))
    try:
        target_chapters = int(payload.get("target_chapters") or 0)
    except (TypeError, ValueError):
        target_chapters = 0
    try:
        words_per_chapter = int(payload.get("words_per_chapter") or 0)
    except (TypeError, ValueError):
        words_per_chapter = 0
    if not target_chapters or not words_per_chapter:
        meta = book_store.load_book_meta(project)
        target_chapters = target_chapters or int(meta.get("target_chapters") or 0)
        words_per_chapter = words_per_chapter or int(meta.get("words_per_chapter") or 0)
    key = _pipeline_job_key(project)
    with _proposal_pipeline_lock:
        existing = _proposal_pipeline_jobs.get(key)
        if existing and existing.get("status") == "running":
            return {"ok": True, "job": dict(existing)}
        _proposal_pipeline_jobs[key] = {"status": "queued", "stage": "queued", "stage_label": "排队中", "detail": "", "done": 0, "total": 1, "message": ""}
    threading.Thread(target=_run_proposal_pipeline, args=(project, guidance, target_words, rounds, workers, target_chapters, words_per_chapter), daemon=True).start()
    return {"ok": True, "job": _get_pipeline_state(key)}


@app.get("/api/proposals/pipeline")
def proposal_pipeline_status(path: str = ""):
    project = str(_project_path(path or "", create=True))
    return {"ok": True, "job": _get_pipeline_state(project)}


# ---------------------------------------------------------------- Agent（Hermes/Codex 等）一键入口

def _agent_payload_int(payload: dict, key: str, default: int = 0, minimum: int = 0) -> int:
    try:
        value = int(float(str(payload.get(key, default) or default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _agent_add_docs(project: str, docs) -> list[str]:
    """把服务器本机上的参考文档读进项目（.txt/.md/.epub），返回入库文件名列表。"""
    if isinstance(docs, str):
        docs = [docs]
    names = []
    for doc in docs or []:
        path = Path(str(doc)).expanduser()
        if not path.is_file():
            raise HTTPException(400, f"文档不存在：{path}")
        lower = path.suffix.lower()
        if lower == ".epub":
            try:
                text = epub_utils.epub_to_text(path.read_bytes())
            except ValueError as exc:
                raise HTTPException(400, f"无法解析 epub：{path}（{exc}）") from exc
        elif lower in (".txt", ".md"):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="gb18030", errors="ignore")
        else:
            raise HTTPException(400, f"仅支持 .txt / .md / .epub：{path}")
        name = f"{path.stem.strip() or 'reference'}.txt"
        book_store.add_reference_file(project, name, text)
        names.append(name)
    return names


def _agent_result_summary(project: str) -> dict:
    book = book_store.load_book(project)
    unit = next((item for item in book.get("units") or [] if item.get("id") == book.get("current")), None)
    merged_route = (unit or {}).get("merged_route") or {}
    chapters = merged_route.get("chapters") or []
    written = sum(1 for item in chapters if str(item.get("draft", "") or "").strip())
    return {
        "title": book.get("title") or "",
        "merged_route_title": merged_route.get("title") or "",
        "chapters": len(chapters),
        "written_chapters": written,
        "target_words": int(book.get("target_words") or 0),
        "target_chapters": int(book.get("target_chapters") or 0),
        "words_per_chapter": int(book.get("words_per_chapter") or 0),
    }


@app.post("/api/agent/run")
def agent_run(payload: dict = Body(...)):
    """外部 Agent 的一键入口：入参考文档 → 跑完整 LangGraph 全流程 → 阻塞等待并返回结果。

    必填：project（项目目录）、guidance（想写什么）。
    规模：target_words / target_chapters / words_per_chapter（未传时用项目里已保存的规模，都要求 > 0）。
    可选：docs（服务器本机路径，可传多个）、rounds、workers、wait（默认 true）、timeout（默认 1800 秒）、poll_interval（默认 2 秒）。
    """
    project_path = _project_path(str(payload.get("project", "")), create=True)
    project = str(project_path)
    guidance = str(payload.get("guidance", "") or "").strip()
    if not guidance:
        raise HTTPException(400, "缺少 guidance：请一句话说明想写什么（题材 / 标签 / 关系走向）")
    names = _agent_add_docs(project, payload.get("docs"))
    meta = book_store.load_book_meta(project)
    target_words = _agent_payload_int(payload, "target_words", int(meta.get("target_words") or 0))
    target_chapters = _agent_payload_int(payload, "target_chapters", int(meta.get("target_chapters") or 0))
    words_per_chapter = _agent_payload_int(payload, "words_per_chapter", int(meta.get("words_per_chapter") or 0))
    if target_words <= 0 or target_chapters <= 0 or words_per_chapter <= 0:
        raise HTTPException(400, "请提供完整的全书规模：target_words / target_chapters / words_per_chapter（均需大于 0）。也可先在项目里保存过规模。")
    book_store.save_book_meta(project, {**meta, "target_words": target_words, "target_chapters": target_chapters, "words_per_chapter": words_per_chapter})
    wait = bool(payload.get("wait", True))
    timeout = max(60, min(3600, _agent_payload_int(payload, "timeout", 1800, minimum=1)))
    poll_interval = max(1, min(30, _agent_payload_int(payload, "poll_interval", 2, minimum=1)))
    started = start_proposal_pipeline({
        "path": project,
        "guidance": guidance,
        "target_words": target_words,
        "target_chapters": target_chapters,
        "words_per_chapter": words_per_chapter,
        "rounds": _agent_payload_int(payload, "rounds", 1, minimum=1),
        "workers": _agent_payload_int(payload, "workers", 4, minimum=1),
    })
    job = started.get("job") or {}
    if not wait:
        return {"ok": True, "project": project, "docs_added": names, "job": job}
    deadline = time.monotonic() + timeout
    while True:
        job = _get_pipeline_state(project)
        status = job.get("status")
        if status in ("done", "error"):
            break
        if time.monotonic() >= deadline:
            return {"ok": False, "error": "全流程超时未完成，可用 /api/agent/status 继续查询", "project": project, "job": job}
        time.sleep(poll_interval)
    if status == "error":
        return {"ok": False, "error": job.get("message") or "全流程失败", "project": project, "job": job}
    return {
        "ok": True,
        "project": project,
        "docs_added": names,
        "job": job,
        "summary": _agent_result_summary(project),
        "book": book_store.load_book(project),
    }


@app.get("/api/agent/status")
def agent_status(path: str = ""):
    """查询某个项目 /api/agent/run 的任务进度。"""
    project = str(_project_path(path or "", create=True))
    return {"ok": True, "project": project, "job": _get_pipeline_state(project)}


# ---------------------------------------------------------------- 章节管理

@app.post("/api/units")
def create_unit(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    return {"ok": True, "book": book_store.create_unit(str(project), payload)}


@app.put("/api/units/{unit_id}")
def update_unit(unit_id: str, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    return {"ok": True, "book": book_store.update_unit(str(project), unit_id, payload)}


@app.delete("/api/units/{unit_id}")
def delete_unit(unit_id: str, path: str):
    project = _project_path(path, create=True)
    return {"ok": True, "book": book_store.delete_unit(str(project), unit_id)}


@app.post("/api/units/{unit_id}/select")
def select_unit(unit_id: str, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    return {"ok": True, "book": book_store.select_unit(str(project), unit_id)}


# ---------------------------------------------------------------- 一键编排（harness）

_auto_outline_jobs: dict[str, dict] = {}
_auto_outline_lock = threading.RLock()


def _auto_outline_job_key(project: str, unit_id: str) -> str:
    return f"{str(project)}::{unit_id}"


def _set_outline_state(key: str, **fields) -> dict:
    with _auto_outline_lock:
        job = _auto_outline_jobs.setdefault(key, {"status": "idle"})
        job.update(fields)
        return dict(job)


def _get_outline_state(key: str) -> dict:
    with _auto_outline_lock:
        return dict(_auto_outline_jobs.get(key, {"status": "idle"}))


def _design_option_text(option: dict) -> str:
    return f"{option.get('title', '')}\n{option.get('summary', '')}\n{option.get('content', '')}".strip()


def _chapter_design_summary(chapter: dict) -> str:
    options = chapter.get("design_options") or []
    if not options:
        return ""
    chosen = str(chapter.get("chosen_design") or options[0].get("key") or "")
    option = next((item for item in options if str(item.get("key")) == chosen), options[0])
    return _design_option_text(option)[-600:]


def _run_auto_outline(project: str, unit_id: str, guidance: str, history: list) -> None:
    """把「大事件目录 → 拆章节 → 内容设计」全分支展开：每套目录方案都往下拆、每套章节方案都做内容设计。"""
    project = str(_project_path(project, create=True))
    key = _auto_outline_job_key(project, unit_id)
    try:
        book = book_store.load_book(project)
        unit = book_store.get_unit(book, unit_id)
        if not unit:
            _set_outline_state(key, status="error", message="全书工作区不存在")
            return
        try:
            _require_ideation(book, unit)
        except HTTPException as exc:
            _set_outline_state(key, status="error", message=str(exc.detail))
            return

        model = _llm("chapter_outline_llm")
        working = book_store.load_unit_working(project, unit_id)

        def save() -> None:
            book_store.save_unit_working(project, unit_id, working)

        base = {
            "world_text": book_store.render_world(book.get("world", {})),
            "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
            "unit_goal": str(unit.get("goal", "") or ""),
            "mainline_text": _book_mainline(book),
            "graph_text": _graph_working_set(book, unit),
            "reference_text": book_store.render_reference_files(project),
            "history": history or [],
            "target_words": int(book.get("target_words") or 0),
            "target_chapters": int(book.get("target_chapters") or 0),
            "words_per_chapter": int(book.get("words_per_chapter") or 0),
        }

        # 1) 大事件目录：多套方案
        _set_outline_state(key, status="running", stage="directory", stage_label="大事件目录", detail="正在生成多套目录方案", done=0, total=1)
        if not working.get("directory_options"):
            result = unit_generation.generate_directory_options(
                world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
                lead_in=str(unit.get("lead_in", "") or ""), guidance=guidance or "",
                history=base["history"], carry_over=unit_generation.render_carry_over(_carry_units(book, unit)),
                graph_text=base["graph_text"], mainline_text=base["mainline_text"], reference_text=base["reference_text"],
                model=model,
            )
            working["directory_options"] = result.get("directory_options", [])
            working["chosen_plan"] = ""
            working["segments"] = []
            save()
        plans = working["directory_options"]
        if not plans:
            _set_outline_state(key, status="error", stage="directory", message="目录方案为空")
            return

        branches = []
        for plan in plans:
            events = [
                {"seq": index, "who": str(step.get("who", "") or "").strip(), "what": str(step.get("what", "") or "").strip(), "chapter_branches": []}
                for index, step in enumerate(plan.get("steps", []), start=1)
            ]
            branches.append({"plan_key": str(plan.get("key")), "plan": plan, "events": events})
        working["outline_tree"] = {"directory_options": plans, "branches": branches}
        save()

        chapter_nodes = sum(len(branch["events"]) for branch in branches)
        total = 1 + chapter_nodes
        done = 1
        _set_outline_state(key, status="running", stage="chapters", stage_label="拆章节", detail=f"{len(branches)} 套目录 × {chapter_nodes} 个事件", done=done, total=total)

        # 2) 每一套目录的每一个事件都拆出多套章节方案
        for branch in branches:
            plan_key = branch["plan_key"]
            directory_text = unit_generation.render_events(branch["events"])
            for event in branch["events"]:
                _set_outline_state(key, status="running", stage="chapters", stage_label="拆章节", detail=f"目录 {plan_key} · 事件 {event['seq']}/{len(branch['events'])}", done=done, total=total)
                result = unit_generation.generate_event_chapter_options(
                    world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
                    mainline_text=base["mainline_text"], directory_text=directory_text,
                    event={"seq": event["seq"], "who": event["who"], "what": event["what"]},
                    history=base["history"], graph_text=base["graph_text"], reference_text=base["reference_text"],
                    model=model,
                )
                for chapter_plan in result.get("chapter_options", []):
                    chapters = [
                        {"seq": index, "who": str(step.get("who", "") or "").strip(), "what": str(step.get("what", "") or "").strip(), "design_options": [], "chosen_design": ""}
                        for index, step in enumerate(chapter_plan.get("steps", []), start=1)
                    ]
                    event["chapter_branches"].append({"plan_key": str(chapter_plan.get("key")), "plan": chapter_plan, "chapters": chapters})
                save()
                done += 1

        design_nodes = sum(len(cb["chapters"]) for branch in branches for event in branch["events"] for cb in event["chapter_branches"])
        total = 1 + chapter_nodes + design_nodes
        done = chapter_nodes + 1

        # 3) 每一套章节方案的每一章都做内容设计
        for branch in branches:
            plan_key = branch["plan_key"]
            for event in branch["events"]:
                for cb in event["chapter_branches"]:
                    view = {"segments": [
                        {
                            "seq": item["seq"], "who": item["who"], "what": item["what"],
                            "chapters": cb["chapters"] if item is event else [],
                        }
                        for item in branch["events"]
                    ]}
                    directory_text = unit_generation.render_book_outline(view)
                    for index, chapter in enumerate(cb["chapters"]):
                        _set_outline_state(key, status="running", stage="design", stage_label="内容设计", detail=f"目录 {plan_key} · 事件 {event['seq']} · 方案 {cb['plan_key']} · 第 {chapter['seq']} 章", done=done, total=total)
                        if not chapter.get("design_options"):
                            previous_summary = _chapter_design_summary(cb["chapters"][index - 1]) if index > 0 else ""
                            result = unit_generation.generate_segment_design_options(
                                world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
                                directory_text=directory_text,
                                segment={"seq": chapter["seq"], "who": chapter["who"], "what": chapter["what"]},
                                previous_summary=previous_summary,
                                history=base["history"], graph_text=base["graph_text"], mainline_text=base["mainline_text"], reference_text=base["reference_text"],
                                model=model,
                            )
                            chapter["design_options"] = result.get("options", [])
                            chapter["chosen_design"] = ""
                            save()
                        if chapter["design_options"] and not chapter.get("chosen_design"):
                            chapter["chosen_design"] = str(chapter["design_options"][0].get("key") or "1")
                            save()
                        done += 1

        final = book_store.load_book(project)
        chapter_branch_nodes = sum(len(event["chapter_branches"]) for branch in branches for event in branch["events"])
        _set_outline_state(key, status="done", stage="done", stage_label="完成", detail=f"已生成 {len(branches)} 套目录 / {chapter_branch_nodes} 套章节拆分 / {design_nodes} 章内容设计", done=total, total=total, result=final)
    except Exception as exc:
        _set_outline_state(key, status="error", message=str(exc))


def _select_best_chapter_plans(working: dict) -> int:
    """按 score 给每个事件自动选定章节方案，返回选定数量。"""
    picked = 0
    for segment in working.get("segments") or []:
        options = [plan for plan in (segment.get("chapter_options") or []) if isinstance(plan, dict) and plan.get("key")]
        if not options:
            continue
        best = max(options, key=lambda plan: (float(plan.get("score") or 0), str(plan.get("key", ""))))
        segment["chosen_chapter_plan"] = str(best.get("key", ""))
        picked += 1
    return picked


def _run_batch_outline(project: str, unit_id: str, guidance: str, history: list, workers: int, force: bool, auto_select: bool = False) -> None:
    """选定大事件目录后，一键并行给每个事件生成多套章节拆分方案。"""
    project = str(_project_path(project, create=True))
    key = _auto_outline_job_key(project, unit_id)
    try:
        book = book_store.load_book(project)
        unit = book_store.get_unit(book, unit_id)
        if not unit:
            _set_outline_state(key, status="error", message="全书工作区不存在")
            return
        if not unit.get("chosen_plan"):
            _set_outline_state(key, status="error", message="请先选定一套大事件目录")
            return
        try:
            _require_ideation(book, unit)
        except HTTPException as exc:
            _set_outline_state(key, status="error", message=str(exc.detail))
            return

        model = _llm("chapter_outline_llm")
        working = book_store.load_unit_working(project, unit_id)
        segments = working.get("segments") or []
        directory_text = unit_generation.render_events(segments)
        base = {
            "world_text": book_store.render_world(book.get("world", {})),
            "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
            "unit_goal": str(unit.get("goal", "") or ""),
            "mainline_text": _book_mainline(book),
            "graph_text": _graph_working_set(book, unit),
            "reference_text": book_store.render_reference_files(project),
            "history": history or [],
            "target_words": int(book.get("target_words") or 0),
            "target_chapters": int(book.get("target_chapters") or 0),
            "words_per_chapter": int(book.get("words_per_chapter") or 0),
        }

        tasks = [segment for segment in segments if force or not segment.get("chapter_options")]
        total = len(tasks)
        if not total:
            if auto_select:
                picked = _select_best_chapter_plans(working)
                book_store.save_unit_working(project, unit_id, working)
                final = book_store.load_book(project)
                _set_outline_state(key, status="done", stage="done", stage_label="完成", detail=f"所有事件都已有章节方案，已按评分选定 {picked} 个事件的路线", done=0, total=0, result=final)
            else:
                _set_outline_state(key, status="done", stage="done", stage_label="完成", detail="所有事件都已有章节方案", done=0, total=0)
            return
        _set_outline_state(key, status="running", stage="chapters", stage_label="拆章节", detail=f"并行生成 {total} 个事件的章节方案", done=0, total=total)

        errors = []

        def run_one(segment):
            seq = segment.get("seq")
            try:
                result = unit_generation.generate_event_chapter_options(
                    world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
                    mainline_text=base["mainline_text"], directory_text=directory_text,
                    event={"seq": seq, "who": segment.get("who", ""), "what": segment.get("what", "")},
                    history=base["history"], graph_text=base["graph_text"], reference_text=base["reference_text"],
                    target_words=base["target_words"],
                    target_chapters=base["target_chapters"], words_per_chapter=base["words_per_chapter"], events_total=total,
                    model=model,
                )
                options = result.get("chapter_options", [])
                with _auto_outline_lock:
                    segment["chapter_options"] = options
                    segment["chosen_chapter_plan"] = ""
                    book_store.save_unit_working(project, unit_id, working)
                return seq, options, None
            except Exception as exc:  # noqa: BLE001 - 单事件失败不阻断其它任务
                return seq, [], exc

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
            futures = [pool.submit(run_one, segment) for segment in tasks]
            for future in concurrent.futures.as_completed(futures):
                seq, options, exc = future.result()
                with _auto_outline_lock:
                    done += 1
                    if exc:
                        errors.append((seq, str(exc)))
                        detail = f"事件 {seq} 生成失败（{done}/{total}），完成后可再点一次补齐"
                    else:
                        detail = f"事件 {seq} 已生成 {len(options)} 套章节方案（{done}/{total}）"
                    state = dict(_auto_outline_jobs.get(key, {}))
                    state.update({"status": "running", "stage": "chapters", "stage_label": "拆章节", "detail": detail, "done": done, "total": total})
                    _auto_outline_jobs[key] = state

        with _auto_outline_lock:
            failed = len(errors)
            if failed:
                if failed < total:
                    _set_outline_state(key, status="done", stage="chapters", stage_label="拆章节", detail=f"完成 {total - failed}/{total} 个事件；{failed} 个失败，可再点一次批量生成补齐", done=total, total=total, message="部分失败：" + "；".join(f"事件{seq}: {message[:120]}" for seq, message in errors[:3]))
                else:
                    _set_outline_state(key, status="error", stage="chapters", message=str(errors[0][1]))
            else:
                if auto_select:
                    picked = _select_best_chapter_plans(working)
                    book_store.save_unit_working(project, unit_id, working)
                    final = book_store.load_book(project)
                    _set_outline_state(key, status="done", stage="done", stage_label="完成", detail=f"{total} 个事件的章节方案全部生成，并按评分自动选定了 {picked} 个事件的路线", done=total, total=total, result=final)
                else:
                    _set_outline_state(key, status="done", stage="done", stage_label="完成", detail=f"{total} 个事件的章节方案全部生成", done=total, total=total, result=book_store.load_book(project))
    except Exception as exc:
        _set_outline_state(key, status="error", message=str(exc))


@app.post("/api/units/{unit_id}/auto-outline")
def start_auto_outline(unit_id: str, payload: dict = Body(...)):
    """启动一键编排任务，立即返回；用 GET /auto-outline 轮询进度与结果。"""
    project = str(_project_path(str(payload.get("path", "")), create=True))
    key = _auto_outline_job_key(project, unit_id)
    with _auto_outline_lock:
        existing = _auto_outline_jobs.get(key)
        if existing and existing.get("status") == "running":
            return {"ok": True, "job": dict(existing)}
        _auto_outline_jobs[key] = {"status": "queued", "stage": "queued", "stage_label": "排队中", "detail": "", "done": 0, "total": 1, "message": ""}
    threading.Thread(
        target=_run_auto_outline,
        args=(project, unit_id, str(payload.get("guidance", "") or ""), payload.get("history") or []),
        daemon=True,
    ).start()
    return {"ok": True, "job": _get_outline_state(key)}


@app.get("/api/units/{unit_id}/auto-outline")
def auto_outline_status(unit_id: str, path: str = ""):
    """查询一键编排任务进度；完成后 job.result 包含完整 book。"""
    project = str(_project_path(str(path or ""), create=True))
    key = _auto_outline_job_key(project, unit_id)
    return {"ok": True, "job": _get_outline_state(key)}


@app.post("/api/units/{unit_id}/batch-outline")
def start_batch_outline(unit_id: str, payload: dict = Body(...)):
    """选定大事件目录后的一键并行拆分：给每个事件生成多套章节方案；用 GET /auto-outline 轮询进度。"""
    project = str(_project_path(str(payload.get("path", "")), create=True))
    key = _auto_outline_job_key(project, unit_id)
    with _auto_outline_lock:
        existing = _auto_outline_jobs.get(key)
        if existing and existing.get("status") == "running":
            return {"ok": True, "job": dict(existing)}
        _auto_outline_jobs[key] = {"status": "queued", "stage": "queued", "stage_label": "排队中", "detail": "", "done": 0, "total": 1, "message": ""}
    threading.Thread(
        target=_run_batch_outline,
        args=(project, unit_id, str(payload.get("guidance", "") or ""), payload.get("history") or [], int(payload.get("workers", 4) or 4), bool(payload.get("force", False)), bool(payload.get("auto_select", False))),
        daemon=True,
    ).start()
    return {"ok": True, "job": _get_outline_state(key)}


# ---------------------------------------------------------------- 目录方案与生成

@app.post("/api/units/{unit_id}/directory-options")
def unit_directory_options(unit_id: str, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    _require_ideation(book, unit)
    model = _llm("chapter_outline_llm")
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "lead_in": str(unit.get("lead_in", "") or ""),
        "carry_over": unit_generation.render_carry_over(_carry_units(book, unit)),
        "graph_text": _graph_working_set(book, unit),
        "mainline_text": _book_mainline(book),
        "reference_text": book_store.render_reference_files(str(project)),
        "guidance": str(payload.get("guidance", "") or ""),
        "history": payload.get("history") or [],
        "target_words": int(book.get("target_words") or 0),
    }
    if payload.get("preview"):
        prompt = unit_generation.build_directory_options_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "directory", "label": "全书大事件目录：A/B/C 方案",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        result = unit_generation.generate_directory_options(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    working = book_store.load_unit_working(str(project), unit_id)
    working["directory_options"] = result.get("directory_options", [])
    working["chosen_plan"] = ""
    working["segments"] = []
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.put("/api/units/{unit_id}/directory")
def choose_directory(unit_id: str, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    plan_key = str(payload.get("plan_key", "") or "")
    plan = next(
        (item for item in unit.get("directory_options", []) if str(item.get("key", "")) == plan_key),
        None,
    )
    if not plan:
        raise HTTPException(400, "目录方案不存在")
    working = book_store.load_unit_working(str(project), unit_id)
    working["chosen_plan"] = plan_key
    working["segments"] = []
    for index, step in enumerate(plan.get("steps", []), start=1):
        working["segments"].append({
            "seq": index,
            "who": str(step.get("who", "") or "").strip(),
            "what": str(step.get("what", "") or "").strip(),
        })
    working["merged_route"] = None
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/units/{unit_id}/segments/{seq}/chapter-options")
def event_chapter_options(unit_id: str, seq: int, payload: dict = Body(...)):
    """让 AI 把一个大事件进一步拆成章节方案（A/B/C），不直接改正文。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    _require_ideation(book, unit)
    event = _event_by_seq(unit, seq)
    model = _llm("chapter_outline_llm")
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "mainline_text": _book_mainline(book),
        "directory_text": unit_generation.render_events(unit.get("segments", [])),
        "event": event,
        "graph_text": _graph_working_set(book, unit),
        "reference_text": book_store.render_reference_files(str(project)),
        "history": payload.get("history") or [],
        "target_words": int(book.get("target_words") or 0),
    }
    if payload.get("preview"):
        prompt = unit_generation.build_event_chapter_options_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "chapter-options", "label": f"事件{seq}：拆章节方案",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        result = unit_generation.generate_event_chapter_options(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    working = book_store.load_unit_working(str(project), unit_id)
    for target in working["segments"]:
        if int(target.get("seq", 0)) == seq:
            target["chapter_options"] = result.get("chapter_options", [])
            target["chosen_chapter_plan"] = ""
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.put("/api/units/{unit_id}/segments/{seq}/chapters")
def choose_event_chapters(unit_id: str, seq: int, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    event = _event_by_seq(unit, seq)
    plan_key = str(payload.get("plan_key", "") or "")
    plan = next(
        (item for item in event.get("chapter_options", []) if str(item.get("key", "")) == plan_key),
        None,
    )
    if not plan:
        raise HTTPException(400, "章节拆分方案不存在")
    working = book_store.load_unit_working(str(project), unit_id)
    for target in working["segments"]:
        if int(target.get("seq", 0)) == seq:
            target["chosen_chapter_plan"] = plan_key
            target["chapters"] = []
            for index, step in enumerate(plan.get("steps", []), start=1):
                target["chapters"].append({
                    "seq": index,
                    "who": str(step.get("who", "") or "").strip(),
                    "what": str(step.get("what", "") or "").strip(),
                    "word_estimate": int(step.get("word_estimate") or 0),
                    "word_ratio": int(step.get("word_ratio") or 0),
                })
    working["merged_route"] = None
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/units/{unit_id}/merge-route")
def unit_merge_route(unit_id: str, payload: dict = Body(...)):
    """把作者逐事件选定的章节方案合并成一条完整路线：查重复、补缺口，输出合并方案。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    _require_ideation(book, unit)
    segments = unit.get("segments") or []
    if not segments:
        raise HTTPException(400, "请先选定大事件目录并生成章节方案")
    missing = [str(segment.get("seq")) for segment in segments if not str(segment.get("chosen_chapter_plan", "") or "")]
    if missing:
        raise HTTPException(400, "还有事件未选定章节方案：事件 " + "、".join(missing))
    model = _llm("chapter_outline_llm")
    try:
        result = unit_generation.generate_merged_route(
            world_text=book_store.render_world(book.get("world", {})),
            cast_text=unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
            unit_goal=str(unit.get("goal", "") or ""),
            mainline_text=_book_mainline(book),
            route_text=unit_generation.render_merged_route_text(segments),
            guidance=str(payload.get("guidance", "") or ""),
            history=payload.get("history") or [],
            reference_text=book_store.render_reference_files(str(project)),
            target_words=int(book.get("target_words") or 0),
            target_chapters=int(book.get("target_chapters") or 0),
            words_per_chapter=int(book.get("words_per_chapter") or 0),
            model=model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    working = book_store.load_unit_working(str(project), unit_id)
    working["merged_route"] = result.get("merged_route")
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/units/{unit_id}/segments/{event_seq}/chapters/{chapter_seq}/design-options")
def chapter_design_options(unit_id: str, event_seq: int, chapter_seq: int, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    _require_ideation(book, unit)
    chapter = _chapter_by_seqs(unit, event_seq, chapter_seq)
    chapter_context = dict(chapter)
    chapter_context["seq"] = _global_chapter_seq(unit, event_seq, chapter_seq)
    model = _llm("chapter_outline_llm")
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "directory_text": unit_generation.render_book_outline(unit),
        "segment": chapter_context,
        "previous_summary": _previous_summary(unit, event_seq, chapter_seq),
        "graph_text": _graph_working_set(book, unit),
        "mainline_text": _book_mainline(book),
        "reference_text": book_store.render_reference_files(str(project)),
        "history": payload.get("history") or [],
    }
    if payload.get("preview"):
        prompt = unit_generation.build_segment_design_options_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "design", "label": f"事件{event_seq} 第{chapter_seq}章：内容设计方案",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        result = unit_generation.generate_segment_design_options(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != event_seq:
            continue
        for target in event.setdefault("chapters", []):
            if int(target.get("seq", 0)) == chapter_seq:
                target["design_options"] = result.get("options", [])
                target["chosen_design"] = ""
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.put("/api/units/{unit_id}/segments/{event_seq}/chapters/{chapter_seq}/design")
def choose_design(unit_id: str, event_seq: int, chapter_seq: int, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    chapter = _chapter_by_seqs(unit, event_seq, chapter_seq)
    option_key = str(payload.get("option_key", "") or "")
    if not any(str(item.get("key", "")) == option_key for item in chapter.get("design_options", [])):
        raise HTTPException(400, "内容方案不存在")
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != event_seq:
            continue
        for target in event.setdefault("chapters", []):
            if int(target.get("seq", 0)) == chapter_seq:
                target["chosen_design"] = option_key
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/units/{unit_id}/segments/{event_seq}/chapters/{chapter_seq}/draft")
def segment_draft(unit_id: str, event_seq: int, chapter_seq: int, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    _require_ideation(book, unit)
    chapter = _chapter_by_seqs(unit, event_seq, chapter_seq)
    chosen = str(chapter.get("chosen_design", "") or "")
    option = next((item for item in chapter.get("design_options", []) if str(item.get("key", "")) == chosen), None)
    if not chosen or not option:
        raise HTTPException(400, "请先选择这一章的内容方案")
    chapter_context = dict(chapter)
    chapter_context["seq"] = _global_chapter_seq(unit, event_seq, chapter_seq)
    model = _llm("prompt_draft_llm")
    design_text = f"{option.get('title', '')}\n{option.get('summary', '')}\n{option.get('content', '')}".strip()
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "directory_text": unit_generation.render_book_outline(unit),
        "segment": chapter_context,
        "design_text": design_text,
        "previous_tail": _previous_tail(unit, event_seq, chapter_seq),
        "word_count": int(unit.get("word_count", 1500)),
        "graph_text": _graph_working_set(book, unit),
        "mainline_text": _book_mainline(book),
        "reference_text": book_store.render_reference_files(str(project)),
        "history": payload.get("history") or [],
        "extra_instruction": str(payload.get("extra_instruction", "") or "").strip(),
    }
    if payload.get("preview"):
        prompt = unit_generation.build_segment_draft_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "draft", "label": f"事件{event_seq} 第{chapter_seq}章：写正文",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        text = unit_generation.generate_segment_draft(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != event_seq:
            continue
        for target in event.setdefault("chapters", []):
            if int(target.get("seq", 0)) == chapter_seq:
                target["draft"] = text
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.put("/api/units/{unit_id}/segments/{event_seq}/chapters/{chapter_seq}/draft")
def save_segment_draft(unit_id: str, event_seq: int, chapter_seq: int, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != event_seq:
            continue
        for target in event.setdefault("chapters", []):
            if int(target.get("seq", 0)) == chapter_seq:
                target["draft"] = str(payload.get("text", "") or "")
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/units/{unit_id}/segments/{event_seq}/chapters")
def add_chapter(unit_id: str, event_seq: int, payload: dict = Body(...)):
    """给一个大事件手动新增一章。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != event_seq:
            continue
        chapters = event.setdefault("chapters", [])
        next_seq = max([int(item.get("seq", 0)) for item in chapters], default=0) + 1
        chapters.append({
            "seq": next_seq,
            "who": str(payload.get("who", "") or "").strip(),
            "what": str(payload.get("what", "") or "").strip(),
        })
        break
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.put("/api/units/{unit_id}/segments/{event_seq}/chapters/{chapter_seq}")
def save_chapter(unit_id: str, event_seq: int, chapter_seq: int, payload: dict = Body(...)):
    """保存作者手动修改后的一章（who / what）。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != event_seq:
            continue
        for target in event.setdefault("chapters", []):
            if int(target.get("seq", 0)) == chapter_seq:
                if "who" in payload:
                    target["who"] = str(payload.get("who", "") or "").strip()
                if "what" in payload:
                    target["what"] = str(payload.get("what", "") or "").strip()
                break
        break
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.delete("/api/units/{unit_id}/segments/{event_seq}/chapters/{chapter_seq}")
def delete_chapter(unit_id: str, event_seq: int, chapter_seq: int, path: str = ""):
    """删除一个大事件下的一章，并重新连续编号。"""
    project = _project_path(str(path or ""), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != event_seq:
            continue
        chapters = [item for item in event.setdefault("chapters", []) if int(item.get("seq", 0)) != chapter_seq]
        for index, item in enumerate(chapters, start=1):
            item["seq"] = index
        event["chapters"] = chapters
        break
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.put("/api/units/{unit_id}/segments/{seq}")
def save_event(unit_id: str, seq: int, payload: dict = Body(...)):
    """保存作者手动修改后的一个大事件（who / what）。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    for event in working["segments"]:
        if int(event.get("seq", 0)) != seq:
            continue
        if "who" in payload:
            event["who"] = str(payload.get("who", "") or "").strip()
        if "what" in payload:
            event["what"] = str(payload.get("what", "") or "").strip()
        break
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.put("/api/units/{unit_id}/summary")
def save_unit_summary(unit_id: str, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    working["unit_summary"] = str(payload.get("text", "") or "")
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/units/{unit_id}/snapshot")
def generate_unit_snapshot(unit_id: str, payload: dict = Body(...)):
    """让 AI 根据本章已锁定的内容起草一份前情快照；返回给前端编辑，不直接保存。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    model = _llm("chapter_outline_llm")
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "directory_text": unit_generation.render_book_outline(unit),
        "drafts_text": _unit_drafts_text(unit),
        "summary_text": str(unit.get("unit_summary", "") or ""),
        "mainline_text": _book_mainline(book),
    }
    if payload.get("preview"):
        prompt = unit_generation.build_unit_snapshot_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "snapshot", "label": "全书收尾：前情快照",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        snapshot = unit_generation.generate_unit_snapshot(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "snapshot": snapshot}


@app.put("/api/units/{unit_id}/snapshot")
def save_unit_snapshot(unit_id: str, payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    working["snapshot"] = payload.get("snapshot") or {}
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


@app.post("/api/units/{unit_id}/atomize")
def atomize_unit(unit_id: str, payload: dict = Body(...)):
    """让 AI 把本章前情快照拆成可单独勾选的原子小事件，返回图谱 facts 草稿（不直接保存）。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit = _require_unit(book, unit_id)
    model = _llm("chapter_outline_llm")
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": unit_generation.render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "directory_text": unit_generation.render_book_outline(unit),
        "drafts_text": _unit_drafts_text(unit),
        "snapshot_text": unit_generation.render_unit_snapshot(unit),
        "graph_text": _render_graph(book.get("graph", {})),
        "mainline_text": _book_mainline(book),
        "unit_seq": int(unit.get("seq", 1)),
    }
    if payload.get("preview"):
        prompt = unit_generation.build_atomize_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "atomize", "label": "GraphRAG：拆分小事件",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        result = unit_generation.generate_atomize(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "facts": result.get("facts", []), "unit_seq": int(unit.get("seq", 1))}


@app.put("/api/units/{unit_id}/working")
def save_unit_working(unit_id: str, payload: dict = Body(...)):
    """保存前端直接修改后的目录方案、小目录或正文等 working 数据。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    working = book_store.load_unit_working(str(project), unit_id)
    for key in ("directory_options", "chosen_plan", "segments", "unit_summary", "snapshot", "carry_units", "carry_facts"):
        if key in payload:
            working[key] = payload[key]
    book_store.save_unit_working(str(project), unit_id, working)
    return _ok(project)


# ---------------------------------------------------------------- 知识图谱

@app.put("/api/graph")
def save_graph(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book_store.save_graph(str(project), payload.get("graph") or {})
    return _ok(project)


@app.post("/api/graph/draft")
def draft_graph(payload: dict = Body(...)):
    """让 AI 根据已写完的章节起草一份知识图谱更新；返回给前端编辑，不直接保存。"""
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    model = _llm("chapter_outline_llm")
    units_text = _finished_units_text(book)
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "characters_text": unit_generation.render_all_characters(book.get("characters", [])),
        "units_text": units_text,
        "graph_text": _render_graph(book.get("graph", {})),
        "mainline_text": _book_mainline(book),
    }
    if payload.get("preview"):
        prompt = unit_generation.build_graph_draft_prompt(**args)
        return {"ok": True, "preview": {
            "kind": "graph", "label": "知识图谱：起草更新",
            "prompt": prompt,
            "request": build_raw_request_preview(model, prompt=prompt),
        }}
    try:
        graph = unit_generation.generate_graph_draft(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
        graph = graph if isinstance(graph, dict) else {}
        graph["facts"] = (book.get("graph") or {}).get("facts", [])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "graph": graph}


def _ask_context(unit: dict | None, scope_text: str) -> str:
    """提问上下文：有作用域时只带章节目标 + 选中的那一份，否则带完整章节上下文。"""
    goal = str((unit or {}).get("goal", "") or "").strip()
    if scope_text:
        return f"这本书要写什么：{goal or '（未填写）'}\n\n作者选中的直接上下文：\n{scope_text}"
    return unit_generation.render_unit_context(unit)

@app.post("/api/ask")
def ask_about_selection(payload: dict = Body(...)):
    project = _project_path(str(payload.get("path", "")), create=True)
    book = book_store.load_book(str(project))
    unit_id = str(payload.get("unit_id", "") or "").strip()
    unit = book_store.get_unit(book, unit_id) if unit_id else None
    action = str(payload.get("action", "") or "").strip()
    scope_text = str(payload.get("scope_text", "") or "").strip()
    selected = str(payload.get("selected", "") or "").strip()
    question = str(payload.get("question", "") or "").strip()
    messages = payload.get("messages") or []
    if not selected:
        raise HTTPException(400, "请先在页面上选中一段文字")
    if not isinstance(messages, list) or not messages:
        messages = [{"role": "user", "content": question or action or "请解释这段"}]
    if action in {"rewrite", "revise"}:
        model = _llm("prompt_draft_llm")
    else:
        model = _llm("chapter_outline_llm")
    args = {
        "world_text": book_store.render_world(book.get("world", {})),
        "characters_text": unit_generation.render_all_characters(book.get("characters", [])),
        "unit_context": _ask_context(unit, scope_text),
        "selected_text": selected,
        "action": action,
        "messages": messages,
        "mainline_text": _book_mainline(book),
    }
    if payload.get("preview"):
        tool_available = unit_generation.is_tool_capable(model)
        msg_list = unit_generation.build_conversation_messages(**args, tool_available=tool_available)
        return {"ok": True, "preview": {
            "kind": "ask", "label": "选中文字：多轮问 AI",
            "prompt": json.dumps(msg_list, ensure_ascii=False),
            "request": build_raw_request_preview(
                model,
                messages=msg_list,
                tools=[unit_generation.rewrite_text] if tool_available else None,
            ),
        }}
    try:
        reply = unit_generation.conversation_reply(
            **args,
            model=model,
            prompt_override=payload.get("prompt_override"),
        )
        if isinstance(reply, dict):
            result = str(reply.get("result", "") or "")
            tool_calls = reply.get("tool_calls") or []
        else:
            result = str(reply or "")
            tool_calls = []
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "result": result, "tool_calls": tool_calls}

# ---------------------------------------------------------------- LLM 日志

_active_requests: dict[str, dict] = {}
_active_lock = threading.RLock()


def _track_llm_event(event: dict) -> None:
    request_id = event.get("request_id")
    if not request_id:
        return
    event_type = event.get("type")
    with _active_lock:
        if event_type == "request":
            _active_requests[request_id] = {
                "label": event.get("label", "LLM 请求"),
                "model": event.get("model", ""),
                "started_at": event.get("timestamp", ""),
            }
        elif event_type in {"response", "failed"}:
            _active_requests.pop(request_id, None)


add_llm_trace_listener(_track_llm_event)


@app.get("/api/logs/llm")
def download_llm_log():
    path = Path(LLM_INTERACTION_LOG)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return FileResponse(path, filename="llm_interactions.log", media_type="text/plain")


@app.get("/api/logs/llm/active")
def active_llm_requests():
    with _active_lock:
        return {"active": [{"request_id": key, **value} for key, value in _active_requests.items()]}


@app.get("/api/logs/llm/tail")
def tail_llm_log(offset: int = 0):
    path = Path(LLM_INTERACTION_LOG)
    if not path.exists():
        return {"offset": 0, "events": []}

    size = path.stat().st_size
    start = offset
    if start < 0 or start > size:
        start = 0

    truncated = False
    if start == 0 and size > _LLM_TAIL_BYTES:
        start = size - _LLM_TAIL_BYTES
        truncated = True

    with open(path, "rb") as file:
        file.seek(start)
        data = file.read()

    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]

    text = data.decode("utf-8", errors="replace")
    if truncated:
        break_at = text.find("\n")
        text = text[break_at + 1:] if break_at != -1 else ""

    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return {"offset": start + len(data), "events": events}


DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def frontend(full_path: str):
        candidate = (DIST / full_path).resolve()
        if full_path and candidate.is_file() and DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_api:app", host=AINOVEL_HOST, port=AINOVEL_PORT, log_level="warning")







