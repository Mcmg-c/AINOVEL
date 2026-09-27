# -*- coding: utf-8 -*-
"""LangGraph 全流程流水线：一个按钮从参考文档一路跑到合并路线。

节点顺序：
  1. generate   —— 并行调用 3 次模型，各自生成一套全书方案（详细主线/世界观/人物/大事件目录）
  2. advocate   —— 每套方案一个「辩护 Agent」立论（并行）
  3. rebuttal   —— 每套方案一个「攻辩 Agent」攻击另外两套并回应质疑（并行）
  4. moderator  —— 「主持人 Agent」听完全场辩论，评分排名并可能融合出新方案
  5. apply      —— 自动采用主持人裁决出的获胜方案，写入书名/主线/世界观/人物/大事件目录
  6. outline    —— 逐事件并行生成 A/B/C 章节拆解方案（带评分），自动选最高分组成整条路线
  7. merge      —— AI 通读整条路线：查重复、补缺口，合并为一条完整路线并为每章命名

前四个节点复用 debate_flow 的多 Agent 辩论节点；每个节点都通过 config 里的
progress(stage, label, detail, done, total) 回调上报进度，供前端做可视化。
"""

import concurrent.futures
import threading

from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from novel_generator import book_store
from novel_generator.debate_flow import (
    DebateState,
    _progress_hook,
    advocate_node,
    generate_node,
    moderator_node,
    rebuttal_node,
)
from novel_generator.unit_generation import (
    generate_event_chapter_options,
    generate_merged_route,
    generate_route_chapter_draft,
    render_cast,
    render_events,
    render_graph_working_set,
    render_merged_route_text,
)

_SAVE_LOCK = threading.RLock()


class FullFlowState(DebateState, total=False):
    project: str
    workers: int
    target_chapters: int
    words_per_chapter: int
    chosen_proposal: dict
    unit_id: str
    segments: list
    picked_count: int
    merged_route: dict
    draft_model: dict
    written_count: int
    error: str


def _book_mainline_text(book: dict) -> str:
    """把书名 + 单主线渲染成提示词片段。"""
    title = str(book.get("title", "") or "").strip()
    mainline = str(book.get("mainline", "") or "").strip()
    parts = []
    if title:
        parts.append(f"书名：{title}")
    if mainline:
        parts.append(f"单主线：{mainline}")
    return "\n".join(parts)


def _cast_names(book: dict, unit: dict) -> list:
    by_id = {item.get("id"): str(item.get("name", "") or "").strip() for item in book.get("characters", [])}
    return [by_id[cast_id] for cast_id in unit.get("cast", []) if by_id.get(cast_id)]


def _graph_text(book: dict, unit: dict) -> str:
    return render_graph_working_set(book.get("graph", {}), _cast_names(book, unit), [], unit.get("carry_facts") or [])


def _pick_winner(state: dict) -> dict | None:
    """根据主持人裁决选出获胜方案：decision=M 用融合方案，否则按 key 找原方案。"""
    proposals = [item for item in state.get("proposals") or [] if isinstance(item, dict)]
    verdict = state.get("verdict") if isinstance(state.get("verdict"), dict) else {}
    merged = state.get("merged") if isinstance(state.get("merged"), dict) else {}
    decision = str(verdict.get("decision", "") or "").strip()
    if decision == "M" and (merged.get("mainline") or merged.get("directory")):
        return dict(merged)
    proposal = next((item for item in proposals if str(item.get("key", "")) == decision), None)
    if proposal:
        return dict(proposal)
    return dict(proposals[0]) if proposals else None


def apply_node(state: FullFlowState, config=None) -> dict:
    """自动采用主持人裁决出的获胜方案，一次写入项目。"""
    progress = _progress_hook(config)
    project = str(state.get("project") or "")
    if not project:
        raise RuntimeError("项目路径为空，无法写入")
    if progress:
        progress("apply", "采用获胜方案", "正在把主持人推荐的方案写入：书名、主线、世界观、人物、大事件目录…", 0, 1)
    proposals = [item for item in state.get("proposals") or [] if isinstance(item, dict)]
    verdict = state.get("verdict") if isinstance(state.get("verdict"), dict) else {}
    merged = state.get("merged") if isinstance(state.get("merged"), dict) else {}
    book_store.save_proposals(project, proposals, debate={
        "cases": state.get("cases") or {},
        "rebuttals": state.get("rebuttals") or {},
        "verdict": verdict,
        "merged": merged,
    })
    proposal = _pick_winner(state)
    if not proposal:
        raise RuntimeError("主持人没有给出可采用的方案，请再跑一次")
    mainline = str(proposal.get("mainline", "") or "").strip()
    if not mainline:
        raise RuntimeError("获胜方案缺少单主线，无法采用")
    steps = []
    for index, step in enumerate(proposal.get("directory") or [], start=1):
        if not isinstance(step, dict):
            continue
        steps.append({
            "seq": int(step.get("seq", index) or index),
            "who": str(step.get("who", "") or "").strip(),
            "what": str(step.get("what", "") or "").strip(),
        })
    if not steps:
        raise RuntimeError("获胜方案缺少大事件目录，无法继续拆分")
    target_words = int(state.get("target_words") or 0)
    book_store.save_book_meta(project, {
        "title": str(proposal.get("title", "") or "").strip(),
        "mainline": mainline,
        "target_words": target_words,
    })
    book_store.save_world(project, proposal.get("world") if isinstance(proposal.get("world"), dict) else {})
    characters = book_store.save_characters(project, proposal.get("characters") or [])
    book = book_store.load_book(project)
    units = book.get("units") or []
    unit_id = book.get("current") or ""
    if not unit_id and units:
        unit_id = str(units[0].get("id", "") or "")
    if not unit_id:
        created = book_store.create_unit(project, {})
        unit_id = created.get("current") or ((created.get("units") or [{}])[0].get("id") or "")
    if not unit_id:
        raise RuntimeError("无法创建全书工作区")
    title = str(proposal.get("title", "") or "").strip() or "方案 A"
    book_store.update_unit(project, unit_id, {
        "title": title,
        "goal": mainline,
        "cast": [character.get("id") for character in characters if character.get("id")],
    })
    with _SAVE_LOCK:
        working = book_store.load_unit_working(project, unit_id)
        working["directory_options"] = [{
            "key": "A",
            "title": title,
            "thesis": str(proposal.get("thesis", "") or "").strip(),
            "angle": "",
            "steps": steps,
        }]
        working["chosen_plan"] = "A"
        working["segments"] = steps
        book_store.save_unit_working(project, unit_id, working)
    if progress:
        progress("apply", "采用获胜方案", f"已采用方案 {verdict.get('decision', '?')}：书名、主线、世界、人物与 {len(steps)} 个事件全部写入", 1, 1)
    return {"chosen_proposal": proposal, "unit_id": unit_id, "segments": steps}


def outline_node(state: FullFlowState, config=None) -> dict:
    """逐事件并行生成 A/B/C 章节拆解方案并打分，自动选最高分组成路线。"""
    progress = _progress_hook(config)
    project = str(state.get("project") or "")
    unit_id = str(state.get("unit_id") or "")
    model = state.get("model") or {}
    workers = max(1, min(8, int(state.get("workers") or 4)))
    book = book_store.load_book(project)
    unit = book_store.get_unit(book, unit_id)
    if not unit:
        raise RuntimeError("全书工作区不存在，无法拆分章节")
    with _SAVE_LOCK:
        working = book_store.load_unit_working(project, unit_id)
        working["segments"] = [dict(item) for item in (working.get("segments") or [])]
        segments = working["segments"]
    if not segments:
        raise RuntimeError("大事件目录为空，无法拆分章节")
    total = len(segments)
    if progress:
        progress("outline", "逐事件拆解评分", f"正在并行拆分 {total} 个事件：每个事件独立调用生成 A/B/C 三套章节方案并打分…", 0, total)
    base = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "mainline_text": _book_mainline_text(book),
        "graph_text": _graph_text(book, unit),
        "reference_text": str(state.get("reference_text", "") or ""),
        "history": [],
        "target_words": int(state.get("target_words") or 0) or int(book.get("target_words") or 0),
        "target_chapters": int(state.get("target_chapters") or 0) or int(book.get("target_chapters") or 0),
        "words_per_chapter": int(state.get("words_per_chapter") or 0) or int(book.get("words_per_chapter") or 0),
    }
    if progress and base["target_chapters"]:
        progress("outline", "逐事件拆解评分", f"按全书约 {base['target_chapters']} 章、每章约 {base['words_per_chapter'] or '?'} 字的规模控制拆分粒度…", 0, total)
    directory_text = render_events(segments)
    errors = []

    def run_one(segment):
        seq = segment.get("seq")
        try:
            result = generate_event_chapter_options(
                world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
                mainline_text=base["mainline_text"], directory_text=directory_text,
                event={"seq": seq, "who": segment.get("who", ""), "what": segment.get("what", "")},
                history=base["history"], graph_text=base["graph_text"],
                reference_text=base["reference_text"], target_words=base["target_words"],
                target_chapters=base["target_chapters"], words_per_chapter=base["words_per_chapter"],
                events_total=total,
                model=model,
            )
            options = result.get("chapter_options", [])
            with _SAVE_LOCK:
                segment["chapter_options"] = options
                segment["chosen_chapter_plan"] = ""
                book_store.save_unit_working(project, unit_id, working)
            return seq, options, None
        except Exception as exc:  # noqa: BLE001 - 单事件失败不阻断其它任务
            return seq, [], exc

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, segment) for segment in segments]
        for future in concurrent.futures.as_completed(futures):
            seq, options, exc = future.result()
            done += 1
            if exc:
                errors.append((seq, str(exc)))
                detail = f"事件 {seq} 生成失败（{done}/{total}），其余继续…"
            else:
                detail = f"事件 {seq} 已生成 {len(options)} 套方案并完成评分（{done}/{total}）"
            if progress:
                progress("outline", "逐事件拆解评分", detail, done, total)

    if errors and len(errors) == total:
        raise RuntimeError(f"章节拆分全部失败：{errors[0][1]}")
    picked = 0
    with _SAVE_LOCK:
        for segment in segments:
            options = [plan for plan in (segment.get("chapter_options") or []) if isinstance(plan, dict) and plan.get("key")]
            if not options:
                continue
            best = max(options, key=lambda plan: (float(plan.get("score") or 0), str(plan.get("key", ""))))
            segment["chosen_chapter_plan"] = str(best.get("key", ""))
            picked += 1
        book_store.save_unit_working(project, unit_id, working)
    if picked < total:
        raise RuntimeError(f"只有 {picked}/{total} 个事件成功生成章节方案，请再跑一次流水线补齐")
    if progress:
        progress("outline", "逐事件拆解评分", f"{total} 个事件全部拆分完毕，最高分路线已自动选定（{picked} 个事件）", total, total)
    return {"picked_count": picked}


def merge_node(state: FullFlowState, config=None) -> dict:
    """通读自动选定的整条路线：查重复、补缺口，合并并给每章命名。"""
    progress = _progress_hook(config)
    project = str(state.get("project") or "")
    unit_id = str(state.get("unit_id") or "")
    model = state.get("model") or {}
    book = book_store.load_book(project)
    unit = book_store.get_unit(book, unit_id)
    if not unit:
        raise RuntimeError("全书工作区不存在，无法合并路线")
    if progress:
        scale_hint = f"，按全书约 {state.get('target_chapters') or 0} 章 / 每章约 {state.get('words_per_chapter') or 0} 字重组" if (state.get("target_chapters") or state.get("words_per_chapter")) else ""
        progress("merge", "校验合并路线", f"正在通读整条路线：查重复、补缺口，逐章收拢子事件{scale_hint}…", 0, 1)
    working = book_store.load_unit_working(project, unit_id)
    segments = working.get("segments") or []
    missing = [str(segment.get("seq")) for segment in segments if not str(segment.get("chosen_chapter_plan", "") or "")]
    if missing:
        raise RuntimeError("还有事件未选定章节方案：事件 " + "、".join(missing))
    result = generate_merged_route(
        world_text=book_store.render_world(book.get("world", {})),
        cast_text=render_cast(book.get("characters", []), unit.get("cast", [])),
        unit_goal=str(unit.get("goal", "") or ""),
        mainline_text=_book_mainline_text(book),
        route_text=render_merged_route_text(segments),
        guidance=str(state.get("guidance", "") or ""),
        history=[],
        reference_text=str(state.get("reference_text", "") or ""),
        target_words=int(state.get("target_words") or 0) or int(book.get("target_words") or 0),
        target_chapters=int(state.get("target_chapters") or 0) or int(book.get("target_chapters") or 0),
        words_per_chapter=int(state.get("words_per_chapter") or 0) or int(book.get("words_per_chapter") or 0),
        model=model,
    )
    merged_route = result.get("merged_route") or {}
    with _SAVE_LOCK:
        working["merged_route"] = merged_route
        book_store.save_unit_working(project, unit_id, working)
    chapters = merged_route.get("chapters") or []
    if progress:
        progress("merge", "校验合并路线", f"合并完成：一条 {len(chapters)} 章的完整路线，每章已命名", 1, 1)
    return {"merged_route": merged_route}


def _route_directory_text(chapters: list) -> str:
    """把合并路线的章节渲染成「第 N 章《标题》：谁 → 做什么」的目录文本。"""
    lines = []
    for chapter in chapters or []:
        if not isinstance(chapter, dict):
            continue
        title = str(chapter.get("title", "") or "").strip()
        who = str(chapter.get("who", "") or "").strip()
        what = str(chapter.get("what", "") or "").strip()
        head = f"第{chapter.get('seq', len(lines) + 1)}章" + (f"《{title}》" if title else "")
        lines.append(f"{head}：{who or '（未写人物）'} → {what or '（未写动作）'}")
    return "\n".join(lines) if lines else "（目录为空）"


def write_node(state: FullFlowState, config=None) -> dict:
    """串行撰写各章正文：下一章务必带上前面所有章节的完整正文，因此不能并行。"""
    progress = _progress_hook(config)
    project = str(state.get("project") or "")
    unit_id = str(state.get("unit_id") or "")
    model = state.get("draft_model") or state.get("model") or {}
    book = book_store.load_book(project)
    unit = book_store.get_unit(book, unit_id)
    if not unit:
        raise RuntimeError("全书工作区不存在，无法写正文")
    working = book_store.load_unit_working(project, unit_id)
    merged_route = working.get("merged_route") or {}
    chapters = [item for item in (merged_route.get("chapters") or []) if isinstance(item, dict)]
    if not chapters:
        raise RuntimeError("合并路线为空，无法写正文；请先跑完拆解与合并")
    total = len(chapters)
    if progress:
        progress("write", "串行撰写各章正文", f"开始逐章写正文（{total} 章，串行执行：每一章都会带上前面所有章节的完整正文）…", 0, total)
    target_words = int(state.get("target_words") or 0) or int(book.get("target_words") or 0)
    base = {
        "world_text": book_store.render_world(book.get("world", {})),
        "cast_text": render_cast(book.get("characters", []), unit.get("cast", [])),
        "unit_goal": str(unit.get("goal", "") or ""),
        "mainline_text": _book_mainline_text(book),
        "graph_text": _graph_text(book, unit),
        "reference_text": str(state.get("reference_text", "") or ""),
        "directory_text": _route_directory_text(chapters),
    }
    previous_parts = []
    done = 0
    for chapter in chapters:
        seq = int(chapter.get("seq") or done + 1)
        title = str(chapter.get("title", "") or "").strip()
        ratio = int(chapter.get("word_ratio") or 0)
        word_count = int(chapter.get("word_estimate") or 0)
        if not word_count and state.get("words_per_chapter"):
            word_count = int(state.get("words_per_chapter") or 0)
        if not word_count and target_words and ratio:
            word_count = round(target_words * ratio / 100)
        if not word_count:
            word_count = 1500
        if progress:
            progress("write", "串行撰写各章正文", f"正在写第 {seq} 章《{title or '未命名'}》（带上前 {done} 章完整正文，目标 {word_count} 字）…", done, total)
        sub_text = ""
        sub_events = chapter.get("sub_events") if isinstance(chapter.get("sub_events"), list) else []
        if sub_events:
            sub_lines = []
            for index, sub in enumerate(sub_events, start=1):
                if not isinstance(sub, dict):
                    continue
                sub_lines.append(f"{index}. {sub.get('who') or '（未写人物）'} → {sub.get('what') or '（未写内容）'}（约 {sub.get('word_estimate', 0)} 字）")
            if sub_lines:
                sub_text = "【本章必须依次覆盖的子事件（逐条写完，不要漏掉任何一条）】\n" + "\n".join(sub_lines)
        text = generate_route_chapter_draft(
            world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
            mainline_text=base["mainline_text"], directory_text=base["directory_text"],
            chapter=chapter, previous_text="\n\n".join(previous_parts), word_count=word_count,
            history=[], graph_text=base["graph_text"], reference_text=base["reference_text"],
            target_words=target_words, model=model, sub_events_text=sub_text,
        )
        text = str(text or "").strip()
        if not text:
            raise RuntimeError(f"第 {seq} 章正文生成失败，请重试")
        chapter["draft"] = text
        done += 1
        previous_parts.append(f"第{seq}章《{title or ''}》\n{text}")
        with _SAVE_LOCK:
            book_store.save_unit_working(project, unit_id, working)
        if progress:
            progress("write", "串行撰写各章正文", f"第 {seq} 章《{title or '未命名'}》已完成（{done}/{total}），下一章将带上前 {done} 章完整正文", done, total)
    return {"written_count": done}


def build_full_graph():
    builder = StateGraph(FullFlowState)
    builder.add_node("generate", generate_node)
    builder.add_node("advocate", advocate_node)
    builder.add_node("rebuttal", rebuttal_node)
    builder.add_node("moderator", moderator_node)
    builder.add_node("apply", apply_node)
    builder.add_node("outline", outline_node)
    builder.add_node("merge", merge_node)
    builder.add_node("write", write_node)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "advocate")
    builder.add_edge("advocate", "rebuttal")
    builder.add_edge("rebuttal", "moderator")
    builder.add_edge("moderator", "apply")
    builder.add_edge("apply", "outline")
    builder.add_edge("outline", "merge")
    builder.add_edge("merge", "write")
    builder.add_edge("write", END)
    return builder.compile()


def run_full_pipeline(
    *,
    project: str,
    model: dict,
    guidance: str = "",
    target_words: int = 0,
    reference_text: str = "",
    rebuttal_rounds: int = 1,
    workers: int = 4,
    target_chapters: int = 0,
    words_per_chapter: int = 0,
    draft_model: dict | None = None,
    progress=None,
) -> dict:
    """执行整条 LangGraph 全流程，返回各阶段结果。"""
    graph = build_full_graph()
    initial = {
        "project": str(project),
        "reference_text": reference_text,
        "guidance": guidance,
        "target_words": int(target_words or 0),
        "proposals": [],
        "model": model,
        "rebuttal_rounds": max(1, min(3, int(rebuttal_rounds or 1))),
        "workers": max(1, min(8, int(workers or 4))),
        "target_chapters": int(target_chapters or 0),
        "words_per_chapter": int(words_per_chapter or 0),
        "draft_model": draft_model or {},
        "cases": {},
        "rebuttals": {},
        "verdict": {},
        "merged": {},
    }
    invoke_config = {"configurable": {"progress": progress}} if callable(progress) else None
    final = graph.invoke(initial, config=invoke_config) if invoke_config else graph.invoke(initial)
    return {
        "proposals": final.get("proposals") or [],
        "cases": final.get("cases") or {},
        "rebuttals": final.get("rebuttals") or {},
        "verdict": final.get("verdict") or {},
        "merged": final.get("merged"),
        "chosen_proposal": final.get("chosen_proposal"),
        "unit_id": final.get("unit_id") or "",
        "segments": final.get("segments") or [],
        "picked_count": final.get("picked_count") or 0,
        "merged_route": final.get("merged_route") or {},
        "written_count": final.get("written_count") or 0,
    }
