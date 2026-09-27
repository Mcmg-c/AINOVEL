# -*- coding: utf-8 -*-
"""把「大事件目录 → 拆章节 → 内容设计」三级展开抽成独立模块，供 CLI 与 Web 共用。

与 Web 端一键编排（web_api._run_auto_outline）保持同一套生成逻辑：
每套目录方案都往下拆、每套章节方案都做内容设计，所有分支完整保留在
working["outline_tree"]，方便后续挑选任意一条路线继续写作。

章节拆分与内容设计支持并行调用（每步一次 LLM 请求），结果统一由主线程
写入与落盘，避免多线程同时改动共享状态。
"""

import concurrent.futures
import threading

from . import book_store
from . import unit_generation


_tree_lock = threading.Lock()


def _book_mainline(book):
    parts = []
    if str(book.get("title") or "").strip():
        parts.append(f"书名：{book['title']}")
    if str(book.get("mainline") or "").strip():
        parts.append(f"单主线：{book['mainline']}")
    return "\n".join(parts)


def _previous_design_summary(chapters, index):
    if index <= 0:
        return ""
    previous = chapters[index - 1]
    options = previous.get("design_options") or []
    chosen = str(previous.get("chosen_design") or (options[0].get("key") if options else "1"))
    option = next((item for item in options if str(item.get("key")) == chosen), options[0] if options else None)
    if not option:
        return ""
    return f"{option.get('title', '')}\n{option.get('summary', '')}\n{option.get('content', '')}".strip()[-600:]


def _chapter_branches_from_plans(options):
    chapter_branches = []
    for chapter_plan in options or []:
        chapter_branches.append({
            "plan_key": str(chapter_plan.get("key")),
            "plan": {
                "key": str(chapter_plan.get("key")),
                "title": str(chapter_plan.get("title") or "").strip(),
                "thesis": str(chapter_plan.get("thesis") or "").strip(),
            },
            "chapters": [
                {
                    "seq": step.get("seq"),
                    "who": str(step.get("who") or "").strip(),
                    "what": str(step.get("what") or "").strip(),
                    "word_estimate": int(step.get("word_estimate") or 0),
                    "word_ratio": int(step.get("word_ratio") or 0),
                    "design_options": [],
                    "chosen_design": "",
                }
                for step in chapter_plan.get("steps", [])
            ],
        })
    return chapter_branches


def _generate_event_chapters(base, directory_text, event, model):
    options = []
    for attempt in range(2):
        result = unit_generation.generate_event_chapter_options(
            world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
            mainline_text=base["mainline_text"], directory_text=directory_text,
            event={"seq": event["seq"], "who": event["who"], "what": event["what"]},
            history=base["history"], graph_text=base["graph_text"], reference_text=base["reference_text"],
            model=model,
        )
        options = [
            plan for plan in result.get("chapter_options", [])
            if (plan or {}).get("steps")
        ]
        if options:
            break
    return _chapter_branches_from_plans(options)


def _event_complete(event):
    chapter_branches = event.get("chapter_branches") or []
    return bool(chapter_branches) and all(item.get("chapters") for item in chapter_branches)


def _salvage_chapter_branches(event):
    """旧数据里「未结构化兜底」的空章节分支，尝试从其 thesis 中救回模型曾返回的 JSON。"""
    repaired = []
    for chapter_branch in event.get("chapter_branches") or []:
        if chapter_branch.get("chapters"):
            repaired.append(chapter_branch)
            continue
        thesis = (chapter_branch.get("plan") or {}).get("thesis") or ""
        parsed = unit_generation._extract_json(thesis)
        plans = parsed.get("plans") if isinstance(parsed, dict) else None
        repaired.extend(_chapter_branches_from_plans(unit_generation._normalize_steps(plans)))
    return repaired


def _design_chapter_branch(base, directory_text, chapter_branch, model, report_progress):
    """给同一套章节方案的每一章做内容设计；上一章的结果作为下一章的前情。"""
    previous_summary = ""
    for index, chapter in enumerate(chapter_branch["chapters"]):
        if not chapter.get("design_options"):
            result = unit_generation.generate_segment_design_options(
                world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
                directory_text=directory_text,
                segment={"seq": chapter["seq"], "who": chapter["who"], "what": chapter["what"]},
                previous_summary=previous_summary,
                history=base["history"], graph_text=base["graph_text"],
                mainline_text=base["mainline_text"], reference_text=base["reference_text"],
                model=model,
            )
            options = result.get("options", [])
            with _tree_lock:
                chapter["design_options"] = options
                if options:
                    chapter["chosen_design"] = str(options[0].get("key") or "1")
            if report_progress:
                report_progress()
        previous_summary = _previous_design_summary(chapter_branch["chapters"], index + 1)


def expand_outline_tree(
    *,
    project, unit_id, model, guidance="", progress=None, plan_keys=None,
    include_design=False, workers=4,
):
    """三级全展开并落盘，返回最新的 book 快照。

    progress(stage, detail, done, total) 供调用方汇报进度；include_design 控制是否继续
    给每章做内容设计（默认只展开到章节方案）；workers 控制并行请求数量。
    每完成一次落盘，中途中断后重跑会自动跳过已生成的部分。
    """
    book = book_store.load_book(project)
    unit = book_store.get_unit(book, unit_id)
    if not unit:
        raise RuntimeError(f"单元不存在：{unit_id}")
    working = book_store.load_unit_working(project, unit_id)

    def report(stage, detail, done=0, total=0):
        if progress:
            progress(stage, detail, done, total)

    def save():
        book_store.save_unit_working(project, unit_id, working)

    base = {
        "world_text": book_store.render_world(book.get("world") or {}),
        "cast_text": unit_generation.render_cast(book.get("characters") or [], unit.get("cast") or []),
        "unit_goal": str(unit.get("goal") or ""),
        "mainline_text": _book_mainline(book),
        "graph_text": "",
        "reference_text": book_store.render_reference_files(project),
        "history": [],
    }

    # 1) 多套大事件目录
    report("directory", "正在生成多套大事件目录方案（单次模型请求，请稍候）")
    if not working.get("directory_options"):
        result = unit_generation.generate_directory_options(
            world_text=base["world_text"], cast_text=base["cast_text"], unit_goal=base["unit_goal"],
            lead_in=str(unit.get("lead_in") or ""), guidance=guidance or "",
            history=base["history"], carry_over="", graph_text=base["graph_text"],
            mainline_text=base["mainline_text"], reference_text=base["reference_text"],
            model=model,
        )
        working["directory_options"] = result.get("directory_options", [])
        working["chosen_plan"] = ""
        working["segments"] = []
        save()
    plans = working["directory_options"] or []
    if not plans:
        raise RuntimeError("目录方案为空")
    if plan_keys:
        plans = [plan for plan in plans if str(plan.get("key")) in set(plan_keys)]
        if not plans:
            raise RuntimeError(f"目录方案为空（plan_keys={','.join(plan_keys)} 未匹配到任何方案）")

    previous_branches = {
        str(item.get("plan_key")): item
        for item in (working.get("outline_tree") or {}).get("branches", [])
    }
    branches = []
    for plan in plans:
        previous = previous_branches.get(str(plan.get("key")))
        if previous and len(previous.get("events", [])) == len(plan.get("steps", [])):
            events = previous["events"]
        else:
            events = [
                {
                    "seq": index,
                    "who": str(step.get("who") or "").strip(),
                    "what": str(step.get("what") or "").strip(),
                    "chapter_branches": [],
                }
                for index, step in enumerate(plan.get("steps", []), start=1)
            ]
        branches.append({"plan_key": str(plan.get("key")), "plan": plan, "events": events})
    all_plans = working["directory_options"] or []
    working["outline_tree"] = {"directory_options": all_plans, "branches": branches}
    save()

    # 已有数据先做离线修复：把「未结构化兜底」的空章节分支从 thesis 中救回
    salvaged = 0
    for branch in branches:
        for event in branch["events"]:
            if _event_complete(event):
                continue
            repaired = _salvage_chapter_branches(event)
            if repaired != event.get("chapter_branches"):
                event["chapter_branches"] = repaired
            if _event_complete(event):
                salvaged += 1
    if salvaged:
        report("chapters", f"已从旧数据恢复 {salvaged} 个事件的章节拆分（无需重新请求模型）")
        save()

    # 2) 每一套目录的每一个事件都拆出多套章节方案（并行请求，主线程落盘）
    tasks = []
    for branch in branches:
        directory_text = unit_generation.render_events(branch["events"])
        for event in branch["events"]:
            if not _event_complete(event):
                tasks.append((branch, event, directory_text))
    chapter_nodes = len(tasks)
    errors = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
        future_map = {
            pool.submit(_generate_event_chapters, base, directory_text, event, model): (branch, event)
            for branch, event, directory_text in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            done += 1
            branch, event = future_map[future]
            try:
                chapter_branches = future.result()
                if not chapter_branches:
                    raise RuntimeError("模型未返回有效的章节方案（已重试）")
                event["chapter_branches"] = chapter_branches
            except Exception as exc:  # noqa: BLE001 - 保留已成功的部分，其余可续跑
                errors.append(exc)
                report("chapters", f"目录 {branch['plan_key']} · 事件 {event['seq']} 失败：{exc}", done, chapter_nodes)
                continue
            report("chapters", f"目录 {branch['plan_key']} · 事件 {event['seq']}/{len(branch['events'])}", done, chapter_nodes)
            save()
    if errors:
        working["outline_tree"] = {"directory_options": working["directory_options"] or [], "branches": branches}
        save()
        raise RuntimeError(f"章节拆分有 {len(errors)} 个事件失败，已完成部分已保存，可加 --resume 续跑：{errors[0]}")

    # 3) 每一套章节方案的每一章都做内容设计（可选）
    if not include_design:
        report("design", "已跳过内容设计（include_design=False）")
        working["outline_tree"] = {"directory_options": working["directory_options"] or [], "branches": branches}
        save()
        return book_store.load_book(project)

    design_tasks = []
    for branch in branches:
        for event in branch["events"]:
            for chapter_branch in event["chapter_branches"]:
                pending = [chapter for chapter in chapter_branch["chapters"] if not chapter.get("design_options")]
                if not pending:
                    continue
                view = {"segments": [
                    {
                        "seq": item["seq"], "who": item["who"], "what": item["what"],
                        "chapters": chapter_branch["chapters"] if item is event else [],
                    }
                    for item in branch["events"]
                ]}
                directory_text = unit_generation.render_book_outline(view)
                design_tasks.append((branch, event, chapter_branch, directory_text, len(pending)))

    design_nodes = sum(item[4] for item in design_tasks)
    done_counter = {"value": 0}
    design_errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
        def tick():
            with _tree_lock:
                done_counter["value"] += 1
                current = done_counter["value"]
            report(
                "design", "并行内容设计中",
                current, design_nodes,
            )

        future_map = {
            pool.submit(_design_chapter_branch, base, directory_text, chapter_branch, model, tick): (branch, event, chapter_branch)
            for branch, event, chapter_branch, directory_text, _ in design_tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            branch, event, chapter_branch = future_map[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                design_errors.append(exc)
                report("design", f"目录 {branch['plan_key']} · 事件 {event['seq']} · 章节方案 {chapter_branch['plan_key']} 失败：{exc}", done_counter["value"], design_nodes)
            with _tree_lock:
                save()
    if design_errors:
        with _tree_lock:
            save()
        raise RuntimeError(f"内容设计有 {len(design_errors)} 组章节失败，已完成部分已保存，可加 --resume 续跑：{design_errors[0]}")

    working["outline_tree"] = {"directory_options": working["directory_options"] or [], "branches": branches}
    save()
    return book_store.load_book(project)
