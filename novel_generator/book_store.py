# -*- coding: utf-8 -*-
"""Web 版单元剧创作台的数据层。

与旧版桌面端（Novel_architecture.txt / unit_plot.txt 等）解耦，
采用更贴近单元剧创作心理模型的目录结构：

    world.json          世界观（类型 + 结构化字段）
    characters.json     人物库（姓名 / 性别 / 年龄 / 一句话）
    units.json          单元索引与元信息
    units/{id}/unit.json 单元内的目录方案、小目录、内容方案与正文
"""

import json
import os
import re
import tempfile
import uuid

WORLD_FILENAME = "world.json"
CHARACTERS_FILENAME = "characters.json"
UNITS_FILENAME = "units.json"
KNOWLEDGE_FILENAME = "knowledge.json"
BOOK_FILENAME = "book.json"
PROPOSALS_FILENAME = "proposals.json"
REFERENCE_DIRNAME = "references"
REFERENCE_INDEX_FILENAME = "references.json"

WORLD_FIELDS = (
    ("genre", "类型"),
    ("premise", "一句话世界前提"),
    ("era_place", "时代与地点"),
    ("rules", "规则与限制"),
    ("tone", "基调"),
)


def _atomic_write_json(path: str, data) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _load_json(path: str, default) -> object:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, type(default)) else default
    except Exception:
        return default


def _project_dir(filepath: str) -> str:
    return os.path.abspath(filepath)


def load_world(filepath: str) -> dict:
    world = _load_json(os.path.join(_project_dir(filepath), WORLD_FILENAME), {})
    return {key: str((world or {}).get(key, "")).strip() for key, _label in WORLD_FIELDS}


def save_world(filepath: str, world: dict) -> dict:
    payload = {key: str((world or {}).get(key, "")).strip() for key, _label in WORLD_FIELDS}
    _atomic_write_json(os.path.join(_project_dir(filepath), WORLD_FILENAME), payload)
    return payload


def is_world_filled(world: dict) -> bool:
    return any(str((world or {}).get(key, "")).strip() for key, _label in WORLD_FIELDS)


def render_world(world: dict) -> str:
    lines = []
    for key, label in WORLD_FIELDS:
        value = str((world or {}).get(key, "")).strip()
        lines.append(f"{label}：{value or '（未填写）'}")
    return "\n".join(lines)


BOOK_FIELDS = (
    ("title", "书名"),
    ("mainline", "单主线"),
)
BOOK_NUMBER_FIELDS = (
    ("target_words", "大概总字数"),
    ("target_chapters", "大概章节数"),
    ("words_per_chapter", "每章大概字数"),
)


def load_book_meta(filepath: str) -> dict:
    meta = _load_json(os.path.join(_project_dir(filepath), BOOK_FILENAME), {})
    result = {key: str((meta or {}).get(key, "")).strip() for key, _label in BOOK_FIELDS}
    for key, _label in BOOK_NUMBER_FIELDS:
        try:
            result[key] = int(float(str((meta or {}).get(key, "") or "").strip() or 0))
        except (TypeError, ValueError):
            result[key] = 0
    return result


def save_book_meta(filepath: str, meta: dict) -> dict:
    payload = {key: str((meta or {}).get(key, "")).strip() for key, _label in BOOK_FIELDS}
    for key, _label in BOOK_NUMBER_FIELDS:
        try:
            payload[key] = int(float(str((meta or {}).get(key, "") or "").strip() or 0))
        except (TypeError, ValueError):
            payload[key] = 0
    _atomic_write_json(os.path.join(_project_dir(filepath), BOOK_FILENAME), payload)
    return payload


def render_book_meta(meta: dict) -> str:
    title = str((meta or {}).get("title", "")).strip()
    mainline = str((meta or {}).get("mainline", "")).strip()
    target_words = int((meta or {}).get("target_words") or 0)
    lines = []
    if title:
        lines.append(f"书名：{title}")
    if mainline:
        lines.append(f"单主线：{mainline}")
    if target_words:
        lines.append(f"全书体量：约 {target_words} 字")
    return "\n".join(lines) if lines else "（未设置）"


def _reference_dir(filepath: str) -> str:
    return os.path.join(_project_dir(filepath), REFERENCE_DIRNAME)


def _sanitize_reference_stored_name(name: str) -> str:
    name = os.path.basename(str(name or "").strip())
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]", "_", name) or "reference.txt"
    if not name.lower().endswith(".txt"):
        name += ".txt"
    return name


def _file_char_count(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            return len(file.read())
    except Exception:
        return 0


def load_reference_files(filepath: str) -> list[dict]:
    """返回参考文件元信息（不含正文），供前端列表与开关展示。"""
    data = _load_json(os.path.join(_project_dir(filepath), REFERENCE_INDEX_FILENAME), {"files": []})
    files = data.get("files", []) if isinstance(data, dict) else []
    directory = _reference_dir(filepath)
    result = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        stored_name = str(item.get("stored_name") or name).strip()
        path = os.path.join(directory, stored_name)
        if not name or not stored_name or not os.path.isfile(path):
            continue
        result.append({
            "name": name,
            "stored_name": stored_name,
            "label": str(item.get("label") or name),
            "enabled": bool(item.get("enabled", True)),
            "size": os.path.getsize(path),
            "chars": _file_char_count(path),
        })
    return result


def _coerce_reference_limit(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def load_reference_limit(filepath: str) -> int:
    data = _load_json(os.path.join(_project_dir(filepath), REFERENCE_INDEX_FILENAME), {"files": []})
    return _coerce_reference_limit((data or {}).get("limit", 0))


def save_reference_limit(filepath: str, limit) -> int:
    data = _load_json(os.path.join(_project_dir(filepath), REFERENCE_INDEX_FILENAME), {"files": []})
    payload = {
        "files": (data or {}).get("files", []) if isinstance(data, dict) else [],
        "limit": _coerce_reference_limit(limit),
    }
    _atomic_write_json(os.path.join(_project_dir(filepath), REFERENCE_INDEX_FILENAME), payload)
    return payload["limit"]


def _persist_reference_files(filepath: str, files: list[dict]) -> list[dict]:
    payload = {
        "files": [
            {
                "name": item["name"],
                "stored_name": str(item.get("stored_name") or item["name"]),
                "label": item.get("label", item["name"]),
                "enabled": bool(item.get("enabled", True)),
            }
            for item in files if isinstance(item, dict) and item.get("name")
        ],
        "limit": load_reference_limit(filepath),
    }
    _atomic_write_json(os.path.join(_project_dir(filepath), REFERENCE_INDEX_FILENAME), payload)
    return load_reference_files(filepath)


def add_reference_file(filepath: str, name: str, content: str) -> list[dict]:
    label = os.path.basename(str(name or "reference.txt").strip()) or "reference.txt"
    directory = _reference_dir(filepath)
    os.makedirs(directory, exist_ok=True)
    base = _sanitize_reference_stored_name(label)
    stored_name = base
    index = 1
    while os.path.exists(os.path.join(directory, stored_name)):
        stem, ext = os.path.splitext(base)
        stored_name = f"{stem}_{index}{ext}"
        index += 1
    with open(os.path.join(directory, stored_name), "w", encoding="utf-8") as file:
        file.write(str(content or ""))
    files = load_reference_files(filepath)
    files.append({"name": label, "stored_name": stored_name, "label": label, "enabled": True})
    return _persist_reference_files(filepath, files)


def delete_reference_file(filepath: str, name: str) -> list[dict]:
    files = load_reference_files(filepath)
    target = next((item for item in files if item.get("name") == name), None)
    if target:
        path = os.path.join(_reference_dir(filepath), target.get("stored_name", ""))
        if os.path.isfile(path):
            os.unlink(path)
    files = [item for item in files if item.get("name") != name]
    return _persist_reference_files(filepath, files)


def toggle_reference_file(filepath: str, name: str, enabled: bool) -> list[dict]:
    files = load_reference_files(filepath)
    for item in files:
        if item.get("name") == name:
            item["enabled"] = bool(enabled)
    return _persist_reference_files(filepath, files)


def _excerpt_text(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= max(1, limit):
        return text
    head = max(0, int(limit * 0.4))
    mid = max(0, int(limit * 0.2))
    tail = max(0, limit - head - mid)
    pieces = [text[:head]]
    if mid:
        start = (len(text) - mid) // 2
        pieces.append(text[start:start + mid])
    if tail:
        pieces.append(text[-tail:])
    return "\n\n……【中间省略】……\n\n".join(pieces)


def render_reference_files(filepath: str, max_chars: int | None = None) -> str:
    """把生效的参考文件拼成提示词上下文；0 或空表示全文，大于 0 表示节选总字符数。"""
    files = [item for item in load_reference_files(filepath) if item.get("enabled")]
    if not files:
        return ""
    limit = load_reference_limit(filepath) if max_chars is None else _coerce_reference_limit(max_chars)
    per_file = max(1, limit // max(1, len(files))) if limit > 0 else 0
    blocks = []
    for item in files:
        path = os.path.join(_reference_dir(filepath), item.get("stored_name", ""))
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                text = file.read()
        except Exception:
            text = ""
        if not text.strip():
            body = "（空文件）"
        else:
            body = text if per_file <= 0 else _excerpt_text(text, per_file)
        blocks.append(f"### {item.get('label', item['name'])}\n{body}")
    return "\n\n".join(blocks).strip()


def load_proposals(filepath: str) -> dict:
    """读取三套方案与最近一次辩论结果；没有则返回空。"""
    data = _load_json(os.path.join(_project_dir(filepath), PROPOSALS_FILENAME), {})
    if not isinstance(data, dict):
        data = {}
    proposals = data.get("proposals") if isinstance(data.get("proposals"), list) else []
    debate = data.get("debate") if isinstance(data.get("debate"), dict) else None
    return {"proposals": proposals, "debate": debate}


def save_proposals(filepath: str, proposals, debate=None) -> dict:
    payload = {
        "proposals": proposals if isinstance(proposals, list) else [],
        "debate": debate if isinstance(debate, dict) else None,
    }
    _atomic_write_json(os.path.join(_project_dir(filepath), PROPOSALS_FILENAME), payload)
    return payload


def _normalize_character(character, fallback_id: str = None) -> dict:
    if not isinstance(character, dict):
        character = {}
    name = str(character.get("name", "")).strip()
    return {
        "id": str(character.get("id") or fallback_id or uuid.uuid4().hex[:8]),
        "name": name,
        "gender": str(character.get("gender", "")).strip(),
        "age": str(character.get("age", "")).strip(),
        "notes": str(character.get("notes", "")).strip(),
    }


def load_characters(filepath: str) -> list[dict]:
    data = _load_json(os.path.join(_project_dir(filepath), CHARACTERS_FILENAME), {"characters": []})
    characters = data.get("characters", []) if isinstance(data, dict) else []
    if not isinstance(characters, list):
        characters = []
    return [_normalize_character(item) for item in characters if _normalize_character(item)["name"]]


def save_characters(filepath: str, characters) -> list[dict]:
    items = characters if isinstance(characters, list) else []
    normalized = [_normalize_character(item) for item in items]
    normalized = [item for item in normalized if item["name"]]
    _atomic_write_json(
        os.path.join(_project_dir(filepath), CHARACTERS_FILENAME),
        {"characters": normalized},
    )
    return normalized


def _normalize_graph(graph) -> dict:
    if not isinstance(graph, dict):
        graph = {}

    def _string_list(key):
        items = graph.get(key, []) if isinstance(graph.get(key, []), list) else []
        return items

    nodes = []
    for item in _string_list("nodes"):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        nodes.append({
            "id": str(item.get("id") or uuid.uuid4().hex[:8]).strip(),
            "kind": str(item.get("kind", "人物") or "").strip() or "人物",
            "name": name,
            "note": str(item.get("note", "") or "").strip(),
            "status": str(item.get("status", "活跃") or "").strip() or "活跃",
        })

    edges = []
    for item in _string_list("edges"):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "") or "").strip()
        target = str(item.get("target", "") or "").strip()
        if not source or not target:
            continue
        edges.append({
            "id": str(item.get("id") or uuid.uuid4().hex[:8]).strip(),
            "source": source,
            "target": target,
            "type": str(item.get("type", "") or "").strip(),
            "note": str(item.get("note", "") or "").strip(),
            "status": str(item.get("status", "") or "").strip(),
        })

    events = []
    for item in _string_list("events"):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "").strip()
        if not title:
            continue
        events.append({
            "id": str(item.get("id") or uuid.uuid4().hex[:8]).strip(),
            "unit_seq": item.get("unit_seq"),
            "title": title,
            "summary": str(item.get("summary", "") or "").strip(),
        })

    facts = []
    for item in _string_list("facts"):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        entities = item.get("entities", []) if isinstance(item.get("entities", []), list) else []
        facts.append({
            "id": str(item.get("id") or uuid.uuid4().hex[:8]).strip(),
            "unit_seq": item.get("unit_seq"),
            "text": text,
            "entities": [str(entity).strip() for entity in entities if str(entity).strip()],
        })

    return {"nodes": nodes, "edges": edges, "events": events, "facts": facts}


def load_graph(filepath: str) -> dict:
    return _normalize_graph(_load_json(os.path.join(_project_dir(filepath), KNOWLEDGE_FILENAME), {}))


def save_graph(filepath: str, graph) -> dict:
    payload = _normalize_graph(graph)
    _atomic_write_json(os.path.join(_project_dir(filepath), KNOWLEDGE_FILENAME), payload)
    return payload


def _unit_working_filename() -> str:
    return "unit.json"


def load_units_index(filepath: str) -> dict:
    data = _load_json(os.path.join(_project_dir(filepath), UNITS_FILENAME), {"units": [], "current": ""})
    units = data.get("units", []) if isinstance(data, dict) else []
    if not isinstance(units, list):
        units = []
    return {"units": [dict(item) for item in units], "current": str(data.get("current", "") or "")}


def save_units_index(filepath: str, units: list, current: str) -> None:
    _atomic_write_json(
        os.path.join(_project_dir(filepath), UNITS_FILENAME),
        {"units": units, "current": current},
    )


def _normalize_chapter(chapter, seq: int) -> dict:
    if not isinstance(chapter, dict):
        chapter = {}
    def _as_int(value):
        try:
            return int(float(str(value or "").replace("%", "").strip()))
        except (TypeError, ValueError):
            return 0
    return {
        "seq": int(chapter.get("seq", seq)),
        "who": str(chapter.get("who", "") or "").strip(),
        "what": str(chapter.get("what", "") or "").strip(),
        "word_estimate": _as_int(chapter.get("word_estimate")),
        "word_ratio": _as_int(chapter.get("word_ratio")),
        "design_options": chapter.get("design_options", []) if isinstance(chapter.get("design_options", []), list) else [],
        "chosen_design": str(chapter.get("chosen_design", "") or ""),
        "draft": str(chapter.get("draft", "") or ""),
    }


def _normalize_chapter_plan(plan, index: int) -> dict | None:
    if not isinstance(plan, dict):
        return None
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    normalized_steps = []
    def _as_int(value):
        try:
            return int(float(str(value or "").replace("%", "").strip()))
        except (TypeError, ValueError):
            return 0
    for step_index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        normalized_steps.append({
            "seq": step_index,
            "who": str(step.get("who", "") or "").strip(),
            "what": str(step.get("what", "") or "").strip(),
            "word_estimate": _as_int(step.get("word_estimate")),
            "word_ratio": _as_int(step.get("word_ratio")),
        })
    return {
        "key": str(plan.get("key") or letters[index % len(letters)]),
        "title": str(plan.get("title", "") or "").strip(),
        "thesis": str(plan.get("thesis", "") or "").strip(),
        "steps": normalized_steps,
    }


def _normalize_chapter_plans(plans) -> list:
    result = []
    for index, plan in enumerate(plans or []):
        normalized = _normalize_chapter_plan(plan, index)
        if normalized:
            result.append(normalized)
    return result


def _normalize_segments(segments):
    """事件列表：每个事件里带 chapters（章节）与 chapter_options（章节拆分方案）。"""
    result = []
    for index, segment in enumerate(segments or [], start=1):
        if not isinstance(segment, dict):
            continue
        raw_chapters = segment.get("chapters", [])
        chapters = raw_chapters if isinstance(raw_chapters, list) else []
        if not chapters and any(segment.get(key) for key in ("design_options", "chosen_design", "draft")):
            chapters = [segment]
        normalized_chapters = [
            _normalize_chapter(chapter, chapter_index)
            for chapter_index, chapter in enumerate(chapters, start=1)
            if isinstance(chapter, dict)
        ]
        result.append({
            "seq": int(segment.get("seq", index)),
            "who": str(segment.get("who", "") or "").strip(),
            "what": str(segment.get("what", "") or "").strip(),
            "chapter_options": _normalize_chapter_plans(segment.get("chapter_options", [])),
            "chosen_chapter_plan": str(segment.get("chosen_chapter_plan", "") or ""),
            "chapters": normalized_chapters,
        })
    return result


def _normalize_snapshot(snapshot) -> dict:
    if not isinstance(snapshot, dict):
        snapshot = {}

    def _string_list(value):
        items = value if isinstance(value, list) else []
        return [str(item).strip() for item in items if str(item).strip()]

    return {
        "beats": _string_list(snapshot.get("beats")),
        "character_states": _string_list(snapshot.get("character_states")),
        "hooks": _string_list(snapshot.get("hooks")),
    }


def _normalize_merged_route(route) -> dict | None:
    """校验合并后的完整路线：逐章规范化，非法输入返回 None。"""
    if not isinstance(route, dict):
        return None

    def _as_int(value):
        try:
            return int(float(str(value or "").replace("%", "").strip()))
        except (TypeError, ValueError):
            return 0

    chapters = []
    if isinstance(route.get("chapters"), list):
        for index, chapter in enumerate(route.get("chapters"), start=1):
            if not isinstance(chapter, dict):
                continue
            sub_events = []
            for sub in (chapter.get("sub_events") if isinstance(chapter.get("sub_events"), list) else []):
                if not isinstance(sub, dict):
                    continue
                try:
                    sub_words = int(float(str(sub.get("word_estimate", "") or "").replace("%", "").strip()))
                except (TypeError, ValueError):
                    sub_words = 0
                try:
                    sub_ratio = int(float(str(sub.get("word_ratio", "") or "").replace("%", "").strip()))
                except (TypeError, ValueError):
                    sub_ratio = 0
                sub_events.append({
                    "who": str(sub.get("who", "") or "").strip(),
                    "what": str(sub.get("what", "") or "").strip(),
                    "word_estimate": sub_words,
                    "word_ratio": sub_ratio,
                })
            chapters.append({
                "seq": _as_int(chapter.get("seq")) or index,
                "title": str(chapter.get("title", "") or "").strip(),
                "who": str(chapter.get("who", "") or "").strip(),
                "what": str(chapter.get("what", "") or "").strip(),
                "word_estimate": _as_int(chapter.get("word_estimate")),
                "word_ratio": _as_int(chapter.get("word_ratio")),
                "sub_events": sub_events,
                "draft": str(chapter.get("draft", "") or "").strip(),
            })
    fixed = []
    if isinstance(route.get("fixed"), list):
        fixed = [str(item).strip() for item in route.get("fixed") if str(item).strip()]
    return {
        "title": str(route.get("title", "") or "").strip(),
        "summary": str(route.get("summary", "") or "").strip(),
        "fixed": fixed,
        "chapters": chapters,
    }


def _empty_snapshot() -> dict:
    return {"beats": [], "character_states": [], "hooks": []}


def load_unit_working(filepath: str, unit_id: str) -> dict:
    path = os.path.join(_project_dir(filepath), "units", unit_id, _unit_working_filename())
    data = _load_json(path, {})
    return {
        "directory_options": data.get("directory_options", []) if isinstance(data, dict) else [],
        "chosen_plan": str(data.get("chosen_plan", "") or ""),
        "segments": _normalize_segments(data.get("segments", [])) if isinstance(data, dict) else [],
        "unit_summary": str(data.get("unit_summary", "") or ""),
        "snapshot": _normalize_snapshot(data.get("snapshot")) if isinstance(data, dict) else _empty_snapshot(),
        "carry_units": [str(item) for item in data.get("carry_units", [])] if isinstance(data, dict) and isinstance(data.get("carry_units", []), list) else [],
        "carry_facts": [str(item) for item in data.get("carry_facts", [])] if isinstance(data, dict) and isinstance(data.get("carry_facts", []), list) else [],
        "outline_tree": data.get("outline_tree") if isinstance(data, dict) else None,
        "merged_route": _normalize_merged_route(data.get("merged_route")) if isinstance(data, dict) else None,
    }


def save_unit_working(filepath: str, unit_id: str, working: dict) -> dict:
    payload = {
        "directory_options": working.get("directory_options", []),
        "chosen_plan": str(working.get("chosen_plan", "") or ""),
        "segments": working.get("segments", []),
        "unit_summary": str(working.get("unit_summary", "") or ""),
        "snapshot": _normalize_snapshot(working.get("snapshot")),
        "carry_units": [str(item) for item in working.get("carry_units", [])],
        "carry_facts": [str(item) for item in working.get("carry_facts", [])],
        "outline_tree": working.get("outline_tree"),
        "merged_route": working.get("merged_route"),
    }
    path = os.path.join(_project_dir(filepath), "units", unit_id, _unit_working_filename())
    _atomic_write_json(path, payload)
    return payload


def _empty_meta(seq: int) -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "seq": seq,
        "title": "",
        "goal": "",
        "cast": [],
        "lead_in": "",
        "word_count": 1500,
    }


def _empty_working() -> dict:
    return {"directory_options": [], "chosen_plan": "", "segments": [], "unit_summary": "", "snapshot": _empty_snapshot(), "carry_units": [], "carry_facts": [], "outline_tree": None, "merged_route": None}


def _renumber(units: list) -> list:
    for index, unit in enumerate(units, start=1):
        unit["seq"] = index
    return units


def load_book(filepath: str) -> dict:
    """返回前端可直接使用的完整项目快照。"""
    index = load_units_index(filepath)
    units = []
    for meta in index["units"]:
        if not isinstance(meta, dict) or not meta.get("id"):
            continue
        unit = dict(meta)
        unit.setdefault("seq", 1)
        unit.setdefault("title", "")
        unit.setdefault("goal", "")
        unit.setdefault("lead_in", "")
        if not isinstance(unit.get("cast"), list):
            unit["cast"] = []
        try:
            unit["word_count"] = int(unit.get("word_count") or 1500)
        except (TypeError, ValueError):
            unit["word_count"] = 1500
        working = load_unit_working(filepath, unit["id"])
        unit.update(working)
        units.append(unit)
    _renumber(units)
    current = index.get("current") or ""
    if current and not any(unit["id"] == current for unit in units):
        current = units[0]["id"] if units else ""
    if not current and units:
        current = units[0]["id"]
    meta = load_book_meta(filepath)
    proposals = load_proposals(filepath)
    return {
        "world": load_world(filepath),
        "characters": load_characters(filepath),
        "graph": load_graph(filepath),
        "units": units,
        "current": current,
        "title": meta["title"],
        "mainline": meta["mainline"],
        "target_words": meta.get("target_words", 0),
        "target_chapters": int((meta or {}).get("target_chapters") or 0),
        "words_per_chapter": int((meta or {}).get("words_per_chapter") or 0),
        "references": load_reference_files(filepath),
        "reference_limit": load_reference_limit(filepath),
        "proposals": proposals["proposals"],
        "debate": proposals["debate"],
    }


def _sync_meta_and_index(filepath: str, units: list, current: str) -> None:
    metas = []
    for unit in units:
        metas.append({key: unit.get(key) for key in ("id", "seq", "title", "goal", "cast", "lead_in", "word_count")})
    save_units_index(filepath, metas, current)


def create_unit(filepath: str, fields: dict) -> dict:
    index = load_units_index(filepath)
    seq = (max((int(unit.get("seq", 0)) for unit in index["units"]), default=0)) + 1
    unit = _empty_meta(seq)
    for key in ("title", "goal", "cast", "lead_in", "word_count"):
        if key in fields:
            unit[key] = fields[key]
    unit["goal"] = str(unit.get("goal", "") or "").strip()
    unit["title"] = str(unit.get("title", "") or "").strip()
    unit["lead_in"] = str(unit.get("lead_in", "") or "").strip()
    unit["word_count"] = int(unit.get("word_count") or 1500)
    if not isinstance(unit.get("cast"), list):
        unit["cast"] = []
    index["units"].append(unit)
    _renumber(index["units"])
    save_unit_working(filepath, unit["id"], _empty_working())
    _sync_meta_and_index(filepath, index["units"], index.get("current") or unit["id"])
    return load_book(filepath)


def update_unit(filepath: str, unit_id: str, fields: dict) -> dict:
    index = load_units_index(filepath)
    for unit in index["units"]:
        if unit.get("id") == unit_id:
            for key in ("title", "goal", "cast", "lead_in", "word_count"):
                if key in fields:
                    unit[key] = fields[key]
            unit["goal"] = str(unit.get("goal", "") or "").strip()
            unit["title"] = str(unit.get("title", "") or "").strip()
            unit["lead_in"] = str(unit.get("lead_in", "") or "").strip()
            unit["word_count"] = int(unit.get("word_count") or 1500)
            if not isinstance(unit.get("cast"), list):
                unit["cast"] = []
            break
    _renumber(index["units"])
    _sync_meta_and_index(filepath, index["units"], index.get("current"))
    return load_book(filepath)


def delete_unit(filepath: str, unit_id: str) -> dict:
    index = load_units_index(filepath)
    index["units"] = [unit for unit in index["units"] if unit.get("id") != unit_id]
    _renumber(index["units"])
    if index.get("current") == unit_id:
        index["current"] = index["units"][0]["id"] if index["units"] else ""
    _sync_meta_and_index(filepath, index["units"], index["current"])
    return load_book(filepath)


def select_unit(filepath: str, unit_id: str) -> dict:
    index = load_units_index(filepath)
    if any(unit.get("id") == unit_id for unit in index["units"]):
        index["current"] = unit_id
        _sync_meta_and_index(filepath, index["units"], unit_id)
    return load_book(filepath)


def get_unit(book: dict, unit_id: str) -> dict | None:
    for unit in book.get("units", []):
        if unit.get("id") == unit_id:
            return unit
    return None



