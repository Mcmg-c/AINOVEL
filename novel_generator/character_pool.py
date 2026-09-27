# -*- coding: utf-8 -*-
"""增量式角色信息池：只保存用户输入或正文中已经出现的事实。"""

import hashlib
import json
import os
import re
import tempfile

from llm_adapters import create_llm_adapter
from novel_generator.common import invoke_with_cleaning
from utils import read_file


POOL_FILENAME = "character_pool.json"
# 事实层：只记录可从正文逐字验证的观察。
ALLOWED_FACT_CATEGORIES = {
    "物品", "状态", "行动决定", "关系", "地点", "已知经历", "身体信息", "公开信息"
}
# 表现层：行为模式，允许记录解读性描述，但必须带原文证据。
# 表现层可以参与推演，但不作为硬设定约束落笔。
ALLOWED_TRAIT_CATEGORIES = {"行为模式", "说话方式", "习惯"}
ALLOWED_FACT_CATEGORIES_ALL = ALLOWED_FACT_CATEGORIES | ALLOWED_TRAIT_CATEGORIES
TRAIT_PREFIX = "trait:"


def _pool_path(filepath: str) -> str:
    return os.path.join(filepath, POOL_FILENAME)


def _fact_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _split_names(names_text: str) -> list[str]:
    return [name.strip() for name in re.split(r"[,，、\n]+", names_text or "") if name.strip()]


def load_character_pool(filepath: str) -> dict:
    path = _pool_path(filepath)
    if not os.path.exists(path):
        return {"version": 1, "active_characters": [], "characters": {}}
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("invalid character pool")
        data.setdefault("version", 1)
        data.setdefault("active_characters", [])
        data.setdefault("characters", {})
        return data
    except Exception:
        return {"version": 1, "active_characters": [], "characters": {}}


def save_character_pool(filepath: str, pool: dict) -> None:
    os.makedirs(filepath, exist_ok=True)
    target = _pool_path(filepath)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=filepath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(pool, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, target)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _find_role_profile(filepath: str, character_name: str) -> str:
    role_root = os.path.join(filepath, "角色库")
    if not os.path.isdir(role_root):
        return ""
    for root, _dirs, files in os.walk(role_root):
        for filename in files:
            if filename.lower().endswith(".txt") and os.path.splitext(filename)[0] == character_name:
                return read_file(os.path.join(root, filename)).strip()
    return ""


def _profile_to_base_facts(profile: str, character_name: str) -> list[dict]:
    facts = []
    for raw_line in (profile or "").splitlines():
        line = raw_line.strip().lstrip("-•* ").strip()
        if not line or re.fullmatch(rf"姓名\s*[:：]\s*{re.escape(character_name)}", line):
            continue
        facts.append({
            "id": _fact_id("base", line),
            "text": line,
            "category": "基础信息",
            "source": "user",
            "chapter": 0,
            "evidence": "用户填写的角色资料",
            "locked": True,
            "selected": True,
        })
    return facts


def initialize_character_pool(filepath: str, names_text: str) -> dict:
    """用角色库中的同名文件初始化基础事实；空白资料不会被 AI 自动补全。"""
    names = _split_names(names_text)
    pool = load_character_pool(filepath)
    pool["active_characters"] = names
    characters = pool.setdefault("characters", {})
    for name in names:
        record = characters.setdefault(name, {"base_facts": [], "unlocked_facts": []})
        existing_selected = {
            fact.get("id"): fact.get("selected", True)
            for fact in record.get("base_facts", [])
        }
        base_facts = _profile_to_base_facts(_find_role_profile(filepath, name), name)
        for fact in base_facts:
            fact["selected"] = existing_selected.get(fact["id"], True)
        record["base_facts"] = base_facts
        record.setdefault("unlocked_facts", [])
    save_character_pool(filepath, pool)
    write_selected_character_state(filepath, pool)
    return pool


def is_fact_selected(fact: dict, chapter: int | None = None) -> bool:
    """全局默认 + 按章覆盖：overrides 里没有该章记录时回退到全局 selected。"""
    if chapter is None:
        return bool(fact.get("selected", True))
    overrides = fact.get("overrides") or {}
    key = str(int(chapter))
    if key in overrides:
        return bool(overrides[key])
    return bool(fact.get("selected", True))


def is_trait_fact(fact: dict) -> bool:
    return str(fact.get("id", "")).startswith(TRAIT_PREFIX)


def render_selected_character_context(filepath: str, chapter: int | None = None) -> str:
    """渲染勾选后的人物上下文。传入章号时应用按章覆盖。

    事实层与表现层分开呈现：事实是硬白名单，表现只供参考。
    """
    pool = load_character_pool(filepath)
    lines = []
    for name in pool.get("active_characters", []):
        record = pool.get("characters", {}).get(name, {})
        selected = [
            fact for fact in record.get("base_facts", []) + record.get("unlocked_facts", [])
            if is_fact_selected(fact, chapter)
        ]
        lines.append(f"{name}：")
        if not selected:
            lines.append("- 暂无已选择信息")
            continue
        for fact in selected:
            source = "基础" if fact.get("chapter", 0) == 0 else f"第{fact.get('chapter')}章"
            category = fact.get("category", "事实")
            text = fact.get("text", "")
            if is_trait_fact(fact):
                lines.append(f"- [表现][{source}][{category}] {text}")
            else:
                lines.append(f"- [{source}][{category}] {text}")
    return "\n".join(lines).strip()


def render_full_character_context(filepath: str) -> str:
    """推演阶段使用：给出人物池的全部事实，让模型自行挑选相关项。

    不做检索，避免语义不相关但剧情相关的特征被静默丢掉。
    落笔阶段仍走 render_selected_character_context 的严格白名单。
    """
    pool = load_character_pool(filepath)
    lines = []
    for name in pool.get("active_characters", []):
        record = pool.get("characters", {}).get(name, {})
        facts = record.get("base_facts", []) + record.get("unlocked_facts", [])
        lines.append(f"{name}：")
        if not facts:
            lines.append("- 暂无信息")
            continue
        for fact in facts:
            source = "基础" if fact.get("chapter", 0) == 0 else f"第{fact.get('chapter')}章"
            category = fact.get("category", "事实")
            layer = "表现" if is_trait_fact(fact) else "事实"
            lines.append(f"- [{layer}][{source}][{category}] {fact.get('text', '')}")
    return "\n".join(lines).strip()


def render_character_context_parts(filepath: str, chapter: int | None = None) -> tuple[str, str]:
    """按层拆分已勾选的人物上下文，供提示词的两个字段分别使用。

    - 事实层：可直接观察的事实，作为落笔的硬白名单。
    - 表现层与作者设定：行为模式与作者显式写入的设定，供参考。

    两层互不重叠，避免同一份内容在提示词里重复出现。
    """
    pool = load_character_pool(filepath)
    fact_lines, trait_lines = [], []
    for name in pool.get("active_characters", []):
        record = pool.get("characters", {}).get(name, {})
        facts = [
            fact for fact in record.get("base_facts", []) + record.get("unlocked_facts", [])
            if is_fact_selected(fact, chapter)
        ]
        name_facts = [
            fact for fact in facts
            if not is_trait_fact(fact) and fact.get("category") != "作者备注"
        ]
        name_traits = [fact for fact in facts if fact not in name_facts]
        if name_facts:
            fact_lines.append(f"{name}：")
            for fact in name_facts:
                source = "基础" if fact.get("chapter", 0) == 0 else f"第{fact.get('chapter')}章"
                fact_lines.append(f"- [{source}][{fact.get('category', '事实')}] {fact.get('text', '')}")
        if name_traits:
            trait_lines.append(f"{name}：")
            for fact in name_traits:
                source = "基础" if fact.get("chapter", 0) == 0 else f"第{fact.get('chapter')}章"
                trait_lines.append(f"- [{source}][{fact.get('category', '设定')}] {fact.get('text', '')}")
    return "\n".join(fact_lines).strip(), "\n".join(trait_lines).strip()


def write_selected_character_state(filepath: str, pool: dict | None = None) -> None:
    if pool is None:
        pool = load_character_pool(filepath)
    content = render_selected_character_context(filepath)
    with open(os.path.join(filepath, "character_state.txt"), "w", encoding="utf-8") as file:
        file.write(content)


def update_fact_selection(
    filepath: str,
    selections: dict[tuple[str, str], bool],
    chapter: int | None = None,
) -> None:
    """更新勾选状态。不传章号时改全局默认，传章号时只写该章的覆盖值。"""
    pool = load_character_pool(filepath)
    for name, record in pool.get("characters", {}).items():
        for fact in record.get("base_facts", []) + record.get("unlocked_facts", []):
            key = (name, fact.get("id", ""))
            if key not in selections:
                continue
            if chapter is None:
                fact["selected"] = bool(selections[key])
            else:
                # 只记录与全局默认不同的例外，避免每条都堆一份冗余覆盖。
                key_name = str(int(chapter))
                overrides = fact.setdefault("overrides", {})
                if bool(selections[key]) == bool(fact.get("selected", True)):
                    overrides.pop(key_name, None)
                    if not overrides:
                        fact.pop("overrides", None)
                else:
                    overrides[key_name] = bool(selections[key])
    save_character_pool(filepath, pool)
    write_selected_character_state(filepath, pool)


def add_author_fact(filepath: str, character_name: str, text: str, selected: bool = False) -> None:
    pool = load_character_pool(filepath)
    if character_name not in pool.get("active_characters", []):
        pool.setdefault("active_characters", []).append(character_name)
    record = pool.setdefault("characters", {}).setdefault(
        character_name, {"base_facts": [], "unlocked_facts": []}
    )
    fact = {
        "id": _fact_id("author", text),
        "text": text.strip(),
        "category": "作者备注",
        "source": "author",
        "chapter": 0,
        "evidence": "作者手动添加",
        "locked": True,
        "selected": selected,
    }
    if not any(item.get("id") == fact["id"] for item in record["unlocked_facts"]):
        record["unlocked_facts"].append(fact)
    save_character_pool(filepath, pool)
    write_selected_character_state(filepath, pool)


def _parse_fact_response(response: str) -> list[dict]:
    text = (response or "").strip().replace("```json", "").replace("```", "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return []
        data = json.loads(text[start:end + 1])
    facts = data.get("facts", []) if isinstance(data, dict) else []
    return facts if isinstance(facts, list) else []


def _evidence_appears_in_chapter(evidence: str, chapter_text: str) -> bool:
    """证据必须能在正文中直接找到，避免模型凭空生成“原文证据”。"""
    cleaned = evidence.strip().strip("‘’“”\"'")
    if not cleaned:
        return False
    compact_evidence = re.sub(r"\s+", "", cleaned)
    compact_chapter = re.sub(r"\s+", "", chapter_text)
    return bool(compact_evidence) and compact_evidence in compact_chapter


def extract_and_merge_chapter_facts(
    *, filepath: str, chapter_number: int, chapter_text: str,
    interface_format: str, api_key: str, base_url: str, model_name: str,
    temperature: float, max_tokens: int, timeout: int = 600,
) -> list[dict]:
    pool = load_character_pool(filepath)
    names = pool.get("active_characters", [])
    if not names or not chapter_text.strip():
        return []
    prompt = f"""
从下面第{chapter_number}章正文中，抽取固定人物已经在正文中明确出现的新事实。

固定人物：{json.dumps(names, ensure_ascii=False)}

严格规则：
1. 只抽取正文明确陈述、角色明确说出或可直接观察到的事实。
2. 禁止推测性格、价值观、隐藏动机、未来命运、最终状态或作者计划。
3. 禁止补充正文没有写出的年龄、职业、关系、技能和背景。
4. 只允许固定人物名单中的人物；临时人物不进入信息池。
5. 每条事实必须提供正文中的简短原文证据。没有新事实就返回空数组。
6. 不要重复已经显然相同的事实。

分类只能从以下选择：
- 事实层：物品、状态、行动决定、关系、地点、已知经历、身体信息、公开信息
- 表现层：行为模式、说话方式、习惯

事实层只记录可直接观察的内容。表现层记录可被反复观察到的行为倾向，
每条必须附上正文中的原文证据；不要用"高冷""温柔"这类空泛定性，
要写成具体行为，例如"被追问时只回单字，不解释"。
禁止推测性格、价值观、隐藏动机、未来命运或作者计划。

只返回合法 JSON：
{{"facts":[{{"character":"人物名","category":"分类","text":"事实或行为模式","evidence":"原文证据"}}]}}

正文：
{chapter_text}
""".strip()
    adapter = create_llm_adapter(
        interface_format=interface_format,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=min(temperature, 0.2),
        max_tokens=max_tokens,
        timeout=timeout,
    )
    response = invoke_with_cleaning(
        adapter, prompt, label=f"第{chapter_number}章定稿：提取角色事实"
    )
    facts = _parse_fact_response(response)
    merged = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("character", "")).strip()
        text = str(item.get("text", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        category = str(item.get("category", "")).strip()
        if (
            name not in names
            or not text
            or category not in ALLOWED_FACT_CATEGORIES_ALL
            or not _evidence_appears_in_chapter(evidence, chapter_text)
        ):
            continue
        record = pool["characters"].setdefault(name, {"base_facts": [], "unlocked_facts": []})
        is_trait = category in ALLOWED_TRAIT_CATEGORIES
        prefix = f"{TRAIT_PREFIX}ch{chapter_number}" if is_trait else f"ch{chapter_number}"
        fact = {
            "id": _fact_id(prefix, text),
            "text": text,
            "category": category,
            "layer": "trait" if is_trait else "fact",
            "source": "chapter",
            "chapter": chapter_number,
            "evidence": evidence,
            "locked": True,
            "selected": True,
        }
        all_facts = record.get("base_facts", []) + record.get("unlocked_facts", [])
        if any(existing.get("id") == fact["id"] for existing in all_facts):
            continue
        record.setdefault("unlocked_facts", []).append(fact)
        merged.append(fact)
    save_character_pool(filepath, pool)
    write_selected_character_state(filepath, pool)
    return merged
