# -*- coding: utf-8 -*-
"""Web 版小说创作台的 AI 生成层。

只做三件确定性的事：
1. 目录方案：就一个章节给出 A/B/C 三套「谁 → 去干什么」的步骤方案；
2. 内容设计：针对某个小目录给出多个展开方案；
3. 正文：按选定的小目录与内容方案写正文。

所有生成都只接收作者已经锁定的世界观、入选人物与本章目标，
不把全书历史上下文塞进提示词，从而控制幻觉。
"""

import json
import re
import uuid
import concurrent.futures

from langchain_core.tools import tool

from llm_adapters import create_llm_adapter
from novel_generator.common import (
    emit_llm_trace,
    invoke_messages_with_cleaning,
    invoke_messages_with_tools_cleaning,
    invoke_with_cleaning,
)


@tool
def rewrite_text(replacement: str, reason: str) -> str:
    """当作者要求改写或修改时，提交对作者选中片段的重写版本供作者确认。replacement 只包含选中的那一小段文字的新写法，不要扩写成整段；reason 是修改理由。"""
    return replacement


def _make_adapter(model: dict, temperature=None):
    return create_llm_adapter(
        interface_format=model.get("interface_format", "OpenAI"),
        api_key=model.get("api_key", ""),
        base_url=model.get("base_url", ""),
        model_name=model.get("model_name", ""),
        temperature=float(model.get("temperature", 0.7)) if temperature is None else temperature,
        max_tokens=int(model.get("max_tokens", 8192)),
        timeout=int(model.get("timeout", 600)),
    )


def _extract_json(text: str):
    """从模型输出中稳健地取出第一个 JSON 对象或数组。"""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def render_cast(characters, cast_ids) -> str:
    """把入选人物渲染成提示词片段；未指定时默认使用人物库全部人物。"""
    if not cast_ids:
        cast_ids = [item.get("id") for item in characters if item.get("id")]
        if not cast_ids:
            return "（未选择人物）"
    by_id = {item.get("id"): item for item in characters}
    lines = []
    for cast_id in cast_ids:
        item = by_id.get(cast_id) or {}
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        parts = [str(item.get("gender", "")).strip(), str(item.get("age", "")).strip()]
        label = "，".join(part for part in parts if part)
        entry = name if not label else f"{name}（{label}）"
        notes = str(item.get("notes", "")).strip()
        lines.append(f"- {entry}" + (f"：{notes}" if notes else ""))
    return "\n".join(lines) if lines else "（未选择人物）"


def _render_directory(segments, include_step_note=True):
    lines = []
    for segment in segments:
        who = str(segment.get("who", "")).strip()
        what = str(segment.get("what", "")).strip()
        suffix = f"：{what}" if what else ""
        target = who or "（未写人物）"
        lines.append(f"{segment.get('seq', len(lines) + 1)}）{target}{suffix}")
    return "\n".join(lines) if lines else "（目录为空）"


def _score_value(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, round(score, 1)))


def _normalize_steps(plans):
    """把模型给出的 plans 规范化为内部结构，并补齐 key。"""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def _as_int(value):
        try:
            return int(float(str(value or "").replace("%", "").strip()))
        except (TypeError, ValueError):
            return 0

    result = []
    for index, plan in enumerate(plans or []):
        if not isinstance(plan, dict):
            continue
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        normalized_steps = []
        for step_index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            normalized_steps.append({
                "seq": step_index,
                "who": str(step.get("who", "")).strip(),
                "what": str(step.get("what", "")).strip(),
                "word_estimate": _as_int(step.get("word_estimate")),
                "word_ratio": _as_int(step.get("word_ratio")),
            })
        result.append({
            "key": str(plan.get("key") or letters[index % len(letters)]),
            "title": str(plan.get("title", "")).strip(),
            "thesis": str(plan.get("thesis", "")).strip(),
            "angle": str(plan.get("angle", "")).strip(),
            "score": _score_value(plan.get("score")),
            "score_reason": str(plan.get("score_reason", "") or "").strip(),
            "steps": normalized_steps,
        })
    return result


def render_history(messages) -> str:
    """把多轮讨论渲染进提示词，作为“可参考但不是硬约束”的上下文。"""
    lines = []
    for message in (messages or [])[-20:]:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        speaker = "你" if str(message.get("role", "")) == "user" else "AI"
        lines.append(f"{speaker}：{content}")
    return "\n".join(lines) if lines else "（无）"


def render_message_list(messages) -> list:
    """把消息渲染成接口可直接使用的 [{"role","content"}] 数组。"""
    result = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user") or "user").strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(message.get("content", "") or "")
        if content.strip():
            result.append({"role": role, "content": content})
    return result


def build_directory_options_prompt(
    *, world_text: str, cast_text: str, unit_goal: str, lead_in: str, guidance: str, history=None, carry_over: str = "", graph_text: str = "", mainline_text: str = "", reference_text: str = "", target_words: int = 0
) -> str:
    """只构建大事件目录方案提示词，不请求模型，供“先预览再发送”使用。"""
    return f"""你是这个小说创作项目的编剧。作者已经锁定了世界观、本书入选人物，以及这本书要写什么。
现在只需要产出「全书大事件目录方案」，不要写正文。

【世界观】
{world_text or '（未填写）'}

【本书单主线（全局方向，每一个大事件都要服务于它，但不要复述主线）】
{mainline_text or '（未设置）'}

【本书入选人物（只允许使用这些人物）】
{cast_text}

【参考文本（别人的书，仅用于借鉴风格、结构、节奏与情节处理，不得直接照抄）】
{reference_text or '（未上传参考文本）'}

【这本书要写什么】
{unit_goal or '（未填写）'}

【本书体量】
全书约 {target_words or '未指定'} 字；事件数与拆分节奏要和这个体量匹配。

【承上：已写好的前情快照（这是已经发生的事实，只引用、不要重写；与本书无关的请忽略）】
{carry_over or '（无）'}

【作者关于开头的说明】
{lead_in or '（无）'}

【图谱工作集（只取与本书人物相关的关系与时序）】
{graph_text or '（暂无图谱）'}

【作者补充】
{guidance or '（无）'}

【近期的讨论与决定（可参考，但不是已发生的事实）】
{render_history(history)}

任务：给出 5 套明显不同的《全书大事件目录》方案（方案 A / B / C / D / E）。每套把整本书拆成 4～8 个前后咬合的大事件，
用更细的“丝线”把“这本书要写什么”串联起来：
- 每个步骤就是一个大事件，必须写清「谁 → 在这个事件里发生什么 / 要做什么」，是事件级的行动线，不要写场景描写。
- 五套方案必须拉开足够大的差异：叙事结构（顺叙／双线交叉／倒叙／单元剧／多视角）、切入角度（案件／情感／悬疑／日常／伏笔）、冲突类型、信息释放顺序至少要有两项明显不同，禁止换汤不换药。
- 每套方案用 angle 标注它的叙事结构或切入角度标签（如“双线交叉”“悬念先置”“倒叙剥茧”），五套的 angle 不应雷同。
- 首事件要有钩子，中段要有起伏，结局要回收主线；每一个大事件都要推动单主线前进。
- 严格只用入选人物及其已知资料，不得新增有姓名的角色，不得替人物补写年龄、职业、秘密、背景。
- 不要写正文，不要解释任务。

只返回 JSON（不要任何其他文字）：
{{"plans":[{{"key":"A","title":"方案名","thesis":"这套方案的串联逻辑","angle":"叙事结构/切入角度标签","steps":[{{"who":"人物名","what":"这个事件发生什么/要做什么"}}]}}]}}
""".strip()


def generate_directory_options(
    *, world_text: str, cast_text: str, unit_goal: str, lead_in: str, guidance: str, history=None, model: dict,
    carry_over: str = "", graph_text: str = "", mainline_text: str = "", reference_text: str = "",
    target_words: int = 0, prompt_override: str | None = None
) -> dict:
    prompt = prompt_override if prompt_override is not None else build_directory_options_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal, lead_in=lead_in,
        guidance=guidance, history=history, carry_over=carry_over, graph_text=graph_text, mainline_text=mainline_text, reference_text=reference_text,
        target_words=target_words,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label="全书大事件目录：A-E 方案", invoke_json=True)
    parsed = _extract_json(result)
    plans = _normalize_steps(parsed.get("plans") if isinstance(parsed, dict) else None)
    if not plans:
        plans = [{
            "key": "A",
            "title": "仅有一条未结构化结果",
            "thesis": result.strip() or "（模型未返回内容）",
            "steps": [],
        }]
    return {"directory_options": plans}


def build_event_chapter_options_prompt(
    *, world_text: str, cast_text: str, unit_goal: str, mainline_text: str = "",
    directory_text: str = "", event: dict, history=None, graph_text: str = "", reference_text: str = "", target_words: int = 0,
    target_chapters: int = 0, words_per_chapter: int = 0, events_total: int = 0
) -> str:
    """构建「事件 → 章节」二级拆分提示词：把一个大事件拆成章节。"""
    seq = (event or {}).get("seq", 1)
    who = str((event or {}).get("who", "") or "").strip()
    what = str((event or {}).get("what", "") or "").strip()
    scale_lines = [f"全书约 {target_words or '未指定'} 字；每章 word_estimate / word_ratio 必须和这个体量匹配（全书 word_ratio 合计 100）。"]
    if target_chapters:
        scale_lines.append(f"全书计划大概 {target_chapters} 章，共 {events_total or '若干'} 个大事件，请按这个总体规模来分配事件与章节，不要拆得过于零碎。")
    if words_per_chapter:
        scale_lines.append(f"每章侧重约 {words_per_chapter} 字：word_estimate 以这个数值为目标，word_ratio 按「每章字数 ÷ {target_words or words_per_chapter}」等比例折算。")
    split_hint = "每套把事件" + str(seq) + "拆成 2～5 个前后咬合的章节"
    if target_chapters and events_total:
        suggested = max(1, round(target_chapters / max(1, events_total)))
        split_hint = "每套把事件" + str(seq) + "拆成 " + str(max(1, suggested - 1)) + "～" + str(max(2, suggested + 1)) + " 个前后咬合的章节（全书计划约 " + str(target_chapters) + " 章）"
    return f"""你是这个小说创作项目的编剧。当前要把整本书里的「事件{seq}」进一步拆成章节，仍然不写正文。

【世界观】
{world_text or '（未填写）'}

【本书单主线（这一事件要对主线有所推进）】
{mainline_text or '（未设置）'}

【本书入选人物】
{cast_text}

【参考文本（别人的书，仅用于借鉴风格、结构、节奏与情节处理，不得直接照抄）】
{reference_text or '（未上传参考文本）'}

【这本书要写什么】
{unit_goal or '（未填写）'}

【本书体量】
{chr(10).join(scale_lines)}

【全书大事件目录（只用于定位当前事件，不要改动）】
{directory_text or '（无）'}

【当前事件】
事件{seq}：{who or '（未写人物）'} → {what or '（未写内容）'}

【图谱工作集（本书相关的人物关系与时序）】
{graph_text or '（暂无图谱）'}

【近期的讨论与决定（可参考，但不是已发生的事实）】
{render_history(history)}

任务：给出 3 套明显不同的「事件{seq}」章节拆分方案（方案 A / B / C）。{split_hint}：
- 每个章节写清「谁 → 在这一章发生什么 / 要做什么」，是章节级的行动线，不要写场景描写。
- 每个章节都要给出 word_estimate（预计字数）与 word_ratio（该章占全书篇幅的百分比估计）。word_estimate 要与全书体量匹配；同一事件内各章 word_ratio 加总应等于该事件应占的篇幅比例，全书所有章节合计 100%。
- 每套方案还要给出 score（0～10，最多一位小数）与 score_reason：综合信息释放节奏、冲突强度、与其他事件的衔接、可执行性来打分并说明理由。
- 首章建立事件目标，末章收束该事件，并给下一事件留一个自然的钩子。
- 三套方案在信息释放顺序、冲突落点、视角与节奏上拉开差异。
- 严格只用入选人物及其已知资料，不新增角色，不替人物补写设定。
- 不要写正文，不要解释任务。

只返回 JSON（不要任何其他文字）：
{{"plans":[{{"key":"A","title":"方案名","thesis":"这套方案的拆分逻辑","score":8.5,"score_reason":"为什么打这个分","steps":[{{"who":"人物名","what":"这一章发生什么/要做什么","word_estimate":3200,"word_ratio":12}}]}}]}}
""".strip()


def generate_event_chapter_options(
    *, world_text: str, cast_text: str, unit_goal: str, mainline_text: str = "",
    directory_text: str = "", event: dict, history=None, graph_text: str = "",
    model: dict, prompt_override: str | None = None, reference_text: str = "", target_words: int = 0,
    target_chapters: int = 0, words_per_chapter: int = 0, events_total: int = 0
) -> dict:
    prompt = prompt_override if prompt_override is not None else build_event_chapter_options_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal, mainline_text=mainline_text,
        directory_text=directory_text, event=event, history=history, graph_text=graph_text, reference_text=reference_text,
        target_words=target_words, target_chapters=target_chapters, words_per_chapter=words_per_chapter, events_total=events_total,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label=f"事件{(event or {}).get('seq', '?')}：拆章节方案", invoke_json=True)
    parsed = _extract_json(result)
    plans = _normalize_steps(parsed.get("plans") if isinstance(parsed, dict) else None)
    if not plans:
        plans = [{
            "key": "A",
            "title": "仅有一条未结构化结果",
            "thesis": result.strip() or "（模型未返回内容）",
            "steps": [],
        }]
    return {"chapter_options": plans}


def build_reference_brief_prompt(*, reference_text: str, guidance: str, target_words: int = 0) -> str:
    """基于作者上传的参考文本，起草新书的单主线、世界观与人物预设。"""
    return f"""你是小说设定策划。作者上传了一段参考文本，并想据此写一部新小说。
请只借鉴参考文本的题材、文风与叙事节奏，**不要照抄其中的人物、情节或专属设定**，然后给出新书设定草案。

【作者意图（最优先，例如题材、标签、关系走向等）】
{guidance or '（未说明）'}

【全书体量】
目标约 {target_words or '未指定'} 字；单主线的复杂度、世界观的信息量与人物数量都要能撑起这个体量，输出时不要写死章节数。

【参考文本节选（别人的书）】
{reference_text or '（未上传参考文本）'}

需要产出的内容：
1. 书名建议（1 个，贴合数据与题材）
2. 单主线（完整说清主角要什么、最大的阻力是什么、情感终点通向哪里；不要压缩成一句话，字数不限，说清楚即可）
3. 世界观：类型 / 前提 / 时代与地点 / 规则限制 / 基调，每个字段写清楚即可，不要限制字数
4. 人物：2～6 个，每人姓名 / 性别 / 年龄 / 介绍，介绍不要压缩成一句话，写清楚即可。若题材涉及情感关系，请在“介绍”或单主线里写清关系张力与攻受定位，例如谁攻谁受、核心矛盾是什么。

只返回 JSON（不要任何其他文字）：
{{"title":"","mainline":"","world":{{"genre":"","premise":"","era_place":"","rules":"","tone":""}},"characters":[{{"name":"","gender":"男","age":"24","notes":""}}]}}
""".strip()


def generate_reference_brief(*, reference_text: str, guidance: str, model: dict, prompt_override: str | None = None, target_words: int = 0) -> dict:
    prompt = prompt_override if prompt_override is not None else build_reference_brief_prompt(
        reference_text=reference_text, guidance=guidance, target_words=target_words,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label="参考文本：生成设定建议", invoke_json=True)
    parsed = _extract_json(result)
    if not isinstance(parsed, dict):
        parsed = {}
    world = parsed.get("world") if isinstance(parsed.get("world"), dict) else {}
    characters = parsed.get("characters") if isinstance(parsed.get("characters"), list) else []
    normalized_chars = []
    for item in characters:
        if not isinstance(item, dict):
            continue
        normalized_chars.append({
            "name": str(item.get("name", "") or "").strip(),
            "gender": str(item.get("gender", "") or "").strip(),
            "age": str(item.get("age", "") or "").strip(),
            "notes": str(item.get("notes", "") or "").strip(),
        })
    brief = {
        "title": str(parsed.get("title", "") or "").strip(),
        "mainline": str(parsed.get("mainline", "") or "").strip(),
        "world": {key: str(world.get(key, "") or "").strip() for key in ("genre", "premise", "era_place", "rules", "tone")},
        "characters": normalized_chars,
    }
    return {"brief": brief}


_PROPOSAL_DIRECTIONS = (
    ("A", "悬念驱动：把核心阴谋/悬案的复杂度、反转密度与推理对抗推到最强；情感线贴着案件推进，整体冷峻克制。"),
    ("B", "情感驱动：把人物关系张力、旧伤、欲说还休与救赎推到最强；案件是映照感情的镜子，情绪浓度最高。"),
    ("C", "结构实验：叙事结构最大胆（双时间线 / 多视角 / 倒叙 / 碎片化回忆）；世界观与谜团的呈现方式最特别，但主线仍然清晰。"),
)


def build_proposal_prompt(*, key: str, direction: str, reference_text: str, guidance: str, target_words: int = 0) -> str:
    """生成单套全书方案（主线 + 世界观 + 人物 + 大事件目录）的提示词。"""
    return f"""你是小说总策划。作者上传了参考文本，想据此写一部新小说。
请只借鉴参考文本的题材、文风与叙事节奏，**不要照抄其中的人物、情节或专属设定**。

这一次你只负责产出第 {key} 套方案，方向定位：
{direction}
按这个定位把这一套做出特点即可，不需要和别的方案比较。

【作者意图（最优先，例如题材、标签、关系走向等）】
{guidance or '（未说明）'}

【全书体量】
目标约 {target_words or '未指定'} 字；主线复杂度、世界观信息量、人物数量和大事件拆分节奏都要能撑起这个体量，不要写死章节数。

【参考文本节选（别人的书）】
{reference_text or '（未上传参考文本）'}

必须产出（全部写全，禁止用概述糊弄）：
1. title：书名，贴合题材。
2. thesis：一句话卖点。
3. mainline：完整单主线，**必须写清楚全部细节，绝对禁止写概述**。依次说明：主角是谁、起点状态（过去经历了什么、现在处境如何）、核心目标（想要什么、为什么现在必须行动）、最大阻力（具体的对手/组织/环境/内心冲突，写实）、过程中的关键转折（至少 2 个，写出由什么引发、带来什么后果）、情感线的完整走向（若有感情线：关系起点、核心矛盾、攻受定位、旧伤如何被揭开、如何走向结局）、最终结局与情感落点。字数不限，越具体越好。
4. world：类型 / 前提 / 时代与地点 / 规则限制 / 基调，五个字段都写清楚。
5. characters：2～6 个人物，每人姓名 / 性别 / 年龄 / 介绍；介绍要写具体，禁止一句话带过。
6. directory：大事件目录，8～14 个事件，每个事件只写「谁 + 做什么」，按时间推进覆盖从开场到结局的全过程。

只返回 JSON（不要任何其他文字）：
{{"key":"{key}","title":"","thesis":"","mainline":"","world":{{"genre":"","premise":"","era_place":"","rules":"","tone":""}},"characters":[{{"name":"","gender":"男","age":"24","notes":""}}],"directory":[{{"seq":1,"who":"","what":""}}]}}
""".strip()


def _normalize_proposal(key: str, parsed) -> dict | None:
    """把一次调用返回的 JSON 规整成单套方案；空壳（既没主线也没事件）返回 None。"""
    if not isinstance(parsed, dict):
        return None
    title = str(parsed.get("title", "") or "").strip()
    mainline = str(parsed.get("mainline", "") or "").strip()
    world = parsed.get("world") if isinstance(parsed.get("world"), dict) else {}
    characters = parsed.get("characters") if isinstance(parsed.get("characters"), list) else []
    normalized_chars = []
    for character in characters:
        if not isinstance(character, dict):
            continue
        normalized_chars.append({
            "name": str(character.get("name", "") or "").strip(),
            "gender": str(character.get("gender", "") or "").strip(),
            "age": str(character.get("age", "") or "").strip(),
            "notes": str(character.get("notes", "") or "").strip(),
        })
    directory = parsed.get("directory") if isinstance(parsed.get("directory"), list) else []
    normalized_steps = []
    for step_index, step in enumerate(directory, start=1):
        if not isinstance(step, dict):
            continue
        normalized_steps.append({
            "seq": int(step.get("seq", step_index) or step_index),
            "who": str(step.get("who", "") or "").strip(),
            "what": str(step.get("what", "") or "").strip(),
        })
    if not mainline and not title and not normalized_steps:
        return None
    return {
        "key": str(parsed.get("key", "") or "").strip() or key,
        "title": title,
        "thesis": str(parsed.get("thesis", "") or "").strip(),
        "mainline": mainline,
        "world": {world_key: str(world.get(world_key, "") or "").strip() for world_key in ("genre", "premise", "era_place", "rules", "tone")},
        "characters": normalized_chars,
        "directory": normalized_steps,
    }


def generate_proposals(*, reference_text: str, guidance: str, model: dict, prompt_override: str | None = None, target_words: int = 0, workers: int = 3) -> dict:
    """分三次独立调用模型，每次生成一套方案；三次并行执行，允许出现重复。"""
    def run_one(direction_key: str, direction: str) -> dict | None:
        prompt = build_proposal_prompt(
            key=direction_key, direction=direction,
            reference_text=reference_text, guidance=guidance, target_words=target_words,
        )
        try:
            result = invoke_with_cleaning(_make_adapter(model), prompt, label=f"参考文本：生成第 {direction_key} 套全书方案", invoke_json=True)
        except Exception:
            return None
        return _normalize_proposal(direction_key, _extract_json(result))

    if prompt_override:
        results = {"A": _normalize_proposal("A", _extract_json(invoke_with_cleaning(
            _make_adapter(model), prompt_override, label="参考文本：生成全书方案", invoke_json=True,
        )))}
    else:
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(3, int(workers or 3)))) as pool:
            futures = {pool.submit(run_one, direction_key, direction): direction_key for direction_key, direction in _PROPOSAL_DIRECTIONS}
            for future in concurrent.futures.as_completed(futures):
                direction_key = futures[future]
                try:
                    item = future.result()
                except Exception:
                    item = None
                if item:
                    results[direction_key] = item
    proposals = [results[direction_key] for direction_key, _direction in _PROPOSAL_DIRECTIONS if direction_key in results]
    return {"proposals": proposals}


def build_unit_snapshot_prompt(
    *, world_text: str = "", cast_text: str = "", unit_goal: str = "",
    directory_text: str = "", drafts_text: str = "", summary_text: str = "", mainline_text: str = ""
) -> str:
    """构建「全书收尾快照」提示词：把整本书压缩成前情，不写新剧情。"""
    return f"""你是这个小说创作项目的司书助手。请把「刚刚写完的整本书」压缩成一段“全书前情快照”，
供以后核对连续性、生成续集或扩展时复用，防止上下文膨胀和事实漂移。不要抒情、不要复述正文。

【世界观】
{world_text or '（未填写）'}

【本书单主线（用于判断哪些结果真正影响后续主线）】
{mainline_text or '（未设置）'}

【本书入选人物】
{cast_text or '（未选择）'}

【这本书要写什么】
{unit_goal or '（未填写）'}

【全书章节目录】
{directory_text or '（无）'}

【全书已写正文（只用于提取事实，不要抄写原文）】
{drafts_text or '（无）'}

【作者留下的全书摘要】
{summary_text or '（无）'}

只输出 JSON（不要任何其他文字）：
{{"beats":["第N章 · 谁 → 做什么：发生了什么/结果"],"character_states":["人物：之前… → 现在…（关系/情绪/目标变化）"],"hooks":["仍未解决、以后可能再被用到的线头"]}}

要求：
- beats 一行一条，只保留仍然影响后续的章节结果，最多 8 条；没有影响就返回空数组。
- character_states 只写全书出场且状态确实发生变化的人物，最多 8 条。
- hooks 只写真正遗留、可复用的未完成线头，没有就返回空数组。
- 不新增人物，不补写设定，不替人物编造过去。""".strip()


def generate_unit_snapshot(
    *, world_text: str = "", cast_text: str = "", unit_goal: str = "",
    directory_text: str = "", drafts_text: str = "", summary_text: str = "",
    mainline_text: str = "", model: dict, prompt_override: str | None = None
) -> dict:
    prompt = prompt_override if prompt_override is not None else build_unit_snapshot_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal,
        directory_text=directory_text, drafts_text=drafts_text, summary_text=summary_text, mainline_text=mainline_text,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label="全书收尾：前情快照", invoke_json=True)
    parsed = _extract_json(result)

    def _list(value):
        items = value if isinstance(value, list) else []
        return [str(item).strip() for item in items if str(item).strip()]

    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "beats": _list(parsed.get("beats")),
        "character_states": _list(parsed.get("character_states")),
        "hooks": _list(parsed.get("hooks")),
    }


def build_atomize_prompt(
    *, world_text: str = "", cast_text: str = "", unit_goal: str = "",
    directory_text: str = "", drafts_text: str = "", snapshot_text: str = "",
    graph_text: str = "", unit_seq: int = 1, mainline_text: str = ""
) -> str:
    """构建「GraphRAG 原子化」提示词：把前情快照拆成可单独勾选的小事件，收进知识图谱。"""
    return f"""你是这个小说项目的知识图谱整理器。现在要把「刚写完的整本书」拆成可单独复用的原子小事件，
写进知识图谱，供以后核对连续性、生成续集或扩展时按需引用，避免把整段历史塞进上下文。

【世界观】
{world_text or '（未填写）'}

【本书单主线（用于判断哪些小事件值得保留）】
{mainline_text or '（未设置）'}

【本书入选人物】
{cast_text or '（未选择）'}

【这本书要写什么】
{unit_goal or '（未填写）'}

【全书章节目录】
{directory_text or '（无）'}

【全书已写正文（只用于提取事实，不要抄写原文）】
{drafts_text or '（无）'}

【全书前情快照（beats/人物状态变化/遗留钩子）】
{snapshot_text or '（无）'}

【现有知识图谱（用于对齐实体名称，不要改图谱，只产出小事件）】
{graph_text or '（空）'}

只输出 JSON（不要任何其他文字）：
{{"facts":[{{"text":"一条原子小事件","entities":["相关节点名称"],"unit_seq":{int(unit_seq)}}}]}}

要求：
- 每条 fact 只含一条最小、不能再拆的信息：谁 + 在何时/何地 + 做了什么/发生了什么 + 结果或当前状态。
- 只从上面的快照和已写正文提取，不脑补、不新增人物、不替人物编造动机。
- entities 只填与本条事实直接相关的实体名（人物/地点/组织/物品/悬念），尽量复用现有图谱里的名称。
- 尽量拆分到「以后生成新章节时可以单独勾选」的粒度，最多 40 条；没有可靠事实就返回空数组。
- 每条 unit_seq 固定为 {int(unit_seq)}。""".strip()


def generate_atomize(
    *, world_text: str = "", cast_text: str = "", unit_goal: str = "",
    directory_text: str = "", drafts_text: str = "", snapshot_text: str = "",
    graph_text: str = "", unit_seq: int = 1, mainline_text: str = "", model: dict, prompt_override: str | None = None
) -> dict:
    prompt = prompt_override if prompt_override is not None else build_atomize_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal,
        directory_text=directory_text, drafts_text=drafts_text, snapshot_text=snapshot_text,
        graph_text=graph_text, unit_seq=unit_seq, mainline_text=mainline_text,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label="GraphRAG：拆分小事件", invoke_json=True)
    parsed = _extract_json(result)
    if not isinstance(parsed, dict):
        parsed = {}

    facts = []
    for item in parsed.get("facts", []) if isinstance(parsed.get("facts", []), list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        entities = item.get("entities", []) if isinstance(item.get("entities", []), list) else []
        facts.append({
            "id": str(item.get("id") or uuid.uuid4().hex[:8]).strip(),
            "unit_seq": unit_seq,
            "text": text,
            "entities": [str(entity).strip() for entity in entities if str(entity).strip()],
        })

    return {"facts": facts}


def build_segment_design_options_prompt(
    *, world_text: str, cast_text: str, unit_goal: str, directory_text: str,
    segment: dict, previous_summary: str, history=None, graph_text: str = "", mainline_text: str = "", reference_text: str = ""
) -> str:
    who = str(segment.get("who", "")).strip()
    what = str(segment.get("what", "")).strip()
    seq = segment.get("seq", 1)
    return f"""你是这个小说创作项目的编剧。当前要为「第{seq}章」产出内容设计方案，仍然不写正文。

【世界观】
{world_text or '（未填写）'}

【本书单主线（这一章要对主线有所推进）】
{mainline_text or '（未设置）'}

【本书入选人物】
{cast_text}

【参考文本（别人的书，仅用于借鉴风格、结构、节奏与情节处理，不得直接照抄）】
{reference_text or '（未上传参考文本）'}

【这本书要写什么】
{unit_goal or '（未填写）'}

【全书章节目录】
{directory_text}

【当前章节】
第{seq}章：{who or '（未写人物）'} → {what or '（未写动作）'}

【图谱工作集（本书相关的人物关系与时序）】
{graph_text or '（暂无图谱）'}

【上一章已确定的内容（用于承接，不重复）】
{previous_summary or '（第一章）'}

【近期的讨论与决定（可参考，但不是已发生的事实）】
{render_history(history)}

任务：给出 3 个明显不同的内容设计方案（方案 1 / 2 / 3），每个方案说明这一章具体怎么展开：
切入点与视角、冲突或阻碍、信息释放、情绪变化、结尾给下一章的钩子。
严格只用入选人物资料，不补写背景与秘密，不写正文。

只返回 JSON（不要任何其他文字）：
{{"options":[{{"key":"1","title":"方案名","summary":"一句话概括","content":"具体展开方案"}}]}}
""".strip()


def generate_segment_design_options(
    *, world_text: str, cast_text: str, unit_goal: str, directory_text: str,
    segment: dict, previous_summary: str, history=None, model: dict,
    graph_text: str = "", mainline_text: str = "", reference_text: str = "",
    prompt_override: str | None = None
) -> dict:
    seq = segment.get("seq", 1)
    prompt = prompt_override if prompt_override is not None else build_segment_design_options_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal, directory_text=directory_text,
        segment=segment, previous_summary=previous_summary, history=history, graph_text=graph_text, mainline_text=mainline_text, reference_text=reference_text,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label=f"第{seq}章：内容设计方案", invoke_json=True)
    parsed = _extract_json(result)
    options = parsed.get("options") if isinstance(parsed, dict) else None
    normalized = []
    for index, option in enumerate(options or [], start=1):
        if not isinstance(option, dict):
            continue
        normalized.append({
            "key": str(option.get("key") or str(index)),
            "title": str(option.get("title", "")).strip(),
            "summary": str(option.get("summary", "")).strip(),
            "content": str(option.get("content", "")).strip(),
        })
    if not normalized:
        normalized = [{
            "key": "1",
            "title": "仅有一条未结构化结果",
            "summary": "",
            "content": result.strip() or "（模型未返回内容）",
        }]
    return {"options": normalized}


def build_segment_draft_prompt(
    *, world_text: str, cast_text: str, unit_goal: str, directory_text: str,
    segment: dict, design_text: str, previous_tail: str, word_count: int, history=None, extra_instruction: str = "", graph_text: str = "", mainline_text: str = "", reference_text: str = ""
) -> str:
    who = str(segment.get("who", "")).strip()
    what = str(segment.get("what", "")).strip()
    seq = segment.get("seq", 1)
    return f"""你是这个小说创作项目的小说写手。请按已经锁定的方案，只写「第{seq}章」的正文。

【世界观】
{world_text or '（未填写）'}

【本书单主线（这一章要对主线有所推进）】
{mainline_text or '（未设置）'}

【本书入选人物】
{cast_text}

【参考文本（别人的书，仅用于借鉴风格、结构、节奏与情节处理，不得直接照抄）】
{reference_text or '（未上传参考文本）'}

【这本书要写什么】
{unit_goal or '（未填写）'}

【全书章节目录】
{directory_text}

【当前章节】
第{seq}章：{who or '（未写人物）'} → {what or '（未写动作）'}

【选定内容方案】
{design_text or '（未选择，请按目录自行展开）'}

【图谱工作集（本书相关的人物关系与时序）】
{graph_text or '（暂无图谱）'}

【上一章正文结尾（仅用于承接语气与动作，不要抄写）】
{previous_tail or '（第一章）'}

【近期的讨论与决定（可参考，但不是已发生的事实）】
{render_history(history)}

【作者本章额外要求】
{extra_instruction or '（无）'}

要求：
- 只写这一章，目标 {word_count} 字左右，不要提前发生下一章。
- 若作者写了额外要求，严格按额外要求调整字数、描写重点或节奏。
- 严格只用入选人物资料，不给人物补写年龄、职业、秘密、背景。
- 结尾自然滑向下一章，但不要把下一章写出来。
- 直接输出正文，不要标题、不要前后解释。
""".strip()


def generate_segment_draft(
    *, world_text: str, cast_text: str, unit_goal: str, directory_text: str,
    segment: dict, design_text: str, previous_tail: str, word_count: int, history=None, model: dict,
    extra_instruction: str = "", graph_text: str = "", mainline_text: str = "", reference_text: str = "",
    prompt_override: str | None = None
) -> str:
    seq = segment.get("seq", 1)
    prompt = prompt_override if prompt_override is not None else build_segment_draft_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal, directory_text=directory_text,
        segment=segment, design_text=design_text, previous_tail=previous_tail, word_count=word_count,
        history=history, extra_instruction=extra_instruction, graph_text=graph_text, mainline_text=mainline_text, reference_text=reference_text,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label=f"第{seq}章：写正文")
    return result.strip()
def build_route_chapter_draft_prompt(
    *, world_text: str, cast_text: str, unit_goal: str, mainline_text: str = "",
    directory_text: str = "", chapter: dict, previous_text: str, word_count: int,
    history=None, graph_text: str = "", reference_text: str = "", target_words: int = 0,
    sub_events_text: str = ""
) -> str:
    """合并路线定稿后，串行撰写某一章的提示词：必须带上前面所有章节的完整正文。"""
    chapter = chapter if isinstance(chapter, dict) else {}
    seq = chapter.get("seq", 1)
    title = str(chapter.get("title", "") or "").strip()
    who = str(chapter.get("who", "") or "").strip()
    what = str(chapter.get("what", "") or "").strip()
    lines = [
        '你是这个小说创作项目的总写手。全书章节目录已经锁定，请按顺序只写「第' + str(seq) + '章」的正文。',
        '',
        '【世界观】',
        world_text or '（未填写）',
        '',
        '【本书单主线（这一章要对主线有所推进）】',
        mainline_text or '（未设置）',
        '',
        '【本书入选人物】',
        cast_text,
        '',
        '【参考文本（别人的书，仅用于借鉴风格、结构、节奏与情节处理，不得直接照抄）】',
        reference_text or '（未上传参考文本）',
        '',
        '【这本书要写什么】',
        unit_goal or '（未填写）',
        '',
        '【全书体量】',
        '全书约 ' + str(target_words or '未指定') + ' 字。',
        '',
        '【全书章节目录（每章一句话行动线，只用于定位进度）】',
        directory_text or '（无）',
        '',
        '【前面已写完的章节（完整正文，必须承接：时间线、人物状态、已回收与未回收的钩子；禁止与前文矛盾或重复）】',
        previous_text or '（这是第一章，没有前文）',
        '',
        '【当前章节】',
        '第' + str(seq) + '章' + ('《' + title + '》' if title else '') + '：' + (who or '（未写人物）') + ' → ' + (what or '（未写动作）'),
        (sub_events_text or ''),
        '',
        '【图谱工作集（本书相关的人物关系与时序）】',
        graph_text or '（暂无图谱）',
        '',
        '【近期的讨论与决定（可参考，但不是已发生的事实）】',
        render_history(history),
        '',
        '要求：',
        '- 只写这一章正文，目标 ' + str(word_count) + ' 字左右，不要提前发生下一章。',
        '- 严格承接前文的每一个钩子与人物状态：时间线连续，人物言行前后一致。',
        '- 已在前文完成的情节不得重复上演；新出现的信息必须与前文设定兼容。',
        '- 严格只用入选人物资料，不新增有姓名的角色，不替人物补写设定。',
        '- 结尾自然滑向下一章，但不要把下一章写出来。',
        '- 直接输出正文，不要标题、不要前后解释。',
    ]
    return "\n".join(lines).strip()


def generate_route_chapter_draft(
    *, world_text: str, cast_text: str, unit_goal: str, mainline_text: str = "",
    directory_text: str = "", chapter: dict, previous_text: str, word_count: int,
    history=None, model: dict, graph_text: str = "", reference_text: str = "", target_words: int = 0,
    prompt_override: str | None = None, sub_events_text: str = ""
) -> str:
    chapter = chapter if isinstance(chapter, dict) else {}
    seq = chapter.get("seq", 1)
    title = str(chapter.get("title", "") or "").strip()
    prompt = prompt_override if prompt_override is not None else build_route_chapter_draft_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal, mainline_text=mainline_text,
        directory_text=directory_text, chapter=chapter, previous_text=previous_text, word_count=word_count,
        history=history, graph_text=graph_text, reference_text=reference_text, target_words=target_words,
        sub_events_text=sub_events_text,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label='合并路线第' + str(seq) + '章' + ('《' + title + '》' if title else '') + '：写正文')
    return result.strip()


def render_directory(segments) -> str:
    """渲染目录为「谁 → 干什么」的编号文本，供生成与展示使用。"""
    return _render_directory(segments)


def render_events(segments) -> str:
    """渲染全书的大事件目录（一级拆分）。"""
    lines = []
    for segment in segments or []:
        who = str(segment.get("who", "") or "").strip()
        what = str(segment.get("what", "") or "").strip()
        suffix = f"：{what}" if what else ""
        lines.append(f"事件{segment.get('seq', len(lines) + 1)}：{who or '（未写人物）'}{suffix}")
    return "\n".join(lines) if lines else "（事件目录为空）"


def render_chapters(chapters) -> str:
    """渲染某个事件下的章节目录（二级拆分）。"""
    lines = []
    for chapter in chapters or []:
        who = str(chapter.get("who", "") or "").strip()
        what = str(chapter.get("what", "") or "").strip()
        suffix = f"：{what}" if what else ""
        lines.append(f"第{chapter.get('seq', len(lines) + 1)}章：{who or '（未写人物）'}{suffix}")
    return "\n".join(lines) if lines else "（尚未拆分章节）"


def render_book_outline(unit) -> str:
    """渲染「大事件 → 章节」两级目录，供章节设计与正文提示词使用。"""
    unit = unit or {}
    blocks = []
    for event in unit.get("segments", []):
        who = str(event.get("who", "") or "").strip()
        what = str(event.get("what", "") or "").strip()
        suffix = f"：{what}" if what else ""
        blocks.append(f"事件{event.get('seq', len(blocks) + 1)}：{who or '（未写人物）'}{suffix}")
        for chapter in event.get("chapters", []):
            ch_who = str(chapter.get("who", "") or "").strip()
            ch_what = str(chapter.get("what", "") or "").strip()
            ch_suffix = f"：{ch_what}" if ch_what else ""
            blocks.append(f"  第{chapter.get('seq', 1)}章：{ch_who or '（未写人物）'}{ch_suffix}")
    return "\n".join(blocks) if blocks else "（全书目录为空）"


def render_merged_route_text(segments) -> str:
    """把作者逐事件选定的章节方案，按全书顺序渲染成一条完整路线文本。"""
    lines = []
    chapter_no = 0
    for segment in segments or []:
        who = str(segment.get("who", "") or "").strip()
        what = str(segment.get("what", "") or "").strip()
        key = str(segment.get("chosen_chapter_plan", "") or "")
        plan = next(
            (item for item in (segment.get("chapter_options") or []) if str(item.get("key", "")) == key),
            None,
        )
        suffix = f"：{what}" if what else ""
        lines.append(f"事件{segment.get('seq', len(lines) + 1)}（{who or '（未写人物）'}{suffix}）· 选定方案 {key or '（未选）'}")
        if plan:
            title = str(plan.get("title", "") or "").strip()
            thesis = str(plan.get("thesis", "") or "").strip()
            note = "｜".join(part for part in (title, thesis) if part)
            lines.append(f"  方案说明：{note or '（无）'}")
            for step in plan.get("steps") or []:
                chapter_no += 1
                step_who = str(step.get("who", "") or "").strip()
                step_what = str(step.get("what", "") or "").strip()
                step_suffix = f"：{step_what}" if step_what else ""
                lines.append(
                    f"  第{chapter_no}章：{step_who or '（未写人物）'}{step_suffix}｜"
                    f"约{step.get('word_estimate', 0)}字 / 占全书{step.get('word_ratio', 0)}%"
                )
        else:
            lines.append("  （该事件还没有选定章节方案）")
    return "\n".join(lines) if lines else "（路线为空）"


def build_merge_route_prompt(
    *, world_text: str, cast_text: str, unit_goal: str, mainline_text: str = "",
    route_text: str, guidance: str = "", history=None, reference_text: str = "", target_words: int = 0,
    target_chapters: int = 0, words_per_chapter: int = 0
) -> str:
    """构建「校验并合并整条路线」的提示词，不请求模型。"""
    scale_notes = [f"全书约 {target_words or '未指定'} 字；合并后的总章数与各章字数比例要和这个体量匹配。"]
    if target_chapters:
        scale_notes.append(f"合并后的总章数要控制在大概 {target_chapters} 章：如果子事件比章数多，一个章节要能包进多个紧邻的子事件；如果子事件比章数少，再按冲突节奏适度拆细。")
    if words_per_chapter:
        scale_notes.append(f"每章目标约 {words_per_chapter} 字：word_estimate 以这个数值为目标，word_ratio 按「{words_per_chapter} ÷ {target_words or words_per_chapter}」等比例折算，全书各章合计 100%。")
    return f"""你是这个小说创作项目的总编剧兼责任编辑。作者已经从每个大事件的章节拆分方案里各选定了一套，组成了一条完整的全书路线。
现在由你来做最后的合并校对：查重复、补缺口，输出一条可以直接使用的完整路线。仍然不写正文。

【世界观】
{world_text or '（未填写）'}

【本书单主线】
{mainline_text or '（未设置）'}

【本书入选人物】
{cast_text}

【这本书要写什么】
{unit_goal or '（未填写）'}

【本书体量】
{chr(10).join(scale_notes)}

【参考文本（仅借鉴风格与结构，不得照抄）】
{reference_text or '（未上传参考文本）'}

【作者选定的完整路线（按事件顺序排列，章号是当前临时编号）】
{route_text}

【作者补充】
{guidance or '（无）'}

【近期的讨论与决定（可参考，但不是已发生的事实）】
{render_history(history)}

任务：
1. 通读整条路线，逐项检查并修正：
- 重复：同一情节点/同一功能被不同章节重复执行（合并重复章节）；
- 缺口：事件之间、章节之间接不上，缺过渡、缺动机、缺启动动作（补过渡章节）；
- 断档：主线推进中断、该收的钩子没回收、人物目标/时间线前后矛盾（补写或修正表述）；
- 冗余：对主线没有推进、可以并入相邻章节的碎片章节（并入，不额外扩写）。
2. 输出合并后的「一条」完整路线：所有章节从 1 开始连续编号，保持作者选定方案的顺序、信息释放顺序与写作意图，只在必要处动刀。
3. 每章仍是「谁 → 在这一章发生什么/要做什么」的行动线，并为每章起一个 4～12 字的章节名（title），不写场景描写、不写正文。多个子事件合进一章时，行动线写成连贯的一小节。
4. 每一章必须给出 sub_events 数组：这一章吸收进来的全部子事件，逐条保留原路线的「谁 + 做什么」和各自的 word_estimate / word_ratio，不许丢、不许合并叙述成模糊摘要；被拆成多章的，把子事件按新的章边界切分到对应章节的 sub_events 里。
5. 章节自身的 word_estimate 以「这一章 sub_events 的 word_estimate 之和」为准（如按每章目标字数做微调，同步调整 sub_events 的分配并保持合计相等）；word_ratio 按占全书比例折算。
6. 全书各章 word_ratio 合计 100；因合并章节而变化的比例同步调整；word_estimate 与全书体量匹配。
7. 严格只用入选人物与已知设定：不新增有姓名的角色，不替人物补写背景、秘密、年龄、职业。
8. fixed 里逐条列出你改了什么（去重/补充/修正各几条），没有改动的方面也要写一句「未发现重复/缺口」。

只返回 JSON（不要任何其他文字）：
{{"title":"合并路线的名字","summary":"校验与合并结论，2-4 句","fixed":["去重：…","补充：…","修正：…"],"chapters":[{{"seq":1,"title":"这一章的章节名","who":"人物名","what":"这一章发生什么/要做什么","word_estimate":3200,"word_ratio":5,"sub_events":[{{"who":"人物名","what":"原来的子事件行动线","word_estimate":1600,"word_ratio":2}}]}}]}}
""".strip()


def generate_merged_route(
    *, world_text: str, cast_text: str, unit_goal: str, mainline_text: str = "",
    route_text: str, guidance: str = "", history=None, reference_text: str = "",
    model: dict, prompt_override: str | None = None, target_words: int = 0,
    target_chapters: int = 0, words_per_chapter: int = 0
) -> dict:
    prompt = prompt_override if prompt_override is not None else build_merge_route_prompt(
        world_text=world_text, cast_text=cast_text, unit_goal=unit_goal, mainline_text=mainline_text,
        route_text=route_text, guidance=guidance, history=history, reference_text=reference_text,
        target_words=target_words, target_chapters=target_chapters, words_per_chapter=words_per_chapter,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label="全书路线：校验去重与合并", invoke_json=True)
    parsed = _extract_json(result)

    def _as_int(value):
        try:
            return int(float(str(value or "").replace("%", "").strip()))
        except (TypeError, ValueError):
            return 0

    chapters = []
    if isinstance(parsed, dict) and isinstance(parsed.get("chapters"), list):
        for index, chapter in enumerate(parsed.get("chapters"), start=1):
            if not isinstance(chapter, dict):
                continue
            sub_events = []
            for sub in (chapter.get("sub_events") if isinstance(chapter.get("sub_events"), list) else []):
                if not isinstance(sub, dict):
                    continue
                sub_events.append({
                    "who": str(sub.get("who", "") or "").strip(),
                    "what": str(sub.get("what", "") or "").strip(),
                    "word_estimate": _as_int(sub.get("word_estimate")),
                    "word_ratio": _as_int(sub.get("word_ratio")),
                })
            chapters.append({
                "seq": _as_int(chapter.get("seq")) or index,
                "title": str(chapter.get("title", "") or "").strip(),
                "who": str(chapter.get("who", "") or "").strip(),
                "what": str(chapter.get("what", "") or "").strip(),
                "word_estimate": _as_int(chapter.get("word_estimate")),
                "word_ratio": _as_int(chapter.get("word_ratio")),
                "sub_events": sub_events,
            })
    fixed = []
    if isinstance(parsed, dict) and isinstance(parsed.get("fixed"), list):
        fixed = [str(item).strip() for item in parsed.get("fixed") if str(item).strip()]
    if not chapters and result.strip():
        return {"merged_route": {
            "title": "合并失败：模型未返回结构化结果",
            "summary": result.strip(),
            "fixed": fixed,
            "chapters": [],
        }}
    return {"merged_route": {
        "title": str((parsed or {}).get("title", "") or "").strip(),
        "summary": str((parsed or {}).get("summary", "") or "").strip(),
        "fixed": fixed,
        "chapters": chapters,
    }}


def render_unit_snapshot(unit) -> str:
    """把整本书渲染成一份紧凑的「前情快照」。"""
    unit = unit or {}
    snapshot = unit.get("snapshot") if isinstance(unit.get("snapshot"), dict) else {}
    title = str(unit.get("title") or "").strip()
    seq = unit.get("seq", 1)
    summary = str(unit.get("unit_summary") or "").strip()
    beats = [str(item).strip() for item in (snapshot.get("beats") or []) if str(item).strip()]
    if not beats:
        beats = []
        for event in unit.get("segments") or []:
            for chapter in event.get("chapters") or []:
                who = str(chapter.get("who", "")).strip() or "（未写人物）"
                what = str(chapter.get("what", "")).strip() or "（未写动作）"
                beats.append(f"第{chapter.get('seq', len(beats) + 1)}章 {who} → {what}")
    states = [str(item).strip() for item in (snapshot.get("character_states") or []) if str(item).strip()]
    hooks = [str(item).strip() for item in (snapshot.get("hooks") or []) if str(item).strip()]

    lines = [f"全书 · {title or '未命名'}"]
    if not beats and summary:
        lines.append(f"全书摘要：{summary}")
    if beats:
        lines.append("各章结果：")
        lines.extend(f"- {beat}" for beat in beats)
    if states:
        lines.append("人物状态变化：")
        lines.extend(f"- {state}" for state in states)
    if hooks:
        lines.append("遗留钩子：")
        lines.extend(f"- {hook}" for hook in hooks)
    return "\n".join(lines)


def render_carry_over(units) -> str:
    """渲染承上上下文：只带选中的历史章节快照，而不带正文全文。"""
    blocks = [render_unit_snapshot(unit) for unit in units if isinstance(unit, dict)]
    return "\n\n".join(blocks).strip() if blocks else "（无）"


def render_graph_working_set(graph, cast_names, carried_unit_seqs, selected_fact_ids=None) -> str:
    """从知识图谱中切出与本章相关的工作集，避免把整张图谱塞进提示词。"""
    graph = graph if isinstance(graph, dict) else {}
    nodes = {item.get("id"): item for item in graph.get("nodes", []) if isinstance(item, dict) and item.get("id")}
    cast_names = [str(name).strip() for name in (cast_names or []) if str(name).strip()]
    cast_set = set(cast_names)
    carried_seqs = {int(seq) for seq in (carried_unit_seqs or []) if str(seq).strip().isdigit()}
    selected_fact_ids = {str(item) for item in (selected_fact_ids or []) if str(item).strip()}

    relevant_ids = {
        node_id
        for node_id, node in nodes.items()
        if str(node.get("name", "")).strip() in cast_set
    }

    lines = []
    node_lines = []
    for node_id, node in nodes.items():
        if node_id in relevant_ids:
            name = str(node.get("name", "")).strip()
            kind = str(node.get("kind", "")).strip()
            note = str(node.get("note", "")).strip()
            status = str(node.get("status", "")).strip()
            label = f"{name}（{kind}）" if kind else name
            node_lines.append("- " + label + (f"：{note}" if note else "") + (f" · {status}" if status else ""))
    if node_lines:
        lines.append("人物与关键要素：")
        lines.extend(node_lines)

    edge_lines = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source not in relevant_ids and target not in relevant_ids:
            continue
        source_name = str((nodes.get(source) or {}).get("name", source)).strip()
        target_name = str((nodes.get(target) or {}).get("name", target)).strip()
        relation_type = str(edge.get("type", "")).strip() or "有关联"
        note = str(edge.get("note", "")).strip()
        status = str(edge.get("status", "")).strip()
        text = f"{source_name} —{relation_type}→ {target_name}"
        if note or status:
            text += "（" + " · ".join(part for part in (note, status) if part) + "）"
        edge_lines.append("- " + text)
    if edge_lines:
        lines.append("人物关系：")
        lines.extend(edge_lines)

    event_lines = []
    for event in graph.get("events", []):
        if not isinstance(event, dict):
            continue
        unit_seq = event.get("unit_seq")
        if carried_seqs and (not str(unit_seq).strip().isdigit() or int(unit_seq) not in carried_seqs):
            continue
        title = str(event.get("title", "")).strip()
        summary = str(event.get("summary", "")).strip()
        if not title:
            continue
        prefix = f"第{unit_seq}章" if str(unit_seq).strip().isdigit() else "前情"
        text = f"{prefix} · {title}" + (f"：{summary}" if summary else "")
        event_lines.append("- " + text)
    if event_lines:
        lines.append("时间线：")
        lines.extend(event_lines)

    fact_lines = []
    for fact in graph.get("facts", []):
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("text", "")).strip()
        if not text:
            continue
        unit_seq = fact.get("unit_seq")
        if selected_fact_ids:
            if str(fact.get("id", "")).strip() not in selected_fact_ids:
                continue
        elif not carried_seqs:
            continue
        elif not str(unit_seq).strip().isdigit() or int(unit_seq) not in carried_seqs:
            continue
        prefix = f"第{unit_seq}章" if str(unit_seq).strip().isdigit() else "前情"
        entities = [str(entity).strip() for entity in (fact.get("entities") or []) if str(entity).strip()]
        text = f"{prefix} · {text}"
        if entities:
            text += "（涉及：" + "、".join(entities) + "）"
        fact_lines.append("- " + text)
    if fact_lines:
        lines.append("原子小事件（已选前情）：")
        lines.extend(fact_lines)

    return "\n".join(lines).strip() if lines else "（暂无图谱）"


def build_graph_draft_prompt(
    *, world_text: str = "", characters_text: str = "", units_text: str = "", graph_text: str = "", mainline_text: str = ""
) -> str:
    """构建图谱草稿提示词：从已有事实中整理人物关系与时间线。"""
    return f"""你是这个小说项目的世界观整理器。请把目前已经写过的章节整理成一张「知识图谱」草稿，用来在后续章节中保持事实一致。
只基于下面给出的信息，不要脑补新的人物、关系或事件。

【世界观】
{world_text or '（未填写）'}

【本书单主线（用于判断哪些关系与事件需要保留）】
{mainline_text or '（未设置）'}

【人物库】
{characters_text or '（空）'}

【已写完的章节】
{units_text or '（无）'}

【现有图谱（在此基础上补充修正，不要丢掉已经确认的事实）】
{graph_text or '（空）'}

只输出 JSON（不要任何其他文字）：
{{"nodes":[{{"id":"可用旧节点ID或留空","kind":"人物/地点/组织/物品/悬念","name":"名称","note":"目前已知信息","status":"活跃/离开/死亡/待定"}}],"edges":[{{"source":"节点名称","target":"节点名称","type":"关系（如 暗恋/敌对/合作/师徒）","note":"说明","status":"现在处于什么状态"}}],"events":[{{"unit_seq":1,"title":"事件名","summary":"一句话结果"}}]}}

要求：
- nodes 只收对后续剧情仍有影响的人物、地点、组织、关键物品与未解决悬念。
- edges 的 source/target 必须填节点名称，不要填编号。
- events 只记章节级大事，按发生顺序，不要写场景细节；每个写完的章节最多 3 条。
- 时间线要和章节内已经确认的目录与快照一致，不要发明剧情。""".strip()


def generate_graph_draft(
    *, world_text: str = "", characters_text: str = "", units_text: str = "", graph_text: str = "",
    mainline_text: str = "", model: dict, prompt_override: str | None = None
) -> dict:
    prompt = prompt_override if prompt_override is not None else build_graph_draft_prompt(
        world_text=world_text, characters_text=characters_text, units_text=units_text, graph_text=graph_text, mainline_text=mainline_text,
    )
    result = invoke_with_cleaning(_make_adapter(model), prompt, label="知识图谱：起草更新", invoke_json=True)
    parsed = _extract_json(result)
    if not isinstance(parsed, dict):
        parsed = {}

    name_to_id = {}
    used_ids = set()
    nodes = []
    for item in parsed.get("nodes", []) if isinstance(parsed.get("nodes", []), list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        node_id = str(item.get("id") or "").strip() or uuid.uuid4().hex[:8]
        while node_id in used_ids:
            node_id = uuid.uuid4().hex[:8]
        used_ids.add(node_id)
        name_to_id[name] = node_id
        nodes.append({
            "id": node_id,
            "kind": str(item.get("kind", "") or "").strip(),
            "name": name,
            "note": str(item.get("note", "") or "").strip(),
            "status": str(item.get("status", "") or "").strip(),
        })

    edges = []
    for item in parsed.get("edges", []) if isinstance(parsed.get("edges", []), list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "") or "").strip()
        target = str(item.get("target", "") or "").strip()
        if not source or not target:
            continue
        edges.append({
            "id": str(item.get("id") or uuid.uuid4().hex[:8]).strip(),
            "source": name_to_id.get(source, source),
            "target": name_to_id.get(target, target),
            "type": str(item.get("type", "") or "").strip(),
            "note": str(item.get("note", "") or "").strip(),
            "status": str(item.get("status", "") or "").strip(),
        })

    events = []
    for item in parsed.get("events", []) if isinstance(parsed.get("events", []), list) else []:
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

    return {"nodes": nodes, "edges": edges, "events": events}


def render_all_characters(characters) -> str:
    """渲染人物库全量资料，供跨页面提问时提供上下文。"""
    lines = []
    for item in characters or []:
        okitem = item if isinstance(item, dict) else {}
        name = str(okitem.get("name", "")).strip()
        if not name:
            continue
        fields = [str(okitem.get("gender", "")).strip(), str(okitem.get("age", "")).strip()]
        label = "，".join(part for part in fields if part)
        entry = name if not label else f"{name}（{label}）"
        notes = str(okitem.get("notes", "")).strip()
        lines.append(f"- {entry}" + (f"：{notes}" if notes else ""))
    return "\n".join(lines) if lines else "（人物库为空）"


def _render_unit_context(unit) -> str:
    unit = unit or {}
    target = str(unit.get("goal", "") or "").strip()
    cast = unit.get("cast", [])
    segments = unit.get("segments", [])
    lines = []
    if target:
        lines.append(f"这本书要写什么：{target}")
    if cast:
        lines.append(f"本书入选人物：{', '.join(str(item) for item in cast)}")
    if segments:
        lines.append("全书章节目录：\n" + render_book_outline(unit))
    return "\n".join(lines) if lines else "（尚未创建全书）"


def answer_question(
    *, world_text: str, characters_text: str, unit_context: str,
    selected_text: str, action: str, question: str, model: dict
) -> str:
    """针对作者在界面上选中的一段文字作答 / 改写 / 修改。

    只就选中文字及其直接上下文回答，不擅自改全书方向。
    """
    action_map = {
        "explain": "解释：这段文字在讲什么、它在这本书里起什么作用、为什么这样安排，尽量说人话。",
        "rewrite": "改写：保留原意，给出一个表达更清楚、更有画面感的版本，并简要说明你改了什么。",
        "revise": "修改：先指出这段可能存在的问题，再给出修改建议和可直接替换的新版本。",
        "answer": "回答：直接回答作者针对这段文字提出的问题。",
    }
    action_text = action_map.get(action, action_map["answer"])
    prompt = f"""你是这个小说创作项目的编剧助手。作者在界面上选中了一段文字，想就这段请教你。

【世界观】
{world_text or '（未填写）'}

【人物库】
{characters_text}

【当前全书上下文】
{unit_context}

【作者选中的文字】
--- 开始 ---
{selected_text or '（空）'}
--- 结束 ---

【本次要做的事】
{action_text}

【作者的问题或补充要求】
{question or '（无）'}

要求：
- 只围绕选中的文字回答，不要擅自扩展世界观、人物设定或未来剧情。
- 若涉及人物，只能使用人物库里已列出的人物资料，不得补写年龄、职业、秘密、背景。
- 若是改写或修改：先给简短说明，再用「——新版开始——」和「——新版结束——」包住可直接替换的文字。
- 若是解释或回答：直接说清楚即可，不要重写选中的文字。

仅输出结果，不要解释任务。
""".strip()
    return invoke_with_cleaning(_make_adapter(model), prompt, label="选中文字：问 AI / 修改").strip()
def render_unit_context(unit) -> str:
    """渲染全书上下文，供提问与跨页面对话使用。"""
    return _render_unit_context(unit)

_ACTION_TEXT = {
    "explain": "解释：这段文字在讲什么、它在这本书里起什么作用、为什么这样安排，尽量说人话。",
    "rewrite": "重写：保留原意，输出一段可以直接替换作者选中片段的新写法，并简要说明你改了什么。",
    "revise": "重写：先指出当前内容可能存在的问题，再输出一段可以直接替换作者选中片段的新写法。",
    "answer": "回答：直接回答作者针对这段文字提出的问题。",
}

def action_description(action: str) -> str:
    return _ACTION_TEXT.get(action, _ACTION_TEXT["answer"])


def is_tool_capable(model: dict) -> bool:
    """判断当前模型适配器是否支持真正的 LangChain bind_tools。"""
    fmt = str(model.get("interface_format", "OpenAI")).strip().lower()
    return fmt in {"openai", "deepseek", "ollama", "ml studio", "azure openai"}


def _format_tool_reply(reason: str, replacement: str) -> str:
    reason = str(reason or "").strip()
    replacement = str(replacement or "").strip()
    if replacement:
        return (reason + "\n\n" if reason else "") + "——新版开始——\n" + replacement + "\n——新版结束——"
    return reason or replacement


def build_conversation_messages(
    *, world_text: str, characters_text: str, unit_context: str,
    selected_text: str, action: str, messages, tool_available: bool = True, mainline_text: str = ""
) -> list:
    """把多轮提问/修改组装成真正的 messages 数组，而不是把历史塞进一段文本。"""
    history = []
    for message in (messages or [])[-40:]:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        role = str(message.get("role", "user") or "user").strip().lower()
        if role in {"user", "assistant"}:
            history.append({"role": role, "content": content})

    if tool_available:
        tool_note = (
            "当用户要求改写或修改时，请调用可用的 `rewrite_text` 工具提交重写建议："
            "`replacement` 只能包含作者选中的那一小段文字的新写法，不要扩写成整段、不要带上原文其余部分；"
            "`reason` 填修改理由。作者确认后才会生效。"
        )
    else:
        tool_note = (
            "当用户要求改写或修改时，先给简短说明，再用「——新版开始——」和「——新版结束——」"
            "包住作者选中片段的新写法，不要扩写成整段。"
        )

    system_prompt = f"""你是这个小说创作项目的编剧助手。作者在界面上选中了一段文字，并已经围绕它开始和你连续对话。

【世界观】
{world_text or '（未填写）'}

【本书单主线】
{mainline_text or '（未设置）'}

【人物库】
{characters_text}

【当前全书上下文】
{unit_context}

【作者最初选中的文字】
--- 开始 ---
{selected_text or '（空）'}
--- 结束 ---

【作者最初想让你做的事】
{action_description(action)}

要求：
- {tool_note}
- 若只是解释、追问或讨论：直接回答，不要重写选中文字。
- 只能使用人物库中已经列出的资料，不得补写年龄、职业、秘密、背景或关系。
- 只围绕选中文字及其直接上下文，不擅自扩展世界观、人物或后续剧情。

下面的对话历史按 role/user/assistant 正常传入，你作为助手回复最后一条用户消息。""".strip()
    return [{"role": "system", "content": system_prompt}, *history]


def conversation_reply(
    *, world_text: str, characters_text: str, unit_context: str,
    selected_text: str, action: str, messages, model: dict,
    mainline_text: str = "", prompt_override: str | None = None
) -> dict:
    """多轮回答：用户选中一段文字后，围绕它连续提问 / 修改，每次带上历史对话。"""
    adapter = _make_adapter(model)
    if prompt_override is not None:
        return {
            "result": invoke_with_cleaning(adapter, prompt_override, label="选中文字：多轮问 AI").strip(),
            "tool_calls": [],
        }
    tool_available = is_tool_capable(model)
    msg_list = build_conversation_messages(
        world_text=world_text, characters_text=characters_text, unit_context=unit_context,
        selected_text=selected_text, action=action, messages=messages,
        tool_available=tool_available, mainline_text=mainline_text,
    )
    if tool_available:
        content, tool_calls = invoke_messages_with_tools_cleaning(
            adapter, msg_list, [rewrite_text], label="选中文字：多轮问 AI"
        )
        for call in tool_calls:
            if call.get("name") == "rewrite_text":
                args = call.get("arguments") or {}
                replacement = str(args.get("replacement") or "").strip()
                result = _format_tool_reply(args.get("reason"), replacement)
                if result:
                    emit_llm_trace(
                        "tool_result",
                        uuid.uuid4().hex[:8],
                        json.dumps({
                            "tool": call.get("name"),
                            "arguments": args,
                            "return": replacement,
                        }, ensure_ascii=False, indent=2),
                        model=str(model.get("model_name", "")),
                        label="选中文字：多轮问 AI",
                    )
                    return {"result": result, "tool_calls": [call]}
        return {"result": content.strip(), "tool_calls": tool_calls}
    return {
        "result": invoke_messages_with_cleaning(adapter, msg_list, label="选中文字：多轮问 AI").strip(),
        "tool_calls": [],
    }



