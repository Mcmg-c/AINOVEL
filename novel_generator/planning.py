# -*- coding: utf-8 -*-
"""章节探索与场景方案生成。"""

import os

from chapter_directory_parser import get_chapter_info_from_blueprint
from llm_adapters import create_llm_adapter
from novel_generator.common import invoke_with_cleaning
from utils import read_file, clear_file_content, save_string_to_txt


def get_exploration_file(filepath: str, chapter_number: int) -> str:
    return os.path.join(
        filepath, "chapter_explorations", f"chapter_{chapter_number}_exploration.txt"
    )


def get_scene_plan_file(filepath: str, chapter_number: int) -> str:
    return os.path.join(filepath, "scene_plans", f"chapter_{chapter_number}_scene_plan.txt")


def _save_result(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clear_file_content(path)
    save_string_to_txt(content.strip(), path)


def _chapter_context(filepath: str, chapter_number: int) -> dict:
    blueprint = read_file(os.path.join(filepath, "Novel_directory.txt"))
    chapter_info = get_chapter_info_from_blueprint(blueprint, chapter_number)
    recent_chapters = []
    chapters_dir = os.path.join(filepath, "chapters")
    for number in range(max(1, chapter_number - 3), chapter_number):
        content = read_file(os.path.join(chapters_dir, f"chapter_{number}.txt")).strip()
        if content:
            recent_chapters.append(f"【第{number}章】\n{content[-5000:]}")
    return {
        "architecture": read_file(os.path.join(filepath, "Novel_architecture.txt")),
        "chapter_info": chapter_info,
        "global_summary": read_file(os.path.join(filepath, "global_summary.txt")),
        "character_state": read_file(os.path.join(filepath, "character_state.txt")),
        "recent_chapters": "\n\n".join(recent_chapters) or "（第一章或暂无已完成前文）",
    }


def _make_adapter(interface_format, api_key, base_url, model_name, temperature, max_tokens, timeout):
    return create_llm_adapter(
        interface_format=interface_format,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def generate_chapter_exploration(
    *, filepath: str, chapter_number: int, user_guidance: str,
    characters_involved: str = "", character_profiles: str = "",
    interface_format: str, api_key: str, base_url: str, model_name: str,
    temperature: float, max_tokens: int, timeout: int = 600,
) -> str:
    """为一章生成多条互不雷同的叙事路线，不提前替用户作最终选择。"""
    context = _chapter_context(filepath, chapter_number)
    if not context["architecture"].strip():
        raise ValueError("缺少 Novel_architecture.txt，请先生成小说架构")
    if not read_file(os.path.join(filepath, "Novel_directory.txt")).strip():
        raise ValueError("缺少 Novel_directory.txt，请先生成章节目录")
    info = context["chapter_info"]
    prompt = f"""
你正在参与小说编剧讨论室。当前任务不是写正文，也不是直接确定唯一方案，
而是为第{chapter_number}章进行 Chapter Exploration（章节创意探索）。

【全书架构】
{context['architecture']}

【当前章节蓝图】
标题：{info.get('chapter_title', '')}
定位：{info.get('chapter_role', '')}
核心作用：{info.get('chapter_purpose', '')}
悬念：{info.get('suspense_level', '')}
伏笔：{info.get('foreshadowing', '')}
反转强度：{info.get('plot_twist_level', '')}
概述：{info.get('chapter_summary', '')}

【全局摘要】
{context['global_summary'] or '（暂无）'}

【角色当前状态】
{context['character_state'] or '（暂无）'}

【固定人物名单】
{characters_involved or '（未指定）'}

【固定人物资料】
{character_profiles or '（未提供）'}

【最近正文】
{context['recent_chapters']}

【作者补充要求】
{user_guidance or '（无）'}

请提出 6 条真正不同的抵达路线。它们必须共享章节蓝图规定的起点、终点和功能，
但中间的发现顺序、人物行动、误判、阻碍、信息释放方式或情绪曲线应明显不同。
不要把同一方案换几个形容词重复输出，也不要写成完整正文。
严格使用固定人物的姓名、身份和关系；禁止擅自改名或新增有姓名的核心人物。
“角色当前状态/固定人物资料”是可用人物事实的完整白名单。未列出的年龄、职业、外貌、
技能、经历、秘密、关系、价值观、隐藏动机和最终状态一律保持未知，不得为了路线完整而补写。
候选路线中的行动与即时选择只是剧情提案，不是已经解锁的人物事实。

每条路线使用以下结构：
## 路线N：名称
- 核心构想
- 起点如何进入事件
- 中间因果链（至少4个连续节点）
- 关键选择或误判
- 转折与章节终点
- 读者体验
- 优点
- 风险/俗套点
- 对后续章节的影响

最后增加：
## 横向比较
比较六条路线在悬念、人物主动性、意外性、可写性和后续潜力上的差异。

## 可融合方向
给出至少3种跨路线融合建议，但不要替作者作最终选择。
""".strip()
    result = invoke_with_cleaning(
        _make_adapter(interface_format, api_key, base_url, model_name, temperature, max_tokens, timeout),
        prompt,
        label=f"第{chapter_number}章：章节探索",
    )
    if result.strip():
        _save_result(get_exploration_file(filepath, chapter_number), result)
    return result


def refine_chapter_exploration(
    *, filepath: str, chapter_number: int, existing_text: str, instruction: str,
    interface_format: str, api_key: str, base_url: str, model_name: str,
    temperature: float, max_tokens: int, timeout: int = 600,
) -> str:
    """根据作者追问、删改或融合要求重写当前探索稿。"""
    prompt = f"""
你正在继续一场章节编剧讨论。以下是第{chapter_number}章现有的 Chapter Exploration：

{existing_text}

作者的新要求：
{instruction}

请根据要求返回一份完整、可替换原文件的新版章节探索。
保留仍有价值的路线，明确落实作者要求；如果要求融合路线，请输出融合后的候选，
但仍然不要写正文或 Scene Plan。
""".strip()
    result = invoke_with_cleaning(
        _make_adapter(interface_format, api_key, base_url, model_name, temperature, max_tokens, timeout),
        prompt,
        label=f"第{chapter_number}章：继续探索",
    )
    if result.strip():
        _save_result(get_exploration_file(filepath, chapter_number), result)
    return result


def generate_scene_plan(
    *, filepath: str, chapter_number: int, exploration_text: str, user_guidance: str,
    characters_involved: str = "", character_profiles: str = "",
    interface_format: str, api_key: str, base_url: str, model_name: str,
    temperature: float, max_tokens: int, timeout: int = 600,
) -> str:
    """将用户编辑确认后的探索结果落实为可执行的逐场景方案。"""
    if not exploration_text.strip():
        raise ValueError("章节探索内容为空，无法生成 Scene Plan")
    context = _chapter_context(filepath, chapter_number)
    info = context["chapter_info"]
    prompt = f"""
请把第{chapter_number}章已经讨论、筛选或融合后的 Chapter Exploration，
转化为 Scene Plan。不要写小说正文，不要重新罗列大量候选路线。

【章节蓝图约束】
标题：{info.get('chapter_title', '')}
定位：{info.get('chapter_role', '')}
核心作用：{info.get('chapter_purpose', '')}
章节概述：{info.get('chapter_summary', '')}
伏笔：{info.get('foreshadowing', '')}

【作者确认/编辑后的探索内容】
{exploration_text}

【固定人物名单】
{characters_involved or '（未指定）'}

【固定人物资料】
{character_profiles or '（未提供）'}

【作者补充要求】
{user_guidance or '（无）'}

输出一份能直接指导正文创作的方案。先写：
## 本章采用路线
用一段话明确本章最终采用的因果路线，以及舍弃了哪些干扰方向。

随后拆成 4—8 个场景。每个场景必须包含：
## 场景N：名称
- 时间与地点
- 出场人物
- 场景目标
- 人物当下目标
- 阻碍/冲突
- 行动与反应的因果链
- 新增或隐藏的信息
- 情绪变化
- 场景结束时发生的状态变化
- 与下一场的钩子

最后写：
## 节奏检查
## 伏笔与回收检查
## 连贯性风险

场景之间必须由因果关系推动，不能只是“然后又发生一件事”。
严格沿用固定人物姓名、身份和关系，不得擅自改名或新增有姓名的核心人物。
“固定人物资料”是可用人物事实的完整白名单；未列出的人物信息保持未知，不得在 Scene Plan
中补写年龄、职业、外貌、技能、经历、秘密、关系、价值观、隐藏动机或最终状态。
方案中的行动、反应和场景结束变化只是待创作内容，不得表述为人物原本就有的设定。
""".strip()
    result = invoke_with_cleaning(
        _make_adapter(interface_format, api_key, base_url, model_name, temperature, max_tokens, timeout),
        prompt,
        label=f"第{chapter_number}章：Scene Plan",
    )
    if result.strip():
        _save_result(get_scene_plan_file(filepath, chapter_number), result)
    return result
