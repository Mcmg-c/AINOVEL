# -*- coding: utf-8 -*-
"""作者输入层：世界观、单元剧情，以及供提示词使用的架构视图组装。

设计原则：世界观与人物设定是纯发明，没有可校验的证据，因此由作者持有，
不由模型生成。AI 只在既定约束内做组合与推演。
"""

import json
import os
import tempfile

from llm_adapters import create_llm_adapter
from novel_generator.common import invoke_with_cleaning
from utils import read_file

WORLD_FILENAME = "world_setting.json"
UNIT_PLOT_FILENAME = "unit_plot.txt"
ARCHITECTURE_FILENAME = "Novel_architecture.txt"

# 结构化字段：空文本框对世界观不友好，拆成字段能保证风格一致。
WORLD_FIELDS = (
    ("premise", "一句话世界前提"),
    ("era_place", "时代与地点"),
    ("rules", "规则与限制"),
    ("tone", "基调"),
)

_PLACEHOLDER = "（未填写）"


def _atomic_write(path: str, content: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_world_setting(filepath: str) -> dict:
    path = os.path.join(filepath, WORLD_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def save_world_setting(filepath: str, data: dict) -> None:
    payload = {key: str((data or {}).get(key, "")).strip() for key, _label in WORLD_FIELDS}
    _atomic_write(os.path.join(filepath, WORLD_FILENAME), json.dumps(payload, ensure_ascii=False, indent=2))


def is_world_setting_filled(filepath: str) -> bool:
    data = load_world_setting(filepath)
    return any(str(data.get(key, "")).strip() for key, _label in WORLD_FIELDS)


def render_world_setting(filepath: str) -> str:
    data = load_world_setting(filepath)
    lines = []
    for key, label in WORLD_FIELDS:
        value = str(data.get(key, "")).strip()
        lines.append(f"{label}：{value or _PLACEHOLDER}")
    return "\n".join(lines)


def load_unit_plot(filepath: str) -> str:
    return read_file(os.path.join(filepath, UNIT_PLOT_FILENAME))


def save_unit_plot(filepath: str, text: str) -> None:
    _atomic_write(os.path.join(filepath, UNIT_PLOT_FILENAME), (text or "").strip())


def has_unit_plot(filepath: str) -> bool:
    return bool(load_unit_plot(filepath).strip())


def build_architecture_view(filepath: str, character_context: str = "", number_of_chapters: int = 0, word_number: int = 0) -> str:
    """把各真源拼装成只读的上下文视图，不做手工编辑。"""
    sections = ["#=== 0) 当前单元设定 ==="]
    if number_of_chapters or word_number:
        sections.append(f"篇幅：约{number_of_chapters}章（每章{word_number}字）")
    sections.append(f"世界观：\n{render_world_setting(filepath)}")
    sections.append("#=== 1) 固定人物资料 ===\n" + (character_context.strip() or "（未选择任何角色事实）"))
    sections.append("#=== 2) 当前单元剧情 ===\n" + (load_unit_plot(filepath).strip() or _PLACEHOLDER))
    return "\n\n".join(sections).strip()


def write_architecture_view(filepath: str, character_context: str = "", number_of_chapters: int = 0, word_number: int = 0) -> str:
    content = build_architecture_view(filepath, character_context, number_of_chapters, word_number)
    _atomic_write(os.path.join(filepath, ARCHITECTURE_FILENAME), content)
    return content


def generate_unit_plot_options(
    *, filepath: str, seed: str, world_setting: str, character_context: str,
    genre: str, user_guidance: str,
    interface_format: str, api_key: str, base_url: str, model_name: str,
    temperature: float, max_tokens: int, timeout: int = 600,
) -> str:
    """就当前单元推演多条剧情走向，不替作者作最终选择。

    推演阶段把人物池的全部事实交给模型，让它自行挑选相关项，
    不做检索，避免语义不相关但剧情相关的特征被静默丢掉。
    """
    prompt = f"""
你正在参与小说单元剧的编剧讨论室。当前任务是剧情推演，不是写正文，
也不是确定唯一方案。

【世界观（作者已确定）】
{world_setting or _PLACEHOLDER}

【人物信息池（作者已确定的全部事实，请自行挑选相关项）】
{character_context or '（暂无人物信息）'}

【类型】
{genre or '（未指定）'}

【作者的初步构想】
{seed or '（作者未提供，请自行提出几个方向）'}

【作者补充要求】
{user_guidance or '（无）'}

请提出 4 条真正不同的单元剧情走向。每条都要给出：
1. 单元命题：这个单元到底在讲什么，1—2 句话。
2. 单元事件：谁遇到了什么具体问题，事件如何闭环。
3. 核心悬念：读者会被什么留住。
4. 人物使用：用到信息池中的哪些人物、哪些事实（只允许引用已列出的事实）。
5. 情感落点：本单元结束时应抵达的情感变化。

要求：
- 四条走向的驱动事件、人物关系张力、悬念类型应明显不同，不要只换形容词。
- 严格使用信息池中已列出的人物姓名与事实；信息池未列出的人物信息一律保持未知，
  不得补写年龄、职业、外貌、技能、经历、秘密、价值观或最终状态。
- 不要写正文，不要拆分章节。
- 最后给出「选择建议」，说明每条走向适合什么样的后续展开，但不替作者决定。

仅输出推演结果，不要解释任务。
""".strip()

    adapter = create_llm_adapter(
        interface_format=interface_format, api_key=api_key, base_url=base_url,
        model_name=model_name, temperature=temperature, max_tokens=max_tokens, timeout=timeout,
    )
    return invoke_with_cleaning(adapter, prompt, label="单元剧情：推演走向")
