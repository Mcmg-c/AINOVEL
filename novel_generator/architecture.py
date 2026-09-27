#novel_generator/architecture.py
# -*- coding: utf-8 -*-
"""
小说总体架构生成（Novel_architecture_generate 及相关辅助函数）
"""
import os
import json
import hashlib
import logging
import traceback
from novel_generator.common import invoke_with_cleaning
from novel_generator.story_inputs import (
    is_world_setting_filled,
    render_world_setting,
    write_architecture_view,
)
from llm_adapters import create_llm_adapter
import prompt_definitions
logging.basicConfig(
    filename='app.log',      # 日志文件名
    filemode='a',            # 追加模式（'w' 会覆盖）
    level=logging.INFO,      # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
from utils import clear_file_content, read_file, save_string_to_txt

def load_partial_architecture_data(filepath: str) -> dict:
    """
    从 filepath 下的 partial_architecture.json 读取已有的阶段性数据。
    如果文件不存在或无法解析，返回空 dict。
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    if not os.path.exists(partial_file):
        return {}
    try:
        with open(partial_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.warning(f"Failed to load partial_architecture.json: {e}")
        return {}

def save_partial_architecture_data(filepath: str, data: dict):
    """
    将阶段性数据写入 partial_architecture.json。
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    try:
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save partial_architecture.json: {e}")

def Novel_architecture_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    topic: str,
    genre: str,
    number_of_chapters: int,
    word_number: int,
    filepath: str,
    user_guidance: str = "",  # 新增参数
    characters_involved: str = "",
    character_profiles: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 600
) -> None:
    """
    架构视图组装：

      1. 单元剧情：读取作者提供的 unit_plot.txt（自填或从 AI 推演选项中选定）
      2. 人物动力：可选补写，仅在提供 user_guidance 时执行
      3. 世界观：读取作者填写的 world_setting.json
      4. 单元事件架构：在以上约束下推导

    世界观与人物设定属于纯发明，没有可校验的证据，因此由作者持有，不由模型生成。
    若在中间步骤报错，已生成的内容写入 partial_architecture.json，下次可从该步骤继续。
    最终把各真源拼装成只读的 Novel_architecture.txt（上下文视图，不手工编辑）。

    角色事实由 character_pool.json 中的作者输入与定稿提取决定。
    """
    os.makedirs(filepath, exist_ok=True)
    partial_data = load_partial_architecture_data(filepath)
    input_signature = hashlib.sha256(
        json.dumps(
            {
                "version": 2,
                "topic": topic,
                "genre": genre,
                "number_of_chapters": number_of_chapters,
                "word_number": word_number,
                "user_guidance": user_guidance,
                "characters_involved": characters_involved,
                "character_profiles": character_profiles,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if partial_data.get("_input_signature") != input_signature:
        partial_data = {"_input_signature": input_signature}
        save_partial_architecture_data(filepath, partial_data)
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    # Step1: 单元剧情由作者持有（自填或从 AI 推演选项中选定），不由模型发明。
    if "core_seed_result" not in partial_data:
        logging.info("Step1: Reading author-provided unit plot ...")
        core_seed_result = read_file(os.path.join(filepath, "unit_plot.txt")).strip()
        if not core_seed_result:
            logging.warning("unit_plot.txt is empty; author must provide the unit plot first.")
            raise ValueError("请先填写或选定单元剧情，再生成架构视图")
        partial_data["core_seed_result"] = core_seed_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step1 already done. Skipping...")

    # Step2: 人物动力属于可选补写，不再是必跑链路。
    # 人物设定由作者持有，这里只补一份"本单元行动空间"的推演稿。
    if user_guidance.strip() and "character_dynamics_result" not in partial_data:
        logging.info("Step2: Generating character_dynamics_prompt (optional) ...")
        prompt_character = prompt_definitions.character_dynamics_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            user_guidance=user_guidance,
            characters_involved=characters_involved or "（未指定）",
            character_profiles=character_profiles or "（未提供）",
        )
        character_dynamics_result = invoke_with_cleaning(
            llm_adapter, prompt_character, label="小说架构：角色动力学"
        )
        if character_dynamics_result.strip():
            partial_data["character_dynamics_result"] = character_dynamics_result
            save_partial_architecture_data(filepath, partial_data)
        else:
            logging.warning("character_dynamics_prompt returned empty; skipping optional step.")
    elif "character_dynamics_result" not in partial_data:
        logging.info("Step2 skipped (no user_guidance provided).")

    # Step3: 世界观由作者持有（结构化字段），不由模型发明。
    if "world_building_result" not in partial_data:
        logging.info("Step3: Reading author-provided world setting ...")
        if not is_world_setting_filled(filepath):
            logging.warning("world_setting.json is empty; author must fill the world setting.")
            raise ValueError("请先填写世界观，再生成架构视图")
        partial_data["world_building_result"] = render_world_setting(filepath)
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step3 already done. Skipping...")

    # Step4: 当前单元事件架构（在作者设定的约束下做推导）
    if "plot_arch_result" not in partial_data:
        logging.info("Step4: Generating plot_architecture_prompt ...")
        prompt_plot = prompt_definitions.plot_architecture_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            character_dynamics=partial_data.get("character_dynamics_result", "（未提供）").strip(),
            world_building=partial_data["world_building_result"].strip(),
            user_guidance=user_guidance,
            characters_involved=characters_involved or "（未指定）",
            character_profiles=character_profiles or "（未提供）",
            number_of_chapters=number_of_chapters,
        )
        plot_arch_result = invoke_with_cleaning(llm_adapter, prompt_plot, label="小说架构：当前单元事件架构")
        if not plot_arch_result.strip():
            logging.warning("plot_architecture_prompt generation failed.")
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["plot_arch_result"] = plot_arch_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step4 already done. Skipping...")

    final_content = write_architecture_view(
        filepath,
        character_context=character_profiles,
        number_of_chapters=number_of_chapters,
        word_number=word_number,
    )
    logging.info("Novel_architecture.txt (context view) rebuilt from author inputs.")

    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    logging.info("Novel_architecture.txt has been generated successfully.")

    partial_arch_file = os.path.join(filepath, "partial_architecture.json")
    if os.path.exists(partial_arch_file):
        os.remove(partial_arch_file)
        logging.info("partial_architecture.json removed (all steps completed).")
