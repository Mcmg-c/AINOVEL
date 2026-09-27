# -*- coding: utf-8 -*-
"""作者输入层与按章覆盖的测试。"""

import tempfile
import unittest
from pathlib import Path

from novel_generator.character_pool import (
    initialize_character_pool,
    load_character_pool,
    render_character_context_parts,
    render_full_character_context,
    update_fact_selection,
)
from novel_generator.story_inputs import (
    build_architecture_view,
    has_unit_plot,
    is_world_setting_filled,
    load_unit_plot,
    load_world_setting,
    render_world_setting,
    save_unit_plot,
    save_world_setting,
    write_architecture_view,
)


class StoryInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.book = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_world_setting_round_trip(self):
        self.assertFalse(is_world_setting_filled(str(self.book)))
        save_world_setting(str(self.book), {
            "premise": "旧城区仍在按十年前的地址投递",
            "era_place": "当代南方老城",
            "rules": "无超自然",
            "tone": "克制",
        })
        self.assertTrue(is_world_setting_filled(str(self.book)))
        world = load_world_setting(str(self.book))
        self.assertEqual("当代南方老城", world["era_place"])
        self.assertIn("时代与地点：当代南方老城", render_world_setting(str(self.book)))

    def test_blank_world_field_renders_placeholder(self):
        save_world_setting(str(self.book), {"premise": "只有前提"})
        text = render_world_setting(str(self.book))
        self.assertIn("只有前提", text)
        self.assertIn("（未填写）", text)

    def test_unit_plot_round_trip(self):
        self.assertFalse(has_unit_plot(str(self.book)))
        save_unit_plot(str(self.book), "  一个反复错投的旧包裹  ")
        self.assertTrue(has_unit_plot(str(self.book)))
        self.assertEqual("一个反复错投的旧包裹", load_unit_plot(str(self.book)))

    def test_architecture_view_is_assembled_from_sources(self):
        save_world_setting(str(self.book), {"premise": "旧城代收点"})
        save_unit_plot(str(self.book), "错投的旧包裹")
        view = write_architecture_view(
            str(self.book), character_context="许知遥：\n- [基础][物品] 旧雨伞",
            number_of_chapters=5, word_number=2300,
        )
        self.assertIn("世界观", view)
        self.assertIn("旧城代收点", view)
        self.assertIn("当前单元剧情", view)
        self.assertIn("错投的旧包裹", view)
        self.assertIn("旧雨伞", view)
        self.assertIn("约5章", view)
        self.assertEqual(
            view, (self.book / "Novel_architecture.txt").read_text(encoding="utf-8")
        )
        self.assertEqual(view, build_architecture_view(
            str(self.book), character_context="许知遥：\n- [基础][物品] 旧雨伞",
            number_of_chapters=5, word_number=2300,
        ))


class PerChapterSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.book = Path(self.temp_dir.name)
        role_dir = self.book / "角色库"
        role_dir.mkdir()
        (role_dir / "许知遥.txt").write_text("性别：男\n住在旧屋", encoding="utf-8")
        initialize_character_pool(str(self.book), "许知遥")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _fact_id(self, index):
        return load_character_pool(str(self.book))["characters"]["许知遥"]["base_facts"][index]["id"]

    def test_chapter_override_does_not_change_global_default(self):
        hidden = self._fact_id(1)
        update_fact_selection(str(self.book), {("许知遥", hidden): False}, chapter=5)

        pool = load_character_pool(str(self.book))
        fact = pool["characters"]["许知遥"]["base_facts"][1]
        self.assertTrue(fact["selected"])
        self.assertEqual({"5": False}, fact["overrides"])

        fact_context, _ = render_character_context_parts(str(self.book), chapter=4)
        self.assertIn("住在旧屋", fact_context)
        fact_context, _ = render_character_context_parts(str(self.book), chapter=5)
        self.assertNotIn("住在旧屋", fact_context)

    def test_global_selection_still_writes_selected(self):
        target = self._fact_id(0)
        update_fact_selection(str(self.book), {("许知遥", target): False})
        fact = load_character_pool(str(self.book))["characters"]["许知遥"]["base_facts"][0]
        self.assertFalse(fact["selected"])
        self.assertNotIn("overrides", fact)

    def test_matching_chapter_value_stores_no_redundant_override(self):
        """只有与全局默认不同的条目才记录覆盖。"""
        first = self._fact_id(0)
        second = self._fact_id(1)
        # first 保持 True（与默认一致），second 改为 False（例外）
        update_fact_selection(
            str(self.book),
            {("许知遥", first): True, ("许知遥", second): False},
            chapter=7,
        )
        facts = load_character_pool(str(self.book))["characters"]["许知遥"]["base_facts"]
        self.assertNotIn("overrides", facts[0])
        self.assertEqual({"7": False}, facts[1]["overrides"])

    def test_resetting_to_default_clears_the_override(self):
        target = self._fact_id(1)
        update_fact_selection(str(self.book), {("许知遥", target): False}, chapter=7)
        update_fact_selection(str(self.book), {("许知遥", target): True}, chapter=7)
        fact = load_character_pool(str(self.book))["characters"]["许知遥"]["base_facts"][1]
        self.assertNotIn("overrides", fact)

    def test_full_context_ignores_selection_for_ideation(self):
        hidden = self._fact_id(1)
        update_fact_selection(str(self.book), {("许知遥", hidden): False})
        full = render_full_character_context(str(self.book))
        # 推演阶段给全量事实，避免相关特征被检索静默丢掉
        self.assertIn("住在旧屋", full)

    def test_fact_and_trait_layers_do_not_overlap(self):
        pool = load_character_pool(str(self.book))
        pool["characters"]["许知遥"]["unlocked_facts"].append({
            "id": "trait:ch3-abc",
            "text": "被追问时只回单字",
            "category": "行为模式",
            "layer": "trait",
            "chapter": 3,
            "evidence": "……随你。",
            "selected": True,
        })
        save = __import__("novel_generator.character_pool", fromlist=["save_character_pool"])
        save.save_character_pool(str(self.book), pool)

        facts, traits = render_character_context_parts(str(self.book))
        self.assertIn("性别：男", facts)
        self.assertNotIn("被追问时只回单字", facts)
        self.assertIn("被追问时只回单字", traits)
        self.assertNotIn("性别：男", traits)


if __name__ == "__main__":
    unittest.main()
