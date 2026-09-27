import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from novel_generator.character_pool import (
    add_author_fact,
    extract_and_merge_chapter_facts,
    initialize_character_pool,
    load_character_pool,
    render_selected_character_context,
    update_fact_selection,
)


class CharacterPoolTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.book_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_blank_character_stays_blank(self):
        pool = initialize_character_pool(str(self.book_dir), "许知遥")

        self.assertEqual([], pool["characters"]["许知遥"]["base_facts"])
        self.assertEqual("许知遥：\n- 暂无已选择信息", render_selected_character_context(str(self.book_dir)))

    def test_role_file_lines_become_user_base_facts(self):
        role_dir = self.book_dir / "角色库"
        role_dir.mkdir()
        (role_dir / "许知遥.txt").write_text(
            "姓名：许知遥\n- 性别：男\n- 恋人五年前失踪\n", encoding="utf-8"
        )

        pool = initialize_character_pool(str(self.book_dir), "许知遥")
        facts = pool["characters"]["许知遥"]["base_facts"]

        self.assertEqual(["性别：男", "恋人五年前失踪"], [fact["text"] for fact in facts])
        self.assertTrue(all(fact["source"] == "user" for fact in facts))

    def test_unchecked_fact_is_excluded_from_context(self):
        role_dir = self.book_dir / "角色库"
        role_dir.mkdir()
        (role_dir / "许知遥.txt").write_text("性别：男\n住在旧屋", encoding="utf-8")
        pool = initialize_character_pool(str(self.book_dir), "许知遥")
        hidden_id = pool["characters"]["许知遥"]["base_facts"][1]["id"]

        update_fact_selection(str(self.book_dir), {("许知遥", hidden_id): False})
        context = render_selected_character_context(str(self.book_dir))

        self.assertIn("性别：男", context)
        self.assertNotIn("住在旧屋", context)

    def test_author_fact_defaults_to_unchecked(self):
        initialize_character_pool(str(self.book_dir), "许知遥")
        add_author_fact(str(self.book_dir), "许知遥", "结局：搬离旧屋")

        pool = load_character_pool(str(self.book_dir))
        fact = pool["characters"]["许知遥"]["unlocked_facts"][0]
        self.assertFalse(fact["selected"])
        self.assertNotIn("搬离旧屋", render_selected_character_context(str(self.book_dir)))

    @patch("novel_generator.character_pool.create_llm_adapter", return_value=object())
    @patch("novel_generator.character_pool.invoke_with_cleaning")
    def test_extraction_requires_allowed_character_category_and_real_evidence(
        self, invoke_mock, _adapter_mock
    ):
        initialize_character_pool(str(self.book_dir), "许知遥")
        invoke_mock.return_value = json.dumps(
            {
                "facts": [
                    {
                        "character": "许知遥",
                        "category": "物品",
                        "text": "拿着旧雨伞",
                        "evidence": "他把旧雨伞靠在门边",
                    },
                    {
                        "character": "陌生人",
                        "category": "物品",
                        "text": "拿着钥匙",
                        "evidence": "陌生人拿着钥匙",
                    },
                    {
                        "character": "许知遥",
                        "category": "性格推测",
                        "text": "天生多疑",
                        "evidence": "他把旧雨伞靠在门边",
                    },
                    {
                        "character": "许知遥",
                        "category": "状态",
                        "text": "准备远行",
                        "evidence": "正文里不存在的证据",
                    },
                ]
            },
            ensure_ascii=False,
        )

        merged = extract_and_merge_chapter_facts(
            filepath=str(self.book_dir),
            chapter_number=1,
            chapter_text="回家后，他把旧雨伞靠在门边。",
            interface_format="OpenAI",
            api_key="",
            base_url="",
            model_name="test",
            temperature=0.7,
            max_tokens=1000,
        )

        self.assertEqual(1, len(merged))
        self.assertEqual("拿着旧雨伞", merged[0]["text"])
        self.assertEqual(1, merged[0]["chapter"])


if __name__ == "__main__":
    unittest.main()
