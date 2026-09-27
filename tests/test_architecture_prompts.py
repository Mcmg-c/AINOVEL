import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from novel_generator.architecture import Novel_architecture_generate
from novel_generator.story_inputs import save_unit_plot, save_world_setting


class ArchitecturePromptTest(unittest.TestCase):
    @patch("novel_generator.architecture.create_llm_adapter", return_value=object())
    @patch("novel_generator.architecture.invoke_with_cleaning")
    def test_uses_author_inputs_and_fixed_cast(self, invoke_mock, _adapter_mock):
        """世界观与单元剧情来自作者输入，模型只负责推导事件架构。"""
        prompts = []

        def respond(_adapter, prompt, **_kwargs):
            prompts.append(prompt)
            return "生成结果"

        invoke_mock.side_effect = respond
        with tempfile.TemporaryDirectory() as temp_dir:
            book = Path(temp_dir)
            (book / "unit_plot.txt").write_text(
                "一个反复错投的旧包裹，让两个固定人物看见一段被忽略的等待。",
                encoding="utf-8",
            )
            save_world_setting(str(book), {
                "premise": "旧城区的代收点仍在按十年前的地址投递。",
                "era_place": "当代，南方老城区",
                "rules": "没有超自然设定",
                "tone": "克制、温情",
            })

            Novel_architecture_generate(
                interface_format="OpenAI",
                api_key="",
                base_url="",
                llm_model="test",
                topic="失物与等待",
                genre="都市温情",
                number_of_chapters=5,
                word_number=2300,
                filepath=temp_dir,
                characters_involved="许知遥、林晚",
                character_profiles="许知遥：地铁司机\n林晚：心理咨询师",
            )
            architecture = (book / "Novel_architecture.txt").read_text(encoding="utf-8")

        # 未提供 user_guidance 时，人物动力这步不再执行
        self.assertEqual(1, len(prompts))
        self.assertIn("旧城区的代收点", prompts[0])
        self.assertIn("许知遥、林晚", prompts[0])
        # 架构视图由作者输入拼装
        self.assertIn("世界观", architecture)
        self.assertIn("当前单元剧情", architecture)
        self.assertIn("反复错投的旧包裹", architecture)

    @patch("novel_generator.architecture.create_llm_adapter", return_value=object())
    def test_requires_author_inputs_before_generating(self, _adapter_mock):
        """缺世界观或单元剧情时给出明确错误，而不是静默失败。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            book = Path(temp_dir)
            with self.assertRaises(ValueError) as ctx:
                Novel_architecture_generate(
                    interface_format="OpenAI", api_key="", base_url="",
                    llm_model="test", topic="t", genre="g", number_of_chapters=5,
                    word_number=2300, filepath=temp_dir,
                )
            self.assertIn("单元剧情", str(ctx.exception))

            save_unit_plot(str(book), "单元剧情已填写")
            with self.assertRaises(ValueError) as ctx:
                Novel_architecture_generate(
                    interface_format="OpenAI", api_key="", base_url="",
                    llm_model="test", topic="t", genre="g", number_of_chapters=5,
                    word_number=2300, filepath=temp_dir,
                )
            self.assertIn("世界观", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
