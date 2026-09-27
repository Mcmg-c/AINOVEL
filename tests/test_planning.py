import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from novel_generator.chapter import _attach_scene_plan
from novel_generator.planning import (
    generate_chapter_exploration,
    generate_scene_plan,
    get_exploration_file,
    get_scene_plan_file,
)


class PlanningFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.book_dir = Path(self.temp_dir.name)
        (self.book_dir / "Novel_architecture.txt").write_text("总体架构", encoding="utf-8")
        (self.book_dir / "Novel_directory.txt").write_text(
            "第1章 重复站台\n本章定位：开篇\n核心作用：确认循环\n本章简述：从上车到确认站台重复",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("novel_generator.planning._make_adapter", return_value=object())
    @patch("novel_generator.planning.invoke_with_cleaning", return_value="## 路线1：广告破绽")
    def test_exploration_is_saved_per_chapter(self, invoke_mock, _adapter):
        result = generate_chapter_exploration(
            filepath=str(self.book_dir), chapter_number=1, user_guidance="",
            interface_format="OpenAI", api_key="", base_url="", model_name="test",
            temperature=0.7, max_tokens=1000,
        )
        self.assertIn("路线1", result)
        self.assertEqual(
            Path(get_exploration_file(str(self.book_dir), 1)).read_text(encoding="utf-8"),
            result,
        )
        prompt = invoke_mock.call_args.args[1]
        self.assertIn("完整白名单", prompt)
        self.assertIn("不得为了路线完整而补写", prompt)

    @patch("novel_generator.planning._make_adapter", return_value=object())
    @patch("novel_generator.planning.invoke_with_cleaning", return_value="## 场景1：登车")
    def test_scene_plan_is_saved_and_attached_to_draft_prompt(self, _invoke, _adapter):
        result = generate_scene_plan(
            filepath=str(self.book_dir), chapter_number=1, exploration_text="采用路线1",
            user_guidance="", interface_format="OpenAI", api_key="", base_url="",
            model_name="test", temperature=0.7, max_tokens=1000,
        )
        plan_path = Path(get_scene_plan_file(str(self.book_dir), 1))
        self.assertEqual(plan_path.read_text(encoding="utf-8"), result)
        final_prompt = _attach_scene_plan("正文提示词", str(self.book_dir), 1)
        self.assertIn("最高优先级", final_prompt)
        self.assertIn("场景1", final_prompt)


if __name__ == "__main__":
    unittest.main()
