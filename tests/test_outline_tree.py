# -*- coding: utf-8 -*-
import os
import json
import tempfile
import unittest
from unittest import mock

from novel_generator import book_store
from novel_generator import outline_tree
from novel_generator import unit_generation
from novel_generator.unit_generation import _extract_json


class ExtractJsonTest(unittest.TestCase):
    def test_extracts_object_with_surrounding_text(self):
        text = '这是方案：\n{"plans":[{"key":"A","steps":[{"who":"甲","what":"做"}]}]}\n以上就是全部方案。'
        self.assertEqual(_extract_json(text)["plans"][0]["key"], "A")

    def test_extracts_object_after_markdown_fence(self):
        text = '```json\n{"plans":[{"key":"B"}]}\n```'
        self.assertEqual(_extract_json(text)["plans"][0]["key"], "B")

    def test_returns_none_without_json(self):
        self.assertIsNone(_extract_json("没有任何 JSON 内容"))


class ExpandOutlineTreeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project = str(self.tmp.name)
        os.makedirs(self.project, exist_ok=True)
        book_store.save_world(self.project, {
            "genre": "悬疑", "premise": "前提", "era_place": "现代都市", "rules": "规则", "tone": "冷峻",
        })
        characters = book_store.save_characters(self.project, [
            {"name": "林岳", "gender": "男", "age": "30", "notes": ""},
            {"name": "顾衍", "gender": "男", "age": "24", "notes": ""},
        ])
        book = book_store.create_unit(self.project, {
            "title": "测试书",
            "goal": "写一个测试主线",
            "cast": [item["id"] for item in characters],
        })
        self.unit_id = book["current"]

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_generators(self):
        directory_mock = mock.patch.object(unit_generation, "generate_directory_options", return_value={
            "directory_options": [
                {"key": "A", "title": "路线A", "thesis": "", "steps": [
                    {"who": "林岳", "what": "事件一"},
                    {"who": "顾衍", "what": "事件二"},
                ]},
                {"key": "B", "title": "路线B", "thesis": "", "steps": [
                    {"who": "林岳", "what": "事件三"},
                ]},
            ],
        }).start()
        chapter_mock = mock.patch.object(unit_generation, "generate_event_chapter_options", return_value={
            "chapter_options": [
                {"key": "x1", "title": "章节方案1", "steps": [
                    {"who": "林岳", "what": "第一章"},
                    {"who": "顾衍", "what": "第二章"},
                ]},
                {"key": "x2", "title": "章节方案2", "steps": [
                    {"who": "顾衍", "what": "另一章"},
                ]},
            ],
        }).start()
        design_mock = mock.patch.object(unit_generation, "generate_segment_design_options", return_value={
            "options": [{"key": "d1", "title": "设计", "summary": "摘要", "content": "内容"}],
        }).start()
        self.addCleanup(mock.patch.stopall)
        return directory_mock, chapter_mock, design_mock

    def test_expands_all_branches_and_persists(self):
        progress = []
        _, chapter_mock, design_mock = self._patch_generators()

        book = outline_tree.expand_outline_tree(
            project=self.project, unit_id=self.unit_id, model={"name": "m"},
            include_design=True,
            progress=lambda stage, detail, done, total: progress.append(stage),
        )

        unit = book_store.get_unit(book, self.unit_id)
        tree = unit.get("outline_tree") or {}
        branches = tree.get("branches", [])
        self.assertEqual([b["plan_key"] for b in branches], ["A", "B"])
        self.assertEqual(len(branches[0]["events"]), 2)
        self.assertEqual(len(branches[1]["events"]), 1)

        for event in branches[0]["events"] + branches[1]["events"]:
            self.assertEqual(len(event["chapter_branches"]), 2)
            for chapter_branch in event["chapter_branches"]:
                self.assertNotIn("steps", chapter_branch["plan"])
                self.assertEqual({key for key in chapter_branch["plan"]}, {"key", "title", "thesis"})
                for chapter in chapter_branch["chapters"]:
                    self.assertEqual([o["key"] for o in chapter["design_options"]], ["d1"])
                    self.assertEqual(chapter["chosen_design"], "d1")

        self.assertEqual(chapter_mock.call_count, 3)
        self.assertEqual(design_mock.call_count, 2 * (2 + 1) + 1 * (2 + 1))
        self.assertTrue(progress)

        persisted = book_store.load_book(self.project)
        persisted_unit = book_store.get_unit(persisted, self.unit_id)
        self.assertEqual(persisted_unit["outline_tree"]["branches"][0]["events"][0]["chapter_branches"][0]["chapters"][0]["chosen_design"], "d1")

    def test_rerun_skips_existing_work(self):
        directory_mock, chapter_mock, design_mock = self._patch_generators()
        outline_tree.expand_outline_tree(project=self.project, unit_id=self.unit_id, model={"name": "m"}, include_design=True)
        first_counts = (directory_mock.call_count, chapter_mock.call_count, design_mock.call_count)

        outline_tree.expand_outline_tree(project=self.project, unit_id=self.unit_id, model={"name": "m"}, include_design=True)
        self.assertEqual(
            (directory_mock.call_count, chapter_mock.call_count, design_mock.call_count),
            first_counts,
        )

    def test_default_stops_at_chapter_plans(self):
        _, chapter_mock, design_mock = self._patch_generators()
        book = outline_tree.expand_outline_tree(project=self.project, unit_id=self.unit_id, model={"name": "m"})

        unit = book_store.get_unit(book, self.unit_id)
        branches = unit["outline_tree"]["branches"]
        self.assertEqual(chapter_mock.call_count, 3)
        self.assertEqual(design_mock.call_count, 0)
        for branch in branches:
            for event in branch["events"]:
                for chapter_branch in event["chapter_branches"]:
                    for chapter in chapter_branch["chapters"]:
                        self.assertEqual(chapter["design_options"], [])
                        self.assertEqual(chapter["chosen_design"], "")

    def test_retries_when_model_returns_unstructured(self):
        good = {
            "chapter_options": [
                {"key": "x1", "title": "章节方案1", "steps": [
                    {"who": "林岳", "what": "第一章"},
                ]},
            ],
        }
        bad = {"chapter_options": [{"key": "A", "title": "仅有一条未结构化结果", "thesis": "原始文本", "steps": []}]}
        mock.patch.object(unit_generation, "generate_directory_options", return_value={
            "directory_options": [
                {"key": "A", "title": "路线A", "thesis": "", "steps": [
                    {"who": "林岳", "what": "事件一"},
                    {"who": "顾衍", "what": "事件二"},
                ]},
            ],
        }).start()
        calls = {}

        def chapter_result(*_args, **kwargs):
            event_seq = kwargs["event"]["seq"]
            calls[event_seq] = calls.get(event_seq, 0) + 1
            return bad if calls[event_seq] == 1 else good

        chapter_mock = mock.patch.object(
            unit_generation, "generate_event_chapter_options",
            side_effect=chapter_result,
        ).start()
        self.addCleanup(mock.patch.stopall)

        book = outline_tree.expand_outline_tree(project=self.project, unit_id=self.unit_id, model={"name": "m"})
        unit = book_store.get_unit(book, self.unit_id)
        for branch in unit["outline_tree"]["branches"]:
            for event in branch["events"]:
                self.assertTrue(event["chapter_branches"])
                for chapter_branch in event["chapter_branches"]:
                    self.assertTrue(chapter_branch["chapters"])
        self.assertEqual(chapter_mock.call_count, 4)

    def test_replaces_unstructured_branch_from_previous_run(self):
        directory_options = [
            {"key": "A", "title": "路线A", "thesis": "", "steps": [
                {"who": "林岳", "what": "事件一"},
                {"who": "顾衍", "what": "事件二"},
            ]},
            {"key": "B", "title": "路线B", "thesis": "", "steps": [
                {"who": "林岳", "what": "事件三"},
            ]},
        ]
        working = book_store.load_unit_working(self.project, self.unit_id)
        working["directory_options"] = directory_options
        working["outline_tree"] = {
            "directory_options": directory_options,
            "branches": [
                {
                    "plan_key": "A",
                    "plan": directory_options[0],
                    "events": [
                        {
                            "seq": 1, "who": "林岳", "what": "事件一",
                            "chapter_branches": [{
                                "plan_key": "A",
                                "plan": {"key": "A", "title": "仅有一条未结构化结果", "thesis": "原始文本"},
                                "chapters": [],
                            }],
                        },
                        {"seq": 2, "who": "顾衍", "what": "事件二", "chapter_branches": []},
                    ],
                },
                {
                    "plan_key": "B",
                    "plan": directory_options[1],
                    "events": [
                        {"seq": 1, "who": "林岳", "what": "事件三", "chapter_branches": []},
                    ],
                },
            ],
        }
        book_store.save_unit_working(self.project, self.unit_id, working)
        self._patch_generators()

        book = outline_tree.expand_outline_tree(project=self.project, unit_id=self.unit_id, model={"name": "m"})
        unit = book_store.get_unit(book, self.unit_id)
        for branch in unit["outline_tree"]["branches"]:
            for event in branch["events"]:
                self.assertTrue(event["chapter_branches"])
                for chapter_branch in event["chapter_branches"]:
                    self.assertTrue(chapter_branch["chapters"])

    def test_salvages_unstructured_branch_without_new_calls(self):
        directory_options = [
            {"key": "A", "title": "路线A", "thesis": "", "steps": [
                {"who": "林岳", "what": "事件一"},
                {"who": "顾衍", "what": "事件二"},
            ]},
            {"key": "B", "title": "路线B", "thesis": "", "steps": [
                {"who": "林岳", "what": "事件三"},
            ]},
        ]
        salvage_json = json.dumps({"plans": [
            {"key": "A", "title": "线性", "thesis": "", "steps": [
                {"who": "林岳", "what": "第一章"},
                {"who": "顾衍", "what": "第二章"},
            ]},
            {"key": "B", "title": "倒悬", "thesis": "", "steps": [
                {"who": "林岳", "what": "第三章"},
            ]},
        ]}, ensure_ascii=False)
        working = book_store.load_unit_working(self.project, self.unit_id)
        working["directory_options"] = directory_options
        working["outline_tree"] = {
            "directory_options": directory_options,
            "branches": [
                {
                    "plan_key": "A",
                    "plan": directory_options[0],
                    "events": [
                        {
                            "seq": 1, "who": "林岳", "what": "事件一",
                            "chapter_branches": [{
                                "plan_key": "A",
                                "plan": {"key": "A", "title": "仅有一条未结构化结果", "thesis": salvage_json},
                                "chapters": [],
                            }],
                        },
                        {"seq": 2, "who": "顾衍", "what": "事件二", "chapter_branches": []},
                    ],
                },
                {
                    "plan_key": "B",
                    "plan": directory_options[1],
                    "events": [
                        {"seq": 1, "who": "林岳", "what": "事件三", "chapter_branches": []},
                    ],
                },
            ],
        }
        book_store.save_unit_working(self.project, self.unit_id, working)
        _, chapter_mock, _ = self._patch_generators()

        book = outline_tree.expand_outline_tree(project=self.project, unit_id=self.unit_id, model={"name": "m"})
        unit = book_store.get_unit(book, self.unit_id)
        branches = unit["outline_tree"]["branches"]
        event1 = branches[0]["events"][0]
        self.assertEqual(len(event1["chapter_branches"]), 2)
        self.assertEqual(len(event1["chapter_branches"][0]["chapters"]), 2)
        self.assertEqual(len(event1["chapter_branches"][1]["chapters"]), 1)
        self.assertEqual(chapter_mock.call_count, 2)

    def test_empty_directory_raises(self):
        mock.patch.object(unit_generation, "generate_directory_options", return_value={"directory_options": []}).start()
        self.addCleanup(mock.patch.stopall)
        with self.assertRaises(RuntimeError):
            outline_tree.expand_outline_tree(project=self.project, unit_id=self.unit_id, model={"name": "m"})


if __name__ == "__main__":
    unittest.main()
