import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from fastapi.testclient import TestClient

import web_api
from novel_generator import debate_flow
from novel_generator import full_flow
from novel_generator import unit_generation as unit_gen


class WebApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web_api.app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project = str(Path(self.temp_dir.name))
        self.client.put(
            "/api/book/meta",
            json={"path": self.project, "title": "错投的旧包裹", "mainline": "旧包裹串起三户人家", "target_words": 80000},
        )
        self._world()
        character = self._characters()[0]
        unit = self._unit(character_id=character["id"])
        self.character_id = character["id"]
        self.unit_id = unit["id"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _world(self):
        return self.client.put(
            "/api/world",
            json={"path": self.project, "world": {"genre": "都市", "premise": "旧城代收点"}},
        ).json()["book"]

    def _characters(self):
        return self.client.put(
            "/api/characters",
            json={"path": self.project, "characters": [
                {"name": "路明飞", "gender": "男", "age": "34", "notes": "嘴硬但心软"},
            ]},
        ).json()["book"]["characters"]

    def _unit(self, character_id):
        return self.client.post(
            "/api/units",
            json={"path": self.project, "title": "错投的旧包裹", "goal": "旧包裹串起三户人家", "cast": [character_id]},
        ).json()["book"]["units"][-1]

    def test_health_and_empty_book(self):
        empty = self.client.get("/api/book", params={"path": str(Path(self.temp_dir.name))}).json()["book"]
        self.assertIn("world", empty)

    def test_render_cast_falls_back_to_all_characters(self):
        text = unit_gen.render_cast([{"id": "a", "name": "甲", "notes": "慢热"}, {"id": "b", "name": "乙"}], [])
        self.assertIn("甲", text)
        self.assertIn("乙", text)

    def test_proposal_prompt_requires_detailed_mainline(self):
        prompt = unit_gen.build_proposal_prompt(key="A", direction="悬念驱动", reference_text="", guidance="都市悬疑", target_words=90000)
        self.assertIn("禁止写概述", prompt)
        self.assertIn("最大阻力", prompt)
        self.assertIn("关键转折", prompt)

    @mock.patch.object(unit_gen, "_make_adapter")
    @mock.patch.object(unit_gen, "invoke_with_cleaning")
    def test_generate_proposals_makes_three_separate_calls(self, invoke_mock, _adapter_mock):
        _adapter_mock.return_value = object()

        def respond(_adapter, prompt, *args, **kwargs):
            if "第 A 套" in prompt:
                key = "A"
            elif "第 B 套" in prompt:
                key = "B"
            else:
                key = "C"
            return json.dumps({
                "key": key, "title": f"{key}书", "mainline": f"{key}的完整详细单主线",
                "directory": [{"seq": 1, "who": f"角色{key}", "what": "采取行动"}],
            }, ensure_ascii=False)

        invoke_mock.side_effect = respond
        result = unit_gen.generate_proposals(reference_text="", guidance="都市悬疑", model={})
        self.assertEqual(3, invoke_mock.call_count)
        self.assertEqual(["A", "B", "C"], [item["key"] for item in result["proposals"]])

    def test_merge_prompt_carries_sub_events_and_scale(self):
        prompt = unit_gen.build_merge_route_prompt(
            world_text="", cast_text="", unit_goal="", route_text="路线",
            target_words=5000, target_chapters=2, words_per_chapter=2500,
        )
        self.assertIn("sub_events", prompt)
        self.assertIn("大概 2 章", prompt)
        self.assertIn("2500", prompt)

    def test_book_meta_round_trip_chapter_scale(self):
        applied = self.client.put("/api/book/meta", json={"path": self.project, "target_words": 80000, "target_chapters": 32, "words_per_chapter": 2500})
        self.assertEqual(200, applied.status_code)
        book = applied.json()["book"]
        self.assertEqual(32, book["target_chapters"])
        self.assertEqual(2500, book["words_per_chapter"])

    def test_chapter_options_prompt_requires_scores(self):
        prompt = unit_gen.build_event_chapter_options_prompt(
            world_text="", cast_text="", unit_goal="", event={"seq": 1, "who": "甲", "what": "行动"}, target_words=90000,
        )
        self.assertIn("score", prompt)
        self.assertIn("score_reason", prompt)

    def test_debate_flow_scores_and_merges(self):
        proposals = [
            {"key": "A", "title": "甲方案", "thesis": "悬念驱动", "mainline": "详细主线A", "directory": [{"seq": 1, "who": "甲", "what": "行动"}]},
            {"key": "B", "title": "乙方案", "thesis": "情感驱动", "mainline": "详细主线B", "directory": [{"seq": 1, "who": "乙", "what": "行动"}]},
            {"key": "C", "title": "丙方案", "thesis": "结构实验", "mainline": "详细主线C", "directory": [{"seq": 1, "who": "丙", "what": "行动"}]},
        ]
        with mock.patch.object(debate_flow, "_make_adapter", return_value=object()), mock.patch.object(debate_flow, "invoke_with_cleaning") as invoke_mock:
            def respond(_adapter, prompt, *args, **kwargs):
                label = str(kwargs.get("label", "") or "")
                if "立论" in label:
                    key = label.replace("三方辩论：方案 ", "").replace(" 立论", "")
                    return f"{key}方案立论文本"
                if "攻辩" in label:
                    return "攻辩文本"
                return json.dumps({
                    "decision": "B",
                    "ranking": [
                        {"key": "B", "score": 9, "strengths": "情感强", "weaknesses": "节奏略慢", "comment": "最贴合作者意图"},
                        {"key": "A", "score": 8, "strengths": "悬念足", "weaknesses": "情感弱", "comment": "可用"},
                        {"key": "C", "score": 7, "strengths": "结构新", "weaknesses": "风险高", "comment": "偏实验"},
                    ],
                    "merged": {"key": "M", "title": "融合本", "mainline": "融合后的完整主线", "directory": [{"seq": 1, "who": "甲", "what": "融合行动"}]},
                }, ensure_ascii=False)
            invoke_mock.side_effect = respond
            result = debate_flow.run_debate(proposals=proposals, guidance="都市悬疑", model={})
        self.assertEqual({"A", "B", "C"}, set(result["cases"].keys()))
        self.assertEqual(3, len(result["verdict"]["ranking"]))
        self.assertEqual("B", result["verdict"]["decision"])
        self.assertEqual("融合本", result["merged"]["title"])

    def test_full_flow_pipeline_applies_outlines_and_merges(self):
        proposals = [
            {
                "key": key, "title": f"方案{key}", "thesis": "卖点", "mainline": f"详细主线{key}",
                "world": {"genre": "都市", "premise": "旧城", "era_place": "凇市", "rules": "雨夜", "tone": "阴冷"},
                "characters": [{"name": f"主角{key}", "gender": "男", "age": "28", "notes": "嘴硬"}],
                "directory": [
                    {"seq": 1, "who": f"主角{key}", "what": "收到包裹"},
                    {"seq": 2, "who": f"主角{key}", "what": "送出回信"},
                ],
            }
            for key in ("A", "B", "C")
        ]
        stages = []

        def collect(stage, label, detail, done, total):
            stages.append(stage)

        def respond(_adapter, prompt, *args, **kwargs):
            label = str(kwargs.get("label", "") or "")
            if "立论" in label:
                return "立论文本"
            if "攻辩" in label:
                return "攻辩文本"
            return json.dumps({
                "decision": "B",
                "ranking": [
                    {"key": "B", "score": 9, "strengths": "情感强", "weaknesses": "节奏略慢", "comment": "最贴合作者意图"},
                ],
                "merged": {"key": "M", "title": "融合本", "mainline": "融合后的完整主线", "directory": [{"seq": 1, "who": "甲", "what": "融合行动"}]},
            }, ensure_ascii=False)

        project = str(Path(tempfile.mkdtemp(dir=self.temp_dir.name)))
        with mock.patch.object(debate_flow, "generate_proposals", return_value={"proposals": proposals}), \
             mock.patch.object(debate_flow, "_make_adapter", return_value=object()), \
             mock.patch.object(debate_flow, "invoke_with_cleaning") as invoke_mock, \
             mock.patch.object(full_flow, "generate_event_chapter_options") as chapter_mock, \
             mock.patch.object(full_flow, "generate_merged_route") as merge_mock, \
             mock.patch.object(full_flow, "generate_route_chapter_draft") as draft_mock:
            invoke_mock.side_effect = respond
            outline_calls = []

            def chapter_respond(**kwargs):
                outline_calls.append(kwargs)
                return {"chapter_options": [
                    {"key": "A", "title": "拆两章", "steps": [
                        {"who": "甲", "what": "第一幕", "word_estimate": 1500, "word_ratio": 25},
                        {"who": "甲", "what": "第二幕", "word_estimate": 1000, "word_ratio": 25},
                    ]},
                    {"key": "B", "score": 9, "title": "拆三章", "steps": [
                        {"who": "甲", "what": "第一幕", "word_estimate": 1000, "word_ratio": 15},
                        {"who": "甲", "what": "第二幕", "word_estimate": 1000, "word_ratio": 20},
                        {"who": "甲", "what": "第三幕", "word_estimate": 500, "word_ratio": 15},
                    ]},
                ]}

            chapter_mock.side_effect = chapter_respond
            merge_calls = []

            def merge_respond(**kwargs):
                merge_calls.append(kwargs)
                return {"merged_route": {
                    "title": "合并路线", "summary": "查重补缺完成", "fixed": ["未发现重复/缺口"],
                    "chapters": [
                        {"seq": 1, "title": "雨夜来客", "who": "甲", "what": "收到包裹", "word_estimate": 2500, "word_ratio": 50, "sub_events": [
                            {"who": "甲", "what": "第一幕", "word_estimate": 1500, "word_ratio": 30},
                            {"who": "甲", "what": "第二幕", "word_estimate": 1000, "word_ratio": 20},
                        ]},
                        {"seq": 2, "title": "旧巷回声", "who": "甲", "what": "追查寄件人", "word_estimate": 2500, "word_ratio": 50, "sub_events": [
                            {"who": "甲", "what": "第三幕", "word_estimate": 2500, "word_ratio": 50},
                        ]},
                    ],
                }}

            merge_mock.side_effect = merge_respond
            previous_texts = []
            sub_event_texts = []

            def draft_respond(**kwargs):
                previous_texts.append(kwargs.get("previous_text") or "")
                sub_event_texts.append(kwargs.get("sub_events_text") or "")
                seq = kwargs.get("chapter") and kwargs["chapter"].get("seq")
                return f"第{seq}章正文示例"

            draft_mock.side_effect = draft_respond
            result = full_flow.run_full_pipeline(
                project=project, model={}, guidance="都市悬疑", target_words=5000,
                reference_text="", rebuttal_rounds=1, workers=2, draft_model={},
                target_chapters=2, words_per_chapter=2500, progress=collect,
            )
        self.assertEqual("B", result["verdict"]["decision"])
        self.assertTrue(result["unit_id"])
        self.assertEqual(2, result["picked_count"])
        self.assertEqual(2, len(result["merged_route"]["chapters"]))
        self.assertEqual(2, result["written_count"])
        self.assertEqual(2, len(previous_texts))
        self.assertEqual("", previous_texts[0])
        self.assertIn("第1章正文示例", previous_texts[1])
        for expected in ("generate", "advocate", "rebuttal", "moderator", "apply", "outline", "merge", "write"):
            self.assertIn(expected, stages)
        for call in outline_calls:
            self.assertEqual(2, call["target_chapters"])
            self.assertEqual(2500, call["words_per_chapter"])
            self.assertEqual(2, call["events_total"])
        self.assertEqual(2, merge_calls[0]["target_chapters"])
        self.assertEqual(2500, merge_calls[0]["words_per_chapter"])
        self.assertIn("第三幕", sub_event_texts[1])
        book = web_api.book_store.load_book(project)
        self.assertEqual("详细主线B", book["mainline"])
        unit = web_api.book_store.get_unit(book, result["unit_id"])
        working = web_api.book_store.load_unit_working(project, result["unit_id"])
        self.assertEqual("A", unit["chosen_plan"])
        for segment in working["segments"]:
            self.assertEqual("B", segment["chosen_chapter_plan"], segment)
        self.assertEqual("合并路线", working["merged_route"]["title"])
        self.assertEqual(["第1章正文示例", "第2章正文示例"], [chapter["draft"] for chapter in working["merged_route"]["chapters"]])
        self.assertEqual(2, len(working["merged_route"]["chapters"][0]["sub_events"]))
        self.assertEqual(1500, working["merged_route"]["chapters"][0]["sub_events"][0]["word_estimate"])

    def test_agent_token_guard(self):
        old = web_api.AINOVEL_TOKEN
        web_api.AINOVEL_TOKEN = "secret-agent-token"
        try:
            denied = self.client.get("/api/agent/status", params={"path": self.project})
            self.assertEqual(401, denied.status_code)
            allowed = self.client.get("/api/agent/status", params={"path": self.project}, headers={"X-AINOVEL-Token": "secret-agent-token"})
            self.assertEqual(200, allowed.status_code)
            self.assertTrue(allowed.json()["ok"])
        finally:
            web_api.AINOVEL_TOKEN = old

    def test_agent_run_requires_scale(self):
        response = self.client.post("/api/agent/run", json={"project": self.project, "guidance": "都市悬疑"})
        self.assertEqual(400, response.status_code)

    @mock.patch.object(web_api.full_flow, "run_full_pipeline")
    def test_agent_run_blocks_until_done(self, pipeline_mock):
        pipeline_mock.side_effect = lambda **kwargs: {
            "proposals": [{"key": "A", "title": "测试本", "mainline": "主线", "directory": []}],
            "cases": {}, "rebuttals": {},
            "verdict": {"decision": "A", "ranking": [{"key": "A", "score": 9}], "summary": ""},
            "merged": None, "chosen_proposal": None, "unit_id": "",
            "segments": [], "picked_count": 0,
            "merged_route": {"chapters": [{"seq": 1, "title": "第一章", "draft": "正文"}]},
            "written_count": 1,
        }
        project = str(Path(tempfile.mkdtemp(dir=self.temp_dir.name)))
        response = self.client.post("/api/agent/run", json={
            "project": project, "guidance": "都市悬疑，双男主",
            "target_words": 80000, "target_chapters": 32, "words_per_chapter": 2500,
            "wait": True, "timeout": 60, "poll_interval": 1,
        })
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertTrue(data["ok"], data)
        self.assertEqual("done", data["job"]["status"])
        self.assertEqual(32, data["summary"]["target_chapters"])
        meta = web_api.book_store.load_book_meta(project)
        self.assertEqual(32, meta["target_chapters"])
        self.assertEqual(2500, meta["words_per_chapter"])

    def test_world_and_characters_round_trip(self):
        data = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        self.assertEqual("都市", data["world"]["genre"])
        self.assertEqual("路明飞", data["characters"][0]["name"])

    @mock.patch.object(unit_gen, "generate_directory_options")
    def test_directory_options_allow_missing_cast_and_goal(self, directory_mock):
        directory_mock.return_value = {"directory_options": [{"key": "A", "title": "直给"}]}
        project = str(Path(tempfile.mkdtemp(dir=self.temp_dir.name)))
        self.client.put("/api/world", json={"path": project, "world": {"genre": "都市"}})
        self.client.put("/api/book/meta", json={"path": project, "target_words": 80000})
        unit = self.client.post("/api/units", json={"path": project}).json()["book"]["units"][-1]
        response = self.client.post(f"/api/units/{unit['id']}/directory-options", json={"path": project})
        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json()["book"]["units"][-1]["directory_options"])

    @mock.patch.object(unit_gen, "generate_proposals")
    def test_proposals_and_apply_round_trip(self, generate_mock):
        generate_mock.return_value = {"proposals": [{
            "key": "A", "title": "雨夜清账人", "thesis": "每场雨都是一笔账",
            "mainline": "旧案与清账会。",
            "world": {"genre": "都市悬疑", "premise": "清账会", "era_place": "凇市", "rules": "雨夜行动", "tone": "阴冷"},
            "characters": [{"name": "沈临", "gender": "男", "age": "34", "notes": "嘴硬"}],
            "directory": [
                {"seq": 1, "who": "沈临", "what": "收到戏票"},
                {"seq": 2, "who": "沈临", "what": "联手谢惊澜"},
            ],
        }]}
        project = str(Path(tempfile.mkdtemp(dir=self.temp_dir.name)))
        self.client.put("/api/book/meta", json={"path": project, "target_words": 80000})
        response = self.client.post("/api/proposals", json={"path": project, "guidance": "都市悬疑双男主", "target_words": 80000})
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(response.json()["proposals"]))
        snapshot = self.client.get("/api/book", params={"path": project}).json()["book"]
        self.assertEqual(1, len(snapshot["proposals"]))
        self.assertIsNone(snapshot["debate"])
        applied = self.client.post("/api/proposals/apply", json={"path": project, "proposal": response.json()["proposals"][0], "target_words": 80000})
        self.assertEqual(200, applied.status_code)
        book = applied.json()["book"]
        self.assertEqual("雨夜清账人", book["title"])
        self.assertEqual("沈临", book["characters"][0]["name"])
        self.assertEqual("清账会", book["world"]["premise"])
        unit = next(item for item in book["units"] if item["id"] == book["current"])
        self.assertEqual("A", unit["chosen_plan"])
        self.assertEqual(2, len(unit["segments"]))
        self.assertEqual("沈临", unit["segments"][0]["who"])

    @mock.patch.object(debate_flow, "run_debate")
    def test_debate_endpoint_persists_result(self, run_mock):
        run_mock.return_value = {
            "cases": {"A": "立论文本"},
            "rebuttals": {"A": ["攻辩文本"]},
            "verdict": {"decision": "A", "ranking": [{"key": "A", "score": 9}], "summary": ""},
            "merged": None,
        }
        project = str(Path(tempfile.mkdtemp(dir=self.temp_dir.name)))
        proposal = {"key": "A", "title": "雨夜清账人", "mainline": "完整主线", "directory": [{"seq": 1, "who": "沈临", "what": "行动"}]}
        response = self.client.post("/api/proposals/debate", json={"path": project, "guidance": "都市悬疑", "proposals": [proposal]})
        self.assertEqual(200, response.status_code)
        self.assertEqual("A", response.json()["debate"]["verdict"]["decision"])
        book = self.client.get("/api/book", params={"path": project}).json()["book"]
        self.assertEqual([proposal], book["proposals"])
        self.assertEqual("A", book["debate"]["verdict"]["decision"])

    @mock.patch.object(unit_gen, "generate_event_chapter_options")
    def test_batch_outline_generates_options_for_all_events(self, chapter_mock):
        with mock.patch.object(unit_gen, "generate_directory_options") as directory_mock:
            directory_mock.return_value = {"directory_options": [
                {"key": "A", "title": "直给", "thesis": "开门见山", "angle": "顺叙推进", "steps": [
                    {"who": "路明飞", "what": "收下包裹"},
                    {"who": "路明飞", "what": "找到寄件人"},
                    {"who": "路明飞", "what": "送出回信"},
                ]},
            ]}
            self.client.post(f"/api/units/{self.unit_id}/directory-options", json={"path": self.project})
            self.client.put(f"/api/units/{self.unit_id}/directory", json={"path": self.project, "plan_key": "A"})

        chapter_mock.return_value = {"chapter_options": [
            {"key": "X", "title": "拆两章", "steps": [
                {"who": "路明飞", "what": "第一幕", "word_estimate": 3000, "word_ratio": 12},
                {"who": "路明飞", "what": "第二幕", "word_estimate": 4000, "word_ratio": 14},
            ]},
        ]}
        started = self.client.post(
            f"/api/units/{self.unit_id}/batch-outline",
            json={"path": self.project, "workers": 3},
        ).json()
        self.assertTrue(started["ok"])

        deadline = time.time() + 15
        job = {}
        while time.time() < deadline:
            job = self.client.get(f"/api/units/{self.unit_id}/auto-outline", params={"path": self.project}).json()["job"]
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        self.assertEqual("done", job["status"], job)

        book = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        unit = next(item for item in book["units"] if item["id"] == self.unit_id)
        self.assertEqual(3, len(unit["segments"]))
        for segment in unit["segments"]:
            self.assertEqual(1, len(segment.get("chapter_options", [])))
        chosen = self.client.put(
            f"/api/units/{self.unit_id}/segments/1/chapters",
            json={"path": self.project, "plan_key": "X"},
        )
        self.assertTrue(chosen.json()["ok"])
        book = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        unit = next(item for item in book["units"] if item["id"] == self.unit_id)
        chapter = unit["segments"][0]["chapters"][0]
        self.assertEqual(3000, chapter["word_estimate"])
        self.assertEqual(12, chapter["word_ratio"])

    @mock.patch.object(unit_gen, "generate_event_chapter_options")
    def test_batch_outline_auto_selects_highest_score(self, chapter_mock):
        with mock.patch.object(unit_gen, "generate_directory_options") as directory_mock:
            directory_mock.return_value = {"directory_options": [
                {"key": "A", "title": "直给", "thesis": "开门见山", "angle": "顺叙推进", "steps": [
                    {"who": "路明飞", "what": "收下包裹"},
                    {"who": "路明飞", "what": "找到寄件人"},
                ]},
            ]}
            self.client.post(f"/api/units/{self.unit_id}/directory-options", json={"path": self.project})
            self.client.put(f"/api/units/{self.unit_id}/directory", json={"path": self.project, "plan_key": "A"})

        def options_for(seq):
            return {"chapter_options": [
                {"key": "X", "title": "低分方案", "score": 2.0, "score_reason": "节奏拖", "steps": [{"who": "路明飞", "what": "第一幕"}]},
                {"key": "Y", "title": "高分方案", "score": 9.0, "score_reason": "反转利落", "steps": [{"who": "路明飞", "what": "第一幕"}]},
                {"key": "Z", "title": "中分方案", "score": 5.5, "score_reason": "平稳", "steps": [{"who": "路明飞", "what": "第一幕"}]},
            ]}
        chapter_mock.side_effect = lambda **kwargs: options_for((kwargs.get("event") or {}).get("seq"))
        started = self.client.post(
            f"/api/units/{self.unit_id}/batch-outline",
            json={"path": self.project, "workers": 3, "auto_select": True},
        ).json()
        self.assertTrue(started["ok"])
        deadline = time.time() + 15
        job = {}
        while time.time() < deadline:
            job = self.client.get(f"/api/units/{self.unit_id}/auto-outline", params={"path": self.project}).json()["job"]
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        self.assertEqual("done", job["status"], job)
        book = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        unit = next(item for item in book["units"] if item["id"] == self.unit_id)
        for segment in unit["segments"]:
            self.assertEqual("Y", segment["chosen_chapter_plan"], segment)

    @mock.patch.object(unit_gen, "generate_merged_route")
    def test_merge_route_requires_all_chosen_then_merges(self, merge_mock):
        with mock.patch.object(unit_gen, "generate_directory_options") as directory_mock, \
             mock.patch.object(unit_gen, "generate_event_chapter_options") as chapter_mock:
            directory_mock.return_value = {"directory_options": [
                {"key": "A", "title": "直给", "thesis": "开门见山", "angle": "顺叙推进", "steps": [
                    {"who": "路明飞", "what": "收下包裹"},
                    {"who": "路明飞", "what": "找到寄件人"},
                ]},
            ]}
            chapter_mock.return_value = {"chapter_options": [
                {"key": "X", "title": "拆两章", "steps": [
                    {"who": "路明飞", "what": "第一幕", "word_estimate": 3000, "word_ratio": 40},
                    {"who": "路明飞", "what": "第二幕", "word_estimate": 4000, "word_ratio": 40},
                ]},
            ]}
            self.client.post(f"/api/units/{self.unit_id}/directory-options", json={"path": self.project})
            self.client.put(f"/api/units/{self.unit_id}/directory", json={"path": self.project, "plan_key": "A"})
            started = self.client.post(f"/api/units/{self.unit_id}/batch-outline", json={"path": self.project, "workers": 2}).json()
            self.assertTrue(started["ok"])
            deadline = time.time() + 15
            while time.time() < deadline:
                job = self.client.get(f"/api/units/{self.unit_id}/auto-outline", params={"path": self.project}).json()["job"]
                if job["status"] in ("done", "error"):
                    break
                time.sleep(0.1)

        blocked = self.client.post(f"/api/units/{self.unit_id}/merge-route", json={"path": self.project})
        self.assertEqual(400, blocked.status_code)
        self.assertIn("未选定", blocked.json()["detail"])

        book = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        unit = next(item for item in book["units"] if item["id"] == self.unit_id)
        for segment in unit["segments"]:
            self.client.put(
                f"/api/units/{self.unit_id}/segments/{segment['seq']}/chapters",
                json={"path": self.project, "plan_key": "X"},
            )

        merge_mock.return_value = {"merged_route": {
            "title": "合成路线",
            "summary": "去重一章，补充一章过渡。",
            "fixed": ["去重：合并两章", "补充：过渡章"],
            "chapters": [
                {"seq": 1, "who": "路明飞", "what": "收下包裹", "word_estimate": 3000, "word_ratio": 30},
                {"seq": 2, "who": "路明飞", "what": "找到寄件人", "word_estimate": 3500, "word_ratio": 35},
            ],
        }}
        merged = self.client.post(f"/api/units/{self.unit_id}/merge-route", json={"path": self.project, "history": []}).json()
        self.assertTrue(merged["ok"])
        unit = next(item for item in merged["book"]["units"] if item["id"] == self.unit_id)
        self.assertEqual("合成路线", unit["merged_route"]["title"])
        self.assertEqual(2, len(unit["merged_route"]["chapters"]))
        self.assertEqual("去重：合并两章", unit["merged_route"]["fixed"][0])
        self.assertEqual(3500, unit["merged_route"]["chapters"][1]["word_estimate"])

    @mock.patch.object(unit_gen, "generate_directory_options")
    def test_directory_design_draft_flow(self, directory_mock):
        directory_mock.return_value = {"directory_options": [
            {"key": "A", "title": "直给", "thesis": "开门见山",
             "steps": [{"who": "路明飞", "what": "收下包裹"}, {"who": "路明飞", "what": "找到寄件人"}]},
            {"key": "B", "title": "倒叙", "thesis": "从结尾切入", "steps": [{"who": "路明飞", "what": "看到结果"}]},
        ]}
        with mock.patch.object(unit_gen, "generate_event_chapter_options") as chapter_mock, \
             mock.patch.object(unit_gen, "generate_segment_design_options") as design_mock, \
             mock.patch.object(unit_gen, "generate_segment_draft") as draft_mock:
            chapter_mock.return_value = {"chapter_options": [
                {"key": "X", "title": "拆两章", "steps": [
                    {"who": "路明飞", "what": "收下包裹"},
                    {"who": "路明飞", "what": "找到寄件人"},
                ]},
            ]}
            design_mock.return_value = {"options": [
                {"key": "1", "title": "慢热", "summary": "细节铺垫", "content": "先写环境"},
                {"key": "2", "title": "冲突", "summary": "直接对峙", "content": "进门就吵"},
            ]}
            draft_mock.return_value = "第一步正文：路明飞收下了包裹。"

            data = self.client.post(f"/api/units/{self.unit_id}/directory-options", json={"path": self.project}).json()["book"]
            self.assertEqual(["A", "B"], [plan["key"] for plan in data["units"][0]["directory_options"]])

            data = self.client.put(f"/api/units/{self.unit_id}/directory", json={"path": self.project, "plan_key": "A"}).json()["book"]
            self.assertEqual("A", data["units"][0]["chosen_plan"])
            self.assertEqual([(1, "路明飞", "收下包裹"), (2, "路明飞", "找到寄件人")],
                             [(seg["seq"], seg["who"], seg["what"]) for seg in data["units"][0]["segments"]])

            data = self.client.post(f"/api/units/{self.unit_id}/segments/1/chapter-options", json={"path": self.project}).json()["book"]
            self.assertEqual(["X"], [plan["key"] for plan in data["units"][0]["segments"][0]["chapter_options"]])

            data = self.client.put(f"/api/units/{self.unit_id}/segments/1/chapters", json={"path": self.project, "plan_key": "X"}).json()["book"]
            self.assertEqual([(1, "路明飞", "收下包裹"), (2, "路明飞", "找到寄件人")],
                             [(ch["seq"], ch["who"], ch["what"]) for ch in data["units"][0]["segments"][0]["chapters"]])

            data = self.client.post(f"/api/units/{self.unit_id}/segments/1/chapters/1/design-options", json={"path": self.project}).json()["book"]
            self.assertEqual(["1", "2"], [option["key"] for option in data["units"][0]["segments"][0]["chapters"][0]["design_options"]])

            data = self.client.put(f"/api/units/{self.unit_id}/segments/1/chapters/1/design", json={"path": self.project, "option_key": "2"}).json()["book"]
            self.assertEqual("2", data["units"][0]["segments"][0]["chapters"][0]["chosen_design"])

            data = self.client.post(f"/api/units/{self.unit_id}/segments/1/chapters/1/draft", json={"path": self.project}).json()["book"]
            self.assertEqual("第一步正文：路明飞收下了包裹。", data["units"][0]["segments"][0]["chapters"][0]["draft"])

    @mock.patch.object(unit_gen, "conversation_reply")
    def test_ask_about_selection(self, reply_mock):
        reply_mock.return_value = "解释：这是中段转折。"
        payload = {
            "path": self.project, "unit_id": self.unit_id,
            "selected": "路明飞收下包裹", "action": "explain",
            "scope_text": "方案 A：路明飞直接敲门送包裹",
            "messages": [
                {"role": "user", "content": "为什么这样安排？"},
                {"role": "assistant", "content": "这是进入事件的钩子。"},
                {"role": "user", "content": "那能不能把它放到第二步？"},
            ],
        }
        response = self.client.post("/api/ask", json=payload)
        self.assertEqual(200, response.status_code)
        self.assertIn("中段转折", response.json()["result"])
        self.assertEqual(3, len(reply_mock.call_args.kwargs["messages"]))
        self.assertIn("作者选中的直接上下文", reply_mock.call_args.kwargs["unit_context"])
        self.assertIn("方案 A：路明飞直接敲门送包裹", reply_mock.call_args.kwargs["unit_context"])

    @mock.patch.object(unit_gen, "generate_directory_options")
    def test_directory_options_preview_returns_editable_prompt(self, generate_mock):
        response = self.client.post(
            f"/api/units/{self.unit_id}/directory-options",
            json={"path": self.project, "preview": True},
        )
        self.assertEqual(200, response.status_code)
        preview = response.json()["preview"]
        self.assertEqual("directory", preview["kind"])
        self.assertIn("这本书要写什么", preview["prompt"])
        self.assertIn("旧包裹串起三户人家", preview["prompt"])
        self.assertIn("POST ", preview["request"])
        self.assertIn("/chat/completions", preview["request"])
        self.assertIn('"messages"', preview["request"])
        generate_mock.assert_not_called()

    def test_ask_preview_uses_selected_scope_only(self):
        response = self.client.post("/api/ask", json={
            "path": self.project, "unit_id": self.unit_id,
            "selected": "路明飞收下包裹", "action": "explain",
            "scope_text": "方案 A：路明飞直接敲门送包裹",
            "messages": [{"role": "user", "content": "为什么这样安排？"}],
            "preview": True,
        })
        self.assertEqual(200, response.status_code)
        prompt = response.json()["preview"]["prompt"]
        self.assertIn("作者选中的直接上下文", prompt)
        self.assertIn("方案 A：路明飞直接敲门送包裹", prompt)
        self.assertIn("rewrite_text", prompt)
        request = response.json()["preview"]["request"]
        self.assertIn('"role": "system"', request)
        self.assertIn('"role": "user"', request)
        self.assertIn('"rewrite_text"', request)
        self.assertIn('"tool_choice": "auto"', request)

    def test_unit_summary_and_delete(self):
        data = self.client.put(f"/api/units/{self.unit_id}/summary", json={"path": self.project, "text": "包裹已送达"}).json()["book"]
        self.assertEqual("包裹已送达", data["units"][0]["unit_summary"])
        removed = self.client.delete(f"/api/units/{self.unit_id}?path={self.project}").json()["book"]
        self.assertEqual([], removed["units"])

    def test_reference_upload_toggle_delete_and_prompt(self):
        created = self.client.post("/api/references", json={"path": self.project, "name": "别人的书.txt", "content": "第一章 起风前，他回到了老城。"}).json()
        self.assertEqual(1, len(created["references"]))
        self.assertTrue(created["references"][0]["enabled"])

        book = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        self.assertEqual(1, len(book["references"]))
        encoded_name = quote("别人的书.txt", safe="")

        with mock.patch.object(unit_gen, "generate_directory_options"):
            self.client.put(f"/api/references/{encoded_name}", json={"path": self.project, "enabled": False})
            preview = self.client.post(f"/api/units/{self.unit_id}/directory-options", json={"path": self.project, "preview": True}).json()["preview"]
            self.assertIn("参考文本", preview["prompt"])
            self.assertIn("未上传参考文本", preview["prompt"])
            self.assertNotIn("别人的书.txt", preview["prompt"])
            self.assertNotIn("起风前", preview["prompt"])

            self.client.put(f"/api/references/{encoded_name}", json={"path": self.project, "enabled": True})
            preview = self.client.post(f"/api/units/{self.unit_id}/directory-options", json={"path": self.project, "preview": True}).json()["preview"]
            self.assertIn("别人的书.txt", preview["prompt"])
            self.assertIn("起风前", preview["prompt"])

        removed = self.client.delete(f"/api/references/{encoded_name}?path={quote(self.project, safe='')}").json()
        self.assertEqual([], removed["references"])

    def test_reference_brief_preview_and_generate(self):
        with mock.patch.object(unit_gen, "generate_reference_brief") as brief_mock:
            brief_mock.return_value = {"brief": {
                "title": "夜航旧书店",
                "mainline": "旧书店店主为找回一封遗信，卷入一桩旧案。",
                "world": {"genre": "都市悬疑", "premise": "旧城代收点", "era_place": "当代南方", "rules": "记忆力会失真", "tone": "克制"},
                "characters": [{"name": "路明飞", "gender": "男", "age": "34", "notes": "嘴硬心软"}],
            }}
            preview = self.client.post("/api/references/brief", json={"path": self.project, "guidance": "都市悬疑，双男主，年下", "preview": True}).json()["preview"]
            self.assertEqual("reference-brief", preview["kind"])
            self.assertIn("参考文本", preview["prompt"])
            self.assertIn("都市悬疑，双男主，年下", preview["prompt"])
            brief_mock.assert_not_called()

            data = self.client.post("/api/references/brief", json={"path": self.project, "guidance": "都市悬疑，双男主，年下"}).json()
            self.assertEqual("夜航旧书店", data["brief"]["title"])
            self.assertEqual("都市悬疑", data["brief"]["world"]["genre"])

        missing = self.client.post("/api/references/brief", json={"path": self.project, "guidance": ""})
        self.assertEqual(400, missing.status_code)

    def test_reference_limit_defaults_to_full_and_saves(self):
        self.client.post("/api/references", json={"path": self.project, "name": "长文.txt", "content": "开始" + "字" * 200 + "结束"})
        book = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        self.assertEqual(0, book["reference_limit"])

        data = self.client.put("/api/reference-limit", json={"path": self.project, "limit": 10}).json()
        self.assertEqual(10, data["reference_limit"])
        book = self.client.get("/api/book", params={"path": self.project}).json()["book"]
        self.assertEqual(10, book["reference_limit"])

    def test_save_unit_working_preserves_plan_edit(self):
        plans = [{"key": "A", "title": "改动后", "thesis": "新逻辑",
                  "steps": [{"seq": 1, "who": "路明飞", "what": "先敲门再送包裹"}]}]
        data = self.client.put(
            f"/api/units/{self.unit_id}/working",
            json={"path": self.project, "directory_options": plans, "chosen_plan": "A", "segments": [], "unit_summary": "修好"},
        ).json()["book"]
        self.assertEqual("改动后", data["units"][0]["directory_options"][0]["title"])

    def test_snapshot_generate_and_save_round_trip(self):
        with mock.patch.object(unit_gen, "generate_unit_snapshot") as snapshot_mock:
            snapshot_mock.return_value = {
                "beats": ["第1步 · 路明飞 → 收下包裹：完成"],
                "character_states": ["路明飞：疏离 → 动摇"],
                "hooks": ["寄件人还没找到"],
            }
            generated = self.client.post(f"/api/units/{self.unit_id}/snapshot", json={"path": self.project}).json()["snapshot"]
            self.assertEqual(["第1步 · 路明飞 → 收下包裹：完成"], generated["beats"])
            self.assertNotIn("outcome", generated)
            saved = self.client.put(
                f"/api/units/{self.unit_id}/snapshot",
                json={"path": self.project, "snapshot": generated},
            ).json()["book"]
            self.assertEqual(["寄件人还没找到"], saved["units"][0]["snapshot"]["hooks"])

    def test_directory_prompt_carries_previous_unit_snapshot(self):
        self.client.put(f"/api/units/{self.unit_id}/snapshot", json={"path": self.project, "snapshot": {
            "beats": ["第1步 · 路明飞 → 收下包裹：包裹已送达"],
            "character_states": [], "hooks": [],
        }})
        second = self.client.post("/api/units", json={
            "path": self.project, "title": "找寄件人", "goal": "找到寄件人", "cast": [self.character_id],
        }).json()["book"]["units"][-1]
        preview = self.client.post(
            f"/api/units/{second['id']}/directory-options", json={"path": self.project, "preview": True}
        ).json()["preview"]
        self.assertIn("前情快照", preview["prompt"])
        self.assertIn("包裹已送达", preview["prompt"])

    def test_graph_round_trip_and_prompt_injection(self):
        graph = {
            "nodes": [
                {"id": "n1", "kind": "人物", "name": "路明飞", "note": "店员", "status": "活跃"},
                {"id": "n2", "kind": "地点", "name": "夜航书店", "note": "", "status": "活跃"},
            ],
            "edges": [{"id": "e1", "source": "n1", "target": "n2", "type": "在", "note": "", "status": ""}],
            "events": [{"id": "t1", "unit_seq": 1, "title": "雨夜开门", "summary": "书店首次出现"}],
        }
        saved = self.client.put("/api/graph", json={"path": self.project, "graph": graph}).json()["book"]
        self.assertEqual("路明飞", saved["graph"]["nodes"][0]["name"])
        preview = self.client.post(
            f"/api/units/{self.unit_id}/directory-options", json={"path": self.project, "preview": True}
        ).json()["preview"]
        self.assertIn("图谱工作集", preview["prompt"])
        self.assertIn("路明飞", preview["prompt"])

    @mock.patch.object(unit_gen, "generate_graph_draft")
    def test_graph_draft_generate(self, draft_mock):
        draft_mock.return_value = {
            "nodes": [{"id": "n1", "kind": "人物", "name": "路明飞", "note": "店员", "status": "活跃"}],
            "edges": [],
            "events": [],
            "facts": [],
        }
        response = self.client.post("/api/graph/draft", json={"path": self.project})
        self.assertEqual(200, response.status_code)
        self.assertEqual("路明飞", response.json()["graph"]["nodes"][0]["name"])

    @mock.patch.object(unit_gen, "generate_atomize")
    def test_atomize_generates_facts(self, atomize_mock):
        atomize_mock.return_value = {"facts": [
            {"id": "f1", "unit_seq": 1, "text": "路明飞收下了旧包裹", "entities": ["路明飞"]},
        ]}
        response = self.client.post(f"/api/units/{self.unit_id}/atomize", json={"path": self.project})
        self.assertEqual(200, response.status_code)
        self.assertEqual("路明飞收下了旧包裹", response.json()["facts"][0]["text"])

    def test_selected_facts_injected_into_directory_prompt(self):
        self.client.put("/api/graph", json={"path": self.project, "graph": {
            "nodes": [{"id": "n1", "kind": "人物", "name": "路明飞", "note": "", "status": "活跃"}],
            "edges": [],
            "events": [],
            "facts": [{"id": "f1", "unit_seq": 1, "text": "路明飞收下了旧包裹", "entities": ["路明飞"]}],
        }})
        second = self.client.post("/api/units", json={
            "path": self.project, "title": "找寄件人", "goal": "找到寄件人", "cast": [self.character_id],
        }).json()["book"]["units"][-1]
        self.client.put(f"/api/units/{second['id']}/working", json={
            "path": self.project, "carry_units": [self.unit_id], "carry_facts": ["f1"],
        })
        preview = self.client.post(
            f"/api/units/{second['id']}/directory-options", json={"path": self.project, "preview": True}
        ).json()["preview"]
        self.assertIn("路明飞收下了旧包裹", preview["prompt"])

    def test_llm_log_tail_is_incremental(self):
        first = self.client.get("/api/logs/llm/tail", params={"offset": 0})
        self.assertEqual(200, first.status_code)
        self.assertIn("events", first.json())

    def test_active_llm_requests_reports_nothing_when_idle(self):
        self.assertEqual([], self.client.get("/api/logs/llm/active").json()["active"])

    @mock.patch.object(unit_gen, "generate_directory_options")
    def test_auto_outline_full_pipeline(self, directory_mock):
        directory_mock.return_value = {"directory_options": [
            {"key": "A", "title": "直给", "thesis": "开门见山", "steps": [
                {"who": "路明飞", "what": "收下包裹"},
                {"who": "路明飞", "what": "找到寄件人"},
            ]},
        ]}
        with mock.patch.object(unit_gen, "generate_event_chapter_options") as chapter_mock, \
             mock.patch.object(unit_gen, "generate_segment_design_options") as design_mock:
            chapter_mock.return_value = {"chapter_options": [
                {"key": "X", "title": "拆两章", "steps": [
                    {"who": "路明飞", "what": "收下包裹"},
                    {"who": "路明飞", "what": "找到寄件人"},
                ]},
            ]}
            design_mock.return_value = {"options": [
                {"key": "1", "title": "慢热", "summary": "细节铺垫", "content": "先写环境"},
                {"key": "2", "title": "冲突", "summary": "直接对峙", "content": "进门就吵"},
            ]}

            started = self.client.post(f"/api/units/{self.unit_id}/auto-outline", json={"path": self.project}).json()
            self.assertEqual("queued", started["job"]["status"])

            job = started["job"]
            deadline = time.time() + 10
            while job["status"] in ("queued", "running") and time.time() < deadline:
                time.sleep(0.05)
                job = self.client.get(f"/api/units/{self.unit_id}/auto-outline", params={"path": self.project}).json()["job"]
            self.assertEqual("done", job["status"], job.get("message"))

            unit = self.client.get("/api/book", params={"path": self.project}).json()["book"]["units"][0]
            tree = unit["outline_tree"]
            self.assertTrue(tree)
            branches = tree["branches"]
            self.assertEqual(1, len(branches))
            self.assertEqual("A", branches[0]["plan_key"])

            events = branches[0]["events"]
            self.assertEqual(2, len(events))
            for event in events:
                self.assertTrue(event["chapter_branches"])

            chapter_branch = events[0]["chapter_branches"][0]
            self.assertEqual("X", chapter_branch["plan_key"])
            self.assertEqual(2, len(chapter_branch["chapters"]))
            self.assertEqual("1", chapter_branch["chapters"][0]["chosen_design"])
            self.assertEqual(2, len(chapter_branch["chapters"][0]["design_options"]))

            self.assertEqual(2, chapter_mock.call_count)
            self.assertEqual(4, design_mock.call_count)


class WebApiGateTest(unittest.TestCase):
    def test_ideation_requires_target_words(self):
        client = TestClient(web_api.app)
        project = tempfile.mkdtemp()
        client.put("/api/world", json={"path": project, "world": {"genre": "都市"}})
        characters = client.put("/api/characters", json={"path": project, "characters": [{"name": "路明飞"}]}).json()["book"]["characters"]
        unit = client.post("/api/units", json={"path": project, "cast": [characters[0]["id"]], "goal": ""}).json()["book"]["units"][-1]
        response = client.post(f"/api/units/{unit['id']}/directory-options", json={"path": project})
        self.assertEqual(400, response.status_code)
        self.assertIn("字数", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

