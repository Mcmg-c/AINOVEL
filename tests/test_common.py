# -*- coding: utf-8 -*-
import importlib.util
import io
from pathlib import Path
import unittest
from contextlib import redirect_stdout

COMMON_PATH = Path(__file__).resolve().parents[1] / "novel_generator" / "common.py"
SPEC = importlib.util.spec_from_file_location("common_under_test", COMMON_PATH)
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)

invoke_with_cleaning = common.invoke_with_cleaning
remove_think_tags = common.remove_think_tags
add_llm_trace_listener = common.add_llm_trace_listener
remove_llm_trace_listener = common.remove_llm_trace_listener


class FakeLLMAdapter:
    def __init__(self, response):
        self.response = response

    def invoke(self, prompt):
        return self.response


class EmptyLLMAdapter(FakeLLMAdapter):
    def get_last_response_diagnostics(self):
        return '{"finish_reason": "length", "output_tokens": 4096}'


class CommonCleaningTest(unittest.TestCase):
    def test_remove_think_tags_strips_reasoning_blocks(self):
        text = "prefix<think>internal reasoning</think>body"

        self.assertEqual("prefixbody", remove_think_tags(text))

    def test_invoke_with_cleaning_removes_think_tags_from_llm_output(self):
        adapter = FakeLLMAdapter(
            "```<think>internal reasoning should be hidden</think>\nChapter text```"
        )

        with redirect_stdout(io.StringIO()):
            result = invoke_with_cleaning(adapter, "write chapter")

        self.assertEqual("Chapter text", result)

    def test_invoke_emits_prompt_waiting_and_response_events(self):
        events = []
        listener = events.append
        add_llm_trace_listener(listener)
        try:
            result = invoke_with_cleaning(
                FakeLLMAdapter("answer"),
                "full prompt",
                label="test stage",
            )
        finally:
            remove_llm_trace_listener(listener)

        self.assertEqual("answer", result)
        self.assertEqual(["request", "attempt", "response"], [event["type"] for event in events])
        self.assertIn('"model"', events[0]["content"])
        self.assertIn('"messages"', events[0]["content"])
        self.assertIn("full prompt", events[0]["content"])
        self.assertEqual("answer", events[-1]["content"])
        self.assertEqual("test stage", events[0]["label"])

    def test_empty_response_trace_contains_provider_diagnostics(self):
        events = []
        listener = events.append
        add_llm_trace_listener(listener)
        try:
            with self.assertRaises(RuntimeError):
                invoke_with_cleaning(
                    EmptyLLMAdapter(""),
                    "full prompt",
                    max_retries=1,
                    label="empty test",
                )
        finally:
            remove_llm_trace_listener(listener)

        retry_event = next(event for event in events if event["type"] == "retry")
        self.assertIn("finish_reason", retry_event["content"])
        self.assertIn("length", retry_event["content"])


if __name__ == "__main__":
    unittest.main()
