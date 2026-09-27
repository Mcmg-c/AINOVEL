# -*- coding: utf-8 -*-
import unittest

from llm_adapters import BaseLLMAdapter


class OkJsonAdapter(BaseLLMAdapter):
    def _invoke_json_request(self, prompt):
        return '{"ok": true}'


class UnsupportedJsonAdapter(BaseLLMAdapter):
    def invoke(self, prompt):
        return "plain"

    def _invoke_json_request(self, prompt):
        raise RuntimeError("400: response_format is not supported")


class FailingJsonAdapter(BaseLLMAdapter):
    def _invoke_json_request(self, prompt):
        raise RuntimeError("server exploded")


class InvokeJsonTest(unittest.TestCase):
    def test_returns_json_when_backend_supports_json_mode(self):
        self.assertEqual('{"ok": true}', OkJsonAdapter().invoke_json("prompt"))

    def test_falls_back_to_normal_invoke_when_backend_rejects_json_mode(self):
        self.assertEqual("plain", UnsupportedJsonAdapter().invoke_json("prompt"))

    def test_reraises_unrelated_errors(self):
        with self.assertRaises(RuntimeError):
            FailingJsonAdapter().invoke_json("prompt")


if __name__ == "__main__":
    unittest.main()
