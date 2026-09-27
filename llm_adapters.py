# llm_adapters.py
# -*- coding: utf-8 -*-
import logging
import json
from typing import Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from google import genai
from google.genai import types
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from openai import OpenAI
import requests


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content or "")


def _normalize_messages(messages) -> list:
    """把上层传入的消息统一成 [{"role","content"}]，忽略空内容。"""
    result = []
    for item in messages or []:
        if isinstance(item, dict):
            content = _stringify_content(item.get("content", ""))
            role = str(item.get("role", "user") or "user").strip().lower()
        else:
            content = _stringify_content(item)
            role = "user"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        if content.strip():
            result.append({"role": role, "content": content})
    return result


def _to_langchain_messages(messages):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    result = []
    for item in _normalize_messages(messages):
        if item["role"] == "system":
            result.append(SystemMessage(content=item["content"]))
        elif item["role"] == "assistant":
            result.append(AIMessage(content=item["content"]))
        else:
            result.append(HumanMessage(content=item["content"]))
    return result


def _extract_langchain_response(adapter, response) -> str:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = getattr(response, "usage_metadata", {}) or {}
    additional = getattr(response, "additional_kwargs", {}) or {}
    diagnostics = {
        "finish_reason": metadata.get("finish_reason"),
        "model": metadata.get("model_name") or metadata.get("model"),
        "usage": usage or metadata.get("token_usage"),
        "additional_fields": sorted(additional.keys()),
    }
    reasoning = additional.get("reasoning_content")
    if reasoning:
        diagnostics["reasoning_content_length"] = len(str(reasoning))
    adapter.last_response_diagnostics = json.dumps(
        {key: value for key, value in diagnostics.items() if value not in (None, {}, [])},
        ensure_ascii=False,
        default=str,
    )
    return _stringify_content(getattr(response, "content", ""))


def _langchain_tool_calls(response) -> list:
    """从 LangChain 的 AIMessage 中取出 tool_calls。"""
    raw_calls = getattr(response, "tool_calls", None) or []
    calls = []
    for item in raw_calls:
        if isinstance(item, dict):
            name = item.get("name")
            args = item.get("args", {})
            if not name:
                function = item.get("function") or {}
                name = function.get("name")
                args = function.get("arguments", {})
        else:
            name = getattr(item, "name", None)
            args = getattr(item, "args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"raw": args}
        calls.append({"name": name, "arguments": args or {}})
    return calls


def _extract_openai_response(adapter, response) -> str:
    choice = response.choices[0] if response and response.choices else None
    usage = getattr(response, "usage", None)
    diagnostics = {
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": getattr(response, "model", None),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
    }
    adapter.last_response_diagnostics = json.dumps(
        {key: value for key, value in diagnostics.items() if value is not None},
        ensure_ascii=False,
        default=str,
    )
    return _stringify_content(choice.message.content if choice else "")


def check_base_url(url: str) -> str:
    """
    处理base_url的规则：
    1. 如果url以#结尾，则移除#并直接使用用户提供的url
    2. 否则检查是否需要添加/v1后缀
    """
    import re
    url = url.strip()
    if not url:
        return url
        
    if url.endswith('#'):
        return url.rstrip('#')
        
    if not re.search(r'/v\d+$', url):
        if '/v1' not in url:
            url = url.rstrip('/') + '/v1'
    return url


def _unsupported_json_mode(exc) -> bool:
    """判断异常是否说明当前后端不支持 JSON 输出模式。"""
    message = str(exc).lower()
    return any(
        token in message
        for token in ("response_format", "json_object", "json schema", "unexpected keyword", "400")
    )


def _build_chat_openai_json_client(adapter):
    """按 adapter 配置创建一个强制 JSON 输出的 LangChain ChatOpenAI 客户端。"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=adapter.model_name,
        api_key=adapter.api_key,
        base_url=adapter.base_url,
        max_tokens=adapter.max_tokens,
        temperature=adapter.temperature,
        timeout=adapter.timeout,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _chat_openai_invoke_json(adapter, prompt, name):
    client = getattr(adapter, "_json_client", None)
    if client is None:
        client = _build_chat_openai_json_client(adapter)
        adapter._json_client = client
    response = client.invoke(prompt)
    if not response:
        logging.warning("No JSON response from %s.", name)
        return ""
    return _extract_langchain_response(adapter, response)


def _openai_sdk_invoke_json(adapter, prompt, system, name):
    """使用适配器自带的 OpenAI SDK 客户端，请求时强制 JSON 输出。"""
    response = adapter._client.chat.completions.create(
        model=adapter.model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=adapter.max_tokens,
        temperature=adapter.temperature,
        timeout=adapter.timeout,
        response_format={"type": "json_object"},
    )
    if not response:
        logging.warning("No JSON response from %s.", name)
        return ""
    return _extract_openai_response(adapter, response)

class BaseLLMAdapter:
    """
    统一的 LLM 接口基类，为不同后端（OpenAI、Ollama、ML Studio、Gemini等）提供一致的方法签名。
    """
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement .invoke(prompt) method.")

    def invoke_json(self, prompt: str) -> str:
        """在请求时强制 JSON 输出；当前后端不支持时自动退化为普通调用。"""
        try:
            return self._invoke_json_request(prompt)
        except Exception as exc:
            if not _unsupported_json_mode(exc):
                raise
            logging.warning("%s 的 JSON 模式不可用，已回退为普通调用：%s", self.__class__.__name__, exc)
            return self.invoke(prompt)

    def _invoke_json_request(self, prompt: str) -> str:
        return self.invoke(prompt)

    def get_last_response_diagnostics(self) -> str:
        return getattr(self, "last_response_diagnostics", "")

    def invoke_messages(self, messages) -> str:
        """默认把结构化消息降级成文本；OpenAI 类适配器会覆盖为真正的 messages 请求。"""
        role_name = {"system": "系统", "assistant": "助手"}
        lines = []
        for item in _normalize_messages(messages):
            speaker = role_name.get(item["role"], "用户")
            lines.append(f"{speaker}：{item['content']}")
        return self.invoke("\n\n".join(lines))

    def invoke_messages_with_tools(self, messages, tools):
        """优先用 LangChain `bind_tools` 调用；不支持工具调用的适配器自动降级。"""
        client = getattr(self, "_client", None)
        bind_tools = getattr(client, "bind_tools", None)
        if not callable(bind_tools):
            return self.invoke_messages(messages), []
        response = client.bind_tools(tools, tool_choice="auto").invoke(_to_langchain_messages(messages))
        if not response:
            logging.warning("No response from %s (bind_tools).", self.__class__.__name__)
            return "", []
        return _extract_langchain_response(self, response), _langchain_tool_calls(response)

class DeepSeekAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def _invoke_json_request(self, prompt: str) -> str:
        return _chat_openai_invoke_json(self, prompt, self.__class__.__name__)

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from DeepSeekAdapter.")
            return ""
        return _extract_langchain_response(self, response)

    def invoke_messages(self, messages) -> str:
        response = self._client.invoke(_to_langchain_messages(messages))
        if not response:
            logging.warning("No response from DeepSeekAdapter.")
            return ""
        return _extract_langchain_response(self, response)

class OpenAIAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def _invoke_json_request(self, prompt: str) -> str:
        return _chat_openai_invoke_json(self, prompt, self.__class__.__name__)

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OpenAIAdapter.")
            return ""
        return _extract_langchain_response(self, response)

    def invoke_messages(self, messages) -> str:
        response = self._client.invoke(_to_langchain_messages(messages))
        if not response:
            logging.warning("No response from OpenAIAdapter.")
            return ""
        return _extract_langchain_response(self, response)

class GeminiAdapter(BaseLLMAdapter):
    """
    适配 Google Gemini (Google Generative AI) 接口
    """

    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # 使用最新的 google-genai 客户端
        self._client = genai.Client(api_key=self.api_key)

    def invoke(self, prompt: str) -> str:
        # 使用当前 google-genai SDK 构造生成参数，API 调用异常保留给上层重试逻辑处理。
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        try:
            text = response.text
        except (ValueError, AttributeError):
            candidate = response.candidates[0] if getattr(response, "candidates", None) else None
            self.last_response_diagnostics = json.dumps({
                "finish_reason": str(getattr(candidate, "finish_reason", "unknown")),
                "safety_ratings": str(getattr(candidate, "safety_ratings", "")),
            }, ensure_ascii=False)
            logging.warning("Gemini response blocked or empty (safety filter).")
            return ""

        if text:
            return text
        else:
            logging.warning("No text response from Gemini API.")
            return ""

    def _invoke_json_request(self, prompt: str) -> str:
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        text = getattr(response, "text", None)
        if text:
            return text
        logging.warning("No JSON text response from Gemini API.")
        return ""

class AzureOpenAIAdapter(BaseLLMAdapter):
    """
    适配 Azure OpenAI 接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        import re
        match = re.match(r'https://(.+?)/openai/deployments/(.+?)/chat/completions\?api-version=(.+)', base_url)
        if match:
            self.azure_endpoint = f"https://{match.group(1)}"
            self.azure_deployment = match.group(2)
            self.api_version = match.group(3)
        else:
            raise ValueError("Invalid Azure OpenAI base_url format")
        
        self.api_key = api_key
        self.model_name = self.azure_deployment
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from AzureOpenAIAdapter.")
            return ""
        return _extract_langchain_response(self, response)

    def invoke_messages(self, messages) -> str:
        response = self._client.invoke(_to_langchain_messages(messages))
        if not response:
            logging.warning("No response from AzureOpenAIAdapter.")
            return ""
        return _extract_langchain_response(self, response)

    def _invoke_json_request(self, prompt: str) -> str:
        client = getattr(self, "_json_client", None)
        if client is None:
            client = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
                api_version=self.api_version,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            self._json_client = client
        response = client.invoke(prompt)
        if not response:
            logging.warning("No JSON response from AzureOpenAIAdapter.")
            return ""
        return _extract_langchain_response(self, response)

class OllamaAdapter(BaseLLMAdapter):
    """
    Ollama 同样有一个 OpenAI-like /v1/chat 接口，可直接使用 ChatOpenAI。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        if self.api_key == '':
            self.api_key= 'ollama'

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OllamaAdapter.")
            return ""
        return _extract_langchain_response(self, response)

    def invoke_messages(self, messages) -> str:
        response = self._client.invoke(_to_langchain_messages(messages))
        if not response:
            logging.warning("No response from OllamaAdapter.")
            return ""
        return _extract_langchain_response(self, response)

    def _invoke_json_request(self, prompt: str) -> str:
        return _chat_openai_invoke_json(self, prompt, self.__class__.__name__)

class MLStudioAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.invoke(prompt)
            if not response:
                logging.warning("No response from MLStudioAdapter.")
                return ""
            return _extract_langchain_response(self, response)
        except Exception as e:
            logging.error(f"ML Studio API 调用超时或失败: {e}")
            raise RuntimeError(f"ML Studio API 调用失败: {e}") from e

    def invoke_messages(self, messages) -> str:
        try:
            response = self._client.invoke(_to_langchain_messages(messages))
            if not response:
                logging.warning("No response from MLStudioAdapter.")
                return ""
            return _extract_langchain_response(self, response)
        except Exception as e:
            logging.error(f"ML Studio API 调用超时或失败: {e}")
            raise RuntimeError(f"ML Studio API 调用失败: {e}") from e

    def _invoke_json_request(self, prompt: str) -> str:
        return _chat_openai_invoke_json(self, prompt, self.__class__.__name__)

class AzureAIAdapter(BaseLLMAdapter):
    """
    适配 Azure AI Inference 接口，用于访问Azure AI服务部署的模型
    使用 azure-ai-inference 库进行API调用
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        import re
        # 匹配形如 https://xxx.services.ai.azure.com/models/chat/completions?api-version=xxx 的URL
        match = re.match(r'https://(.+?)\.services\.ai\.azure\.com(?:/models)?(?:/chat/completions)?(?:\?api-version=(.+))?', base_url)
        if match:
            # endpoint需要是形如 https://xxx.services.ai.azure.com/models 的格式
            self.endpoint = f"https://{match.group(1)}.services.ai.azure.com/models"
            # 如果URL中包含api-version参数，使用它；否则使用默认值
            self.api_version = match.group(2) if match.group(2) else "2024-05-01-preview"
        else:
            raise ValueError("Invalid Azure AI base_url format. Expected format: https://<endpoint>.services.ai.azure.com/models/chat/completions?api-version=xxx")
        
        self.base_url = self.endpoint  # 存储处理后的endpoint URL
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.complete(
                messages=[
                    SystemMessage("You are a helpful assistant."),
                    UserMessage(prompt)
                ]
            )
            if response and response.choices:
                choice = response.choices[0]
                self.last_response_diagnostics = json.dumps({
                    "finish_reason": str(getattr(choice, "finish_reason", "")),
                    "usage": str(getattr(response, "usage", "")),
                }, ensure_ascii=False)
                return _stringify_content(choice.message.content)
            else:
                logging.warning("No response from AzureAIAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Azure AI Inference API 调用失败: {e}")
            raise RuntimeError(f"Azure AI Inference API 调用失败: {e}") from e

    def _invoke_json_request(self, prompt: str) -> str:
        try:
            response = self._client.complete(
                messages=[
                    SystemMessage("You are a helpful assistant."),
                    UserMessage(prompt)
                ],
                response_format={"type": "json_object"}
            )
            if response and response.choices:
                choice = response.choices[0]
                self.last_response_diagnostics = json.dumps({
                    "finish_reason": str(getattr(choice, "finish_reason", "")),
                    "usage": str(getattr(response, "usage", "")),
                }, ensure_ascii=False)
                return _stringify_content(choice.message.content)
            else:
                logging.warning("No JSON response from AzureAIAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Azure AI Inference JSON 调用失败: {e}")
            raise

# 火山引擎实现
class VolcanoEngineAIAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout  # 添加超时配置
        )
    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if not response:
                logging.warning("No response from VolcanoEngineAdapter.")
                return ""
            return _extract_openai_response(self, response)
        except Exception as e:
            logging.error(f"火山引擎API调用超时或失败: {e}")
            raise RuntimeError(f"火山引擎 API 调用失败: {e}") from e

    def invoke_messages(self, messages) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=_normalize_messages(messages),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if not response:
                logging.warning("No response from VolcanoEngineAdapter.")
                return ""
            return _extract_openai_response(self, response)
        except Exception as e:
            logging.error(f"火山引擎API调用超时或失败: {e}")
            raise RuntimeError(f"火山引擎 API 调用失败: {e}") from e

    def _invoke_json_request(self, prompt: str) -> str:
        return _openai_sdk_invoke_json(
            self, prompt, "你是DeepSeek，是一个 AI 人工智能助手", self.__class__.__name__
        )

class SiliconFlowAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout  # 添加超时配置
        )
    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if not response:
                logging.warning("No response from SiliconFlowAdapter.")
                return ""
            return _extract_openai_response(self, response)
        except Exception as e:
            logging.error(f"硅基流动API调用超时或失败: {e}")
            raise RuntimeError(f"硅基流动 API 调用失败: {e}") from e

    def invoke_messages(self, messages) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=_normalize_messages(messages),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if not response:
                logging.warning("No response from SiliconFlowAdapter.")
                return ""
            return _extract_openai_response(self, response)
        except Exception as e:
            logging.error(f"硅基流动API调用超时或失败: {e}")
            raise RuntimeError(f"硅基流动 API 调用失败: {e}") from e

    def _invoke_json_request(self, prompt: str) -> str:
        return _openai_sdk_invoke_json(
            self, prompt, "你是DeepSeek，是一个 AI 人工智能助手", self.__class__.__name__
        )
# grok實現
class GrokAdapter(BaseLLMAdapter):
    """
    适配 xAI Grok API
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are Grok, created by xAI."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if response and response.choices:
                return _extract_openai_response(self, response)
            else:
                logging.warning("No response from GrokAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Grok API 调用失败: {e}")
            raise RuntimeError(f"Grok API 调用失败: {e}") from e

    def invoke_messages(self, messages) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=_normalize_messages(messages),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if response and response.choices:
                return _extract_openai_response(self, response)
            else:
                logging.warning("No response from GrokAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Grok API 调用失败: {e}")
            raise RuntimeError(f"Grok API 调用失败: {e}") from e

    def _invoke_json_request(self, prompt: str) -> str:
        return _openai_sdk_invoke_json(
            self, prompt, "You are Grok, created by xAI.", self.__class__.__name__
        )

def create_llm_adapter(
    interface_format: str,
    base_url: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int
) -> BaseLLMAdapter:
    """
    工厂函数：根据 interface_format 返回不同的适配器实例。
    """
    fmt = interface_format.strip().lower()
    if fmt == "deepseek":
        return DeepSeekAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "openai":
        return OpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "azure openai":
        return AzureOpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "azure ai":
        return AzureAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "ollama":
        return OllamaAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "ml studio":
        return MLStudioAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "gemini":
        return GeminiAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "阿里云百炼":
        return OpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "火山引擎":
        return VolcanoEngineAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "硅基流动":
        return SiliconFlowAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "grok":
        return GrokAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    else:
        raise ValueError(f"Unknown interface_format: {interface_format}")
