#novel_generator/common.py
# -*- coding: utf-8 -*-
"""
通用重试、清洗、日志工具
"""
import logging
import json
import os
import re
import threading
import time
import traceback
import uuid
from datetime import datetime

from langchain_core.utils.function_calling import convert_to_openai_tool

logging.basicConfig(
    filename='app.log',      # 日志文件名
    filemode='a',            # 追加模式（'w' 会覆盖）
    level=logging.INFO,      # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_llm_trace_lock = threading.RLock()
_llm_trace_listeners = []
LLM_INTERACTION_LOG = os.path.abspath("llm_interactions.log")


def _write_llm_trace_to_file(event: dict):
    """把完整 LLM 交互写入独立日志，不再依赖界面标签页。"""
    try:
        with open(LLM_INTERACTION_LOG, "a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("Failed to write LLM interaction log")


def _tool_schema_for_log(tool) -> dict:
    try:
        return convert_to_openai_tool(tool)
    except Exception:
        name = getattr(tool, "name", str(tool))
        description = getattr(tool, "description", "") or ""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


def _request_payload_for_log(llm_adapter, messages: list, tools: list | None = None) -> dict:
    payload = {
        "model": str(getattr(llm_adapter, "model_name", "") or llm_adapter.__class__.__name__),
        "messages": messages,
        "max_tokens": int(getattr(llm_adapter, "max_tokens", 0) or 0),
        "temperature": float(getattr(llm_adapter, "temperature", 0) or 0),
    }
    if tools:
        payload["tools"] = [_tool_schema_for_log(tool) for tool in tools]
        payload["tool_choice"] = "auto"
    return payload


def add_llm_trace_listener(listener):
    """注册 LLM 交互监听器。监听器会收到可直接展示的事件字典。"""
    with _llm_trace_lock:
        if listener not in _llm_trace_listeners:
            _llm_trace_listeners.append(listener)


def remove_llm_trace_listener(listener):
    with _llm_trace_lock:
        if listener in _llm_trace_listeners:
            _llm_trace_listeners.remove(listener)


def emit_llm_trace(event_type: str, request_id: str, content: str = "", **metadata):
    event = {
        "type": event_type,
        "request_id": request_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": content or "",
        **metadata,
    }
    _write_llm_trace_to_file(event)
    with _llm_trace_lock:
        listeners = list(_llm_trace_listeners)
    for listener in listeners:
        try:
            listener(event)
        except Exception:
            logging.exception("LLM trace listener failed")
def call_with_retry(func, max_retries=3, sleep_time=2, fallback_return=None, **kwargs):
    """
    通用的重试机制封装。
    :param func: 要执行的函数
    :param max_retries: 最大重试次数
    :param sleep_time: 重试前的等待秒数
    :param fallback_return: 如果多次重试仍失败时的返回值
    :param kwargs: 传给func的命名参数
    :return: func的结果，若失败则返回 fallback_return
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            logging.warning(f"[call_with_retry] Attempt {attempt} failed with error: {e}")
            traceback.print_exc()
            if attempt < max_retries:
                time.sleep(sleep_time)
            else:
                logging.error("Max retries reached, returning fallback_return.")
                return fallback_return

def remove_think_tags(text: str) -> str:
    """移除 <think>...</think> 包裹的内容"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def is_llm_io_debug_enabled() -> bool:
    return os.getenv("AI_NOVEL_DEBUG_LLM_IO", "").strip().lower() in {"1", "true", "yes", "on"}

def log_llm_io(label: str, content: str):
    """默认只记录长度，避免日志泄露完整提示词、正文或用户私有素材。"""
    content = content or ""
    if is_llm_io_debug_enabled():
        logging.info(
            f"\n[#########################################  {label}  #########################################]\n{content}\n"
        )
    else:
        logging.info("[LLM IO] %s length=%s", label, len(content))

def debug_log(prompt: str, response_content: str):
    log_llm_io("Prompt", prompt)
    log_llm_io("Response", response_content)

def invoke_with_cleaning(llm_adapter, prompt: str, max_retries: int = 3, label: str = "LLM 请求", invoke_json: bool = False) -> str:
    """调用 LLM 并清理返回结果"""
    request_id = uuid.uuid4().hex[:8]
    model_name = getattr(llm_adapter, "model_name", llm_adapter.__class__.__name__)
    request_text = json.dumps(
        _request_payload_for_log(llm_adapter, [{"role": "user", "content": prompt}]),
        ensure_ascii=False,
        indent=2,
    )
    emit_llm_trace(
        "request",
        request_id,
        request_text,
        model=str(model_name),
        label=label,
        max_retries=max_retries,
    )
    log_llm_io("Prompt", prompt)

    retry_count = 0

    while retry_count < max_retries:
        try:
            emit_llm_trace(
                "attempt",
                request_id,
                f"正在等待模型响应（第 {retry_count + 1}/{max_retries} 次）",
                attempt=retry_count + 1,
                max_retries=max_retries,
                model=str(model_name),
                label=label,
            )
            result = llm_adapter.invoke_json(prompt) if invoke_json else llm_adapter.invoke(prompt)
            log_llm_io("Response", result)

            # 清理结果中的特殊格式标记
            result = remove_think_tags(result).replace("```", "").strip()
            if result:
                emit_llm_trace("response", request_id, result, model=str(model_name), label=label)
                return result
            retry_count += 1
            diagnostics = ""
            if hasattr(llm_adapter, "get_last_response_diagnostics"):
                diagnostics = llm_adapter.get_last_response_diagnostics() or ""
            elif hasattr(llm_adapter, "last_response_diagnostics"):
                diagnostics = str(llm_adapter.last_response_diagnostics or "")
            empty_message = "接口调用成功，但模型响应正文为空"
            if diagnostics:
                empty_message += f"\n响应诊断：{diagnostics}"
            emit_llm_trace(
                "retry",
                request_id,
                empty_message + ("\n准备重试" if retry_count < max_retries else "\n已达到最大重试次数"),
                attempt=retry_count,
                max_retries=max_retries,
                model=str(model_name),
                label=label,
            )
            logging.warning(
                "LLM 返回空内容 (%s/%s). diagnostics=%s",
                retry_count,
                max_retries,
                diagnostics or "unavailable",
            )
        except Exception as e:
            retry_count += 1
            emit_llm_trace(
                "error",
                request_id,
                str(e),
                attempt=retry_count,
                max_retries=max_retries,
                will_retry=retry_count < max_retries,
                model=str(model_name),
                label=label,
            )
            logging.error(f"调用失败 ({retry_count}/{max_retries}): {str(e)}")
            if retry_count >= max_retries:
                emit_llm_trace("failed", request_id, str(e), model=str(model_name), label=label)
                raise

    emit_llm_trace("failed", request_id, "多次重试后仍未获得有效响应", model=str(model_name), label=label)
    raise RuntimeError(f"LLM 在 {max_retries} 次重试后仍返回空内容")


def invoke_messages_with_cleaning(llm_adapter, messages: list, max_retries: int = 3, label: str = "LLM 请求") -> str:
    """以结构化 messages 调用 LLM，并复用请求日志与重试逻辑。"""
    request_id = uuid.uuid4().hex[:8]
    model_name = getattr(llm_adapter, "model_name", llm_adapter.__class__.__name__)
    request_text = json.dumps(_request_payload_for_log(llm_adapter, messages), ensure_ascii=False, indent=2)
    emit_llm_trace(
        "request",
        request_id,
        request_text,
        model=str(model_name),
        label=label,
        max_retries=max_retries,
    )
    log_llm_io("Messages", request_text)

    invoker = getattr(llm_adapter, "invoke_messages", None)
    if not callable(invoker):
        invoker = lambda messages: llm_adapter.invoke("\n\n".join(
            f"{item.get('role', 'user')}：{item.get('content', '')}" if isinstance(item, dict) else str(item)
            for item in messages
        ))

    retry_count = 0
    while retry_count < max_retries:
        try:
            emit_llm_trace(
                "attempt",
                request_id,
                f"正在等待模型响应（第 {retry_count + 1}/{max_retries} 次）",
                attempt=retry_count + 1,
                max_retries=max_retries,
                model=str(model_name),
                label=label,
            )
            result = invoker(messages)
            log_llm_io("Response", result)
            result = remove_think_tags(result).replace("```", "").strip()
            if result:
                emit_llm_trace("response", request_id, result, model=str(model_name), label=label)
                return result
            retry_count += 1
            diagnostics = ""
            if hasattr(llm_adapter, "get_last_response_diagnostics"):
                diagnostics = llm_adapter.get_last_response_diagnostics() or ""
            elif hasattr(llm_adapter, "last_response_diagnostics"):
                diagnostics = str(llm_adapter.last_response_diagnostics or "")
            empty_message = "接口调用成功，但模型响应正文为空"
            if diagnostics:
                empty_message += f"\n响应诊断：{diagnostics}"
            emit_llm_trace(
                "retry",
                request_id,
                empty_message + ("\n准备重试" if retry_count < max_retries else "\n已达到最大重试次数"),
                attempt=retry_count,
                max_retries=max_retries,
                model=str(model_name),
                label=label,
            )
            logging.warning(
                "LLM 返回空内容 (%s/%s). diagnostics=%s",
                retry_count,
                max_retries,
                diagnostics or "unavailable",
            )
        except Exception as e:
            retry_count += 1
            emit_llm_trace(
                "error",
                request_id,
                str(e),
                attempt=retry_count,
                max_retries=max_retries,
                will_retry=retry_count < max_retries,
                model=str(model_name),
                label=label,
            )
            logging.error(f"调用失败 ({retry_count}/{max_retries}): {str(e)}")
            if retry_count >= max_retries:
                emit_llm_trace("failed", request_id, str(e), model=str(model_name), label=label)
                raise

    emit_llm_trace("failed", request_id, "多次重试后仍未获得有效响应", model=str(model_name), label=label)
    raise RuntimeError(f"LLM 在 {max_retries} 次重试后仍返回空内容")


def invoke_messages_with_tools_cleaning(llm_adapter, messages: list, tools: list, max_retries: int = 3, label: str = "LLM 请求"):
    """以 LangChain `bind_tools` 调用支持工具调用的适配器，返回 (正文, tool_calls)。"""
    invoker = getattr(llm_adapter, "invoke_messages_with_tools", None)
    if not callable(invoker):
        return invoke_messages_with_cleaning(llm_adapter, messages, max_retries=max_retries, label=label), []

    request_id = uuid.uuid4().hex[:8]
    model_name = getattr(llm_adapter, "model_name", llm_adapter.__class__.__name__)
    request_text = json.dumps(
        _request_payload_for_log(llm_adapter, messages, tools),
        ensure_ascii=False,
        indent=2,
    )
    emit_llm_trace(
        "request",
        request_id,
        request_text,
        model=str(model_name),
        label=label,
        max_retries=max_retries,
    )
    log_llm_io("Messages", request_text)

    retry_count = 0
    while retry_count < max_retries:
        try:
            emit_llm_trace(
                "attempt",
                request_id,
                f"正在等待模型响应（第 {retry_count + 1}/{max_retries} 次）",
                attempt=retry_count + 1,
                max_retries=max_retries,
                model=str(model_name),
                label=label,
            )
            content, tool_calls = invoker(messages, tools)
            content = remove_think_tags(content or "").replace("```", "").strip()
            if content or tool_calls:
                response_text = json.dumps(
                    {"content": content, "tool_calls": tool_calls},
                    ensure_ascii=False,
                    indent=2,
                )
                log_llm_io("Response", response_text)
                emit_llm_trace(
                    "response",
                    request_id,
                    response_text,
                    model=str(model_name),
                    label=label,
                    tool_calls=len(tool_calls),
                )
                return content, tool_calls
            retry_count += 1
            diagnostics = ""
            if hasattr(llm_adapter, "get_last_response_diagnostics"):
                diagnostics = llm_adapter.get_last_response_diagnostics() or ""
            elif hasattr(llm_adapter, "last_response_diagnostics"):
                diagnostics = str(llm_adapter.last_response_diagnostics or "")
            empty_message = "接口调用成功，但模型响应正文为空"
            if diagnostics:
                empty_message += f"\n响应诊断：{diagnostics}"
            emit_llm_trace(
                "retry",
                request_id,
                empty_message + ("\n准备重试" if retry_count < max_retries else "\n已达到最大重试次数"),
                attempt=retry_count,
                max_retries=max_retries,
                model=str(model_name),
                label=label,
            )
            logging.warning("LLM bind_tools 返回空内容 (%s/%s). diagnostics=%s", retry_count, max_retries, diagnostics or "unavailable")
        except Exception as e:
            retry_count += 1
            emit_llm_trace(
                "error",
                request_id,
                str(e),
                attempt=retry_count,
                max_retries=max_retries,
                will_retry=retry_count < max_retries,
                model=str(model_name),
                label=label,
            )
            logging.error(f"工具调用失败 ({retry_count}/{max_retries}): {str(e)}")
            if retry_count >= max_retries:
                emit_llm_trace("failed", request_id, str(e), model=str(model_name), label=label)
                raise

    emit_llm_trace("failed", request_id, "多次重试后仍未获得有效响应", model=str(model_name), label=label)
    raise RuntimeError(f"LLM 在 {max_retries} 次重试后仍返回空内容")
