"""Conversion between LLM Mesh messages and the Ollama chat API."""
from __future__ import annotations

import json
from typing import Any


def _text_and_images(message: dict) -> tuple[str, list[str]]:
    text = message.get("content") or ""
    images = []
    for part in message.get("parts") or []:
        if part.get("text"):
            text += ("\n" if text else "") + part["text"]
        if part.get("inlineImage"):
            images.append(part["inlineImage"])
    return text, images


def to_ollama_messages(messages: list[dict]) -> list[dict]:
    """LLM Mesh chat messages -> Ollama messages (tool outputs become one message each)."""
    out: list[dict] = []
    names: dict[str, str] = {}  # tool call id -> function name, for tool messages
    for m in messages:
        role = m.get("role", "user")
        if m.get("toolOutputs"):
            for t in m["toolOutputs"]:
                msg = {"role": "tool", "content": t.get("output", "")}
                if t.get("callId") in names:
                    msg["tool_name"] = names[t["callId"]]
                out.append(msg)
            continue
        text, images = _text_and_images(m)
        msg: dict[str, Any] = {"role": role, "content": text}
        if images:
            msg["images"] = images
        if m.get("toolCalls"):
            calls = []
            for c in m["toolCalls"]:
                fn = c.get("function", {})
                names[c.get("id", "")] = fn.get("name", "")
                args = fn.get("arguments") or "{}"
                calls.append({"function": {"name": fn.get("name"),
                                           "arguments": json.loads(args) if isinstance(args, str) else args}})
            msg["tool_calls"] = calls
        out.append(msg)
    return out


def with_system_instruction(messages: list[dict], instruction: str) -> list[dict]:
    """Prepends the instruction, or appends it to the system message already present."""
    if not instruction:
        return messages
    if messages and messages[0]["role"] == "system":
        first = dict(messages[0], content=f"{messages[0]['content']}\n\n{instruction}".strip())
        return [first] + messages[1:]
    return [{"role": "system", "content": instruction}] + messages


def to_ollama_tools(tools: list[dict] | None) -> list[dict]:
    result = []
    for t in tools or []:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        result.append({"type": "function", "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }})
    return result


def to_mesh_tool_calls(calls: list[dict] | None) -> list[dict]:
    """Ollama tool calls -> LLM Mesh function tool calls (arguments as a JSON string)."""
    return [{
        "type": "function",
        "id": f"call_{i}",
        "index": i,
        "function": {"name": c["function"]["name"],
                     "arguments": json.dumps(c["function"].get("arguments") or {})},
    } for i, c in enumerate(calls or [])]
