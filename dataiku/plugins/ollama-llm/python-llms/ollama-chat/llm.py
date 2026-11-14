import requests
from dataiku.llm.python import BaseLLM

from ollama_llm.conversion import (to_mesh_tool_calls, to_ollama_messages, to_ollama_tools,
                                   with_system_instruction)


class OllamaChat(BaseLLM):
    def __init__(self):
        self.base_url = self.model = None

    def set_config(self, config: dict, plugin_config: dict) -> None:
        self.base_url = (config.get("base_url") or "http://localhost:11434").rstrip("/")
        self.model = config.get("model") or "qwen2.5:7b"
        self.num_ctx = int(config.get("num_ctx") or 8192)
        self.timeout = int(config.get("timeout") or 600)
        self.system_prompt = (config.get("system_prompt") or "").strip()

    def process(self, query, settings, trace):
        settings = settings or {}
        options = {"num_ctx": self.num_ctx}
        for mesh_key, ollama_key in (("temperature", "temperature"), ("topP", "top_p"),
                                     ("topK", "top_k"), ("maxOutputTokens", "num_predict"),
                                     ("stopSequences", "stop")):
            if settings.get(mesh_key) is not None:
                options[ollama_key] = settings[mesh_key]

        body = {"model": self.model, "messages": with_system_instruction(
                    to_ollama_messages(query["messages"]), self.system_prompt),
                "stream": False, "options": options}
        tools = to_ollama_tools(settings.get("tools"))
        if tools:
            body["tools"] = tools
        response_format = settings.get("responseFormat") or {}
        if response_format.get("schema"):
            body["format"] = response_format["schema"]  # structured output
        elif "json" in str(response_format.get("type", "")).lower():
            body["format"] = "json"  # JSON mode

        r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()

        message = out.get("message", {})
        result = {
            "text": message.get("content") or "",
            "promptTokens": out.get("prompt_eval_count", 0),
            "completionTokens": out.get("eval_count", 0),
            "estimatedCost": 0,
        }
        if message.get("tool_calls"):
            result["toolCalls"] = to_mesh_tool_calls(message["tool_calls"])
        return result
