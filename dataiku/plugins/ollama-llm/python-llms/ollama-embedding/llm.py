import requests
from dataiku.llm.python import BaseEmbeddingModel


class OllamaEmbedding(BaseEmbeddingModel):
    def __init__(self):
        self.base_url = self.model = None

    def set_config(self, config: dict, plugin_config: dict) -> None:
        self.base_url = (config.get("base_url") or "http://localhost:11434").rstrip("/")
        self.model = config.get("model") or "bge-m3"
        self.timeout = int(config.get("timeout") or 120)

    def process(self, query, settings, trace):
        r = requests.post(f"{self.base_url}/api/embed",
                          json={"model": self.model, "input": query.get("text") or ""},
                          timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        return {"embedding": out["embeddings"][0],
                "promptTokens": out.get("prompt_eval_count", 0),
                "estimatedCost": 0}
