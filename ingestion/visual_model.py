"""Provider-neutral visual-model interface and concrete adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import requests
from langchain_core.messages import HumanMessage

from ingestion.models import TableValidation, VisualAnalysis


VisualTask = Literal["figure", "table"]
VisualValue = VisualAnalysis | TableValidation


class VisualModelError(RuntimeError):
    """Normalized failure raised by any visual-model adapter."""


def _strict_output_schema(schema: dict) -> dict:
    """Require explicit values instead of silently accepting model defaults."""

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        value.pop("default", None)
        for child in value.values():
            visit(child)
        properties = value.get("properties")
        if value.get("type") == "object" and isinstance(properties, dict):
            value["required"] = list(properties)
            value["additionalProperties"] = False

    visit(schema)
    return schema


@dataclass(frozen=True)
class VisualRequest:
    """One schema-constrained visual enrichment request."""

    task: VisualTask
    prompt: str
    image_base64: str = ""
    media_type: str = "image/png"


@dataclass(frozen=True)
class VisualResponse:
    """Provider-independent visual result and accounting metadata."""

    value: VisualValue
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class OllamaVisualModel:
    """Ollama adapter implementing the visual-model interface."""

    provider = "ollama"

    def __init__(self, settings):
        self.model_id = str(settings.ollama_visual_model)
        self.cache_id = f"ollama:{self.model_id}"
        self._base_url = str(settings.ollama_base_url).rstrip("/")
        self._timeout = float(settings.ollama_visual_timeout_seconds)
        self._context_length = max(
            4096,
            int(getattr(settings, "ollama_visual_context_length", 8192)),
        )
        self._max_tokens = max(
            1,
            int(getattr(settings, "bedrock_visual_max_tokens", 3500)),
        )
        self._keep_alive = str(getattr(settings, "ollama_keep_alive", "5m"))

    def analyze(self, request: VisualRequest) -> VisualResponse:
        schema_model = VisualAnalysis if request.task == "figure" else TableValidation
        message: dict = {"role": "user", "content": request.prompt}
        if request.image_base64:
            message["images"] = [request.image_base64]
        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self.model_id,
                    "messages": [message],
                    "format": _strict_output_schema(
                        schema_model.model_json_schema()
                    ),
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    "options": {
                        "temperature": 0,
                        "num_ctx": self._context_length,
                        "num_predict": self._max_tokens,
                    },
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty response")
            value = schema_model.model_validate(json.loads(content))
        except requests.RequestException as exc:
            detail = str(exc)
            if exc.response is not None:
                try:
                    detail = str(exc.response.json().get("error") or detail)
                except ValueError:
                    pass
            raise VisualModelError(
                f"Ollama visual request failed for {self.model_id}: {detail}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise VisualModelError(
                f"Ollama returned an invalid {request.task} response for "
                f"{self.model_id}: {exc}"
            ) from exc
        return VisualResponse(
            value=value,
            input_tokens=int(payload.get("prompt_eval_count", 0) or 0),
            output_tokens=int(payload.get("eval_count", 0) or 0),
            latency_ms=float(payload.get("total_duration", 0) or 0) / 1_000_000,
        )


class BedrockVisualModel:
    """Bedrock adapter implementing the same visual-model interface."""

    provider = "bedrock"

    def __init__(self, settings):
        self.model_id = str(settings.bedrock_visual_model_id)
        self.cache_id = f"bedrock:{self.model_id}"
        self._settings = settings
        self._models: dict[VisualTask, object] = {}

    def _model_for(self, task: VisualTask):
        if task not in self._models:
            from ingestion.bedrock import (
                get_table_validation_model,
                get_visual_analysis_model,
            )

            factory = (
                get_visual_analysis_model
                if task == "figure"
                else get_table_validation_model
            )
            self._models[task] = factory(self._settings)
        return self._models[task]

    def analyze(self, request: VisualRequest) -> VisualResponse:
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        if request.image_base64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": request.media_type,
                        "data": request.image_base64,
                    },
                }
            )
        response = self._model_for(request.task).invoke(
            [HumanMessage(content=content)]
        )
        raw = None
        parsed = response
        if isinstance(response, dict) and "parsed" in response:
            parsing_error = response.get("parsing_error")
            if parsing_error:
                if isinstance(parsing_error, BaseException):
                    raise parsing_error
                raise RuntimeError(str(parsing_error))
            parsed = response.get("parsed")
            raw = response.get("raw")
        schema_model = VisualAnalysis if request.task == "figure" else TableValidation
        if not isinstance(parsed, schema_model):
            parsed = schema_model.model_validate(parsed)
        usage = getattr(raw, "usage_metadata", None) or {}
        response_metadata = getattr(raw, "response_metadata", None) or {}
        if not usage and isinstance(response_metadata, dict):
            usage = response_metadata.get("usage", {}) or response_metadata.get(
                "usage_metadata", {}
            )
        return VisualResponse(
            value=parsed,
            input_tokens=int(
                usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
            ),
            output_tokens=int(
                usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
            ),
        )


def create_visual_model(settings):
    """Create the configured visual-model adapter."""

    provider = str(getattr(settings, "visual_model_provider", "bedrock")).lower()
    if provider == "ollama":
        return OllamaVisualModel(settings)
    if provider == "bedrock":
        return BedrockVisualModel(settings)
    raise ValueError(f"Unsupported visual model provider: {provider!r}")
