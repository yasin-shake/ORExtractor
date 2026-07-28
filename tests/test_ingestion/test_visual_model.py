import base64
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from ingestion.models import VisualAnalysis
from ingestion.visual_model import VisualModelError, VisualRequest, create_visual_model
from rag_app import load_settings


@contextmanager
def _ollama_server(response_content: dict, *, status: int = 200):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self.server.request_body = json.loads(self.rfile.read(length))
            response_body = (
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(response_content),
                    },
                    "prompt_eval_count": 23,
                    "eval_count": 11,
                    "total_duration": 1_500_000,
                }
                if status < 400
                else response_content
            )
            payload = json.dumps(response_body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_ollama_visual_model_returns_schema_valid_analysis():
    expected = {
        "figure_type": "bar_chart",
        "caption": "Annual production",
        "description": "Production rises from 2025 to 2027.",
        "contains_quantitative_data": True,
        "reconstruction_supported": False,
        "reconstruction_method": "none",
        "values_are_estimated": False,
        "confidence": 0.93,
        "warnings": [],
        "labels": ["2025", "2026", "2027"],
        "chart": None,
        "diagram": None,
    }
    with _ollama_server(expected) as server:
        settings = SimpleNamespace(
            visual_model_provider="ollama",
            ollama_base_url=f"http://127.0.0.1:{server.server_port}",
            ollama_visual_model="qwen3-vl:8b-instruct-q8_0",
            ollama_visual_timeout_seconds=5,
            ollama_visual_context_length=8192,
            ollama_keep_alive="5m",
            bedrock_visual_max_tokens=3500,
        )
        model = create_visual_model(settings)
        result = model.analyze(
            VisualRequest(
                task="figure",
                prompt="Analyse this chart.",
                image_base64=base64.b64encode(b"png-bytes").decode("ascii"),
                media_type="image/png",
            )
        )

    assert isinstance(result.value, VisualAnalysis)
    assert result.value.description == expected["description"]
    assert result.input_tokens == 23
    assert result.output_tokens == 11
    assert result.latency_ms == 1.5

    body = server.request_body
    assert body["model"] == "qwen3-vl:8b-instruct-q8_0"
    assert body["stream"] is False
    assert body["format"]["title"] == "VisualAnalysis"
    assert set(body["format"]["required"]) == set(
        body["format"]["properties"]
    )
    chart_schema = body["format"]["$defs"]["ChartSpecification"]
    assert set(chart_schema["required"]) == set(chart_schema["properties"])
    assert all(
        "default" not in property_schema
        for property_schema in body["format"]["properties"].values()
    )
    assert body["options"]["num_ctx"] == 8192
    assert body["options"]["num_predict"] == 3500
    assert body["messages"][0]["images"] == [
        base64.b64encode(b"png-bytes").decode("ascii")
    ]


def test_ollama_visual_model_normalizes_remote_errors():
    with _ollama_server({"error": "model is unavailable"}, status=503) as server:
        settings = SimpleNamespace(
            visual_model_provider="ollama",
            ollama_base_url=f"http://127.0.0.1:{server.server_port}",
            ollama_visual_model="qwen3-vl:test",
            ollama_visual_timeout_seconds=5,
            ollama_keep_alive="5m",
        )
        model = create_visual_model(settings)
        with pytest.raises(
            VisualModelError,
            match="qwen3-vl:test.*model is unavailable",
        ):
            model.analyze(VisualRequest(task="table", prompt="Validate."))


def test_visual_model_settings_select_local_qwen(monkeypatch):
    monkeypatch.setenv("VISUAL_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:31415")
    monkeypatch.setenv("OLLAMA_VISUAL_MODEL", "qwen3-vl:test")
    monkeypatch.setenv("OLLAMA_VISUAL_TIMEOUT_SECONDS", "47")
    monkeypatch.setenv("OLLAMA_VISUAL_CONTEXT_LENGTH", "6144")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "9m")
    monkeypatch.setenv("VISUAL_MODEL_CONCURRENCY", "1")

    settings = load_settings()

    assert settings.visual_model_provider == "ollama"
    assert settings.ollama_base_url == "http://127.0.0.1:31415"
    assert settings.ollama_visual_model == "qwen3-vl:test"
    assert settings.ollama_visual_timeout_seconds == 47
    assert settings.ollama_visual_context_length == 6144
    assert settings.ollama_keep_alive == "9m"
    assert settings.visual_model_concurrency == 1


def test_settings_repr_redacts_api_credentials(monkeypatch):
    secrets = {
        "OPENAI_API_KEY": "openai-secret-sentinel",
        "DOCLING_SERVE_API_KEY": "docling-secret-sentinel",
        "MINERU_API_TOKEN": "mineru-secret-sentinel",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)

    rendered = repr(load_settings())

    assert all(value not in rendered for value in secrets.values())
