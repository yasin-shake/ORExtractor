from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

import local_embeddings
import rag_app
from local_embeddings import embedder_signature, embedding_signature


class _FakeEmbeddings(Embeddings):
    def __init__(self, provider: str, model: str, dimensions: int = 3):
        self.orextractor_embedding_signature = embedding_signature(
            provider=provider,
            model=model,
            dimensions=dimensions,
        )
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dimensions - 1) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (self.dimensions - 1)


def _settings(tmp_path=None):
    return SimpleNamespace(
        embedding_provider="qwen",
        embedding_fallback_provider="openai",
        local_embed_model="Qwen/Qwen3-Embedding-0.6B",
        local_embed_device="cuda",
        local_embed_batch_size=16,
        local_embed_max_length=512,
        local_embed_dimensions=1024,
        local_embed_dtype="float16",
        local_embed_query_instruction="Retrieve NI 43-101 passages",
        embed_model="text-embedding-3-small",
        embed_batch_size=64,
        openai_embed_dimensions=1536,
        openai_base_url=None,
        openai_api_key="test-key",
        chroma_dir=tmp_path,
        collection_name="test_embeddings",
    )


def test_qwen_startup_failure_uses_openai_fallback(monkeypatch):
    class _BrokenQwen:
        def __init__(self, **kwargs):
            raise RuntimeError("CUDA unavailable")

    fallback = _FakeEmbeddings("openai", "text-embedding-3-small", 3)
    monkeypatch.setattr(local_embeddings, "QwenLocalEmbeddings", _BrokenQwen)
    monkeypatch.setattr(rag_app, "_openai_embedder", lambda settings: fallback)
    rag_app._EMBEDDER_INSTANCES.clear()

    settings = _settings()
    selected = rag_app.get_embedder(settings)

    assert selected is fallback
    assert settings.resolved_embedding_provider == "openai"
    assert settings.resolved_embedding_signature == embedder_signature(fallback)


def test_vectorstore_rejects_embedding_space_mismatch(tmp_path):
    settings = _settings(tmp_path)
    qwen = _FakeEmbeddings("qwen", "qwen-test", 3)
    vectorstore = rag_app.get_vectorstore(settings, qwen)
    vectorstore.add_texts(["mineral resource estimate"], ids=["chunk-1"])

    openai = _FakeEmbeddings("openai", "openai-test", 3)
    with pytest.raises(RuntimeError, match="incompatible embedding space"):
        rag_app.get_vectorstore(settings, openai)


def test_empty_collection_can_adopt_new_embedding_signature(tmp_path):
    settings = _settings(tmp_path)
    rag_app.get_vectorstore(settings, _FakeEmbeddings("qwen", "qwen-test", 3))
    vectorstore = rag_app.get_vectorstore(
        settings,
        _FakeEmbeddings("openai", "openai-test", 3),
    )

    assert "openai-test" in vectorstore._collection.metadata[
        "orextractor_embedding_signature"
    ]
