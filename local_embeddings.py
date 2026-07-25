"""Local embedding backends and stable vector-space identity helpers."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Iterable

from langchain_core.embeddings import Embeddings


DEFAULT_QWEN_QUERY_INSTRUCTION = (
    "Given a technical due diligence question about an NI 43-101 mining "
    "report, retrieve relevant report passages that answer the question"
)


def embedding_signature(
    *,
    provider: str,
    model: str,
    dimensions: int,
    query_instruction: str = "",
    max_length: int | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "dimensions": int(dimensions),
        "normalize": bool(normalize),
    }
    if query_instruction:
        signature["query_instruction"] = query_instruction
    if max_length is not None:
        signature["max_length"] = int(max_length)
    return signature


def signature_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def embedder_signature(embedder: Embeddings) -> dict[str, Any]:
    value = getattr(embedder, "orextractor_embedding_signature", None)
    if not isinstance(value, dict):
        raise RuntimeError(
            "Embedding backend does not expose an ORExtractor vector-space signature."
        )
    return dict(value)


class QwenLocalEmbeddings(Embeddings):
    """Qwen3 dense embeddings on local PyTorch with retrieval instructions."""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        dimensions: int = 1024,
        query_instruction: str = DEFAULT_QWEN_QUERY_INSTRUCTION,
        dtype: str = "float16",
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self.dimensions = max(1, int(dimensions))
        self.query_instruction = query_instruction.strip()
        self.dtype_name = dtype.strip().lower()
        self._lock = threading.RLock()

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen embeddings require torch and transformers. "
                "Install the project requirements before retrying."
            ) from exc

        requested_device = device.strip().lower() or "auto"
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "LOCAL_EMBED_DEVICE requests CUDA, but CUDA-enabled PyTorch "
                "is not available."
            )
        if requested_device not in {"cpu", "cuda"} and not requested_device.startswith(
            "cuda:"
        ):
            raise ValueError(
                "LOCAL_EMBED_DEVICE must be auto, cpu, cuda, or cuda:N; "
                f"got {device!r}."
            )
        self.device = requested_device

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype_name not in dtype_map:
            raise ValueError(
                "LOCAL_EMBED_DTYPE must be float16, bfloat16, or float32; "
                f"got {dtype!r}."
            )
        torch_dtype = (
            torch.float32 if self.device == "cpu" else dtype_map[self.dtype_name]
        )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        self._model = (
            AutoModel.from_pretrained(self.model_name, dtype=torch_dtype)
            .to(self.device)
            .eval()
        )
        hidden_size = int(getattr(self._model.config, "hidden_size", 0) or 0)
        if hidden_size and self.dimensions > hidden_size:
            raise ValueError(
                f"LOCAL_EMBED_DIMENSIONS={self.dimensions} exceeds the model "
                f"hidden size {hidden_size}."
            )

        self.orextractor_embedding_signature = embedding_signature(
            provider="qwen",
            model=self.model_name,
            dimensions=self.dimensions,
            query_instruction=self.query_instruction,
            max_length=self.max_length,
            normalize=True,
        )

    def _last_token_pool(self, hidden_states, attention_mask):
        torch = self._torch
        if attention_mask[:, -1].sum() == attention_mask.shape[0]:
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            sequence_lengths,
        ]

    def _encode(self, texts: Iterable[str]) -> list[list[float]]:
        values = [str(text) for text in texts]
        if not values:
            return []
        torch = self._torch
        vectors: list[list[float]] = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = self._tokenizer(
                    values[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                output = self._model(**batch)
                pooled = self._last_token_pool(
                    output.last_hidden_state,
                    batch["attention_mask"],
                )
                pooled = pooled[:, : self.dimensions]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.float().cpu().tolist())
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        query = str(text)
        if self.query_instruction:
            query = f"Instruct: {self.query_instruction}\nQuery:{query}"
        return self._encode([query])[0]

