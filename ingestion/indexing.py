"""Stable document IDs and bounded vector-store writes."""

from __future__ import annotations

import hashlib
import time
from typing import Any, List

from langchain_core.documents import Document
from openai import APIConnectionError, APIError, RateLimitError
from tqdm import tqdm


def build_doc_id(
    source: str,
    page: int,
    chunk_idx: int,
    chunk: str,
) -> str:
    digest = hashlib.md5(
        chunk.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return f"{source}:p{page}:c{chunk_idx}:{digest}"


def add_documents_with_retry(
    vectorstore: Any,
    docs: List[Document],
    ids: List[str],
    batch_size: int,
    retries: int = 3,
) -> None:
    if batch_size <= 0:
        raise ValueError(
            "RAG_UPSERT_BATCH_SIZE must be greater than 0."
        )
    total = len(docs)
    batch_iter = tqdm(
        range(0, total, batch_size),
        desc="  Embedding & upserting",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for start in batch_iter:
        end = min(start + batch_size, total)
        batch_iter.set_postfix(chunks=f"{end}/{total}")
        docs_batch = docs[start:end]
        ids_batch = ids[start:end]
        attempt = 0
        while True:
            try:
                vectorstore.add_documents(
                    documents=docs_batch,
                    ids=ids_batch,
                )
                break
            except (
                APIConnectionError,
                RateLimitError,
                APIError,
            ) as exc:
                attempt += 1
                if attempt >= retries:
                    raise RuntimeError(
                        "Vectorstore add_documents failed for chunks "
                        f"{start}-{end - 1}: {exc}"
                    ) from exc
                wait_seconds = 2 ** attempt
                tqdm.write(
                    f"  Retry {attempt}/{retries - 1} for chunks "
                    f"{start}-{end - 1} after error: {exc}. "
                    f"Waiting {wait_seconds}s..."
                )
                time.sleep(wait_seconds)

