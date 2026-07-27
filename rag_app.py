import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from ingestion.indexing import (
    add_documents_with_retry as _add_documents_with_retry,
    build_doc_id,
)
from ingestion.sources import (
    filesystem_path,
    iter_pdf_paths,
    pdf_source_id,
    source_output_path,
)

# When this file is executed directly, ingestion modules import helpers using
# ``from rag_app import ...``. Alias the running module so Python does not load
# a second copy (and, critically, a second local embedding model).
if __name__ == "__main__":
    sys.modules.setdefault("rag_app", sys.modules[__name__])

_MAX_HISTORY_TURNS = 10  # user+assistant pairs to keep in conversation history

# Domain instruction shared by the CLI, API and Streamlit chat surfaces.
SYSTEM_INSTRUCTION = (
    "You are a technical due diligence assistant specialised in NI 43-101 mineral project "
    "reports (Form 43-101F1). Use chapter-directed retrieval: cite the NI Item number and "
    "section title (e.g. Item 14 — Mineral Resource Estimates) alongside page numbers. "
    "Answer using only the supplied context. When you state figures such as resource "
    "tonnages, grades, contained metal, cut-off grades, NPV, IRR or capital costs, quote "
    "them exactly as written and include their units. Where possible, cite the relevant "
    "report section, NI Item, page number and the Qualified Person responsible. "
    "When comparing to peers, state typical ranges and flag outliers. "
    "If the context is incomplete, say what is known and what is unknown. "
    "If the context is not relevant, say you do not know."
)

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
        "shall", "can", "need", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below", "between", "under",
        "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or", "because", "until",
        "while", "this", "that", "these", "those", "what", "which", "who", "whom", "i", "you", "he",
        "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "its", "our",
        "their",
    }
)


@dataclass
class Settings:
    openai_api_key: str
    openai_base_url: Optional[str]
    embed_model: str
    aws_region: str
    bedrock_model_id: str
    knowledge_dir: Path
    extra_pdf_dirs: List[Path]
    chroma_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    upsert_batch_size: int
    top_k: int
    extracted_dir: Path
    extract_top_k: int
    spatial_dir: Path = Path("spatial_data")
    ingestion_pipeline_enabled: bool = True
    ingestion_pipeline_queue_size: int = 2
    embedding_provider: str = "qwen"
    embedding_fallback_provider: str = "openai"
    openai_embed_dimensions: int = 1536
    local_embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    local_embed_device: str = "cuda"
    local_embed_batch_size: int = 16
    local_embed_max_length: int = 512
    local_embed_dimensions: int = 1024
    local_embed_dtype: str = "float16"
    local_embed_query_instruction: str = (
        "Given a technical due diligence question about an NI 43-101 mining "
        "report, retrieve relevant report passages that answer the question"
    )
    # Parser-neutral document and visual ingestion
    ingestion_backend: str = "docling"
    parser_primary: str = "docling"
    force_parser: str = ""
    parser_fallback: str = "mineru"
    parser_fallback_enabled: bool = True
    parser_min_text_page_coverage: float = 0.90
    parser_max_empty_page_ratio: float = 0.10
    parser_max_replacement_char_ratio: float = 0.01
    parser_min_table_valid_ratio: float = 0.80
    parser_require_picture_crops: bool = False
    parser_min_cache_quality_score: float = 0.90
    parser_min_page_count_agreement: float = 0.90
    parser_require_fallback_ready: bool = True
    docling_execution_mode: str = "local"
    docling_serve_url: Optional[str] = None
    docling_serve_api_key: str = ""
    docling_do_ocr: bool = True
    docling_ocr_backend: str = "onnxruntime"
    docling_ocr_languages: str = "english"
    docling_force_full_page_ocr: bool = False
    docling_ocr_bitmap_area_threshold: float = 0.05
    docling_do_table_structure: bool = True
    docling_table_mode: str = "accurate"
    docling_text_first_table_mode: str = "fast"
    docling_generate_page_images: bool = True
    docling_generate_picture_images: bool = True
    docling_images_scale: float = 1.0
    docling_ocr_batch_size: int = 2
    docling_layout_batch_size: int = 2
    docling_table_batch_size: int = 1
    docling_queue_max_size: int = 2
    docling_page_batch_size: int = 2
    docling_num_threads: int = 4
    docling_device: str = "auto"
    docling_adaptive_ocr: bool = True
    docling_native_text_min_chars: int = 80
    docling_native_text_coverage: float = 0.98
    docling_native_text_max_empty_pages: int = 2
    docling_batch_fallback_enabled: bool = True
    docling_safe_batch_size: int = 1
    docling_fast_table_max_pages: int = 20
    docling_profiling: bool = True
    docling_converter_cache_size: int = 2
    docling_heading_hierarchy: bool = True
    docling_timeout_seconds: int = 900
    docling_hard_timeout_seconds: int = 900
    docling_process_isolation: bool = True
    docling_segment_min_pages: int = 300
    docling_segment_pages: int = 100
    docling_max_pages: int = 1000
    docling_max_file_mb: int = 2048
    docling_model_artifact_revision: str = ""
    mineru_execution_mode: str = "service"
    mineru_api_url: Optional[str] = None
    mineru_api_token: str = ""
    mineru_command: str = "mineru"
    mineru_backend: str = "pipeline"
    mineru_timeout_seconds: int = 1800
    artifact_dir: Path = Path("ingestion_artifacts")
    ingest_work_dir: Path = Path(".ingestion_work")
    bedrock_visual_model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    bedrock_visual_max_tokens: int = 3500
    bedrock_visual_concurrency: int = 8
    bedrock_visual_confidence_threshold: float = 0.85
    visual_min_width: int = 250
    visual_min_height: int = 150
    visual_max_width: int = 4096
    visual_max_height: int = 4096
    visual_max_calls_per_report: int = 30
    visual_max_table_calls_per_report: int = 20
    visual_max_figure_calls_per_report: int = 10
    visual_token_budget_per_report: int = 350000
    visual_reconstruct_charts: bool = True
    visual_reconstruct_diagrams: bool = True
    visual_enrichment_enabled: bool = True
    langsmith_tracing: bool = False
    langsmith_project: str = "orextractor-ingestion"
    langsmith_trace_content: bool = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    if not Path(".env").exists():
        print("Warning: no .env file found, using defaults. Copy .env.example to .env to configure.")
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    backend = os.getenv("INGESTION_BACKEND", "docling").strip().lower() or "docling"
    if backend not in {"docling", "mineru"}:
        raise RuntimeError(
            f"INGESTION_BACKEND must be 'docling' or 'mineru', got {backend!r}"
        )
    return Settings(
        openai_api_key=openai_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        aws_region=os.getenv("AWS_REGION", "us-east-2"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "arn:aws:bedrock:us-east-2:387653681033:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        knowledge_dir=Path(os.getenv("RAG_KNOWLEDGE_DIR", "knowledge")),
        extra_pdf_dirs=[
            Path(p.strip())
            for p in os.getenv("RAG_EXTRA_PDF_DIRS", "").split(";")
            if p.strip()
        ],
        chroma_dir=Path(os.getenv("RAG_CHROMA_DIR", ".chroma_db")),
        collection_name=os.getenv("RAG_COLLECTION_NAME", "ni43101_knowledge"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1400")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
        embed_batch_size=int(os.getenv("RAG_EMBED_BATCH_SIZE", "64")),
        upsert_batch_size=int(os.getenv("RAG_UPSERT_BATCH_SIZE", "128")),
        top_k=int(os.getenv("RAG_TOP_K", "8")),
        extracted_dir=Path(os.getenv("RAG_EXTRACTED_DIR", "extracted_data")),
        extract_top_k=int(os.getenv("NI43101_EXTRACT_TOP_K", "12")),
        spatial_dir=Path(os.getenv("RAG_SPATIAL_DIR", "spatial_data")),
        ingestion_pipeline_enabled=_env_bool("INGEST_PIPELINE_ENABLED", True),
        ingestion_pipeline_queue_size=max(
            1, int(os.getenv("INGEST_PIPELINE_QUEUE_SIZE", "2"))
        ),
        embedding_provider=os.getenv(
            "EMBEDDING_PROVIDER", "qwen"
        ).strip().lower() or "qwen",
        embedding_fallback_provider=os.getenv(
            "EMBEDDING_FALLBACK_PROVIDER", "openai"
        ).strip().lower(),
        openai_embed_dimensions=int(
            os.getenv("OPENAI_EMBED_DIMENSIONS", "1536")
        ),
        local_embed_model=os.getenv(
            "LOCAL_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"
        ).strip(),
        local_embed_device=os.getenv(
            "LOCAL_EMBED_DEVICE", "cuda"
        ).strip().lower(),
        local_embed_batch_size=int(
            os.getenv("LOCAL_EMBED_BATCH_SIZE", "16")
        ),
        local_embed_max_length=int(
            os.getenv("LOCAL_EMBED_MAX_LENGTH", "512")
        ),
        local_embed_dimensions=int(
            os.getenv("LOCAL_EMBED_DIMENSIONS", "1024")
        ),
        local_embed_dtype=os.getenv(
            "LOCAL_EMBED_DTYPE", "float16"
        ).strip().lower(),
        local_embed_query_instruction=os.getenv(
            "LOCAL_EMBED_QUERY_INSTRUCTION",
            (
                "Given a technical due diligence question about an NI 43-101 "
                "mining report, retrieve relevant report passages that answer "
                "the question"
            ),
        ).strip(),
        ingestion_backend=backend,
        parser_primary=os.getenv("PARSER_PRIMARY", "docling").strip().lower() or "docling",
        force_parser=os.getenv("FORCE_PARSER", "").strip().lower(),
        parser_fallback=os.getenv("PARSER_FALLBACK", "mineru").strip().lower(),
        parser_fallback_enabled=_env_bool("PARSER_FALLBACK_ENABLED", True),
        parser_min_text_page_coverage=float(
            os.getenv("PARSER_MIN_TEXT_PAGE_COVERAGE", "0.90")
        ),
        parser_max_empty_page_ratio=float(
            os.getenv("PARSER_MAX_EMPTY_PAGE_RATIO", "0.10")
        ),
        parser_max_replacement_char_ratio=float(
            os.getenv("PARSER_MAX_REPLACEMENT_CHAR_RATIO", "0.01")
        ),
        parser_min_table_valid_ratio=float(
            os.getenv("PARSER_MIN_TABLE_VALID_RATIO", "0.80")
        ),
        parser_require_picture_crops=_env_bool(
            "PARSER_REQUIRE_PICTURE_CROPS", False
        ),
        parser_min_cache_quality_score=float(
            os.getenv("PARSER_MIN_CACHE_QUALITY_SCORE", "0.90")
        ),
        parser_min_page_count_agreement=float(
            os.getenv("PARSER_MIN_PAGE_COUNT_AGREEMENT", "0.90")
        ),
        parser_require_fallback_ready=_env_bool(
            "PARSER_REQUIRE_FALLBACK_READY", True
        ),
        docling_execution_mode=os.getenv(
            "DOCLING_EXECUTION_MODE", "local"
        ).strip().lower(),
        docling_serve_url=os.getenv("DOCLING_SERVE_URL", "").strip() or None,
        docling_serve_api_key=os.getenv("DOCLING_SERVE_API_KEY", "").strip(),
        docling_do_ocr=_env_bool("DOCLING_DO_OCR", True),
        docling_ocr_backend=os.getenv(
            "DOCLING_OCR_BACKEND", "onnxruntime"
        ).strip().lower(),
        docling_ocr_languages=os.getenv(
            "DOCLING_OCR_LANGUAGES", "english"
        ).strip(),
        docling_force_full_page_ocr=_env_bool(
            "DOCLING_FORCE_FULL_PAGE_OCR", False
        ),
        docling_ocr_bitmap_area_threshold=float(
            os.getenv("DOCLING_OCR_BITMAP_AREA_THRESHOLD", "0.05")
        ),
        docling_do_table_structure=_env_bool("DOCLING_DO_TABLE_STRUCTURE", True),
        docling_table_mode=os.getenv("DOCLING_TABLE_MODE", "accurate").strip(),
        docling_text_first_table_mode=os.getenv(
            "DOCLING_TEXT_FIRST_TABLE_MODE", "fast"
        ).strip().lower(),
        docling_generate_page_images=_env_bool(
            "DOCLING_GENERATE_PAGE_IMAGES", True
        ),
        docling_generate_picture_images=_env_bool(
            "DOCLING_GENERATE_PICTURE_IMAGES", True
        ),
        docling_images_scale=float(os.getenv("DOCLING_IMAGES_SCALE", "1.0")),
        docling_ocr_batch_size=int(os.getenv("DOCLING_OCR_BATCH_SIZE", "2")),
        docling_layout_batch_size=int(os.getenv("DOCLING_LAYOUT_BATCH_SIZE", "2")),
        docling_table_batch_size=int(os.getenv("DOCLING_TABLE_BATCH_SIZE", "1")),
        docling_queue_max_size=int(os.getenv("DOCLING_QUEUE_MAX_SIZE", "2")),
        docling_page_batch_size=int(os.getenv("DOCLING_PAGE_BATCH_SIZE", "2")),
        docling_num_threads=int(os.getenv("DOCLING_NUM_THREADS", "4")),
        docling_device=os.getenv("DOCLING_DEVICE", "auto").strip().lower() or "auto",
        docling_adaptive_ocr=_env_bool("DOCLING_ADAPTIVE_OCR", True),
        docling_native_text_min_chars=int(
            os.getenv("DOCLING_NATIVE_TEXT_MIN_CHARS", "80")
        ),
        docling_native_text_coverage=float(
            os.getenv("DOCLING_NATIVE_TEXT_COVERAGE", "0.98")
        ),
        docling_native_text_max_empty_pages=int(
            os.getenv("DOCLING_NATIVE_TEXT_MAX_EMPTY_PAGES", "2")
        ),
        docling_batch_fallback_enabled=_env_bool(
            "DOCLING_BATCH_FALLBACK_ENABLED", True
        ),
        docling_safe_batch_size=max(
            1, int(os.getenv("DOCLING_SAFE_BATCH_SIZE", "1"))
        ),
        docling_fast_table_max_pages=max(
            0, int(os.getenv("DOCLING_FAST_TABLE_MAX_PAGES", "20"))
        ),
        docling_profiling=_env_bool("DOCLING_PROFILING", True),
        docling_converter_cache_size=max(
            1, int(os.getenv("DOCLING_CONVERTER_CACHE_SIZE", "2"))
        ),
        docling_heading_hierarchy=_env_bool("DOCLING_HEADING_HIERARCHY", True),
        docling_timeout_seconds=int(os.getenv("DOCLING_TIMEOUT_SECONDS", "900")),
        docling_hard_timeout_seconds=int(
            os.getenv("DOCLING_HARD_TIMEOUT_SECONDS", "900")
        ),
        docling_process_isolation=_env_bool(
            "DOCLING_PROCESS_ISOLATION", True
        ),
        docling_segment_min_pages=max(
            0, int(os.getenv("DOCLING_SEGMENT_MIN_PAGES", "300"))
        ),
        docling_segment_pages=max(
            0, int(os.getenv("DOCLING_SEGMENT_PAGES", "100"))
        ),
        docling_max_pages=int(os.getenv("DOCLING_MAX_PAGES", "1000")),
        docling_max_file_mb=int(os.getenv("DOCLING_MAX_FILE_MB", "2048")),
        docling_model_artifact_revision=os.getenv(
            "DOCLING_MODEL_ARTIFACT_REVISION", ""
        ).strip(),
        mineru_execution_mode=os.getenv(
            "MINERU_EXECUTION_MODE", "service"
        ).strip().lower(),
        mineru_api_url=os.getenv("MINERU_API_URL", "").strip() or None,
        mineru_api_token=os.getenv("MINERU_API_TOKEN", "").strip(),
        mineru_command=os.getenv("MINERU_COMMAND", "mineru").strip() or "mineru",
        mineru_backend=os.getenv("MINERU_BACKEND", "pipeline").strip() or "pipeline",
        mineru_timeout_seconds=int(os.getenv("MINERU_TIMEOUT_SECONDS", "1800")),
        artifact_dir=Path(os.getenv("RAG_ARTIFACT_DIR", "ingestion_artifacts")),
        ingest_work_dir=Path(
            os.getenv("RAG_INGEST_WORK_DIR", ".ingestion_work")
        ),
        bedrock_visual_model_id=os.getenv(
            "BEDROCK_VISUAL_MODEL_ID",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ),
        bedrock_visual_max_tokens=int(os.getenv("BEDROCK_VISUAL_MAX_TOKENS", "3500")),
        bedrock_visual_concurrency=int(os.getenv("BEDROCK_VISUAL_CONCURRENCY", "8")),
        bedrock_visual_confidence_threshold=float(
            os.getenv("BEDROCK_VISUAL_CONFIDENCE_THRESHOLD", "0.85")
        ),
        visual_min_width=int(os.getenv("VISUAL_MIN_WIDTH", "250")),
        visual_min_height=int(os.getenv("VISUAL_MIN_HEIGHT", "150")),
        visual_max_width=int(os.getenv("VISUAL_MAX_WIDTH", "4096")),
        visual_max_height=int(os.getenv("VISUAL_MAX_HEIGHT", "4096")),
        visual_max_calls_per_report=int(os.getenv("VISUAL_MAX_CALLS_PER_REPORT", "30")),
        visual_max_table_calls_per_report=int(
            os.getenv("VISUAL_MAX_TABLE_CALLS_PER_REPORT", "20")
        ),
        visual_max_figure_calls_per_report=int(
            os.getenv("VISUAL_MAX_FIGURE_CALLS_PER_REPORT", "10")
        ),
        visual_token_budget_per_report=int(
            os.getenv("VISUAL_TOKEN_BUDGET_PER_REPORT", "350000")
        ),
        visual_reconstruct_charts=_env_bool("VISUAL_RECONSTRUCT_CHARTS", True),
        visual_reconstruct_diagrams=_env_bool("VISUAL_RECONSTRUCT_DIAGRAMS", True),
        visual_enrichment_enabled=_env_bool("VISUAL_ENRICHMENT_ENABLED", True),
        langsmith_tracing=_env_bool("LANGSMITH_TRACING", False),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "orextractor-ingestion"),
        langsmith_trace_content=_env_bool("LANGSMITH_TRACE_CONTENT", False),
    )


def get_vectorstore(settings: Settings, embedder: Embeddings) -> Chroma:
    from local_embeddings import embedder_signature, signature_json

    signature = signature_json(embedder_signature(embedder))
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embedder,
        collection_metadata={
            "hnsw:space": "cosine",
            "orextractor_embedding_signature": signature,
        },
    )
    collection = vectorstore._collection
    metadata = dict(collection.metadata or {})
    recorded = metadata.get("orextractor_embedding_signature")
    count = int(collection.count())
    if count and recorded != signature:
        recorded_label = recorded or "legacy/unknown"
        raise RuntimeError(
            "The existing Chroma collection uses an incompatible embedding "
            f"space ({recorded_label}). The configured backend resolves to "
            f"{signature}. Rebuild it with: python rag_app.py ingest --rebuild "
            "--parser docling --fallback mineru"
        )
    if not count and recorded != signature:
        # Chroma rejects hnsw:* keys on modify even when their value is
        # unchanged. Omitting them preserves the collection's configured
        # distance function while updating mutable identity metadata.
        mutable_metadata = {
            key: value
            for key, value in metadata.items()
            if not str(key).startswith("hnsw:")
        }
        mutable_metadata["orextractor_embedding_signature"] = signature
        collection.modify(metadata=mutable_metadata)
    return vectorstore


_EMBEDDER_INSTANCES: dict[tuple, Embeddings] = {}


def _openai_embedder(settings: Settings) -> OpenAIEmbeddings:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured, so the OpenAI embedding "
            "fallback is unavailable."
        )
    kwargs = {
        "api_key": settings.openai_api_key,
        "model": settings.embed_model,
        "chunk_size": settings.embed_batch_size,
    }
    if settings.openai_embed_dimensions > 0:
        kwargs["dimensions"] = settings.openai_embed_dimensions
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    embedder = _CachedOpenAIEmbeddings(**kwargs)
    from local_embeddings import embedding_signature

    object.__setattr__(
        embedder,
        "orextractor_embedding_signature",
        embedding_signature(
            provider="openai",
            model=settings.embed_model,
            dimensions=settings.openai_embed_dimensions,
            normalize=True,
        ),
    )
    return embedder


def get_embedder(settings: Settings) -> Embeddings:
    from local_embeddings import QwenLocalEmbeddings, embedder_signature

    provider = str(settings.embedding_provider).strip().lower()
    fallback = str(settings.embedding_fallback_provider).strip().lower()
    valid = {"qwen", "openai", ""}
    if provider not in valid - {""}:
        raise ValueError(
            f"EMBEDDING_PROVIDER must be qwen or openai, got {provider!r}."
        )
    if fallback not in valid:
        raise ValueError(
            "EMBEDDING_FALLBACK_PROVIDER must be openai, qwen, or empty, "
            f"got {fallback!r}."
        )
    cache_key = (
        provider,
        fallback,
        settings.local_embed_model,
        settings.local_embed_device,
        settings.local_embed_batch_size,
        settings.local_embed_max_length,
        settings.local_embed_dimensions,
        settings.local_embed_dtype,
        settings.local_embed_query_instruction,
        settings.embed_model,
        settings.openai_embed_dimensions,
        settings.openai_base_url,
        hashlib.sha256(settings.openai_api_key.encode()).hexdigest()[:12],
    )
    cached = _EMBEDDER_INSTANCES.get(cache_key)
    if cached is not None:
        signature = embedder_signature(cached)
        settings.resolved_embedding_provider = signature["provider"]
        settings.resolved_embedding_model = signature["model"]
        settings.resolved_embedding_signature = signature
        return cached

    def create(selected: str) -> Embeddings:
        if selected == "openai":
            remote = _openai_embedder(settings)
            # Validate credentials/connectivity before a rebuild can remove an
            # existing collection. This query is retained in the process cache.
            remote.embed_query("ORExtractor embedding backend health check")
            return remote
        if selected == "qwen":
            local = QwenLocalEmbeddings(
                model_name=settings.local_embed_model,
                device=settings.local_embed_device,
                batch_size=settings.local_embed_batch_size,
                max_length=settings.local_embed_max_length,
                dimensions=settings.local_embed_dimensions,
                query_instruction=settings.local_embed_query_instruction,
                dtype=settings.local_embed_dtype,
            )
            # Eager health check: fallback is allowed only before an index is
            # opened or built, never after vectors from another space exist.
            local.embed_query("ORExtractor embedding backend health check")
            return local
        raise ValueError(f"Unsupported embedding provider: {selected!r}")

    try:
        embedder = create(provider)
    except Exception as primary_error:
        if not fallback or fallback == provider:
            raise
        try:
            embedder = create(fallback)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Primary embedding provider {provider!r} failed: "
                f"{primary_error}. Fallback {fallback!r} also failed: "
                f"{fallback_error}."
            ) from fallback_error
        print(
            f"Warning: embedding provider {provider!r} failed ({primary_error}); "
            f"using startup fallback {fallback!r}.",
        )

    signature = embedder_signature(embedder)
    settings.resolved_embedding_provider = signature["provider"]
    settings.resolved_embedding_model = signature["model"]
    settings.resolved_embedding_signature = signature
    _EMBEDDER_INSTANCES[cache_key] = embedder
    print(
        "Embedding backend: "
        f"{signature['provider']} / {signature['model']} "
        f"({signature['dimensions']} dimensions)"
    )
    return embedder


def get_chat_model(
    settings: Settings, max_tokens: int = 4096, temperature: float = 0.35
) -> ChatBedrockConverse:
    # Spatial extraction passes a higher limit: a collar table with hundreds of
    # holes cannot fit its structured output inside the 4096-token chat default.
    # It also passes temperature=0 — run-to-run variance in table extraction
    # showed up as whole datasets appearing/disappearing between identical runs.
    from botocore.config import Config

    # botocore's default 60s read timeout is shorter than a large structured
    # extraction takes to generate — long-context passes were dying mid-response.
    return ChatBedrockConverse(
        model_id=settings.bedrock_model_id,
        provider="amazon",
        temperature=temperature,
        max_tokens=max_tokens,
        region_name=settings.aws_region,
        config=Config(read_timeout=300, connect_timeout=15, retries={"max_attempts": 2, "mode": "adaptive"}),
    )


def _question_keywords(question: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", question.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _keyword_overlap_score(doc: str, keywords: set[str]) -> int:
    if not keywords:
        return 0
    lower = doc.lower()
    return sum(1 for w in keywords if w in lower)


def rerank_chunks(
    documents: List[str],
    metadatas: List[dict],
    distances: Optional[List[float]],
    question: str,
    top_k: int,
) -> Tuple[List[str], List[dict]]:
    keywords = _question_keywords(question)
    dists = distances if distances is not None else [0.0] * len(documents)
    scored = list(zip(documents, metadatas, dists))
    # Table chunks (type="table") carry a +2 keyword bonus: numeric tables hold
    # the precise figures that semantic embeddings represent only loosely.
    def _rank_key(doc: str, meta: dict, dist: float) -> tuple:
        kw = _keyword_overlap_score(doc, keywords)
        bonus = 2 if meta.get("type") == "table" else 0
        return (-(kw + bonus), dist)
    scored.sort(key=lambda x: _rank_key(*x))
    top = scored[:top_k]
    return [t[0] for t in top], [t[1] for t in top]


_EMBED_CACHE: dict[tuple, list] = {}


class _CachedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings with a process-level embed_query cache.

    Avoids re-embedding identical query strings across extraction passes and
    multi-report runs in extract_all().
    """

    def embed_query(self, text: str) -> list:  # type: ignore[override]
        key = (
            str(getattr(self, "model", "")),
            getattr(self, "dimensions", None),
            str(getattr(self, "openai_api_base", "")),
            text,
        )
        if key not in _EMBED_CACHE:
            _EMBED_CACHE[key] = super().embed_query(text)
        return _EMBED_CACHE[key]


def ingest(
    settings: Settings,
    rebuild: bool = False,
    *,
    only_file: Optional[str] = None,
    enable_visuals: bool = True,
    partition_only: bool = False,
    reprocess_visuals: bool = False,
    backend: Optional[str] = None,
):
    """Ingest PDFs into Chroma through the selected parser pipeline."""
    from ingestion.pipeline import IngestionPipeline
    from ingestion.runtime import IngestionRuntime

    chosen = (backend or settings.ingestion_backend or "docling").lower()
    if chosen not in {"docling", "mineru"}:
        raise ValueError(f"Parser must be 'docling' or 'mineru', got {chosen!r}")
    settings.ingestion_backend = chosen
    settings.parser_primary = chosen
    if chosen == "mineru":
        settings.force_parser = "mineru"
    pipeline = IngestionPipeline(
        settings,
        enable_visuals=enable_visuals,
        partition_only=partition_only,
        runtime=IngestionRuntime(
            get_embedder=get_embedder,
            get_vectorstore=get_vectorstore,
        ),
    )
    return pipeline.ingest_all(
        rebuild=rebuild,
        only_file=only_file,
        reprocess_visuals=reprocess_visuals,
    )


def inspect_elements(settings: Settings, pdf_file: str) -> dict:
    """Parse a PDF and print canonical element and routing diagnostics."""
    from ingestion.pipeline import IngestionPipeline

    pdf_paths = list(iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs))
    requested = str(pdf_file).replace("\\", "/").casefold()
    matches = [
        p
        for p in pdf_paths
        if pdf_source_id(
            p,
            settings.knowledge_dir,
            settings.extra_pdf_dirs,
        ).casefold()
        == requested
    ]
    if not matches:
        matches = [
            p
            for p in pdf_paths
            if p.name.casefold() == Path(pdf_file).name.casefold()
            or p.stem.casefold() == Path(pdf_file).stem.casefold()
        ]
    if not matches:
        path = Path(pdf_file)
        if path.exists():
            matches = [path]
    if not matches:
        raise FileNotFoundError(f"PDF not found: {pdf_file}")
    if len(matches) > 1:
        choices = ", ".join(
            pdf_source_id(p, settings.knowledge_dir, settings.extra_pdf_dirs)
            for p in matches
        )
        raise ValueError(f"Ambiguous PDF name {pdf_file!r}; use one of: {choices}")
    pipeline = IngestionPipeline(settings, enable_visuals=False, partition_only=True)
    source_file = pdf_source_id(
        matches[0],
        settings.knowledge_dir,
        settings.extra_pdf_dirs,
    )
    return pipeline.inspect_elements(
        filesystem_path(matches[0]),
        source_file=source_file,
        artifact_dir=source_output_path(settings.artifact_dir, source_file, ""),
    )


def compare_parsers(settings: Settings, pdf_file: str) -> dict:
    """Run Docling and MinerU diagnostically without writing production vectors."""
    from copy import copy

    from ingestion.parsers.router import get_parser_router

    pdf_paths = list(iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs))
    requested = str(pdf_file).replace("\\", "/").casefold()
    matches = [
        p
        for p in pdf_paths
        if pdf_source_id(
            p,
            settings.knowledge_dir,
            settings.extra_pdf_dirs,
        ).casefold()
        == requested
    ]
    if not matches:
        matches = [
            p
            for p in pdf_paths
            if p.name.casefold() == Path(pdf_file).name.casefold()
            or p.stem.casefold() == Path(pdf_file).stem.casefold()
        ]
    direct = Path(pdf_file)
    if not matches and direct.exists():
        matches = [direct]
    if not matches:
        raise FileNotFoundError(f"PDF not found: {pdf_file}")
    if len(matches) > 1:
        choices = ", ".join(
            pdf_source_id(p, settings.knowledge_dir, settings.extra_pdf_dirs)
            for p in matches
        )
        raise ValueError(f"Ambiguous PDF name {pdf_file!r}; use one of: {choices}")

    comparison: dict[str, dict] = {}
    source_file = pdf_source_id(
        matches[0],
        settings.knowledge_dir,
        settings.extra_pdf_dirs,
    )
    for parser_name in ("docling", "mineru"):
        diagnostic = copy(settings)
        diagnostic.force_parser = parser_name
        diagnostic.parser_primary = parser_name
        diagnostic.parser_fallback_enabled = False
        result = get_parser_router(diagnostic).parse(
            filesystem_path(matches[0]),
            source_file=source_file,
            artifact_dir=source_output_path(
                diagnostic.artifact_dir,
                source_file,
                "",
            ),
        )
        comparison[parser_name] = {
            "status": result.status,
            "version": result.parser_version,
            "duration_ms": result.duration_ms,
            "page_count": result.page_count,
            "elements": len(result.elements),
            "quality": result.quality.model_dump(mode="json"),
            "errors": result.errors,
            "artifacts": result.artifact_paths,
        }
    return {"file": source_file, "parsers": comparison}

def reindex_chapters(settings: Settings, vectorstore: Chroma) -> None:
    """Rebuild chapter indexes directly from documents already in Chroma."""
    from chapter_index import (
        build_chapter_index_from_documents,
        save_chapter_index,
        tag_documents_with_items,
    )

    pdf_paths = list(iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs))
    if not pdf_paths:
        print("No PDFs found to reindex.")
        return

    updated = 0
    for pdf_path in tqdm(pdf_paths, desc="Reindexing chapters", unit="pdf"):
        source_file = pdf_source_id(
            pdf_path,
            settings.knowledge_dir,
            settings.extra_pdf_dirs,
        )
        try:
            existing = vectorstore._collection.get(
                where={"source": source_file},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            tqdm.write(f"  {source_file}: could not read Chroma records ({exc})")
            continue
        ids = list(existing.get("ids") or [])
        texts = list(existing.get("documents") or [])
        raw_metadatas = list(existing.get("metadatas") or [])
        if not ids or len(ids) != len(texts):
            continue
        docs = [
            Document(page_content=text, metadata=dict(metadata or {}))
            for text, metadata in zip(texts, raw_metadatas)
        ]
        chapters = build_chapter_index_from_documents(docs)
        docs = tag_documents_with_items(docs, chapters)
        save_chapter_index(settings.extracted_dir, source_file, chapters)
        metadatas = [dict(document.metadata) for document in docs]
        try:
            vectorstore._collection.update(ids=ids, metadatas=metadatas)
            updated += len(ids)
            tqdm.write(f"  {source_file}: {len(chapters)} chapters, {len(ids)} chunks tagged")
        except Exception as exc:
            tqdm.write(f"  {source_file}: metadata update failed ({exc}); re-ingest recommended")

    print(f"Chapter reindex complete. Updated metadata on {updated} chunks.")


def _build_chroma_filter(
    filter_sources: Optional[List[str]] = None,
    filter_items: Optional[List[int]] = None,
    filter_types: Optional[List[str]] = None,
) -> Optional[dict]:
    clauses: List[dict] = []
    if filter_sources and len(filter_sources) == 1:
        clauses.append({"source": filter_sources[0]})
    elif filter_sources:
        clauses.append({"source": {"$in": list(filter_sources)}})
    if filter_items:
        items = [int(i) for i in filter_items if int(i) > 0]
        if len(items) == 1:
            clauses.append({"ni_item": items[0]})
        elif items:
            clauses.append({"ni_item": {"$in": items}})
    if filter_types:
        if len(filter_types) == 1:
            clauses.append({"type": filter_types[0]})
        else:
            clauses.append({"type": {"$in": list(filter_types)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _retrieve_and_rerank(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    filter_sources: Optional[List[str]] = None,
    filter_items: Optional[List[int]] = None,
    filter_types: Optional[List[str]] = None,
) -> Tuple[List[str], List[dict]]:
    """Shared retrieval + rerank core. Returns (documents, metadatas) lists."""
    try:
        count = vectorstore._collection.count()
    except Exception:
        return [], []
    if count == 0:
        return [], []

    fetch_n = min(max(top_k * 4, top_k + 12), count)
    chroma_filter = _build_chroma_filter(filter_sources, filter_items, filter_types)
    results = vectorstore.similarity_search_with_score(
        question, k=fetch_n, filter=chroma_filter
    )
    documents = [r[0].page_content for r in results]
    metadatas  = [r[0].metadata    for r in results]
    distances  = [float(r[1])      for r in results]
    if not documents:
        return [], []
    if len(distances) != len(documents):
        distances = None
    return rerank_chunks(documents, metadatas, distances, question, top_k)


def _format_parts(documents: List[str], metadatas: List[dict]) -> List[str]:
    """Format each (doc, metadata) pair into a labelled context string."""
    parts = []
    for idx, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        ni_item = metadata.get("ni_item") or 0
        section = metadata.get("section_title") or ""
        item_label = f"Item {ni_item} ({section})" if ni_item else "section unknown"
        parts.append(f"[{idx}] {source} {item_label} page {page}\n{doc}")
    return parts


def query_chunks(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    filter_sources: Optional[List[str]] = None,
    filter_items: Optional[List[int]] = None,
    filter_types: Optional[List[str]] = None,
) -> Tuple[List[str], List[dict]]:
    """Return individual labelled chunk strings and their metadata dicts.

    Unlike query_context, chunks are never joined — safe for per-chunk
    deduplication in the extractor without risking misalignment on blank lines.
    """
    documents, metadatas = _retrieve_and_rerank(
        vectorstore, question, top_k, filter_sources, filter_items, filter_types
    )
    if not documents:
        return [], []
    return _format_parts(documents, metadatas), metadatas


def query_context(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    filter_sources: Optional[List[str]] = None,
    filter_items: Optional[List[int]] = None,
) -> Tuple[str, List[dict]]:
    """Return retrieved context as a single joined string (used by chat/API)."""
    documents, metadatas = _retrieve_and_rerank(
        vectorstore, question, top_k, filter_sources, filter_items=filter_items
    )
    if not documents:
        return "", []
    parts = _format_parts(documents, metadatas)
    return "\n\n".join(parts), metadatas


def query_by_items(
    vectorstore: Chroma,
    question: str,
    items: List[int],
    top_k: int,
    filter_sources: Optional[List[str]] = None,
) -> Tuple[str, List[dict]]:
    """Retrieve context scoped to specific NI 43-101 Items.

    Uses a single Chroma query with a $in filter across all requested items
    rather than N separate queries, cutting embedding API calls from N to 1.
    _build_chroma_filter already emits {"ni_item": {"$in": items}} when
    len(items) > 1, so no special-casing is needed here.
    """
    if not items:
        return query_context(vectorstore, question, top_k, filter_sources)
    docs, metas = _retrieve_and_rerank(
        vectorstore, question, top_k, filter_sources, filter_items=items
    )
    allowed_items = {int(item) for item in items if int(item) > 0}
    scoped_pairs = []
    for document, metadata in zip(docs, metas):
        try:
            item = int(metadata.get("ni_item") or 0)
        except (TypeError, ValueError):
            item = 0
        if item in allowed_items:
            scoped_pairs.append((document, metadata))
    if not scoped_pairs:
        return "", []
    docs = [document for document, _ in scoped_pairs]
    metas = [metadata for _, metadata in scoped_pairs]
    parts = _format_parts(docs, metas)
    return "\n\n".join(parts), metas


def _index_is_empty(vectorstore: Chroma) -> bool:
    """Safe wrapper around Chroma's internal count — insulates against API changes."""
    try:
        return vectorstore._collection.count() == 0
    except Exception:
        return True


def _is_short_greeting_or_thanks(text: str) -> bool:
    t = text.lower().strip()
    if len(t) > 50:
        return False
    return bool(re.match(r"^(hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay)\b[\s!.?]*$", t))


def _history_to_text(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "None"
    out: List[str] = []
    for user_q, assistant_a in history[-_MAX_HISTORY_TURNS:]:
        out.append(f"User: {user_q}\nAssistant: {assistant_a}")
    return "\n\n".join(out)


def build_chat_prompt(
    question: str,
    context: str,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Assemble the JSON chat prompt shared across CLI, API and Streamlit."""
    prompt_payload = {
        "instruction": SYSTEM_INSTRUCTION,
        "context": context,
        "question": question,
    }
    if history is not None:
        prompt_payload["history"] = _history_to_text(history)
    return (
        "Follow the instruction and answer clearly.\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=True, indent=2)}"
    )


def chat(settings: Settings, vectorstore: Chroma, llm: ChatBedrockConverse) -> None:
    if _index_is_empty(vectorstore):
        print("Vector index is empty. Run ingest first:")
        print("  python rag_app.py ingest")
        return

    conversation: List[Tuple[str, str]] = []
    print("NI 43-101 RAG chat ready. Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if _is_short_greeting_or_thanks(question):
            print(
                "\nAssistant: I answer questions from the indexed NI 43-101 reports in "
                f"`{settings.knowledge_dir}`. Ask something specific about the project, "
                "resources, economics or geology."
            )
            continue

        t_total_start = time.perf_counter()
        t_retrieval_start = time.perf_counter()
        context, metadatas = query_context(vectorstore, question, settings.top_k)
        t_retrieval = time.perf_counter() - t_retrieval_start
        if not context:
            print("\nAssistant: I could not find relevant context in the indexed reports.")
            continue

        prompt = build_chat_prompt(question, context, history=conversation)

        t_gen_start = time.perf_counter()
        response = llm.invoke([HumanMessage(content=prompt)])
        t_generation = time.perf_counter() - t_gen_start
        t_total = time.perf_counter() - t_total_start

        answer = str(response.content)
        conversation.append((question, answer))
        if len(conversation) > _MAX_HISTORY_TURNS:
            conversation = conversation[-_MAX_HISTORY_TURNS:]

        print(f"\nAssistant: {answer}")
        print(
            f"\nTiming: retrieval {t_retrieval:.2f}s | generation {t_generation:.2f}s | total {t_total:.2f}s"
        )
        print("\nSources:")
        seen = set()
        for metadata in metadatas:
            key = (metadata.get("source"), metadata.get("page"), metadata.get("chunk"))
            if key in seen:
                continue
            seen.add(key)
            print(
                f"- {metadata.get('source', 'unknown')} page {metadata.get('page', '?')} "
                f"(chunk {metadata.get('chunk', '?')})"
            )


def run_extract_spatial(
    settings: Settings,
    vectorstore: Chroma,
    target_file: Optional[str] = None,
) -> None:
    """CLI entry point for spatial/geological-model extraction.

    Builds its own LLM with a raised output limit (collar tables with hundreds
    of holes overflow the 4096-token chat default). Existing spatial files are
    skipped so a re-run never clobbers reviewed (`confirmed`) or digitized data —
    delete the JSON to force re-extraction.
    """
    from extractor import extract_spatial, save_spatial_extraction

    if _index_is_empty(vectorstore):
        print("Vector index is empty. Run ingest first:")
        print("  python rag_app.py ingest")
        return

    llm = get_chat_model(settings, max_tokens=16000, temperature=0.0)
    if target_file:
        targets = [target_file]
    else:
        targets = [
            pdf_source_id(
                path,
                settings.knowledge_dir,
                settings.extra_pdf_dirs,
            )
            for path in iter_pdf_paths(
                settings.knowledge_dir,
                settings.extra_pdf_dirs,
            )
        ]

    done = 0
    for name in targets:
        out_file = source_output_path(settings.spatial_dir, name, ".json")
        if out_file.exists():
            print(f"Already extracted (delete {out_file} to redo), skipping: {name}")
            continue
        print(f"Spatial extraction: {name}")
        extraction = extract_spatial(settings, vectorstore, llm, name)
        out_path = save_spatial_extraction(settings, extraction)
        print(
            f"  OK {len(extraction.boreholes)} boreholes, "
            f"{len(extraction.lithology_intervals)} lithology intervals, "
            f"{len(extraction.stratigraphic_pile)} strat units, "
            f"{len(extraction.orientations)} orientations, "
            f"{len(extraction.faults)} faults -> {out_path}"
        )
        done += 1
    print(f"Spatial extraction complete. Extracted {done} report(s). Review and set 'confirmed' before modeling.")


def run_extract(
    settings: Settings,
    vectorstore: Chroma,
    llm: ChatBedrockConverse,
    target_file: Optional[str] = None,
) -> None:
    """CLI entry point for structured extraction."""
    from extractor import extract_all, extract_report

    if _index_is_empty(vectorstore):
        print("Vector index is empty. Run ingest first:")
        print("  python rag_app.py ingest")
        return

    if target_file:
        out_file = source_output_path(settings.extracted_dir, target_file, ".json")
        if out_file.exists():
            print(f"Already extracted, skipping: {target_file}")
            return
        report = extract_report(settings, vectorstore, llm, target_file)
        out_path = save_extraction(settings, report)
        print(f"Extracted {target_file} -> {out_path}")
    else:
        done = 0
        for _, report in extract_all(settings, vectorstore, llm, skip_existing=True):
            out_path = save_extraction(settings, report)
            tqdm.write(f"  ✓ Saved -> {out_path}")
            done += 1
        print(f"Extraction complete. Extracted {done} report(s).")


def save_extraction(settings: Settings, report) -> Path:
    """Persist a NI43101Report model to extracted_data/{stem}.json."""
    source_file = report.source_file or "report"
    out_path = source_output_path(settings.extracted_dir, source_file, ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NI 43-101 RAG with Docling primary parsing, MinerU fallback, "
            "LangChain, Claude Haiku, LangSmith, and Chroma."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs into vector store.")
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing vectors before ingesting.",
    )
    ingest_parser.add_argument(
        "--file",
        default=None,
        help="Ingest a single PDF by filename.",
    )
    ingest_parser.add_argument(
        "--parser",
        choices=["docling", "mineru"],
        default=None,
        help="Override the primary parser for this run.",
    )
    ingest_parser.add_argument(
        "--fallback",
        choices=["mineru", "none"],
        default=None,
        help="Select the quality-gated fallback parser.",
    )
    ingest_parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Retain degraded primary output instead of invoking MinerU.",
    )
    ingest_parser.add_argument(
        "--force-parser",
        choices=["docling", "mineru"],
        default=None,
        help="Run exactly one parser and bypass fallback routing.",
    )
    ingest_parser.add_argument(
        "--no-visual-enrichment",
        action="store_true",
        help="Skip Bedrock/Claude Haiku visual enrichment.",
    )
    ingest_parser.add_argument(
        "--partition-only",
        action="store_true",
        help="Partition and normalize only; do not enrich, chunk, or upsert.",
    )
    ingest_parser.add_argument(
        "--reprocess-visuals",
        action="store_true",
        help="Force visual enrichment even when the PDF fingerprint is unchanged.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-elements",
        help="Partition a PDF and print element / NI Item diagnostics.",
    )
    inspect_parser.add_argument(
        "--file",
        required=True,
        help="PDF filename (in knowledge dirs) or path.",
    )
    compare_parser = subparsers.add_parser(
        "compare-parsers",
        help="Benchmark Docling and MinerU without writing vectors.",
    )
    compare_parser.add_argument(
        "--file",
        required=True,
        help="PDF filename (in knowledge dirs) or path.",
    )

    subparsers.add_parser("chat", help="Start interactive CLI chat.")
    extract_parser = subparsers.add_parser(
        "extract", help="Extract structured NI 43-101 data from ingested reports."
    )
    extract_parser.add_argument(
        "--file",
        default=None,
        help="Extract a single report by filename (default: all ingested reports).",
    )
    extract_parser.add_argument(
        "--spatial",
        action="store_true",
        help=(
            "Extract spatial/geological-model data (boreholes, lithology, stratigraphy, "
            "orientations, faults) to RAG_SPATIAL_DIR instead of the standard extraction."
        ),
    )
    subparsers.add_parser(
        "reindex-chapters",
        help="Rebuild NI Item chapter indexes and patch chunk metadata without re-embedding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        if args.command == "ingest":
            selected_parser = args.force_parser or args.parser
            if selected_parser:
                settings.parser_primary = selected_parser
                settings.ingestion_backend = selected_parser
            if args.force_parser:
                settings.force_parser = args.force_parser
                settings.parser_fallback_enabled = False
            if args.fallback:
                settings.parser_fallback = "" if args.fallback == "none" else args.fallback
                settings.parser_fallback_enabled = args.fallback != "none"
            if args.no_fallback:
                settings.parser_fallback_enabled = False
            ingest(
                settings,
                rebuild=args.rebuild,
                only_file=args.file,
                enable_visuals=not args.no_visual_enrichment,
                partition_only=args.partition_only,
                reprocess_visuals=args.reprocess_visuals,
                backend=selected_parser,
            )
        elif args.command == "inspect-elements":
            import json

            info = inspect_elements(settings, args.file)
            print(json.dumps(info, indent=2))
        elif args.command == "compare-parsers":
            info = compare_parsers(settings, args.file)
            print(json.dumps(info, indent=2))
        else:
            embedder = get_embedder(settings)
            vectorstore = get_vectorstore(settings, embedder)
            llm = get_chat_model(settings)
            if args.command == "chat":
                chat(settings, vectorstore, llm)
            elif args.command == "extract":
                if args.spatial:
                    run_extract_spatial(settings, vectorstore, target_file=args.file)
                else:
                    run_extract(settings, vectorstore, llm, target_file=args.file)
            elif args.command == "reindex-chapters":
                reindex_chapters(settings, vectorstore)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
