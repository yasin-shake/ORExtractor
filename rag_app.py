import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chromadb
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError, APIError, RateLimitError

_MAX_HISTORY_TURNS = 10  # user+assistant pairs to keep in conversation history

# Domain instruction shared by the CLI, API and Streamlit chat surfaces.
SYSTEM_INSTRUCTION = (
    "You are a technical assistant specialised in NI 43-101 mineral project reports. "
    "Answer using only the supplied context. When you state figures such as resource "
    "tonnages, grades, contained metal, cut-off grades, NPV, IRR or capital costs, quote "
    "them exactly as written and include their units. Where possible, cite the relevant "
    "report section, page number and the Qualified Person responsible. If the context is "
    "incomplete, say what is known and what is unknown. If the context is not relevant, "
    "say you do not know."
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


def load_settings() -> Settings:
    if not Path(".env").exists():
        print("Warning: no .env file found, using defaults. Copy .env.example to .env to configure.")
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings. Add it to .env before running.")
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
        upsert_batch_size=int(os.getenv("RAG_UPSERT_BATCH_SIZE", "24")),
        top_k=int(os.getenv("RAG_TOP_K", "8")),
        extracted_dir=Path(os.getenv("RAG_EXTRACTED_DIR", "extracted_data")),
        extract_top_k=int(os.getenv("NI43101_EXTRACT_TOP_K", "12")),
    )


def _openai_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def check_openai_connectivity(settings: Settings) -> None:
    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    try:
        response = requests.get(
            f"{base_url}/models",
            headers=_openai_headers(settings.openai_api_key),
            timeout=10,
        )
        if response.status_code >= 400:
            detail = response.text[:300]
            raise RuntimeError(f"OpenAI connectivity check failed ({response.status_code}): {detail}")
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenAI connectivity check failed: {exc}") from exc


def get_vectorstore(settings: Settings, embedder: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embedder,
    )


def get_embedder(settings: Settings) -> OpenAIEmbeddings:
    kwargs = {
        "api_key": settings.openai_api_key,
        "model": settings.embed_model,
        "chunk_size": settings.embed_batch_size,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


def get_chat_model(settings: Settings) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model_id=settings.bedrock_model_id,
        provider="amazon",
        temperature=0.35,
        max_tokens=4096,
        region_name=settings.aws_region,
    )


def iter_pdf_paths(knowledge_dir: Path, extra_dirs: Optional[List[Path]] = None) -> Iterable[Path]:
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {knowledge_dir}")
    paths: List[Path] = list(knowledge_dir.glob("*.pdf"))
    for d in (extra_dirs or []):
        if d.exists():
            paths.extend(d.glob("*.pdf"))
    return sorted(set(paths), key=lambda p: p.name)


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


def build_doc_id(source: str, page: int, chunk_idx: int, chunk: str) -> str:
    digest = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:12]
    return f"{source}:p{page}:c{chunk_idx}:{digest}"


def _add_documents_with_retry(
    vectorstore: Chroma,
    docs: List[Document],
    ids: List[str],
    batch_size: int,
    retries: int = 3,
) -> None:
    if batch_size <= 0:
        raise ValueError("RAG_UPSERT_BATCH_SIZE must be greater than 0.")

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
                vectorstore.add_documents(documents=docs_batch, ids=ids_batch)
                break
            except (APIConnectionError, RateLimitError, APIError) as exc:
                attempt += 1
                if attempt >= retries:
                    raise RuntimeError(
                        f"Vectorstore add_documents failed for chunks {start}-{end - 1}: {exc}"
                    ) from exc
                wait_seconds = 2 ** attempt
                tqdm.write(
                    f"  Retry {attempt}/{retries - 1} for chunks {start}-{end - 1} "
                    f"after error: {exc}. Waiting {wait_seconds}s..."
                )
                time.sleep(wait_seconds)


# ── Ingestion manifest helpers ────────────────────────────────────────────────
# The manifest is a JSON file stored alongside the Chroma index that maps each
# PDF filename to a fingerprint (mtime_ns:size).  On the next ingest run, any
# file whose fingerprint matches is skipped — no re-parsing, no re-embedding.

def _file_fingerprint(path: Path) -> str:
    s = path.stat()
    return f"{s.st_mtime_ns}:{s.st_size}"


def _load_manifest(chroma_dir: Path) -> dict:
    p = chroma_dir / "ingest_manifest.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_manifest(chroma_dir: Path, manifest: dict) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "ingest_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


_parser_logs_suppressed: bool = False


def _suppress_parser_logs() -> None:
    global _parser_logs_suppressed
    if _parser_logs_suppressed:
        return
    import logging
    for _noisy in ("marker", "surya", "texify", "transformers", "datasets",
                   "PIL", "huggingface_hub", "filelock"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    _parser_logs_suppressed = True


def _chunk_ocr_text(text: str, chunk_size: int) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: List[str] = []
    buf = ""
    for p in paras:
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = p[:chunk_size] if len(p) > chunk_size else p
    if buf:
        out.append(buf)
    return [c for c in out if c.strip()]


_marker_converter = None


def _get_marker_converter():
    """Build (once) and cache the marker-pdf converter configured for chunk output.

    The chunk renderer flattens every page into a list of top-level blocks, each
    carrying its ``block_type``, fully-assembled ``html`` and 0-indexed ``page``.
    """
    global _marker_converter
    if _marker_converter is not None:
        return _marker_converter

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    config_parser = ConfigParser({
        "output_format": "chunks",
        "disable_image_extraction": True,
        # These NI 43-101 PDFs are digital (embedded text layer), so trust the
        # provider text and skip Surya's recognition model — the slow step that
        # otherwise re-OCRs every page. Scanned PDFs will yield no text and are
        # handled by the empty-result skip in ingest().
        "disable_ocr": True,
    })
    _marker_converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    return _marker_converter


def _html_to_text(raw_html: str) -> str:
    """Flatten a marker HTML block into plain text suitable for chunking."""
    import html as _htmllib

    text = re.sub(r"(?is)<\s*br\s*/?>", "\n", raw_html)
    text = re.sub(r"(?is)</\s*(p|div|li|h[1-6]|tr|table)\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _htmllib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_pdf_to_documents(
    pdf_path: Path,
    chunk_size: int,
) -> List[Document]:
    _suppress_parser_logs()
    tqdm.write("  Converting with marker-pdf …")
    try:
        converter = _get_marker_converter()
        rendered = converter(str(pdf_path))
        blocks = list(getattr(rendered, "blocks", None) or [])
    except Exception as exc:
        tqdm.write(f"  marker-pdf conversion failed: {exc}")
        return []

    if not blocks:
        tqdm.write("  marker-pdf produced no output")
        return []

    docs: List[Document] = []
    page_texts: dict[int, List[str]] = {}
    table_items: List[tuple] = []
    image_items: List[tuple] = []

    _SKIP_TYPES = {"Equation", "PageHeader", "PageFooter", "Handwriting"}
    _TABLE_TYPES = {"Table", "TableGroup", "TableOfContents", "Form"}
    # Figures/pictures are kept for their caption (and, if enabled, any LLM
    # description) rather than discarded — the surrounding <img> tag is stripped
    # by _html_to_text, leaving the textual context behind.
    _IMAGE_TYPES = {"Picture", "PictureGroup", "Figure", "FigureGroup"}

    for block in blocks:
        page_no = (getattr(block, "page", 0) or 0) + 1  # marker is 0-indexed
        block_type = str(getattr(block, "block_type", ""))
        if block_type in _SKIP_TYPES:
            continue
        raw_html = (getattr(block, "html", "") or "").strip()
        if not raw_html:
            continue
        if block_type in _TABLE_TYPES:
            table_items.append((raw_html, page_no))
        elif block_type in _IMAGE_TYPES:
            text = _html_to_text(raw_html)
            if text:
                image_items.append((text, page_no))
        else:
            text = _html_to_text(raw_html)
            if text:
                page_texts.setdefault(page_no, []).append(text)

    for page_no, texts in sorted(page_texts.items()):
        for c_idx, chunk in enumerate(_chunk_ocr_text("\n\n".join(texts), chunk_size)):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": pdf_path.name, "page": page_no, "chunk": c_idx, "type": "text"},
            ))

    for t_idx, (table_content, page_no) in enumerate(table_items):
        docs.append(Document(
            page_content=table_content,
            metadata={"source": pdf_path.name, "page": page_no, "chunk": 1000 + t_idx, "type": "table"},
        ))

    for i_idx, (image_content, page_no) in enumerate(image_items):
        docs.append(Document(
            page_content=image_content,
            metadata={"source": pdf_path.name, "page": page_no, "chunk": 2000 + i_idx, "type": "image"},
        ))

    return docs


def ingest(settings: Settings, rebuild: bool = False) -> None:
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if rebuild:
        try:
            chroma_client.delete_collection(name=settings.collection_name)
            print("Rebuilt vector index (old records removed).")
        except Exception as exc:
            print(f"Warning: could not delete collection '{settings.collection_name}': {exc}")
    embedder = get_embedder(settings)
    vectorstore = get_vectorstore(settings, embedder)

    pdf_paths = list(iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs))
    if not pdf_paths:
        dirs = [settings.knowledge_dir] + list(settings.extra_pdf_dirs)
        print(f"No PDFs found in {', '.join(str(d) for d in dirs)}")
        return

    # Load the change-detection manifest; wipe it when rebuilding from scratch.
    manifest = {} if rebuild else _load_manifest(settings.chroma_dir)

    upserted = 0
    skipped = 0
    pdf_bar = tqdm(pdf_paths, desc="Ingesting PDFs", unit="pdf", dynamic_ncols=True)
    for pdf_path in pdf_bar:
        pdf_bar.set_postfix(file=pdf_path.name[:40])

        fp = _file_fingerprint(pdf_path)
        if manifest.get(pdf_path.name) == fp:
            tqdm.write(f"\nSkipping {pdf_path.name} (unchanged since last ingest)")
            skipped += 1
            continue

        tqdm.write(f"\nParsing {pdf_path.name}...")
        docs = parse_pdf_to_documents(pdf_path, settings.chunk_size)
        if not docs:
            tqdm.write(f"  No extractable content found in {pdf_path.name}.")
            continue

        ids: List[str] = []
        for d in docs:
            page_num = d.metadata.get("page", -1)
            if not isinstance(page_num, int):
                page_num = -1
            chunk_idx = d.metadata.get("chunk", 0)
            if not isinstance(chunk_idx, int):
                chunk_idx = 0
            ids.append(build_doc_id(pdf_path.name, page_num, chunk_idx, d.page_content))
        _add_documents_with_retry(
            vectorstore=vectorstore,
            docs=docs,
            ids=ids,
            batch_size=settings.upsert_batch_size,
        )
        upserted += len(ids)
        manifest[pdf_path.name] = fp
        _save_manifest(settings.chroma_dir, manifest)
        tqdm.write(f"  Done — {len(ids)} chunks upserted.")

    parts = [f"Upserted {upserted} chunks"]
    if skipped:
        parts.append(f"skipped {skipped} unchanged")
    print(f"\nIngestion complete. {', '.join(parts)} in '{settings.collection_name}'.")


def _retrieve_and_rerank(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    filter_sources: Optional[List[str]] = None,
) -> Tuple[List[str], List[dict]]:
    """Shared retrieval + rerank core. Returns (documents, metadatas) lists."""
    try:
        count = vectorstore._collection.count()
    except Exception:
        return [], []
    if count == 0:
        return [], []

    fetch_n = min(max(top_k * 4, top_k + 12), count)
    chroma_filter = None
    if filter_sources and len(filter_sources) == 1:
        chroma_filter = {"source": filter_sources[0]}
    elif filter_sources:
        chroma_filter = {"source": {"$in": list(filter_sources)}}
    results = vectorstore.similarity_search_with_score(question, k=fetch_n, filter=chroma_filter)
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
        parts.append(f"[{idx}] {source} page {page}\n{doc}")
    return parts


def query_chunks(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    filter_sources: Optional[List[str]] = None,
) -> Tuple[List[str], List[dict]]:
    """Return individual labelled chunk strings and their metadata dicts.

    Unlike query_context, chunks are never joined — safe for per-chunk
    deduplication in the extractor without risking misalignment on blank lines.
    """
    documents, metadatas = _retrieve_and_rerank(vectorstore, question, top_k, filter_sources)
    if not documents:
        return [], []
    return _format_parts(documents, metadatas), metadatas


def query_context(
    vectorstore: Chroma,
    question: str,
    top_k: int,
    filter_sources: Optional[List[str]] = None,
) -> Tuple[str, List[dict]]:
    """Return retrieved context as a single joined string (used by chat/API)."""
    documents, metadatas = _retrieve_and_rerank(vectorstore, question, top_k, filter_sources)
    if not documents:
        return "", []
    parts = _format_parts(documents, metadatas)
    return "\n\n".join(parts), metadatas


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
        out_file = settings.extracted_dir / f"{Path(target_file).stem}.json"
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
    settings.extracted_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(report.source_file).stem if report.source_file else "report"
    out_path = settings.extracted_dir / f"{stem}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NI 43-101 RAG + structured extraction with marker-pdf + OpenAI embeddings "
            "+ Claude (Anthropic) + Chroma."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs into vector store.")
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing vectors before ingesting.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        check_openai_connectivity(settings)
        if args.command == "ingest":
            ingest(settings, rebuild=args.rebuild)
        else:
            embedder = get_embedder(settings)
            vectorstore = get_vectorstore(settings, embedder)
            llm = get_chat_model(settings)
            if args.command == "chat":
                chat(settings, vectorstore, llm)
            elif args.command == "extract":
                run_extract(settings, vectorstore, llm, target_file=args.file)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
