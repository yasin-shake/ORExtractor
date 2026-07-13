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
    olmocr_server_url: Optional[str] = None
    olmocr_api_key: str = ""
    olmocr_model: str = "allenai/olmOCR-7B-0225-preview"
    olmocr_workers: int = 4
    spatial_dir: Path = Path("spatial_data")


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
        olmocr_server_url=os.getenv("OLMOCR_SERVER_URL", "").strip() or None,
        olmocr_api_key=os.getenv("OLMOCR_API_KEY", "").strip(),
        olmocr_model=os.getenv("OLMOCR_MODEL", "allenai/olmOCR-7B-0225-preview"),
        olmocr_workers=int(os.getenv("OLMOCR_WORKERS", "4")),
        spatial_dir=Path(os.getenv("RAG_SPATIAL_DIR", "spatial_data")),
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
    return _CachedOpenAIEmbeddings(**kwargs)


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


_EMBED_CACHE: dict[str, list] = {}


class _CachedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings with a process-level embed_query cache.

    Avoids re-embedding identical query strings across extraction passes and
    multi-report runs in extract_all().
    """

    def embed_query(self, text: str) -> list:  # type: ignore[override]
        if text not in _EMBED_CACHE:
            _EMBED_CACHE[text] = super().embed_query(text)
        return _EMBED_CACHE[text]


# ── olmocr prompt (YAML response format, compatible with allenai/olmOCR-7B) ───
try:
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as _build_olmocr_prompt
    _OLMOCR_PROMPT: str = _build_olmocr_prompt()
except Exception:
    _OLMOCR_PROMPT = (
        "Below is an image of a page from a document. Extract all visible text in reading order.\n"
        "Convert tables to HTML (<table>...</table>). Use LaTeX for equations.\n"
        "Respond with ONLY a YAML document containing exactly these fields:\n"
        "primary_language: <ISO 639-1 code>\n"
        "is_rotation_valid: <true or false>\n"
        "rotation_correction: <0, 90, 180, or 270>\n"
        "is_table: <true if most of the page is a table>\n"
        "is_diagram: <true if most of the page is a figure or diagram>\n"
        "natural_text: |\n"
        "  <extracted page text>\n"
    )


def _olmocr_render_page_b64(pdf_path: Path, page_num: int, longest_dim: int = 1920) -> str:
    """Render a PDF page to a base64-encoded PNG using PyMuPDF (no poppler needed)."""
    import fitz  # PyMuPDF — already a project dependency
    import base64

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_num - 1]  # page_num is 1-indexed
        rect = page.rect
        scale = longest_dim / max(rect.width, rect.height)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()
    return base64.b64encode(png_bytes).decode("utf-8")


def _olmocr_parse_yaml_response(raw: str) -> Optional[dict]:
    """Parse an olmocr YAML response, stripping markdown fences if present."""
    import yaml

    text = raw.strip()
    text = re.sub(r"^```(?:yaml)?\s*\n?", "", text, flags=re.I)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else None
    except yaml.YAMLError:
        return None


def _olmocr_process_page(
    client,        # pre-instantiated OpenAI client (shared across the thread pool)
    model: str,
    pdf_path: Path,
    page_num: int,
) -> Tuple[int, str, bool, bool]:
    """Call the olmocr-compatible inference server for a single PDF page.

    Returns (page_num, natural_text, is_table, is_diagram).
    Falls back to empty string on any failure.
    """
    try:
        b64 = _olmocr_render_page_b64(pdf_path, page_num)
    except Exception as exc:
        tqdm.write(f"    page {page_num}: render failed — {exc}")
        return page_num, "", False, False

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": _OLMOCR_PROMPT},
                ],
            }],
            temperature=0.1,
            max_tokens=8192,
        )
        raw = resp.choices[0].message.content or ""
        data = _olmocr_parse_yaml_response(raw)
        if not data:
            tqdm.write(f"    page {page_num}: YAML parse failed, raw snippet: {raw[:120]!r}")
            return page_num, "", False, False
        return (
            page_num,
            data.get("natural_text") or "",
            bool(data.get("is_table", False)),
            bool(data.get("is_diagram", False)),
        )
    except Exception as exc:
        tqdm.write(f"    page {page_num}: inference call failed — {exc}")
        return page_num, "", False, False


def _pymupdf_fallback_text(pdf_path: Path) -> dict[int, str]:
    """Extract text from each page using PyMuPDF's text layer (instant, no GPU)."""
    import fitz

    result: dict[int, str] = {}
    try:
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                result[i] = text
        doc.close()
    except Exception as exc:
        tqdm.write(f"  PyMuPDF fallback failed: {exc}")
    return result


def _chunk_ocr_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
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
                if overlap:
                    tail = buf[-overlap:].strip()
                    next_start = f"{tail}\n\n{p}".strip() if tail else p
                else:
                    next_start = p
                buf = next_start[:chunk_size] if len(next_start) > chunk_size else next_start
            else:
                buf = p[:chunk_size] if len(p) > chunk_size else p
    if buf:
        out.append(buf)
    return [c for c in out if c.strip()]


def _postprocess_olmocr_page(natural_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Split olmocr natural_text into (body_text, table_markdown).

    olmocr embeds HTML tables inline inside the page markdown. We extract
    them as separate table documents (consistent with the pre-olmocr pipeline)
    and return the remaining prose separately.
    """
    tables: List[str] = []
    body = natural_text

    for m in re.finditer(r"(?is)<table[^>]*>.*?</table>", natural_text):
        tables.append(_html_table_to_markdown(m.group(0)))

    body = re.sub(r"(?is)<table[^>]*>.*?</table>", "", body)
    # Drop bare image references  ![alt](path.png)  left by olmocr figure handling
    body = re.sub(r"!\[.*?\]\([^)]*\.png[^)]*\)", "", body)
    body = body.strip() or None
    return body, "\n\n".join(tables) if tables else None


def _html_to_text(raw_html: str) -> str:
    """Flatten a marker HTML block into plain text suitable for chunking."""
    import html as _htmllib

    text = re.sub(r"(?is)<\s*br\s*/?>", "\n", raw_html)
    text = re.sub(r"(?is)</\s*(p|div|li|h[1-6]|tr|table)\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _htmllib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _html_table_to_markdown(raw_html: str) -> str:
    """Convert a marker-pdf table HTML block to pipe-delimited markdown.

    The LLM reads tabular columns directly from markdown rather than having to
    mentally parse HTML tag nesting, reducing grade-misalignment errors and
    token usage.  Falls back to plain text if no <tr>/<td> rows are found.
    """
    import html as _htmllib

    rows: List[List[str]] = []
    for row_match in re.finditer(r"(?is)<tr[^>]*>(.*?)</tr>", raw_html):
        cells = re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", row_match.group(1))
        row = []
        for cell in cells:
            cell_text = re.sub(r"(?s)<[^>]+>", " ", cell)
            cell_text = _htmllib.unescape(cell_text)
            cell_text = re.sub(r"\s+", " ", cell_text).strip()
            row.append(cell_text.replace("|", "\\|"))
        if any(row):
            rows.append(row)

    if not rows:
        return _html_to_text(raw_html)

    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    lines: List[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "|".join(" --- " for _ in row) + "|")
    return "\n".join(lines)


def parse_pdf_to_documents(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int = 0,
    settings: Optional["Settings"] = None,
) -> List[Document]:
    docs: List[Document] = []

    # ── olmocr path ──────────────────────────────────────────────────────────
    if settings and settings.olmocr_server_url:
        import fitz

        tqdm.write("  Converting with olmocr …")
        try:
            doc_fitz = fitz.open(str(pdf_path))
            n_pages = len(doc_fitz)
            doc_fitz.close()
        except Exception as exc:
            tqdm.write(f"  Could not open PDF: {exc}")
            return []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from openai import OpenAI as _OAIClient

        # Instantiate once — all worker threads share the HTTP connection pool.
        olmocr_client = _OAIClient(
            api_key=settings.olmocr_api_key or "olmocr",
            base_url=settings.olmocr_server_url,
            timeout=120.0,
        )

        page_results: dict[int, Tuple[str, bool, bool]] = {}
        with ThreadPoolExecutor(max_workers=settings.olmocr_workers) as pool:
            futs = {
                pool.submit(
                    _olmocr_process_page,
                    olmocr_client,
                    settings.olmocr_model,
                    pdf_path,
                    pn,
                ): pn
                for pn in range(1, n_pages + 1)
            }
            for fut in as_completed(futs):
                pn, text, is_tbl, is_diag = fut.result()
                page_results[pn] = (text, is_tbl, is_diag)

        table_items: List[Tuple[str, int]] = []
        for page_no in sorted(page_results):
            natural_text, is_tbl, is_diag = page_results[page_no]
            if not natural_text:
                continue
            body, tables_md = _postprocess_olmocr_page(natural_text)
            if tables_md:
                table_items.append((tables_md, page_no))
            if body and not is_diag:
                for c_idx, chunk in enumerate(_chunk_ocr_text(body, chunk_size, chunk_overlap)):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_no,
                            "chunk": c_idx,
                            "type": "text",
                            "ni_item": 0,
                            "section_title": "",
                        },
                    ))

        for t_idx, (table_content, page_no) in enumerate(table_items):
            docs.append(Document(
                page_content=table_content,
                metadata={
                    "source": pdf_path.name,
                    "page": page_no,
                    "chunk": 1000 + t_idx,
                    "type": "table",
                    "ni_item": 0,
                    "section_title": "",
                },
            ))

    # ── PyMuPDF fallback (no olmocr server configured) ───────────────────────
    else:
        tqdm.write("  OLMOCR_SERVER_URL not set — using PyMuPDF text-layer fallback")
        page_texts = _pymupdf_fallback_text(pdf_path)
        if not page_texts:
            tqdm.write(f"  No extractable text in {pdf_path.name}")
            return []
        for page_no, text in sorted(page_texts.items()):
            for c_idx, chunk in enumerate(_chunk_ocr_text(text, chunk_size, chunk_overlap)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_no,
                        "chunk": c_idx,
                        "type": "text",
                        "ni_item": 0,
                        "section_title": "",
                    },
                ))

    if not docs:
        tqdm.write(f"  No content extracted from {pdf_path.name}")
        return []

    from chapter_index import (
        build_chapter_index_from_documents,
        save_chapter_index,
        tag_documents_with_items,
    )

    chapters = build_chapter_index_from_documents(docs)
    docs = tag_documents_with_items(docs, chapters)
    extracted_dir = settings.extracted_dir if settings else Path(os.getenv("RAG_EXTRACTED_DIR", "extracted_data"))
    save_chapter_index(extracted_dir, pdf_path.name, chapters)
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
        docs = parse_pdf_to_documents(pdf_path, settings.chunk_size, settings.chunk_overlap, settings)
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


def reindex_chapters(settings: Settings, vectorstore: Chroma) -> None:
    """Re-parse PDFs to rebuild chapter indexes and patch chunk ni_item metadata."""
    from chapter_index import reindex_chapters_for_pdf

    pdf_paths = list(iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs))
    if not pdf_paths:
        print("No PDFs found to reindex.")
        return

    updated = 0
    for pdf_path in tqdm(pdf_paths, desc="Reindexing chapters", unit="pdf"):
        parse_fn = lambda p, cs: parse_pdf_to_documents(p, cs, settings.chunk_overlap, settings)
        chapters, docs = reindex_chapters_for_pdf(
            pdf_path,
            settings.chunk_size,
            parse_fn,
            settings.extracted_dir,
        )
        if not docs:
            continue
        ids: List[str] = []
        metadatas: List[dict] = []
        for d in docs:
            page_num = int(d.metadata.get("page", -1) or -1)
            chunk_idx = int(d.metadata.get("chunk", 0) or 0)
            ids.append(build_doc_id(pdf_path.name, page_num, chunk_idx, d.page_content))
            metadatas.append(dict(d.metadata))
        try:
            vectorstore._collection.update(ids=ids, metadatas=metadatas)
            updated += len(ids)
            tqdm.write(f"  {pdf_path.name}: {len(chapters)} chapters, {len(ids)} chunks tagged")
        except Exception as exc:
            tqdm.write(f"  {pdf_path.name}: metadata update failed ({exc}); re-ingest recommended")

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
    try:
        results = vectorstore.similarity_search_with_score(
            question, k=fetch_n, filter=chroma_filter
        )
    except Exception:
        # Fallback when ni_item metadata missing on older index entries
        chroma_filter = _build_chroma_filter(filter_sources, None, filter_types)
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
    if not docs:
        return query_context(vectorstore, question, top_k, filter_sources)
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
        targets = [p.name for p in iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs)]

    done = 0
    for name in targets:
        out_file = settings.spatial_dir / f"{Path(name).stem}.json"
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
            "NI 43-101 RAG + structured extraction with olmocr + OpenAI embeddings "
            "+ Claude via AWS Bedrock + Chroma."
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
            check_openai_connectivity(settings)
            ingest(settings, rebuild=args.rebuild)
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
