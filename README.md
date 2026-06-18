# NI 43-101 RAG & Extraction

A research and due-diligence tool for NI 43-101 mineral project technical reports. Drop PDFs into a knowledge directory, ingest them into a vector database, then interrogate them via natural-language chat, a structured screener dashboard, or a REST API.

Built for the full spectrum of NI 43-101 users — investors and fund managers comparing project economics, investment bankers running due diligence, securities regulators checking disclosure compliance, M&A advisors benchmarking comparable transactions, lenders and royalty companies sizing project risk, and qualified persons reviewing resource estimate methodology.

> **Background** — NI 43-101 is the Canadian standard for disclosure of mineral project information, introduced after the 1997 Bre-X fraud to give investors standardized, QP-certified resource estimates. Every structured field extracted by this tool maps directly to a disclosure requirement in the standard.

## Architecture

```
PDF files (knowledge/ + extra dirs)
  → parse page-by-page (PyMuPDF; OCR fallback via pdf2image + pytesseract)
  → chunk & embed (OpenAI text-embedding-3-small)
  → store (Chroma persistent vector DB)

Chat query  → similarity search + keyword rerank → Claude (AWS Bedrock) → answer + source citations
Extraction  → per-topic retrieval (11 topic groups) → Claude structured output → NI43101Report JSON
Dashboard   → REST API → interactive HTML (portfolio screener · map · section detail · chat)
```

## Prerequisites

- Python 3.10+
- **OpenAI API key** — used for embeddings only (`text-embedding-3-small`)
- **AWS credentials** with Bedrock access — used for generation (chat) and structured extraction (Claude Sonnet 4 via Bedrock inference profile)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Copy and configure the environment file:

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Edit `.env` and set your `OPENAI_API_KEY` and AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optionally `AWS_SESSION_TOKEN` for STS). All other variables have sensible defaults.

## Workflow

### 1 — Add reports

Drop NI 43-101 PDF files into `knowledge/`. Multiple directories are supported via `RAG_EXTRA_PDF_DIRS` (see Configuration).

### 2 — Ingest

```bash
python rag_app.py ingest
```

Pages are parsed with PyMuPDF, OCR-rescued where needed, chunked, embedded, and upserted into Chroma. Progress is shown with tqdm bars (per-file, per-batch) including ETA.

To wipe and rebuild the index (e.g. after changing chunk settings):

```bash
python rag_app.py ingest --rebuild
```

After upgrading to chapter-aware ingestion, tag existing chunks with NI Item metadata without re-embedding:

```bash
python rag_app.py reindex-chapters
```

### 3 — Extract structured data

```bash
python rag_app.py extract                    # all ingested reports
python rag_app.py extract --file report.pdf  # a single report
```

Each report is processed through focused extraction passes (identity, resources/reserves, economics/technical, geology/environment, plus portfolio metadata tags). Retrieval uses NI 43-101 Item-aligned queries where available. Results are written to `extracted_data/{stem}.json`.

If Bedrock rate limits are hit, the extractor backs off automatically (60 s → 120 s → …) and pauses 30 s between reports.

### 4a — HTML Dashboard

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000/dashboard`.

#### Dashboard sections

| Section | Description |
|---------|-------------|
| **Dashboard** | KPI cards (NPV, IRR, CapEx, Mine Life, Payback, M+I Resources) · resource breakdown chart · financial snapshot · full resource/reserve table with row-level filter |
| **Portfolio** | Screener table across all extracted reports — filter by commodity, country, stage; search by project / company; click any row to drill into the full report |
| **Resources & Reserves** | Detailed resource and reserve tables grouped by category with tonnage and grade roll-ups |
| **Economics** | Full economics card (study type, NPV, IRR, CapEx, OpEx, mine life, throughput, strip ratio, recovery, royalties, metal price assumptions) |
| **Property & Geology** | Location, coordinates, jurisdiction, exchange, tenure, infrastructure, deposit type, host rock, mineralisation, structural controls, historical production |
| **Exploration & QPs** | Drilling statistics, notable intercepts, geophysical surveys, sampling methods, Qualified Person panel |
| **Map** | Leaflet world map — markers sized by M+I tonnage, coloured by primary commodity; filter by commodity and stage; popup with project summary and "View full report" link |
| **Chat** | Agentic NI 43-101 due diligence chat with chapter-directed retrieval (Items 1–27), peer benchmarking, red-flag detection, and Go/No-Go assessment; client-side conversation memory; optional single-report filter |

Additional topbar controls:
- **⚙ Extract All** — triggers extraction for every ingested PDF from the UI (no CLI needed)
- **⚙ Extract** — re-extracts the currently selected report
- **↓ Export** — downloads the current extraction as JSON
- **Light / dark mode** toggle

### 4b — Streamlit UI

```bash
streamlit run streamlit_app.py
```

- **Ask** tab: RAG chat with answers and matched source-page thumbnails
- **Reports** tab: run extraction and browse structured data

### 4c — REST API

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`    | `/api/documents` | List all ingested PDF filenames |
| `POST`   | `/api/ingest` | Upload PDFs (multipart) and ingest |
| `POST`   | `/api/ingest/rebuild` | Wipe and rebuild the vector index |
| `DELETE` | `/api/documents/{filename}` | Remove a PDF and rebuild |
| `POST`   | `/api/chat` | Agentic Q&A; body `{question, pdf_filter?, history?}`; returns `{answer, sources, routed_items, flags, peer_summary, assessment}` |
| `POST`   | `/api/reindex-chapters` | Rebuild NI Item tags on chunks without re-embedding |
| `GET`    | `/api/reports` | List all structured extractions |
| `GET`    | `/api/reports/{filename}` | Get a single structured extraction |
| `POST`   | `/api/extract` | Run extraction on one ingested report; body `{filename}` |
| `POST`   | `/api/extract/all` | Run extraction on all ingested reports |

If `API_KEY` is set in `.env`, every request must include `X-API-Key: <key>`. The `/dashboard` route is always public.

### 4d — CLI chat

```bash
python rag_app.py chat
```

Type `exit` or `quit` to stop.

### 4e — Agent chat (LangGraph test harness)

With `AGENT_CHAT=1` and `AGENT_MODE=langgraph` in `.env`:

```bash
# Single question (all reports)
python agent_chat.py "Are QAQC results acceptable and complete?"

# Scoped to one report
python agent_chat.py "Is the cut-off grade reasonable compared with peers?" --file my_report.pdf
```

The agent calls six tools in a ReAct loop (max 5 rounds): `route_question`, `get_routing_playbook`, `search_by_items`, `get_extraction`, `find_peer_reports`, `benchmark_field`.

Set `AGENT_MODE=pipeline` to use the deterministic fallback (no tool-calling).

Via API / dashboard: `POST /api/chat` with `{ "question": "...", "pdf_filter": ["report.pdf"], "history": [] }`. The response includes `routed_items`, `tool_calls`, `flags`, `peer_summary`, and `assessment`.

## Extracted schema

Every field is optional — the extractor returns `null` or `[]` rather than fabricating values when a section is absent from the report.

### `NI43101Report` (top level)

| Field | Type | Description |
|-------|------|-------------|
| `source_file` | str | Source PDF filename |
| `report_title` | str | Full report title |
| `report_date` | str | Report / effective date |
| `report_purpose` | str | Trigger event (e.g. "Initial NI 43-101", "Updated resource estimate", "PEA") |
| `previous_resource_date` | str | Effective date of the estimate this report supersedes — enables change tracking |
| `issuer` | str | Company on whose behalf the report was prepared |
| `authors` | list[str] | Authoring firms or individuals |
| `qualified_persons` | list[QualifiedPerson] | QP name, credentials, responsibility |
| `property_info` | PropertyInfo | Location, coordinates, stage, exchange |
| `geology` | GeologySummary | Deposit type, host rock, mineralisation |
| `exploration` | ExplorationSummary | Drilling, sampling, geophysics |
| `mineral_resources` | list[MineralResource] | One row per category/zone line item |
| `mineral_reserves` | list[MineralReserve] | One row per category/zone line item |
| `economics` | EconomicParameters | NPV, IRR, CapEx, OpEx, mine life, … |
| `mining_method` | MiningMethod | Method, rate, strip ratio, equipment |
| `processing_method` | ProcessingMethod | Method, throughput, recoveries |
| `environmental` | EnvironmentalSummary | Permits, studies, indigenous consultation, political risk |
| `summary` | str | 3–5 sentence narrative project summary |

### `PropertyInfo` — key fields

| Field | Description |
|-------|-------------|
| `project_name`, `country`, `region` | Identity and location |
| `latitude`, `longitude` | Decimal-degree coordinates for map plotting (converted from DMS if needed) |
| `jurisdiction` | Mining jurisdiction as "Region, Country" |
| `exchange_listed` | Stock exchange ticker (e.g. `TSX-V`, `ASX`) |
| `project_stage` | Development stage: Grassroots → Exploration → PEA → Pre-Feasibility → Feasibility → Construction → Operating → Care & Maintenance |
| `commodities` | Primary and by-product commodities |
| `ownership`, `tenure_status` | Ownership and mineral tenure |

### `MineralResource` / `MineralReserve`

Each row maps to one line in the resource or reserve table. The `grades` field is a list of `GradeEntry` objects — one per commodity column — supporting polymetallic deposits (e.g. Cu%, Au g/t, Ag g/t on the same row).

### `EnvironmentalSummary` — new fields

| Field | Description |
|-------|-------------|
| `indigenous_consultation` | FPIC / duty-to-consult / IBA / community opposition status |
| `political_risk_flags` | List of explicit risk statements (resource nationalism, conflict zones, mining-code changes) |

## Docker

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. PDFs in `knowledge/` and extractions in `extracted_data/` are bind-mounted into the container; the Chroma index is persisted in a named Docker volume.

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Embeddings. |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_BASE_URL` | OpenAI default | Override for compatible APIs |
| `AWS_REGION` | `us-east-2` | AWS region for Bedrock |
| `AWS_ACCESS_KEY_ID` | — | **Required.** AWS credential |
| `AWS_SECRET_ACCESS_KEY` | — | **Required.** AWS credential |
| `AWS_SESSION_TOKEN` | _(unset)_ | Required only for temporary STS credentials |
| `BEDROCK_MODEL_ID` | Claude Sonnet 4 inference profile ARN | Bedrock model ID or cross-region inference profile ARN |
| `API_KEY` | _(unset)_ | REST API key (`X-API-Key`). Unset = open local access. |
| `RAG_KNOWLEDGE_DIR` | `knowledge` | Primary PDF directory |
| `RAG_EXTRA_PDF_DIRS` | _(unset)_ | Additional read-only PDF directories, semicolon-separated |
| `RAG_CHROMA_DIR` | `.chroma_db` | Persistent vector store location |
| `RAG_COLLECTION_NAME` | `ni43101_knowledge` | Chroma collection name |
| `RAG_CHUNK_SIZE` | `1400` | Max characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RAG_EMBED_BATCH_SIZE` | `64` | Embedding batch size |
| `RAG_UPSERT_BATCH_SIZE` | `24` | Chroma upsert batch size |
| `RAG_TOP_K` | `8` | Chunks retrieved per chat query |
| `RAG_EXTRACTED_DIR` | `extracted_data` | Where structured extractions are saved |
| `NI43101_EXTRACT_TOP_K` | `12` | Chunks per topic query fed to the extractor — lower to `6` if hitting rate limits |

## Notes

- **PDF parsing** — PyMuPDF (fitz) provides fast, exact 1-indexed page numbers. Pages with CID-encoded (unreadable) fonts are automatically OCR'd via `pdf2image` + `pytesseract`. On Windows install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add it to PATH; the Docker image includes it.
- **Rate limiting** — Bedrock throttles at a per-account tokens-per-minute quota. The extractor uses exponential backoff starting at 60 s and pauses 30 s between reports. Reduce `NI43101_EXTRACT_TOP_K` to shrink context size if limits persist.
- **Extraction accuracy** — the extractor never invents values: any field absent from the retrieved context is returned as `null` or `[]`. Fields like coordinates, project stage, exchange listing, and political risk flags require that the report context contains the relevant disclosure.
- **Map view** — requires `latitude` and `longitude` to have been extracted. Re-run extraction on existing reports (⚙ Extract) after upgrading to populate the new coordinate fields.
- **Chat** — answers are grounded in retrieved context only; Claude will state explicitly when context is absent or insufficient.
- **Chroma private API** — `vectorstore._collection.count()` is wrapped in `_index_is_empty()` to insulate against future Chroma API changes. All empty-index checks go through this wrapper.
