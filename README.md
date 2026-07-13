# ORExtractor — NI 43-101 RAG & Extraction

A research and due-diligence tool for NI 43-101 mineral project technical reports. Drop PDFs into a knowledge directory, ingest them into a vector database, then interrogate them via an agentic due-diligence chat, a structured screener dashboard, or a REST API.

Built for the full spectrum of NI 43-101 users — investors and fund managers comparing project economics, investment bankers running due diligence, securities regulators checking disclosure compliance, M&A advisors benchmarking comparable transactions, lenders and royalty companies sizing project risk, and qualified persons reviewing resource estimate methodology.

> **Background** — NI 43-101 is the Canadian standard for disclosure of mineral project information, introduced after the 1997 Bre-X fraud to give investors standardized, QP-certified resource estimates. Its disclosure requirements are organized into 27 numbered "Items" (Item 1 Summary … Item 27 References); this tool routes retrieval, extraction, and due-diligence checklists against those same Item numbers.

## Capabilities

- **VLM-powered PDF ingestion** — parses NI 43-101 PDFs page-by-page via [olmocr](https://github.com/allenai/olmocr) (`allenai/olmOCR-7B-0225-preview`); HTML tables embedded in the model output are converted to pipe-delimited markdown; figure/diagram pages are indexed if they contain extractable text. Falls back to PyMuPDF text-layer extraction when no inference server is configured (`OLMOCR_SERVER_URL` unset). Every chunk is tagged with its NI Item number and section title, embedded with OpenAI, and stored in a persistent Chroma vector index with configurable overlap (`RAG_CHUNK_OVERLAP`). Unchanged files are skipped on re-ingest via a content-fingerprint manifest.
- **Structured extraction** — a 5-pass Claude (Bedrock) pipeline turns each report into a typed `NI43101Report` JSON object (identity, resources/reserves, economics/technical, geology/exploration/environmental, and portfolio metadata tags), never fabricating values for sections that aren't in the report. Each pass fans out its NI Item/topic queries concurrently for speed.
- **Agentic due-diligence chat** — a LangGraph ReAct agent with 6 tools (question routing, Item-scoped retrieval, extraction lookup, peer discovery, cross-report benchmarking, DD playbooks) that cites NI Item + page numbers, raises red flags from a due-diligence playbook, and issues a Go / Conditional Go / Further Work / No-Go assessment.
- **Portfolio screener dashboard** — a single-page HTML/JS app (no build step) with a KPI home page, sortable/filterable portfolio table, side-by-side report comparison (up to 4 reports), a Leaflet world map of every geolocated project, per-report resource/economics/geology/exploration views, and an embedded 3D geological model viewer (`spatial_data/` HTML models).
- **REST API** — FastAPI service exposing ingestion, extraction, chat (including streaming SSE), and report retrieval, optionally protected by an API key.
- **Streamlit alternative UI** — a lighter-weight chat + extraction browser with source-page thumbnails rendered from the original PDF.
- **CLI** — ingest, extract, chat, and chapter-reindex commands for scripted/offline use.

## Architecture

```
PDF files (knowledge/ + extra dirs)
  → olmocr VLM inference per page (natural_text + HTML tables; PyMuPDF fallback when no GPU server)
  → chunk (paragraph-aware) + tag each chunk with its NI Item # and section title (chapter_index.py)
  → embed (OpenAI text-embedding-3-small) → store (Chroma persistent vector DB)
  → change-detection manifest skips unchanged PDFs on re-ingest

Agentic chat  → LangGraph ReAct agent (6 tools: route_question, get_routing_playbook,
                search_by_items, get_extraction, find_peer_reports, benchmark_field)
              → Claude (AWS Bedrock) tool-calling loop → cited answer + flags + Go/No-Go assessment
              (falls back to a deterministic pipeline, or a dry-run tools-only mode, if needed)

Extraction    → 5 focused passes, each retrieving only its topic/Item-scoped chunks
              → Claude structured output (schemas.py) per pass → field-by-field merge → NI43101Report JSON

Dashboard     → REST API → interactive HTML (KPI home · portfolio screener · compare · map ·
                3D models · per-report drill-down · chat)
```

## Prerequisites

- Python 3.10+
- **OpenAI API key** — required for `ingest` only (embeddings, `text-embedding-3-small`). The `extract`, `chat`, and `reindex-chapters` commands do not contact OpenAI and can run without a valid key.
- **AWS credentials** with Bedrock access — used for generation (chat) and structured extraction (Claude Sonnet 4 via a Bedrock cross-region inference profile, by default)
- **olmocr inference server** — ingestion calls an OpenAI-compatible endpoint running `allenai/olmOCR-7B-0225-preview` (set `OLMOCR_SERVER_URL`). If unset, ingestion falls back to PyMuPDF's text layer (fast, good for digital PDFs; skips scanned pages). A GPU with ≥15 GB VRAM (e.g. RTX 5070 Ti) is sufficient for local serving via `vllm`.

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

Each PDF is processed page-by-page through the olmocr pipeline:

- **VLM inference** — each page image is sent to an `allenai/olmOCR-7B-0225-preview` inference server (`OLMOCR_SERVER_URL`). Pages are dispatched concurrently (`OLMOCR_WORKERS`, default 4). The model returns per-page YAML containing `natural_text`, `is_table`, and `is_diagram` flags.
- **Text chunks** — `natural_text` is split with configurable size (`RAG_CHUNK_SIZE`, default 1400 chars) and overlap (`RAG_CHUNK_OVERLAP`, default 150 chars).
- **Table chunks** — HTML `<table>` blocks embedded in the olmocr output are extracted, converted to pipe-delimited markdown, and stored as separate table-type chunks so the LLM reads columns directly.
- **Fallback** — if `OLMOCR_SERVER_URL` is unset, `PyMuPDF`'s text layer is used instead (instant; works well for digital PDFs with a text layer).

After chunking, `chapter_index.py` scans every chunk's full text for "Item N — …" headings to build a page-range chapter index per report, then tags each chunk's metadata with `ni_item` and `section_title`. Chunks are embedded in batches (`RAG_EMBED_BATCH_SIZE`) and upserted into Chroma with a process-level embedding cache to avoid re-embedding identical query strings. Progress is shown with tqdm bars (per-file) including ETA. PDFs unchanged since the last run (by content fingerprint) are skipped automatically.

To wipe and rebuild the index (e.g. after changing chunk settings):

```bash
python rag_app.py ingest --rebuild
```

After upgrading to chapter-aware ingestion, or if heading detection needs a refresh, re-tag existing chunks with NI Item metadata without re-embedding:

```bash
python rag_app.py reindex-chapters
```

### 3 — Extract structured data

```bash
python rag_app.py extract                    # all ingested reports
python rag_app.py extract --file report.pdf  # a single report
```

Each report runs through 5 focused extraction passes (identity → resources/reserves → economics/technical → geology/exploration/environmental → portfolio metadata tags), retrieving only the chunks relevant to that pass via NI Item-aligned or topic queries that are fanned out **concurrently** (`ThreadPoolExecutor`, 6 workers per pass) so each Claude call sees a small, targeted context window. Reports themselves are processed concurrently (5 workers) with per-call exponential backoff on Bedrock rate limits. Results are written to `extracted_data/{stem}.json`.

The extracted JSON uses `primary_mining_method` (string tag, e.g. `"open pit"`) as the portfolio metadata field for peer matching, while the structured `mining_method` field holds the full `MiningMethod` object (method, rate, strip ratio, equipment).

### 4a — HTML Dashboard

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000/dashboard`.

#### Dashboard sections

| Section | Description |
|---------|-------------|
| **Home** | Portfolio-wide KPI cards, commodity distribution chart, project-stage breakdown, top countries, top 10 projects by post-tax NPV, and a recent-reports list |
| **Dashboard (Overview)** | KPI cards (NPV, IRR, CapEx, Mine Life, Payback, M+I Resources) for the selected report · resource breakdown chart · financial snapshot · full resource/reserve table with row-level filter |
| **Portfolio** | Sortable, searchable screener table across all extracted reports — filter by commodity, country, stage; search by project / company; select up to 4 rows to compare; click any row to drill into the full report; CSV export |
| **Compare** | Side-by-side comparison of up to 4 selected reports (company, country, stage, commodity, deposit type, M+I tonnes, NPV, IRR, CapEx, mine life, QP count) |
| **Resources & Reserves** | Detailed resource and reserve tables grouped by category with tonnage and grade roll-ups |
| **Economics** | Full economics card (study type, NPV, IRR, CapEx, OpEx, mine life, throughput, strip ratio, recovery, royalties, metal price assumptions) |
| **Property & Geology** | Location, coordinates, jurisdiction, exchange, tenure, infrastructure, deposit type, host rock, mineralisation, structural controls, historical production |
| **Exploration & QPs** | Drilling statistics, notable intercepts, geophysical surveys, sampling methods, Qualified Person panel |
| **Map** | Leaflet world map — markers sized by M+I tonnage, coloured by primary commodity; filter by commodity and stage; popup with project summary and "View full report" link |
| **3D Models** | Embedded iframe viewer for standalone 3D geological model HTML files stored in `spatial_data/`; tab-switch between models (Mosquito Hill, Reid, Rocky Shore / Mosquito Reid MRE) with a loading overlay |
| **Chat** | Agentic NI 43-101 due diligence chat (streamed via SSE, markdown-rendered) with Item-directed retrieval, peer benchmarking, red-flag detection, and Go/No-Go assessment; client-side conversation memory; optional single-report filter |

Additional topbar controls:
- **⚙ Extract All** — triggers extraction for every ingested PDF from the UI (no CLI needed)
- **⚙ Extract** — re-extracts the currently selected report
- **↓ Export** — downloads the current extraction as JSON, or the filtered portfolio as CSV
- Sidebar search-as-you-type across all extracted reports
- **Light / dark mode** toggle
- Optional `X-API-Key` field, sent with every request when `API_KEY` is configured server-side

### 4b — Streamlit UI

```bash
streamlit run streamlit_app.py
```

- **Ask** tab: agentic chat with answers and matched source-page thumbnails (rendered on the fly from the source PDF with PyMuPDF)
- **Reports** tab: run extraction and browse structured data

### 4c — REST API

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`    | `/dashboard` | Serves the HTML dashboard (always public, no API key) |
| `GET`    | `/api/documents` | List all ingested PDF filenames |
| `POST`   | `/api/ingest` | Upload PDFs (multipart) and ingest |
| `POST`   | `/api/ingest/rebuild` | Wipe and rebuild the vector index |
| `DELETE` | `/api/documents/{filename}` | Remove a PDF and rebuild |
| `POST`   | `/api/chat` | Agentic Q&A; body `{question, pdf_filter?, history?}`; returns `{answer, sources, routed_items, cross_check_items, flags, peer_summary, assessment, tool_calls}` |
| `POST`   | `/api/chat/stream` | Same as `/api/chat` but streamed as Server-Sent Events (`status`, `meta`, `token`, `done`, `error`) |
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

### 4e — Agent chat CLI (standalone test harness)

```bash
# Single question (all reports)
python agent_chat.py "Are QAQC results acceptable and complete?"

# Scoped to one report
python agent_chat.py "Is the cut-off grade reasonable compared with peers?" --file my_report.pdf

# Run the tool chain without calling Bedrock (routing/retrieval/benchmark smoke test)
python agent_chat.py "..." --dry-run
```

Prints the routed NI Items, the sequence of tools called, the assessment (if any), the answer, and the source citations.

## RAG + agentic chat process

Chat is served by `agent_chat.py` and is enabled by default (`AGENT_CHAT=1`). Three modes, selected by environment variable:

- **`AGENT_MODE=langgraph`** (default) — a LangGraph `create_react_agent` loop (max 5 tool-call rounds) with 6 tools bound to the current request:
  1. **`route_question`** — maps the question (+ recent conversation history) to primary/cross-check NI Item numbers and an optional benchmark template, using a question-pattern lookup table (`data/routing_guide.json`, generated by `generate_routing_guide.py` from a BMRC due-diligence memo).
  2. **`get_routing_playbook`** — returns the Extract/Compare/Flag due-diligence checklist for the routed Items.
  3. **`search_by_items`** — retrieves chunks filtered by NI Item metadata (falls back to unfiltered similarity search if nothing matches).
  4. **`get_extraction`** — loads the structured JSON extraction for the scoped report (resources, NPV, metadata) instead of re-deriving it from raw text.
  5. **`find_peer_reports`** — filters the extracted-data portfolio by commodity, country, deposit type, mining method, and/or study stage to find comparable projects.
  6. **`benchmark_field`** — statistically compares a numeric field (cut-off grade, NPV, IRR, CapEx, OpEx, recovery, dilution, mining recovery) across the target report and its peers, flagging outliers.

  The non-streaming `/api/chat` endpoint (and the `agent_chat.py` CLI) call this true multi-round ReAct loop. The final answer is post-processed to extract red flags (playbook flags whose keywords appear in the answer, plus benchmark outliers) and to infer a **Go / Conditional Go / Further Work / No-Go** assessment for decision-oriented questions.
- **`AGENT_MODE=pipeline`** — a deterministic, non-tool-calling fallback: runs the same tool chain directly (route → playbook → item search → extraction/benchmark as applicable), then makes one LLM call over the assembled context. Used automatically if the LangGraph agent errors out and no partial context was gathered.
- **`AGENT_DRY_RUN=1`** — runs the tool chain only (routing, retrieval, benchmarking) and returns a formatted summary with no Bedrock call at all — useful for testing retrieval/routing without burning LLM tokens.

The dashboard's **`/api/chat/stream`** SSE endpoint deliberately does *not* run the interactive ReAct loop (tool calls can't be streamed token-by-token) — it runs the same deterministic tool-gathering pass as the pipeline mode (route → playbook → item search → extraction/benchmark) up front, then streams only the final synthesis call token-by-token, emitting `status`/`meta`/`token`/`done`/`error` SSE events.

Retrieval underneath every tool call is `_retrieve_and_rerank` in `rag_app.py`: Chroma similarity search over-fetches (`top_k * 4`, capped by the metadata filter), then reranks with a keyword-overlap signal blended against vector distance before truncating to `top_k`.

## Extraction method

`extractor.py` runs **5 focused passes** per report, each bound to the full `NI43101Report` schema (`schemas.py`) via `with_structured_output`, but instructed to populate only a disjoint subset of fields:

| Pass | Populates | Retrieval |
|------|-----------|-----------|
| **identity** | report metadata, `property_info`, `qualified_persons`, `authors`, `summary` | topic queries: executive_summary, front_matter, property |
| **resources** | `mineral_resources`, `mineral_reserves` (every category/zone row, with per-commodity `GradeEntry` items) | topic queries: resources, reserves |
| **economics** | `economics`, `mining_method`, `processing_method` | topic queries: economics, mining, processing |
| **technical** | `geology`, `exploration`, `environmental` (incl. indigenous consultation, political risk flags) | topic queries: geology, exploration, environmental |
| **metadata** | portfolio tags: `study_stage`, `deposit_type`, `primary_mining_method` (string, e.g. `"open pit"`), `processing_route`, `ore_type`, `cutoff_type`, `economic_year`, `effective_date`, `primary_commodity` — used for peer filtering and benchmarking | NI Item-aligned queries for Items 1, 2, 4, 7, 8, 14–17, 21, 22 |

Each pass's context is assembled by fanning out its topic/Item queries concurrently (`ThreadPoolExecutor`), de-duplicating chunks by `(source, page, chunk)`, and filtering to the single report being extracted. The five partial `NI43101Report` objects are then merged field-by-field — the first non-empty value across passes wins — into the final report. Splitting the context this way keeps each Claude call small (~25–80 unique chunks instead of 150–250), reduces per-call token usage by roughly 75%, and means a single pass hitting a rate limit doesn't block the whole report. If a Bedrock 429 is hit, `_invoke_with_backoff` retries with exponential backoff starting at 60s. Reports are processed 5-at-a-time (`ThreadPoolExecutor`) across `extract_all`.

The extractor is instructed to never invent values, to preserve units exactly as written, to convert DMS coordinates to decimal degrees, to create one `MineralResource`/`MineralReserve` row per table line item (with one `GradeEntry` per commodity column for polymetallic deposits), and to infer the portfolio metadata tags used for peer matching.

## Extracted schema

Every field is optional — the extractor returns `null` or `[]` rather than fabricating values when a section is absent from the report. All models are defined in `schemas.py`.

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
| `study_stage`, `deposit_type`, `primary_mining_method`, `processing_route`, `ore_type`, `cutoff_type`, `economic_year`, `effective_date`, `primary_commodity` | str | Portfolio metadata tags used for peer filtering and benchmarking |

### `PropertyInfo` — key fields

| Field | Description |
|-------|-------------|
| `project_name`, `country`, `region` | Identity and location |
| `latitude`, `longitude` | Decimal-degree coordinates for map plotting (converted from DMS if needed) |
| `jurisdiction` | Mining jurisdiction as "Region, Country" |
| `exchange_listed` | Stock exchange ticker (e.g. `TSX-V`, `ASX`) |
| `project_stage` | Development stage: Grassroots → Exploration → Resource Definition → PEA → Pre-Feasibility → Feasibility → Permitted → Construction → Operating → Care & Maintenance → Closed |
| `commodities` | Primary and by-product commodities |
| `ownership`, `tenure_status` | Ownership and mineral tenure |
| `accessibility`, `infrastructure`, `area_hectares` | Access, available infrastructure, property area |

### `MineralResource` / `MineralReserve`

Each row maps to one line in the resource or reserve table. The `grades` field is a list of `GradeEntry` objects — one per commodity column — supporting polymetallic deposits (e.g. Cu%, Au g/t, Ag g/t on the same row). `GradeEntry` carries `commodity`, `grade_value`, `grade_unit`, `contained_metal`, `contained_metal_unit`.

### `EconomicParameters`, `MiningMethod`, `ProcessingMethod`, `GeologySummary`, `ExplorationSummary`

Study type/date, pre- and post-tax NPV, IRR, payback, discount rate, initial/sustaining/total CapEx, OpEx, mine life, throughput, strip ratio, recovery rate, royalties, and metal price assumptions; mining method, rate, dilution, mine recovery, key equipment; processing method, throughput, per-commodity recoveries, concentrate grade, tailings management; deposit type, geological age, host rock, mineralisation style, structural controls, alteration, historical production; drill hole count/metres, drilling types, last program date, sampling/QAQC methods, notable intercepts, geophysical surveys.

### `EnvironmentalSummary`

| Field | Description |
|-------|-------------|
| `permit_status`, `key_permits_required`, `environmental_studies_completed` | Permitting status and outstanding requirements |
| `tailings_facility`, `water_management`, `closure_cost` | Environmental infrastructure and closure economics |
| `indigenous_consultation` | FPIC / duty-to-consult / IBA / community opposition status |
| `political_risk_flags` | List of explicit risk statements (resource nationalism, conflict zones, mining-code changes) |

### `QualifiedPerson`

`name`, `credentials` (e.g. P.Geo, P.Eng), `responsibility` (sections of the report this QP is accountable for).

## Files

| File | Purpose |
|------|---------|
| `rag_app.py` | Settings/env loading, olmocr PDF parsing (PyMuPDF fallback) → chunking → chapter tagging, Chroma ingest/upsert, retrieval + keyword rerank, chat prompt building, CLI entry point (`ingest`, `chat`, `extract`, `reindex-chapters`) |
| `schemas.py` | Pydantic models for the structured `NI43101Report` and all nested sub-models |
| `extractor.py` | 5-pass structured extraction pipeline, NI Item/topic-aligned context gathering, partial-result merging, rate-limit backoff, batch extraction across the portfolio |
| `chapter_index.py` | Detects "Item N — …" headings in parsed chunks, builds a per-PDF page-range chapter index, tags chunks with `ni_item`/`section_title` metadata |
| `routing_guide.py` | Loads `data/routing_guide.json` (NI Item definitions, question→Item routing table, due-diligence playbooks, benchmark templates) and resolves a question to primary/cross-check Items |
| `generate_routing_guide.py` | One-off script that generates `data/routing_guide.json` from the BMRC due-diligence memo content |
| `agent_chat.py` | LangGraph ReAct agent, its 6 tools, the deterministic pipeline fallback, dry-run mode, SSE streaming, flag/assessment inference |
| `benchmark.py` | Portfolio peer filtering (`find_peer_reports`) and cross-report numeric field benchmarking with outlier detection (`benchmark_field`) |
| `api.py` | FastAPI app — dashboard serving, document/ingest/extract/chat/reindex endpoints, API-key auth, SSE streaming |
| `dashboard.html` | Single-file HTML/CSS/JS portfolio dashboard (Leaflet map, Chart-free CSS charts, markdown-rendered chat via `marked.js`) |
| `streamlit_app.py` | Streamlit alternative UI — agentic chat with source-page thumbnails, extraction browser |
| `data/routing_guide.json` | Generated NI Item metadata, routing matrices, and benchmark templates consumed by `routing_guide.py` |
| `tests/test_routing.py`, `tests/test_benchmark.py`, `tests/test_agent_tools.py` | Pytest unit tests for Item routing, peer benchmarking, and agent tool functions (no LLM calls) |
| `spatial_data/` | Standalone 3D geological model HTML files (Plotly-based) displayed in the dashboard's **3D Models** page via iframe |
| `Dockerfile`, `docker-compose.yml` | Container build and a read-only EC2 deployment profile (API + dashboard only; ingestion/OCR is expected to run off-server) |
| `requirements.txt` | Pinned dependency list (see Libraries below) |

## Libraries

| Purpose | Libraries |
|---------|-----------|
| Vector store & embeddings | `chromadb`, `openai`, `langchain-core`, `langchain-openai`, `langchain-chroma` |
| LLM (Claude via AWS Bedrock) | `langchain-aws`, `boto3` |
| Agent | `langgraph` (ReAct tool-calling loop) |
| PDF parsing | `olmocr` (VLM-based per-page inference via OpenAI-compatible server; `allenai/olmOCR-7B-0225-preview`), `PyMuPDF` (`fitz` — text-layer fallback for ingestion + source-page thumbnails in Streamlit), `pyyaml` (parse olmocr YAML responses) |
| API server | `fastapi`, `uvicorn[standard]`, `anyio`, `python-multipart` |
| UI | `streamlit` |
| Utilities | `tqdm`, `python-dotenv`, `requests`, `pydantic>=2` |
| Testing | `pytest` |

## Docker

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. The bundled `docker-compose.yml` targets a **read-only deployment**: `knowledge/` and `extracted_data/` are bind-mounted read-only (ingestion/extraction are expected to run off-server against pre-built data), while `.chroma_db/` is writable because Chroma opens its SQLite store with write access even for read-only queries. The container image does not bundle an olmocr inference server — ingestion is expected to run off-server (pointing `OLMOCR_SERVER_URL` at a local or remote GPU endpoint). The PyMuPDF fallback works inside the container for digital PDFs.

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
| `BEDROCK_MODEL_ID` | Claude Sonnet 4 cross-region inference profile ARN | Bedrock model ID or inference profile ARN used for both chat and extraction |
| `API_KEY` | _(unset)_ | REST API key (`X-API-Key`). Unset = open local access. |
| `RAG_KNOWLEDGE_DIR` | `knowledge` | Primary PDF directory (uploads/deletes always target this dir) |
| `RAG_EXTRA_PDF_DIRS` | _(unset)_ | Additional read-only PDF directories, semicolon-separated |
| `RAG_CHROMA_DIR` | `.chroma_db` | Persistent vector store location |
| `RAG_COLLECTION_NAME` | `ni43101_knowledge` | Chroma collection name |
| `RAG_CHUNK_SIZE` | `1400` | Max characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RAG_EMBED_BATCH_SIZE` | `64` | Embedding batch size |
| `RAG_UPSERT_BATCH_SIZE` | `24` | Chroma upsert batch size |
| `RAG_TOP_K` | `8` | Chunks retrieved per non-agentic chat query |
| `RAG_EXTRACTED_DIR` | `extracted_data` | Where structured extractions (and chapter indexes) are saved |
| `NI43101_EXTRACT_TOP_K` | `12` | Chunks per topic/Item query fed to the extractor — lower to `6` if hitting rate limits |
| `OLMOCR_SERVER_URL` | _(unset)_ | OpenAI-compatible URL of the olmocr inference server (e.g. `http://localhost:8000/v1`). Unset = PyMuPDF text-layer fallback |
| `OLMOCR_API_KEY` | _(unset)_ | API key for the olmocr server (leave blank for local/unauthenticated endpoints) |
| `OLMOCR_MODEL` | `allenai/olmOCR-7B-0225-preview` | Model name passed to the inference server |
| `OLMOCR_WORKERS` | `4` | Parallel page requests per PDF |
| `AGENT_CHAT` | `1` | `1` = agentic chat (LangGraph/pipeline); `0` = plain RAG chat |
| `AGENT_MODE` | `langgraph` | `langgraph` = ReAct tool-calling agent; `pipeline` = deterministic tool-chain + single LLM call |
| `AGENT_DRY_RUN` | _(unset)_ | `1` = run agent tools only, skip the Bedrock call entirely (routing/retrieval/benchmark smoke test) |

## Testing

```bash
pytest
```

`tests/test_routing.py`, `tests/test_benchmark.py`, and `tests/test_agent_tools.py` cover NI Item routing/chapter tagging, peer-finding/benchmarking, and agent tool functions — none of them call Bedrock or OpenAI.

## Notes

- **PDF parsing** — `olmocr` (`allenai/olmOCR-7B-0225-preview`) processes each page as an image via an OpenAI-compatible inference server. The model returns YAML with `natural_text` (full page markdown) and flags like `is_table`/`is_diagram`. HTML `<table>` blocks within `natural_text` are extracted and stored as separate table-type chunks; figure/diagram pages are indexed as text if `natural_text` is non-empty. When `OLMOCR_SERVER_URL` is unset the pipeline falls back to PyMuPDF's text layer — fast and accurate for digitally-born PDFs, but skips scanned pages silently.
- **NI Item tagging** — after parsing, `chapter_index.py` scans each chunk's full content for "Item N — …" headings to build a page-range chapter index per report, then tags every chunk's `ni_item` and `section_title` metadata. This powers Item-scoped retrieval in both extraction and agentic chat.
- **Chunk overlap** — `RAG_CHUNK_OVERLAP` (default 150 chars) is applied during paragraph chunking: the tail of each completed chunk is prepended to the next chunk so context around paragraph boundaries is not lost at retrieval time.
- **Embedding cache** — `OpenAIEmbeddings` is wrapped in a process-level query cache (`_CachedOpenAIEmbeddings`) so repeated queries across extraction passes and tool calls don't make redundant API calls.
- **OpenAI connectivity** — the `check_openai_connectivity()` call at startup runs only for the `ingest` command. The `extract`, `chat`, and `reindex-chapters` commands do not touch OpenAI and can run without a live API connection.
- **Rate limiting** — Bedrock throttles at a per-account tokens-per-minute quota. The extractor uses exponential backoff starting at 60s per pass. Reduce `NI43101_EXTRACT_TOP_K` to shrink context size if limits persist.
- **Extraction accuracy** — the extractor never invents values: any field absent from the retrieved context is returned as `null` or `[]`. Fields like coordinates, project stage, exchange listing, and political risk flags require that the report context contains the relevant disclosure.
- **Map view** — requires `latitude` and `longitude` to have been extracted. Re-run extraction on existing reports (⚙ Extract) after upgrading to populate missing coordinate fields.
- **Chat** — answers are grounded in retrieved/tool context only; the agent states explicitly when context is absent or insufficient, and the streamed response falls back to an offline tool-output summary if the Bedrock call itself fails mid-stream.
- **Chroma private API** — `vectorstore._collection.count()` is wrapped in `_index_is_empty()` to insulate against future Chroma API changes. All empty-index checks go through this wrapper.
