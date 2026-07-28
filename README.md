# ORExtractor — NI 43-101 RAG & Extraction

A research and due-diligence tool for NI 43-101 mineral project technical reports. Drop PDFs into a knowledge directory, ingest them into a vector database, then interrogate them via an agentic due-diligence chat, a structured screener dashboard, or a REST API.

Built for the full spectrum of NI 43-101 users — investors and fund managers comparing project economics, investment bankers running due diligence, securities regulators checking disclosure compliance, M&A advisors benchmarking comparable transactions, lenders and royalty companies sizing project risk, and qualified persons reviewing resource estimate methodology.

> **Background** — NI 43-101 is the Canadian standard for disclosure of mineral project information, introduced after the 1997 Bre-X fraud to give investors standardized, QP-certified resource estimates. Its disclosure requirements are organized into 27 numbered "Items" (Item 1 Summary … Item 27 References); this tool routes retrieval, extraction, and due-diligence checklists against those same Item numbers.

## Capabilities

- **Layout-aware PDF ingestion** — Docling is the primary parser for text, hierarchy, reading order, tables, formulas, figures, captions, and page provenance. Deterministic quality gates invoke an isolated MinerU fallback only when needed. LangChain carries canonical chunks to Chroma, Claude Haiku enriches selected visuals, and LangSmith traces content-free stage metrics by default.
- **Structured extraction** — a 5-pass Claude (Bedrock) pipeline turns each report into a typed `NI43101Report` JSON object (identity, resources/reserves, economics/technical, geology/exploration/environmental, and portfolio metadata tags), never fabricating values for sections that aren't in the report. Each pass fans out its NI Item/topic queries concurrently for speed.
- **Agentic due-diligence chat** — a LangGraph ReAct agent with 6 tools (question routing, Item-scoped retrieval, extraction lookup, peer discovery, cross-report benchmarking, DD playbooks) that cites NI Item + page numbers, raises red flags from a due-diligence playbook, and issues a Go / Conditional Go / Further Work / No-Go assessment.
- **Portfolio screener dashboard** — a single-page HTML/JS app (no build step) with a KPI home page, sortable/filterable portfolio table, side-by-side report comparison (up to 4 reports), a Leaflet world map of every geolocated project, per-report resource/economics/geology/exploration views, and an embedded 3D geological model viewer (`spatial_data/` HTML models).
- **REST API** — FastAPI service exposing ingestion, extraction, chat (including streaming SSE), and report retrieval, optionally protected by an API key.
- **Streamlit alternative UI** — a lighter-weight chat + extraction browser with source-page thumbnails rendered from the original PDF.
- **CLI** — ingest, extract, chat, and chapter-reindex commands for scripted/offline use.

## Architecture

```
PDF files (knowledge/ + extra dirs)
  → Docling primary parse → deterministic quality gates → MinerU fallback when required
  → parser-neutral text/table/figure elements + lossless retained artifacts
  → selective Claude Haiku enrichment → deterministic validation/reconstruction
  → section-aware chunk + NI Item/section tagging (chapter_index.py consistency pass)
  → embed (local Qwen3; OpenAI startup fallback) → store (Chroma persistent vector DB)
  → versioned partition/enrichment caches + manifest skip unchanged work

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
- **CUDA-capable PyTorch** — recommended for local Qwen embeddings. CPU is supported with `LOCAL_EMBED_DEVICE=cpu`, but is slower.
- **OpenAI API key** — optional when local Qwen starts successfully; required only when OpenAI is selected or used as the startup fallback.
- **AWS credentials** with Bedrock access — used for selective visual enrichment (Claude Haiku), chat, and structured extraction. Credentials are read through the normal AWS credential chain.
- **Docling** — installed by `requirements.txt` and runs locally by default. `DOCLING_EXECUTION_MODE=serve` can use a Docling Serve `/v1/convert/file` deployment.
- **MinerU fallback** — use a service via `MINERU_API_URL` (recommended), or install `requirements-mineru.txt` in a separate environment and select CLI mode. It is intentionally not imported into the main application.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Install the optional LangChain Docling comparison adapter only when running
parser/chunker benchmarks:

```bash
pip install -r requirements-benchmark.txt
```

Copy and configure the environment file:

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Edit `.env` and set AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optionally `AWS_SESSION_TOKEN` for STS). Keep `EMBEDDING_PROVIDER=qwen` for local embeddings. Set `OPENAI_API_KEY` if you want the configured OpenAI startup fallback.

## Workflow

### 1 — Add reports

Drop NI 43-101 PDF files anywhere under `knowledge/`. Discovery is recursive,
so category/year subfolders are ingested automatically. Multiple additional
directory trees are supported via `RAG_EXTRA_PDF_DIRS` (see Configuration).
Nested reports use their path relative to the configured root as the stable
source ID, which prevents equal filenames in different folders from colliding.

### 2 — Ingest

```bash
python rag_app.py ingest
```

Set `INGESTION_BACKEND=docling` to use the layout-aware pipeline. Docling's lossless JSON and Markdown, normalized elements, table/figure crops, quality report, and parser-selection decision are retained. If configured gates fail, MinerU runs on the whole document and the router selects one complete canonical stream by deterministic score. The selected stream is converted to standard LangChain `Document` records with parser provenance.

Multi-report ingestion uses a bounded three-stage pipeline: a persistent,
killable Docling process parses report N with reusable converters, a worker enriches report
N-1 through the configured visual provider, and a single indexing worker embeds/upserts report N-2.
Only the indexing worker writes to Chroma. Adaptive OCR skips RapidOCR when
nearly every page already has native text; scanned or mixed documents retain
GPU OCR. Long reports are checkpointed into page ranges, and a hard timeout
terminates and replaces a stuck Docling process instead of leaking native
threads. Source PDFs are exposed to Docling through short ASCII staging paths,
which avoids Windows path-length and Unicode backend failures. Per-report
Docling timings and runtime decisions are saved in
`parsers/docling/profiling.json`.

When MinerU fallback is enabled, startup now requires a configured MinerU
service or CLI. Degraded and timed-out results remain retryable and are not
stored as completed parser caches or resumable manifests.

Qwen3-VL runs locally through Ollama by default and is called only for routed
figures and important or suspicious tables. Set `VISUAL_MODEL_PROVIDER=bedrock`
to use the retained Bedrock adapter. A failed visual call is recorded per
element and does not stop normal text ingestion. Reconstruction is deterministic
through Plotly or Graphviz, never executes model-generated code, and rejects
geological maps, cross-sections, mine plans, block models, and similar technical
geometry.

Visual crops wholly contained in the top or bottom 10% page-margin bands are
treated as header/footer furniture: they are removed before enrichment,
chunking, and parser-cache persistence, and their generated crop files are
deleted. Remaining crops are deduplicated by content and a conservative
perceptual comparison. Repeated body occurrences retain their own report
context but share one canonical image and only the canonical occurrence is
eligible for visual-model enrichment. Visual context includes the caption,
leading and trailing narrative, and matching references such as
`Figure 16-27`, `Figure 16.27`, or `Figure 16 27` found elsewhere in the report.

Partition output, normalized elements, structured model responses, final chunks, and reconstruction audit manifests are stored below `RAG_ARTIFACT_DIR`. Versioned cache keys include source, image/context, parser, model, prompt, and schema versions. Chroma records retain scalar metadata for source, page, type, NI Item, section title, element ID, and asset provenance.

To wipe and rebuild the index (e.g. after changing chunk settings):

```bash
python rag_app.py ingest --rebuild --file report.pdf --parser docling
python rag_app.py ingest --file report.pdf --parser docling --partition-only
python rag_app.py ingest --file report.pdf --parser docling --no-fallback
python rag_app.py ingest --file report.pdf --force-parser mineru
python rag_app.py ingest --reprocess-visuals --parser docling
python rag_app.py inspect-elements --file report.pdf
python rag_app.py compare-parsers --file report.pdf
```

The first command deletes the configured Chroma collection and rebuilds it from exactly one PDF. Use `--partition-only` first when validating parser artifacts without embeddings, visual-model calls, or Chroma writes.
For a nested report, pass its relative source path, for example
`--file "Sedar_2024/April/report.pdf"`. A basename remains valid when it is
unique across all configured PDF trees.

For the fastest time to a searchable full-corpus index, run a text-first pass
and enrich visuals afterward:

```bash
python rag_app.py ingest --rebuild --parser docling --fallback mineru --no-visual-enrichment
python rag_app.py ingest --parser docling --fallback mineru
```

Both passes are resumable. If interrupted, rerun the same command without
`--rebuild`; completed reports are skipped using the manifest.
The text-first pass uses `DOCLING_TEXT_FIRST_TABLE_MODE=fast`; the visual pass
returns to `DOCLING_TABLE_MODE=accurate`. Existing accurate results satisfy a
fast-pass request, while degraded reports are automatically retried.

The embedding backend is selected once at process startup. If local Qwen fails
its health check, ORExtractor uses the configured OpenAI fallback before opening
or rebuilding Chroma. It never mixes providers within a collection: the
provider, model, dimensions, query instruction, and maximum token length are
stored as collection metadata. Changing any of them requires `--rebuild`; an
incompatible existing collection is rejected with a rebuild command.

Benchmark the configured visual model without opening or changing Chroma:

```bash
ollama pull qwen3-vl:8b-instruct-q8_0
python rag_app.py benchmark-visuals --provider ollama \
  --model qwen3-vl:8b-instruct-q8_0 --real-samples 20
```

The command always runs eight generated gold cases and can add a deterministic
sample of retained real parser artifacts. It writes JSON and Markdown results
under `benchmark_results/visual` unless `--output-dir` is supplied. Retained
real cases are marked unverified; they are not counted as gold until a human
reviewer supplies expectations.

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

If `API_KEY` is set in `.env`, every `/api/*` request must include
`X-API-Key: <key>`. The `/dashboard` route and read-only
`/spatial_data/*` model files remain public.

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
| `rag_app.py` | Settings/env loading, Chroma ingest/retrieval, chat prompt building, and CLI entry points |
| `ingestion/` | Parser-neutral schemas; Docling and MinerU adapters; quality routing; hierarchy/context; LangChain chunking; Bedrock enrichment; caches; evaluation; manifests; and LangSmith telemetry |
| `api_routers/` | Document, ingestion, Chroma export/info, reports, and chat API routers |
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
| Vector store & embeddings | `chromadb`, `transformers`, local Qwen3, `openai`, `langchain-core`, `langchain-openai`, `langchain-chroma` |
| LLM (Claude via AWS Bedrock) | `langchain-aws`, `boto3` |
| Agent | `langgraph` (ReAct tool-calling loop) |
| PDF parsing | `docling` (primary), isolated `mineru` (fallback), `PyMuPDF` (rendering and diagnostics) |
| Visual enrichment/reconstruction | `langchain-aws`, Claude Haiku on Bedrock, `pillow`, `tenacity`, `plotly`, `kaleido`, `graphviz` |
| Observability | `langsmith` (content hidden by default) |
| API server | `fastapi`, `uvicorn[standard]`, `anyio`, `python-multipart` |
| UI | `streamlit` |
| Utilities | `tqdm`, `python-dotenv`, `requests`, `pydantic>=2` |
| Testing | `pytest` |

## Docker

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. The bundled `docker-compose.yml` targets a **read-only deployment**: `knowledge/` and `extracted_data/` are bind-mounted read-only (ingestion/extraction are expected to run off-server against pre-built data), while `.chroma_db/` is writable because Chroma opens its SQLite store with write access even for read-only queries.

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `qwen` | Primary embedding backend: `qwen` or `openai` |
| `EMBEDDING_FALLBACK_PROVIDER` | `openai` | Startup fallback backend; empty disables fallback |
| `LOCAL_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Hugging Face model used for local embeddings |
| `LOCAL_EMBED_DEVICE` | `cuda` | Local embedding device: `auto`, `cpu`, `cuda`, or `cuda:N` |
| `LOCAL_EMBED_BATCH_SIZE` | `16` | Local inference batch size |
| `LOCAL_EMBED_MAX_LENGTH` | `512` | Maximum tokens per local embedding input |
| `LOCAL_EMBED_DIMENSIONS` | `1024` | Number of local embedding dimensions |
| `LOCAL_EMBED_DTYPE` | `float16` | GPU inference dtype (`float16`, `bfloat16`, or `float32`) |
| `LOCAL_EMBED_QUERY_INSTRUCTION` | NI 43-101 retrieval instruction | Instruction prepended to queries, not documents |
| `OPENAI_API_KEY` | — | Required only when OpenAI is selected or used as fallback |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI fallback model |
| `OPENAI_EMBED_DIMENSIONS` | `1536` | OpenAI embedding dimensions |
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
| `RAG_UPSERT_BATCH_SIZE` | `128` | Chroma upsert batch size; OpenAI requests are internally split by `RAG_EMBED_BATCH_SIZE` |
| `RAG_TOP_K` | `8` | Chunks retrieved per non-agentic chat query |
| `RAG_EXTRACTED_DIR` | `extracted_data` | Where structured extractions (and chapter indexes) are saved |
| `NI43101_EXTRACT_TOP_K` | `12` | Chunks per topic/Item query fed to the extractor — lower to `6` if hitting rate limits |
| `INGEST_PIPELINE_ENABLED` | `true` | Overlap parse, enrichment, and indexing through bounded single-owner stages |
| `INGEST_PIPELINE_QUEUE_SIZE` | `2` | Reports buffered between stages; total in-flight work remains bounded |
| `INGESTION_BACKEND` | `docling` | `docling` primary or `mineru` forced |
| `PARSER_FALLBACK` | `mineru` | Quality-gated fallback parser |
| `PARSER_FALLBACK_ENABLED` | `true` | Allow deterministic fallback routing |
| `DOCLING_EXECUTION_MODE` | `local` | `local` or `serve` |
| `DOCLING_SERVE_URL` | _(unset)_ | Docling Serve base URL |
| `DOCLING_OCR_BACKEND` | `onnxruntime` | RapidOCR backend: `onnxruntime`, `openvino`, `paddle`, or `torch`; use `torch` with CUDA PyTorch for GPU OCR |
| `DOCLING_OCR_LANGUAGES` | `english` | Comma-separated OCR languages |
| `DOCLING_FORCE_FULL_PAGE_OCR` | `false` | OCR every page region; leave disabled to skip unnecessary OCR on native PDF text |
| `DOCLING_OCR_BITMAP_AREA_THRESHOLD` | `0.05` | Minimum bitmap-area ratio that triggers adaptive OCR |
| `DOCLING_IMAGES_SCALE` | `1.0` | Render scale for retained page/figure images |
| `DOCLING_OCR_BATCH_SIZE` | `2` | OCR target batch; native-memory failures automatically fall back to the safe batch |
| `DOCLING_LAYOUT_BATCH_SIZE` | `2` | Layout stage target batch |
| `DOCLING_TABLE_BATCH_SIZE` | `1` | Table stage batch size |
| `DOCLING_QUEUE_MAX_SIZE` | `2` | Maximum buffered pages between Docling stages |
| `DOCLING_PAGE_BATCH_SIZE` | `2` | Target PDF page batch; automatically falls back on native-memory failures |
| `DOCLING_NUM_THREADS` | `4` | Docling accelerator worker threads |
| `DOCLING_DEVICE` | `auto` | Accelerator: `auto`, `cpu`, `cuda`, `mps`, or `xpu` |
| `DOCLING_ADAPTIVE_OCR` | `true` | Bypass OCR per document when native-text coverage passes the configured gate |
| `DOCLING_NATIVE_TEXT_MIN_CHARS` | `80` | Minimum non-whitespace characters for a native-text page |
| `DOCLING_NATIVE_TEXT_COVERAGE` | `0.98` | Native-text page ratio required to bypass OCR |
| `DOCLING_NATIVE_TEXT_MAX_EMPTY_PAGES` | `2` | Maximum non-native pages allowed when bypassing OCR |
| `DOCLING_BATCH_FALLBACK_ENABLED` | `true` | Retry native-memory failures with the safe batch and retain safe mode afterward |
| `DOCLING_SAFE_BATCH_SIZE` | `1` | Low-memory retry batch size |
| `DOCLING_FAST_TABLE_MAX_PAGES` | `20` | Use fast table mode for short reports; longer reports retain the configured mode |
| `DOCLING_TEXT_FIRST_TABLE_MODE` | `fast` | Table mode used by `--no-visual-enrichment`; `configured` preserves `DOCLING_TABLE_MODE` |
| `DOCLING_PROFILING` | `true` | Persist Docling timings and effective runtime decisions per report |
| `DOCLING_CONVERTER_CACHE_SIZE` | `2` | Reusable Docling pipeline variants retained in memory |
| `DOCLING_PROCESS_ISOLATION` | `true` | Run Docling in a persistent process that can be terminated safely |
| `DOCLING_HARD_TIMEOUT_SECONDS` | `900` | Parent-enforced deadline per document or page segment |
| `DOCLING_SEGMENT_MIN_PAGES` | `300` | Page count at which checkpointed segment processing begins |
| `DOCLING_SEGMENT_PAGES` | `100` | Pages processed per checkpointed Docling request |
| `MINERU_EXECUTION_MODE` | `service` | `service` or isolated `cli` |
| `MINERU_API_URL` | _(unset)_ | MinerU service endpoint |
| `PARSER_MIN_TEXT_PAGE_COVERAGE` | `0.90` | Fallback text-page coverage gate; calibrate on corpus |
| `PARSER_MAX_EMPTY_PAGE_RATIO` | `0.10` | Fallback empty-page gate |
| `PARSER_MAX_REPLACEMENT_CHAR_RATIO` | `0.01` | Text corruption gate |
| `PARSER_MIN_TABLE_VALID_RATIO` | `0.80` | Table structure gate |
| `PARSER_MIN_CACHE_QUALITY_SCORE` | `0.90` | Minimum parser score that may be cached or marked complete |
| `PARSER_MIN_PAGE_COUNT_AGREEMENT` | `0.90` | Minimum observed/expected page agreement for resumable completion |
| `PARSER_REQUIRE_FALLBACK_READY` | `true` | Fail before ingestion when the enabled MinerU adapter is unavailable |
| `RAG_ARTIFACT_DIR` | `ingestion_artifacts` | Retained source crops, raw/normalized partition output, enrichments, chunks, and audit manifests |
| `RAG_INGEST_WORK_DIR` | `.ingestion_work` | Temporary short-path PDF aliases used by native parsers |
| `VISUAL_MODEL_PROVIDER` | `ollama` in `.env.example` | Visual enrichment provider: `ollama` or `bedrock` |
| `VISUAL_MODEL_CONCURRENCY` | `1` | Maximum concurrent calls to the selected visual provider |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama API endpoint |
| `OLLAMA_VISUAL_MODEL` | `qwen3-vl:8b-instruct-q8_0` | Local vision model and quantization |
| `OLLAMA_VISUAL_TIMEOUT_SECONDS` | `300` | Per-call timeout for local visual analysis |
| `OLLAMA_VISUAL_CONTEXT_LENGTH` | `8192` | Context window for schema, prompt, image tokens, and output |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long Ollama retains the model after a request |
| `BEDROCK_VISUAL_MODEL_ID` | Claude 3.5 Haiku inference profile | Separate Bedrock model/inference profile for visual ingestion |
| `BEDROCK_VISUAL_MAX_TOKENS` | `3500` | Maximum output tokens per visual/table call |
| `BEDROCK_VISUAL_CONCURRENCY` | `8` | Legacy fallback when `VISUAL_MODEL_CONCURRENCY` is unset |
| `BEDROCK_VISUAL_CONFIDENCE_THRESHOLD` | `0.85` | Minimum confidence for reconstruction |
| `VISUAL_MAX_CALLS_PER_REPORT` | `30` | Combined per-report visual/table call limit |
| `VISUAL_MAX_TABLE_CALLS_PER_REPORT` | `20` | Table-validation share of the report call budget |
| `VISUAL_MAX_FIGURE_CALLS_PER_REPORT` | `10` | Figure-analysis share of the report call budget |
| `VISUAL_TOKEN_BUDGET_PER_REPORT` | `350000` | Conservative per-report visual token budget |
| `LANGSMITH_TRACING` | `false` | Enable ingestion traces |
| `LANGSMITH_TRACE_CONTENT` | `false` | Include document content in traces; disabled by default |
| `AGENT_CHAT` | `1` | `1` = agentic chat (LangGraph/pipeline); `0` = plain RAG chat |
| `AGENT_MODE` | `langgraph` | `langgraph` = ReAct tool-calling agent; `pipeline` = deterministic tool-chain + single LLM call |
| `AGENT_DRY_RUN` | _(unset)_ | `1` = run agent tools only, skip the Bedrock call entirely (routing/retrieval/benchmark smoke test) |

## Testing

```bash
pytest
```

`tests/test_routing.py`, `tests/test_benchmark.py`, and `tests/test_agent_tools.py` cover NI Item routing/chapter tagging, peer-finding/benchmarking, and agent tool functions — none of them call Bedrock or OpenAI.

## Notes

- **PDF parsing** — Docling preserves lossless parser JSON and source artifacts; MinerU is invoked only on explicit request or deterministic quality failure.
- **NI Item tagging** — after parsing, `chapter_index.py` scans each chunk's full content for "Item N — …" headings to build a page-range chapter index per report, then tags every chunk's `ni_item` and `section_title` metadata. This powers Item-scoped retrieval in both extraction and agentic chat.
- **Chunk overlap** — `RAG_CHUNK_OVERLAP` (default 150 chars) is applied during paragraph chunking: the tail of each completed chunk is prepended to the next chunk so context around paragraph boundaries is not lost at retrieval time.
- **Embedding cache** — `OpenAIEmbeddings` is wrapped in a process-level query cache (`_CachedOpenAIEmbeddings`) so repeated queries across extraction passes and tool calls don't make redundant API calls.
- **Embedding startup checks** — the selected local or OpenAI embedding backend is health-checked before a rebuild can delete the existing collection.
- **Rate limiting** — Bedrock throttles at a per-account tokens-per-minute quota. The extractor uses exponential backoff starting at 60s per pass. Reduce `NI43101_EXTRACT_TOP_K` to shrink context size if limits persist.
- **Extraction accuracy** — the extractor never invents values: any field absent from the retrieved context is returned as `null` or `[]`. Fields like coordinates, project stage, exchange listing, and political risk flags require that the report context contains the relevant disclosure.
- **Map view** — requires `latitude` and `longitude` to have been extracted. Re-run extraction on existing reports (⚙ Extract) after upgrading to populate missing coordinate fields.
- **Chat** — answers are grounded in retrieved/tool context only; the agent states explicitly when context is absent or insufficient, and the streamed response falls back to an offline tool-output summary if the Bedrock call itself fails mid-stream.
- **Chroma private API** — `vectorstore._collection.count()` is wrapped in `_index_is_empty()` to insulate against future Chroma API changes. All empty-index checks go through this wrapper.
