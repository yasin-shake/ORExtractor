"""Structured extraction of NI 43-101 data from ingested reports.

Extraction runs as five focused passes, each targeting a distinct subset of the
schema with a proportionally smaller context window:

1. **Identity** (executive summary, front matter, property) → report metadata,
   property info, qualified persons, project summary.
2. **Resources & Reserves** (resources, reserves topics) → all resource and
   reserve table rows, one MineralResource/MineralReserve per line item.
3. **Economics & Technical** (economics, mining, processing topics) → NPV/IRR/
   CapEx, mining method, processing method.
4. **Geology & Environment** (geology, exploration, environmental topics) →
   deposit type, drilling stats, permitting, indigenous consultation, risk flags.
5. **Portfolio Metadata** (project stage, deposit type, primary mining method,
   commodity tags, political risk) → scalar tag fields used for peer matching
   and portfolio screening.

Each pass uses ``with_structured_output`` bound to the full ``NI43101Report``
schema but is instructed to populate only its designated fields; the five partial
results are merged field-by-field (first non-empty value wins) into the final
report object.  Splitting the context reduces per-call token usage by ~80 %,
avoids monolithic-context degradation, and means a single-pass rate-limit retry
does not block the entire extraction.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm
from langchain_aws import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from rag_app import (
    Settings,
    iter_pdf_paths,
    pdf_source_id,
    query_chunks,
    source_output_path,
)
from schemas import NI43101Report

# ---------------------------------------------------------------------------
# Rate-limit retry helper
# ---------------------------------------------------------------------------

def _invoke_with_backoff(structured_llm, messages, max_retries: int = 6):
    """Invoke a LangChain LLM with exponential backoff on 429 rate-limit errors."""
    delay = 60  # initial wait (seconds) — gives the rate-limit window time to refill
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke(messages)
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = "rate_limit" in msg or "429" in msg or "rate limit" in msg
            is_timeout = "read timeout" in msg or "timed out" in msg
            if is_rate_limit and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)  # 60 → 120 → 240 → …
                print(
                    f"  Rate limit hit — waiting {wait}s before retry "
                    f"{attempt + 1}/{max_retries - 1} …",
                    flush=True,
                )
                time.sleep(wait)
            elif is_timeout and attempt < max_retries - 1:
                # Transient network stall, not throttling — retry quickly.
                print(
                    f"  Read timeout — retrying in 15s "
                    f"({attempt + 1}/{max_retries - 1}) …",
                    flush=True,
                )
                time.sleep(15)
            else:
                raise

# NI 43-101 Item-aligned retrieval queries (BMRC chapter routing guide).
_ITEM_QUERIES: dict[int, List[str]] = {
    1: ["executive summary project overview key risks recommendations viability"],
    2: ["introduction qualified person QP effective date site visit issuer"],
    3: ["reliance on other experts legal tenure tax environment permitting"],
    4: ["property description location tenure ownership royalty NSR encumbrance licence"],
    5: ["accessibility climate infrastructure power water road port logistics"],
    6: ["history historical exploration production prior estimates"],
    7: ["geological setting mineralization host rock alteration structural controls"],
    8: ["deposit type genetic model analogue deposit classification"],
    9: ["exploration mapping geochemistry geophysics trenching sampling"],
    10: ["drilling drill holes spacing orientation recovery downhole survey intercept"],
    11: ["sample preparation analyses security QAQC CRM blank duplicate laboratory"],
    12: ["data verification QP validation database integrity site visit independent"],
    13: ["metallurgical testing testwork recovery ore type deleterious elements"],
    14: ["mineral resource estimate cut-off classification variography density block model"],
    15: ["mineral reserve estimate proven probable modifying factors cut-off conversion"],
    16: ["mining methods open pit underground dilution recovery strip ratio schedule"],
    17: ["recovery methods processing plant flowsheet throughput reagents concentrate"],
    18: ["project infrastructure tailings waste dump power water port logistics"],
    19: ["market studies contracts payability concentrate pricing offtake TC RC"],
    20: ["environmental permitting social impact community ESG closure resettlement"],
    21: ["capital operating costs CAPEX OPEX contingency sustaining closure accuracy"],
    22: ["economic analysis NPV IRR payback sensitivity discount rate metal price"],
    23: ["adjacent properties nearby deposits mines analogues"],
    24: ["other relevant data information project specific technical"],
    25: ["interpretation conclusions QP risks confidence materiality"],
    26: ["recommendations work program budget next stage study requirements"],
    27: ["references citations source reports bibliography"],
}

# Legacy topic groups retained for backward-compatible extraction passes.
_TOPIC_QUERIES: dict[str, List[str]] = {
    "executive_summary": [
        "executive summary highlights key results overview project description findings",
        "project highlights summary key metrics conclusions NI 43-101",
    ],
    "front_matter": [
        "report title effective date issuer qualified person author",
        "NI 43-101 technical report prepared for company",
    ],
    "property": [
        "property location country region coordinates latitude longitude ownership claims tenure",
        "project area hectares mineral tenure permitting commodities exchange listed TSX",
        "site access road infrastructure power water logistics facilities",
        "project stage exploration PEA pre-feasibility feasibility construction operating",
        "jurisdiction province state mining district regulatory authority",
    ],
    "geology": [
        "deposit type host rock mineralization style structural controls alteration",
        "geological setting mineralization geology summary geological age",
        "historical production past mining operations previous operator",
    ],
    "exploration": [
        "exploration drilling drill holes metres drilled sampling assay QA QC",
        "significant drill intercepts diamond RC drilling program",
        "geophysical survey IP airborne magnetics CSAMT ground survey",
    ],
    "resources": [
        "mineral resource estimate measured indicated inferred tonnes grade contained metal cut-off",
        "resource estimate table effective date cut-off grade",
        "total measured indicated inferred global mineral resource estimate",
    ],
    "reserves": [
        "mineral reserve estimate proven probable tonnes grade contained metal cut-off",
        "ore reserve table proven probable",
    ],
    "mining": [
        "mining method open pit underground stope dilution mine recovery mining rate equipment fleet",
        "mine design pit slope optimization underground development extraction sequence",
    ],
    "processing": [
        "processing method metallurgical test work recovery flotation leach CIL concentrate grade",
        "process plant throughput metallurgy recoveries by commodity reagents",
    ],
    "economics": [
        "net present value NPV internal rate of return IRR payback discount rate",
        "initial capital cost CAPEX operating cost OPEX AISC mine life metal price assumptions",
        "economic analysis pre-tax post-tax feasibility PEA strip ratio throughput",
        "royalty NSR NPI gross revenue royalty rate",
    ],
    "environmental": [
        "environmental assessment permit status baseline studies tailings closure reclamation bond",
        "environmental impact assessment key permits water management closure cost bond",
    ],
}

_EXTRACTION_INSTRUCTION = (
    "You are a meticulous mining analyst extracting structured data from an NI 43-101 "
    "technical report. Use ONLY the provided context excerpts. Rules:\n"
    "1. Populate every field you can find; leave a field null (or empty list) when absent.\n"
    "2. Never invent or estimate values.\n"
    "3. Preserve units exactly as written; strip thousands separators from numbers.\n"
    "4. For resource/reserve tables: create one MineralResource/MineralReserve entry per "
    "category+zone line item. For each row populate the `grades` list with one GradeEntry "
    "per commodity column (e.g. Cu%, Au g/t, Ag g/t all become separate GradeEntry items). "
    "Set the top-level `commodity` field to the primary/dominant commodity for that row.\n"
    "5. For economics: capture strip_ratio, throughput_tpd, recovery_rate, royalties and "
    "study_effective_date in addition to NPV/IRR/CAPEX.\n"
    "6. Populate mining_method, processing_method, and environmental sections when the "
    "context contains information about mine design, process plant, or permitting.\n"
    "7. Write a concise 3-5 sentence project summary in the `summary` field.\n"
    "8. Coordinates — extract property_info.latitude and property_info.longitude as "
    "decimal-degree floats. Convert DMS (e.g. '49°30′N, 117°45′W') to decimal "
    "(49.500, -117.750). If only a bounding box is given, use the centroid.\n"
    "9. property_info.project_stage — infer from context: look for phrases like "
    "'Preliminary Economic Assessment', 'Pre-Feasibility Study', 'Feasibility Study', "
    "'construction', 'producing mine', 'care and maintenance', 'exploration stage'.\n"
    "10. property_info.exchange_listed — look for stock ticker/exchange references on "
    "the cover page or in the issuer description (e.g. 'TSX-V: ABC', 'listed on the TSX').\n"
    "11. property_info.jurisdiction — extract the mining jurisdiction as 'Region, Country' "
    "(e.g. 'Ontario, Canada', 'Nevada, USA').\n"
    "12. report_purpose — identify why the report was commissioned (trigger event); "
    "report_previous_resource_date — the effective date of the estimate this report updates.\n"
    "13. environmental.indigenous_consultation — capture any FPIC, duty-to-consult, IBA "
    "or community-opposition notes.\n"
    "14. environmental.political_risk_flags — list explicit risk statements about resource "
    "nationalism, political instability, mining-code changes, or social licence issues.\n"
    "15. Metadata tags — populate study_stage, deposit_type, primary_mining_method, processing_route, "
    "ore_type, cutoff_type, economic_year, effective_date, primary_commodity from context."
)


# ---------------------------------------------------------------------------
# Extraction pass definitions
# ---------------------------------------------------------------------------
# Each pass targets a slice of the schema.  The 'focus' string is injected into
# the HumanMessage so Claude knows exactly which fields to populate and which to
# leave empty — keeping each context window small and the model's attention tight.

_EXTRACTION_PASSES: List[dict] = [
    {
        "name": "identity",
        "topics": ["executive_summary", "front_matter", "property"],
        "focus": (
            "Extract ONLY these fields: report_title, report_date, report_purpose, "
            "previous_resource_date, issuer, authors, qualified_persons, "
            "property_info (all sub-fields: project_name, country, region, coordinates, "
            "latitude, longitude, jurisdiction, exchange_listed, project_stage, "
            "area_hectares, ownership, commodities, tenure_status, accessibility, "
            "infrastructure), and summary. "
            "Leave mineral_resources, mineral_reserves, economics, mining_method, "
            "processing_method, environmental, geology, and exploration null or empty."
        ),
    },
    {
        "name": "resources",
        "topics": ["resources", "reserves"],
        "focus": (
            "Extract ONLY: mineral_resources and mineral_reserves. "
            "Capture every single row in every resource and reserve table, "
            "including sub-total and total rows if present. "
            "For polymetallic rows populate grades with one GradeEntry per commodity column. "
            "Leave all other top-level fields null or empty."
        ),
    },
    {
        "name": "economics",
        "topics": ["economics", "mining", "processing"],
        "focus": (
            "Extract ONLY: economics (study_type, study_effective_date, pre_tax_npv, "
            "post_tax_npv, irr, payback_years, discount_rate, initial_capex, "
            "sustaining_capex, total_capex, opex, mine_life_years, throughput_tpd, "
            "strip_ratio, recovery_rate, royalties, metal_price_assumptions), "
            "mining_method, and processing_method. "
            "Leave all other top-level fields null or empty."
        ),
    },
    {
        "name": "technical",
        "topics": ["geology", "exploration", "environmental"],
        "focus": (
            "Extract ONLY: geology (deposit_type, geological_age, host_rock, "
            "mineralization_style, structural_controls, alteration, historical_production), "
            "exploration (total_drill_holes, total_metres_drilled, drilling_types, "
            "last_program_date, sampling_methods, notable_intercepts, geophysical_surveys), "
            "and environmental (permit_status, key_permits_required, "
            "environmental_studies_completed, tailings_facility, water_management, "
            "closure_cost, indigenous_consultation, political_risk_flags). "
            "Leave all other top-level fields null or empty."
        ),
    },
    {
        "name": "metadata",
        "topics": [],
        "items": [1, 2, 4, 7, 8, 14, 15, 16, 17, 21, 22],
        "focus": (
            "Extract ONLY portfolio metadata tags: study_stage, deposit_type, primary_mining_method, "
            "processing_route, ore_type, cutoff_type, economic_year, effective_date, "
            "primary_commodity. Infer from report purpose, property stage, geology deposit "
            "type, mining/processing sections, resource cut-off type, and economics. "
            "Leave all other fields null or empty."
        ),
    },
]


# ---------------------------------------------------------------------------
# Context gathering (per-topic-group)
# ---------------------------------------------------------------------------

def _gather_context_for_topics(
    settings: Settings,
    vectorstore: Chroma,
    filename: str,
    topics: List[str],
    items: Optional[List[int]] = None,
    per_query_k: Optional[int] = None,
) -> str:
    """Retrieve and de-duplicate topic- or Item-relevant chunks."""
    per_query_k = per_query_k or settings.extract_top_k
    seen_chunks: set = set()
    blocks: List[str] = []

    def _fetch_item(item_num: int) -> Tuple[int, List[Tuple[str, dict]]]:
        queries = _ITEM_QUERIES.get(item_num, [])
        out: List[Tuple[str, dict]] = []
        for q in queries:
            parts, metadatas = query_chunks(
                vectorstore, q, per_query_k, filter_sources=[filename]
            )
            for part, meta in zip(parts, metadatas):
                if meta.get("ni_item") and int(meta.get("ni_item") or 0) not in (0, item_num):
                    continue
                out.append((part, meta))
        return item_num, out

    def _fetch_topic(topic: str) -> Tuple[str, List[Tuple[str, dict]]]:
        queries = _TOPIC_QUERIES.get(topic, [])
        out: List[Tuple[str, dict]] = []
        for q in queries:
            parts, metadatas = query_chunks(
                vectorstore, q, per_query_k, filter_sources=[filename]
            )
            for part, meta in zip(parts, metadatas):
                out.append((part, meta))
        return topic, out

    if items:
        item_results: dict = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_fetch_item, n): n for n in items}
            for fut in as_completed(futs):
                item_num, pairs = fut.result()
                item_results[item_num] = pairs
        for item_num in items:
            item_parts: List[str] = []
            for part, meta in item_results.get(item_num, []):
                key = (meta.get("source"), meta.get("page"), meta.get("chunk"))
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
                item_parts.append(part)
            if item_parts:
                blocks.append(f"### Context for: Item {item_num}\n" + "\n\n".join(item_parts))
        return "\n\n".join(blocks)

    topic_results: dict = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch_topic, t): t for t in topics}
        for fut in as_completed(futs):
            topic, pairs = fut.result()
            topic_results[topic] = pairs
    for topic in topics:
        topic_parts: List[str] = []
        for part, meta in topic_results.get(topic, []):
            key = (meta.get("source"), meta.get("page"), meta.get("chunk"))
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            topic_parts.append(part)
        if topic_parts:
            blocks.append(f"### Context for: {topic}\n" + "\n\n".join(topic_parts))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Partial-result merger
# ---------------------------------------------------------------------------

def _merge_partials(source_file: str, *partials: NI43101Report) -> NI43101Report:
    """Merge pass-level partial extractions into one report.

    Iterates over every field in NI43101Report; the first non-empty value found
    across the partials (in pass order) wins.  Because each pass is designed to
    populate a disjoint set of fields, collisions are rare and intentional order
    (identity first) resolves them sensibly.
    """
    merged: dict = {"source_file": source_file}
    for partial in partials:
        for field_name in NI43101Report.model_fields:
            if field_name in merged:
                continue
            val = getattr(partial, field_name, None)
            if val is None:
                continue
            if isinstance(val, list) and len(val) == 0:
                continue
            merged[field_name] = val
    return NI43101Report(**merged)


# ---------------------------------------------------------------------------
# Single-report extraction
# ---------------------------------------------------------------------------

def extract_report(
    settings: Settings,
    vectorstore: Chroma,
    llm: ChatBedrockConverse,
    filename: str,
) -> NI43101Report:
    """Extract a structured NI43101Report via four focused passes.

    Each pass retrieves context only for its topic group (~25–80 unique chunks
    instead of 150–250), reducing per-call token usage by ~75 % and keeping
    Claude's attention tight on the designated fields.
    """
    structured_llm = llm.with_structured_output(NI43101Report)
    partials: List[NI43101Report] = []

    for pass_def in _EXTRACTION_PASSES:
        context = _gather_context_for_topics(
            settings,
            vectorstore,
            filename,
            pass_def.get("topics", []),
            items=pass_def.get("items"),
        )
        if not context.strip():
            continue
        messages = [
            SystemMessage(content=_EXTRACTION_INSTRUCTION),
            HumanMessage(
                content=(
                    f"FOCUS FOR THIS PASS: {pass_def['focus']}\n\n"
                    "Extract NI 43-101 fields from the context below. "
                    f"Set source_file to '{filename}'.\n\n"
                    f"{json.dumps({'source_file': filename, 'context': context}, ensure_ascii=True, indent=2)}"
                )
            ),
        ]
        partial = _invoke_with_backoff(structured_llm, messages)
        partials.append(partial)

    if not partials:
        return NI43101Report(source_file=filename)

    return _merge_partials(filename, *partials)


_EXTRACT_WORKERS = 5  # concurrent reports; tune down if hitting Bedrock TPM limits


def extract_all(
    settings: Settings,
    vectorstore: Chroma,
    llm: ChatBedrockConverse,
    skip_existing: bool = True,
) -> List[Tuple[str, NI43101Report]]:
    """Extract structured data for every PDF in the knowledge directory.

    Reports are processed concurrently (_EXTRACT_WORKERS at a time).  Each
    worker runs the four-pass extraction independently; real rate-limit errors
    are caught by _invoke_with_backoff and retried with exponential back-off,
    so no pre-emptive sleep is needed.

    Args:
        skip_existing: When True (default), PDFs whose extracted JSON already
            exists in ``settings.extracted_dir`` are skipped.  Pass False to
            force a full re-extraction (e.g. after a schema upgrade).
    """
    try:
        pdf_paths = list(iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs))
    except FileNotFoundError:
        pdf_paths = []

    to_process: List[Tuple[Path, str]] = []
    for pdf_path in pdf_paths:
        source_file = pdf_source_id(
            pdf_path,
            settings.knowledge_dir,
            settings.extra_pdf_dirs,
        )
        if skip_existing:
            out_file = source_output_path(
                settings.extracted_dir,
                source_file,
                ".json",
            )
            if out_file.exists():
                tqdm.write(f"  ↷ {pdf_path.name} (already extracted, skipping)")
                continue
        to_process.append((pdf_path, source_file))

    results: List[Tuple[str, NI43101Report]] = []
    bar = tqdm(total=len(to_process), desc="Extracting", unit="report")

    def _worker(item: Tuple[Path, str]) -> Tuple[str, NI43101Report]:
        pdf_path, source_file = item
        tqdm.write(f"  → {pdf_path.name}")
        report = extract_report(settings, vectorstore, llm, source_file)
        return source_file, report

    with ThreadPoolExecutor(max_workers=_EXTRACT_WORKERS) as pool:
        futures = {pool.submit(_worker, item): item for item in to_process}
        for future in as_completed(futures):
            try:
                name, report = future.result()
                results.append((name, report))
            except Exception as exc:
                pdf_path, _ = futures[future]
                tqdm.write(f"  ✗ {pdf_path.name} failed: {exc}")
            finally:
                bar.update(1)

    bar.close()
    return results


def load_extraction(settings: Settings, filename: str) -> Optional[dict]:
    """Load a previously saved extraction JSON by source filename."""
    path = source_output_path(settings.extracted_dir, filename, ".json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_extractions(settings: Settings) -> List[dict]:
    """Return all saved extraction JSON documents."""
    if not settings.extracted_dir.exists():
        return []
    out: List[dict] = []
    for path in sorted(settings.extracted_dir.rglob("*.json")):
        if path.name.endswith("_chapters.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                out.append(payload)
        except (json.JSONDecodeError, OSError):
            continue
    return out


# ---------------------------------------------------------------------------
# Spatial / geological-model extraction (SpatialExtraction, spatial_data/)
# ---------------------------------------------------------------------------
# Guarded import: spatial_schemas.py is optional. The standard NI43101
# extraction pipeline works fine without it; only extract_spatial() will
# fail (with a clear ImportError) if the file is absent.
try:
    from spatial_schemas import SpatialExtraction
except ImportError:
    SpatialExtraction = None  # type: ignore[assignment,misc]
# Separate from the 5-pass NI43101Report pipeline: the output is its own
# top-level object with per-record source provenance and a human review gate.
#
# Retrieval deliberately does NOT scope the drilling queries to NI Items:
# collar/survey/lithology tables usually live in unnumbered appendices, which
# chapter_index tags as ni_item=27 (the last detected chapter's page range is
# open-ended) — an Item-scoped query would be blind to them. Instead each
# query runs twice — once restricted to table chunks (type="table"), once
# unrestricted — and the results are deduplicated.

_SPATIAL_INSTRUCTION = (
    "You are a meticulous mining geologist extracting SPATIAL / GEOLOGICAL-MODEL data "
    "from an NI 43-101 technical report, for 3D geological model reconstruction. Use "
    "ONLY the provided context excerpts. Rules:\n"
    "1. Never invent or estimate values. A field absent from the context is null (or an "
    "empty list). A wrong-but-plausible coordinate is worse than a missing one.\n"
    "2. Every record MUST have a `source` citing where it came from — table number and "
    "page where visible (e.g. 'Table 10-1, page 87').\n"
    "3. Collar tables: one Borehole per data row. Record x, y, z_collar, azimuth, dip and "
    "total_depth_m exactly as printed — strip thousands separators, do not flip dip "
    "signs, do not convert between grids.\n"
    "4. Downhole surveys: attach SurveyPoint rows to the matching hole_id. If no survey "
    "data exists for a hole, leave survey_points empty — never assume values.\n"
    "5. Lithology logs: one LithologyInterval per logged interval row. If the report has "
    "no lithology log tables (common — logs stay in company databases), capture mineralized "
    "intercept tables (hole, from, to, zone/unit) as LithologyInterval rows instead: use the "
    "zone/unit/rock name as unit_name and cite the intercept table in source. Capture EVERY "
    "row present in the context, including continuation pages of the same table. When the "
    "deposit/zone a table belongs to is stated (table title or nearby text), qualify "
    "unit_name with it — e.g. 'Porphyry (Mosquito Hill)'. When the zone is NOT "
    "identifiable, still capture every row with the plain unit name — never skip a row "
    "because its zone is unclear.\n"
    "6. Stratigraphic pile: order_top_down 1 = youngest / structurally topmost. Only "
    "assign an ordering the report states or clearly implies (e.g. a stratigraphic "
    "column or 'X overlies Y' statements).\n"
    "7. Orientations: extract dip and dip_direction only where explicitly stated. If "
    "only strike is given, apply the right-hand rule (dip direction = strike + 90 "
    "degrees) ONLY when the dip side is unambiguous, and quote the original strike "
    "text in `source`.\n"
    "8. coordinate_system: record the grid exactly as stated (UTM zone and datum, or "
    "'local mine grid'). If different tables use different grids, say so in `notes`.\n"
    "9. If a table has more rows than you can output, extract the first 200 rows and "
    "state in `notes` how many rows were omitted and from which table.\n"
    "10. Never set `confirmed` and never populate cross_section_points — both belong to "
    "the human review / digitizing workflow, not to extraction."
)

_SPATIAL_PASSES: List[dict] = [
    {
        "name": "drilling",
        "queries": [
            "drill hole collar coordinates easting northing elevation azimuth dip depth table",
            "collar location table UTM coordinates drill hole",
            "downhole survey deviation azimuth dip measured depth",
            "lithology log drill hole interval from to rock unit",
            "significant intercepts drill hole from to interval width grade zone table",
        ],
        "filter_types": ["table"],
        "filter_items": None,
        "focus": (
            "Extract ONLY: coordinate_system, boreholes (with survey_points where a "
            "downhole survey table exists), and lithology_intervals. "
            "Leave stratigraphic_pile, orientations, and faults empty."
        ),
    },
    {
        "name": "structure",
        "queries": [
            "stratigraphy stratigraphic column formation member sequence youngest oldest overlies",
            "structural geology bedding foliation dip strike dip direction measurements",
            "fault shear zone displacement offset orientation",
        ],
        "filter_types": None,
        "filter_items": [7, 8],
        "focus": (
            "Extract ONLY: coordinate_system, stratigraphic_pile (in age order), "
            "orientations, and faults. Leave boreholes and lithology_intervals empty."
        ),
    },
]

_SPATIAL_MAX_CONTEXT_CHARS = 100_000


def _gather_spatial_context(
    settings: Settings,
    vectorstore: Chroma,
    filename: str,
    pass_def: dict,
) -> str:
    """Retrieve context for one spatial pass.

    Each query runs a filtered variant (table-only or Item-scoped, per the pass)
    plus an unrestricted variant, deduplicated by (source, page, chunk) — the
    filter buys precision, the unrestricted run catches content the filter would
    miss (collar data in prose, structure data in appendices tagged Item 27).
    """
    per_query_k = settings.extract_top_k
    seen: set = set()
    blocks: List[str] = []

    for q in pass_def["queries"]:
        variants: List[dict] = []
        if pass_def.get("filter_types"):
            variants.append({"filter_types": pass_def["filter_types"]})
        if pass_def.get("filter_items"):
            variants.append({"filter_items": pass_def["filter_items"]})
        variants.append({})

        query_parts: List[str] = []
        for kwargs in variants:
            parts, metadatas = query_chunks(
                vectorstore, q, per_query_k, filter_sources=[filename], **kwargs
            )
            for part, meta in zip(parts, metadatas):
                key = (meta.get("source"), meta.get("page"), meta.get("chunk"))
                if key in seen:
                    continue
                seen.add(key)
                query_parts.append(part)
        if query_parts:
            blocks.append(f"### Context for: {q}\n" + "\n\n".join(query_parts))

    context = "\n\n".join(blocks)
    if len(context) > _SPATIAL_MAX_CONTEXT_CHARS:
        tqdm.write(
            f"  Spatial pass '{pass_def['name']}': context truncated "
            f"{len(context)} -> {_SPATIAL_MAX_CONTEXT_CHARS} chars"
        )
        context = context[:_SPATIAL_MAX_CONTEXT_CHARS] + "\n… [context truncated]"
    return context


def _merge_spatial(source_file: str, *partials: SpatialExtraction) -> SpatialExtraction:
    """Merge spatial pass results: lists concatenate, scalars first-non-empty.

    Unlike _merge_partials, lists are concatenated rather than first-wins —
    both passes may legitimately contribute to the same list (e.g. orientations
    mentioned alongside a collar table). `confirmed` always resets to False:
    a fresh extraction invalidates any prior human review.
    """
    merged: dict = {"source_file": source_file, "confirmed": False}
    notes: List[str] = []
    for partial in partials:
        for field_name in SpatialExtraction.model_fields:
            if field_name in ("source_file", "confirmed"):
                continue
            val = getattr(partial, field_name, None)
            if val is None or (isinstance(val, list) and not val):
                continue
            if field_name == "notes":
                if val not in notes:
                    notes.append(val)
            elif isinstance(val, list):
                merged[field_name] = merged.get(field_name, []) + val
            elif field_name not in merged:
                merged[field_name] = val
    if notes:
        merged["notes"] = " | ".join(notes)
    return SpatialExtraction(**merged)


def extract_spatial(
    settings: Settings,
    vectorstore: Chroma,
    llm: ChatBedrockConverse,
    filename: str,
) -> SpatialExtraction:
    """Extract spatial/geological-model data via two focused passes.

    Pass the LLM in with a raised max_tokens (see get_chat_model): a collar
    table with hundreds of holes cannot fit its structured output in the
    4096-token chat default.
    """
    structured_llm = llm.with_structured_output(SpatialExtraction)
    partials: List[SpatialExtraction] = []

    for pass_def in _SPATIAL_PASSES:
        context = _gather_spatial_context(settings, vectorstore, filename, pass_def)
        if not context.strip():
            continue
        messages = [
            SystemMessage(content=_SPATIAL_INSTRUCTION),
            HumanMessage(
                content=(
                    f"FOCUS FOR THIS PASS: {pass_def['focus']}\n\n"
                    "Extract spatial geological-model data from the context below. "
                    f"Set source_file to '{filename}'.\n\n"
                    f"{json.dumps({'source_file': filename, 'context': context}, ensure_ascii=True, indent=2)}"
                )
            ),
        ]
        partial = _invoke_with_backoff(structured_llm, messages)
        partials.append(partial)

    if not partials:
        return SpatialExtraction(source_file=filename)

    return _merge_spatial(filename, *partials)


def save_spatial_extraction(settings: Settings, extraction: SpatialExtraction) -> Path:
    """Persist a SpatialExtraction to spatial_data/{stem}.json.

    Overwrites unconditionally — including any `confirmed` flag or digitized
    cross_section_points in the old file. Callers that must not clobber
    reviewed data should check for the file first (run_extract skips existing).
    """
    source_file = extraction.source_file or "report"
    out_path = source_output_path(settings.spatial_dir, source_file, ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_spatial_extraction(settings: Settings, filename: str) -> Optional[dict]:
    """Load a previously saved spatial extraction JSON by source filename."""
    path = source_output_path(settings.spatial_dir, filename, ".json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
