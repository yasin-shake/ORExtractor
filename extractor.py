"""Structured extraction of NI 43-101 data from ingested reports.

Extraction runs as four focused passes, each targeting a distinct subset of the
schema with a proportionally smaller context window:

1. **Identity** (executive summary, front matter, property) → report metadata,
   property info, qualified persons, project summary.
2. **Resources & Reserves** (resources, reserves topics) → all resource and
   reserve table rows, one MineralResource/MineralReserve per line item.
3. **Economics & Technical** (economics, mining, processing topics) → NPV/IRR/
   CapEx, mining method, processing method.
4. **Geology & Environment** (geology, exploration, environmental topics) →
   deposit type, drilling stats, permitting, indigenous consultation, risk flags.

Each pass uses ``with_structured_output`` bound to the full ``NI43101Report``
schema but is instructed to populate only its designated fields; the four partial
results are merged field-by-field (first non-empty value wins) into the final
report object.  Splitting the context reduces per-call token usage by ~75 %,
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

from rag_app import Settings, iter_pdf_paths, query_chunks
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
            if is_rate_limit and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)  # 60 → 120 → 240 → …
                print(
                    f"  Rate limit hit — waiting {wait}s before retry "
                    f"{attempt + 1}/{max_retries - 1} …",
                    flush=True,
                )
                time.sleep(wait)
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
    "15. Metadata tags — populate study_stage, deposit_type, mining_method, processing_route, "
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
            "Extract ONLY portfolio metadata tags: study_stage, deposit_type, mining_method, "
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

    if items:
        for item_num in items:
            queries = _ITEM_QUERIES.get(item_num, [])
            item_parts: List[str] = []
            for q in queries:
                parts, metadatas = query_chunks(
                    vectorstore, q, per_query_k, filter_sources=[filename]
                )
                for part, meta in zip(parts, metadatas):
                    if meta.get("ni_item") and int(meta.get("ni_item") or 0) not in (0, item_num):
                        continue
                    key = (meta.get("source"), meta.get("page"), meta.get("chunk"))
                    if key in seen_chunks:
                        continue
                    seen_chunks.add(key)
                    item_parts.append(part)
            if item_parts:
                blocks.append(f"### Context for: Item {item_num}\n" + "\n\n".join(item_parts))
        return "\n\n".join(blocks)

    for topic in topics:
        queries = _TOPIC_QUERIES.get(topic, [])
        topic_parts: List[str] = []
        for q in queries:
            parts, metadatas = query_chunks(
                vectorstore, q, per_query_k, filter_sources=[filename]
            )
            for part, meta in zip(parts, metadatas):
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

    to_process: List = []
    for pdf_path in pdf_paths:
        if skip_existing:
            out_file = settings.extracted_dir / f"{pdf_path.stem}.json"
            if out_file.exists():
                tqdm.write(f"  ↷ {pdf_path.name} (already extracted, skipping)")
                continue
        to_process.append(pdf_path)

    results: List[Tuple[str, NI43101Report]] = []
    bar = tqdm(total=len(to_process), desc="Extracting", unit="report")

    def _worker(pdf_path: Path) -> Tuple[str, NI43101Report]:
        tqdm.write(f"  → {pdf_path.name}")
        report = extract_report(settings, vectorstore, llm, pdf_path.name)
        return pdf_path.name, report

    with ThreadPoolExecutor(max_workers=_EXTRACT_WORKERS) as pool:
        futures = {pool.submit(_worker, p): p for p in to_process}
        for future in as_completed(futures):
            try:
                name, report = future.result()
                results.append((name, report))
            except Exception as exc:
                pdf_path = futures[future]
                tqdm.write(f"  ✗ {pdf_path.name} failed: {exc}")
            finally:
                bar.update(1)

    bar.close()
    return results


def load_extraction(settings: Settings, filename: str) -> Optional[dict]:
    """Load a previously saved extraction JSON by source filename."""
    stem = Path(filename).stem
    path = settings.extracted_dir / f"{stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_extractions(settings: Settings) -> List[dict]:
    """Return all saved extraction JSON documents."""
    if not settings.extracted_dir.exists():
        return []
    out: List[dict] = []
    for path in sorted(settings.extracted_dir.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out
