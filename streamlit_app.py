import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import streamlit as st
from langchain_core.messages import HumanMessage

from agent_chat import ChatTurn, agent_chat_enabled, run_agent_chat
from extractor import extract_report, list_extractions
from rag_app import (
    _is_short_greeting_or_thanks,
    build_chat_prompt,
    get_chat_model,
    get_embedder,
    get_vectorstore,
    iter_pdf_paths,
    load_settings,
    pdf_source_id,
    query_context,
    save_extraction,
)

_CSS = """
<style>
/* ── Metric cards ─────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,200,0.2);
    border-radius: 10px;
    padding: 14px 16px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.65;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
    font-weight: 700;
}

/* ── Gallery thumbnails ───────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 2px solid transparent;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
[data-testid="stImage"] img:hover {
    border-color: #7c9ff8;
    box-shadow: 0 4px 16px rgba(124,159,248,0.3);
    transform: scale(1.01);
}

/* ── Buttons ──────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    border-radius: 8px;
    font-weight: 600;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.18);
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px);
}

/* ── Expanders ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-radius: 8px !important;
}

/* ── Source badge ─────────────────────────────────────────────────── */
.src-badge {
    display: inline-block;
    background: rgba(124,159,248,0.1);
    border: 1px solid rgba(124,159,248,0.28);
    border-radius: 6px;
    padding: 3px 10px;
    margin: 2px 3px;
    font-size: 0.77rem;
    font-family: monospace;
    white-space: nowrap;
}

/* ── Slide counter ────────────────────────────────────────────────── */
.slide-count {
    text-align: center;
    font-size: 0.85rem;
    opacity: 0.65;
    padding: 8px 0;
}
</style>
"""


def _conversation_pairs(messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    current_user = None
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            current_user = msg.get("content", "")
        elif role == "assistant" and current_user is not None:
            pairs.append((current_user, msg.get("content", "")))
            current_user = None
    return pairs


@st.cache_resource(show_spinner=False)
def _load_runtime():
    settings = load_settings()
    embedder = get_embedder(settings)
    vectorstore = get_vectorstore(settings, embedder)
    llm = get_chat_model(settings)
    return settings, vectorstore, llm


@st.cache_data(show_spinner=False)
def _render_pdf_page(knowledge_dirs: tuple, source_file: str, page_number: int, zoom: float = 1.7) -> bytes:
    pdf_path = next(
        (Path(d) / source_file for d in knowledge_dirs if (Path(d) / source_file).exists()),
        None,
    )
    if pdf_path is None or page_number < 1:
        return b""
    with fitz.open(str(pdf_path)) as doc:
        if page_number > len(doc):
            return b""
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")


def _source_items(metadatas: List[dict]) -> List[Tuple[str, int, int]]:
    seen = set()
    out: List[Tuple[str, int, int]] = []
    for metadata in metadatas:
        source = str(metadata.get("source") or metadata.get("file") or "unknown")
        page = metadata.get("page", -1)
        chunk = metadata.get("chunk", -1)
        if not isinstance(page, int):
            page = -1
        if not isinstance(chunk, int):
            chunk = -1
        key = (source, page, chunk)
        if key in seen:
            continue
        seen.add(key)
        out.append((source, page, chunk))
    return out


# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------

_SUGGESTIONS = [
    "Is the resource classification defensible given drill spacing?",
    "Are QAQC results acceptable and complete?",
    "Is the cut-off grade reasonable compared with peer reports?",
    "What are the key technical red flags?",
    "Should this project be Go, Conditional Go, Further Work, or No-Go?",
]


def render_ask_tab(settings, vectorstore, llm) -> None:
    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### PDF Sources")
        try:
            available_pdfs = [
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
        except FileNotFoundError:
            available_pdfs = []

        if available_pdfs:
            st.caption(f"{len(available_pdfs)} report{'s' if len(available_pdfs) != 1 else ''} indexed")
            selected_pdfs: List[str] = st.multiselect(
                "Filter reports:",
                options=available_pdfs,
                default=available_pdfs,
                help="Narrow the search to specific reports. Leave all selected to search everything.",
            )
            if not selected_pdfs:
                st.caption("⚠️ No filter — all documents searched.")
        else:
            st.info("No PDFs found in the knowledge directory.")
            selected_pdfs = []

        st.divider()

        if st.session_state.get("messages"):
            if st.button("🗑️ Clear chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.session_state.viewer_index = 0
                st.rerun()

    # ── Session state ───────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "viewer_index" not in st.session_state:
        st.session_state.viewer_index = 0

    # ── Welcome screen (shown when chat is empty) ───────────────────────
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center; padding:40px 20px 20px;">
                <div style="font-size:3rem;">⛏️</div>
                <h3 style="margin:8px 0 4px 0;">Ask anything about your NI 43-101 reports</h3>
                <p style="opacity:0.6; font-size:0.9rem;">
                    Resources &nbsp;·&nbsp; Reserves &nbsp;·&nbsp; Economics &nbsp;·&nbsp;
                    Geology &nbsp;·&nbsp; Exploration
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, suggestion in enumerate(_SUGGESTIONS):
            with cols[i % 2]:
                if st.button(suggestion, use_container_width=True, key=f"suggest_{i}"):
                    st.session_state["_pending_question"] = suggestion
                    st.rerun()

    # ── Chat history ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input ───────────────────────────────────────────────────────────
    user_question = st.chat_input("Ask a question about the NI 43-101 reports …")
    if user_question is None:
        user_question = st.session_state.pop("_pending_question", None)

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            if _is_short_greeting_or_thanks(user_question):
                answer = (
                    "I answer NI 43-101 due diligence questions using chapter-directed retrieval. "
                    "Ask about resources, QAQC, cut-off grades, economics, red flags, or peer benchmarks."
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                filter_sources: Optional[List[str]] = selected_pdfs if selected_pdfs else None
                history = [
                    ChatTurn(role=m["role"], content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                if agent_chat_enabled():
                    with st.spinner("Running chapter-directed agent …"):
                        result = run_agent_chat(
                            settings,
                            vectorstore,
                            llm,
                            user_question,
                            pdf_filter=filter_sources,
                            history=history,
                        )
                    answer = result.answer
                    st.markdown(answer)
                    if result.routed_items:
                        st.caption(
                            "Items searched: "
                            + ", ".join(f"Item {i}" for i in result.routed_items)
                        )
                    if result.assessment:
                        st.info(f"Assessment: {result.assessment}")
                    if result.flags:
                        with st.expander("Flags"):
                            for f in result.flags:
                                st.markdown(f"- {f}")
                    if result.peer_summary:
                        with st.expander("Peer benchmark"):
                            st.markdown(result.peer_summary)
                    metadatas = result.sources
                else:
                    with st.spinner("Retrieving relevant context …"):
                        context, metadatas = query_context(
                            vectorstore,
                            user_question,
                            settings.top_k,
                            filter_sources=filter_sources,
                        )
                    if not context:
                        answer = "I could not find relevant context in the indexed reports."
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        metadatas = []
                    else:
                        history_pairs = _conversation_pairs(st.session_state.messages[:-1])
                        prompt = build_chat_prompt(user_question, context, history=history_pairs)
                        answer = st.write_stream(
                            chunk.content
                            for chunk in llm.stream([HumanMessage(content=prompt)])
                        )
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    source_items = _source_items(metadatas)
                    st.session_state.last_sources = source_items
                    st.session_state.viewer_index = 0

                    if source_items:
                        n = len(source_items)
                        with st.expander(f"📄 {n} source{'s' if n != 1 else ''} cited", expanded=False):
                            badges = "".join(
                                f'<span class="src-badge">{src} · p.{page}</span>'
                                for src, page, _ in source_items
                                if page >= 1
                            )
                            st.markdown(badges, unsafe_allow_html=True)

    # ── Source gallery ──────────────────────────────────────────────────
    if st.session_state.last_sources:
        st.divider()
        gallery_items: List[Tuple[bytes, str]] = []
        for idx, (src, page, _chunk) in enumerate(st.session_state.last_sources[:8], start=1):
            if page < 1:
                continue
            all_dirs = tuple(str(d) for d in [settings.knowledge_dir] + list(settings.extra_pdf_dirs))
            img = _render_pdf_page(all_dirs, src, page)
            if not img:
                continue
            gallery_items.append((img, f"{idx}. {src} p.{page}"))

        if gallery_items:
            st.subheader("Source pages")
            n_cols = min(4, len(gallery_items))
            cols = st.columns(n_cols)
            for idx, (img, caption) in enumerate(gallery_items):
                with cols[idx % n_cols]:
                    st.image(img, caption=caption, use_container_width=True)
                    if st.button("↗ View", key=f"open_slide_{idx}", use_container_width=True):
                        st.session_state.viewer_index = idx
                        st.rerun()

            st.markdown("#### Slideshow")
            active_idx = st.session_state.viewer_index % len(gallery_items)
            active_img, active_caption = gallery_items[active_idx]
            st.image(active_img, caption=active_caption, use_container_width=True)

            prev_col, counter_col, next_col = st.columns([1, 2, 1])
            with prev_col:
                if st.button("← Previous", key="prev_slide", use_container_width=True):
                    st.session_state.viewer_index = (active_idx - 1) % len(gallery_items)
                    st.rerun()
            with counter_col:
                st.markdown(
                    f'<div class="slide-count">Slide {active_idx + 1} / {len(gallery_items)}</div>',
                    unsafe_allow_html=True,
                )
            with next_col:
                if st.button("Next →", key="next_slide", use_container_width=True):
                    st.session_state.viewer_index = (active_idx + 1) % len(gallery_items)
                    st.rerun()


# ---------------------------------------------------------------------------
# Reports tab
# ---------------------------------------------------------------------------


def _fmt(value, default: str = "—") -> str:
    if value is None or value == "" or value == []:
        return default
    return str(value)


def _render_property(info: Optional[dict]) -> None:
    if not info:
        st.caption("No property information extracted.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Project", _fmt(info.get("project_name")))
    c2.metric("Country", _fmt(info.get("country")))
    c3.metric("Region", _fmt(info.get("region")))
    d1, d2, d3 = st.columns(3)
    d1.metric("Commodities", _fmt(", ".join(info.get("commodities") or [])))
    d2.metric("Area (ha)", _fmt(info.get("area_hectares")))
    d3.metric("Tenure", _fmt(info.get("tenure_status")))
    if info.get("coordinates"):
        st.caption(f"Coordinates: {info['coordinates']}")
    if info.get("ownership"):
        st.caption(f"Ownership: {info['ownership']}")


def _render_estimate_table(title: str, rows: List[dict]) -> None:
    if not rows:
        st.caption(f"No {title.lower()} extracted.")
        return
    table = [
        {
            "Category": _fmt(r.get("category")),
            "Commodity": _fmt(r.get("commodity")),
            "Zone": _fmt(r.get("zone")),
            "Cut-off": _fmt(r.get("cut_off_grade")),
            "Tonnes": _fmt(r.get("tonnes")),
            "Grade": f"{_fmt(r.get('grade'))} {_fmt(r.get('grade_unit'), '')}".strip(),
            "Contained metal": f"{_fmt(r.get('contained_metal'))} {_fmt(r.get('contained_metal_unit'), '')}".strip(),
            "Effective date": _fmt(r.get("effective_date")),
        }
        for r in rows
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_economics(econ: Optional[dict]) -> None:
    if not econ:
        st.caption("No economic parameters extracted.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Study type", _fmt(econ.get("study_type")))
    c2.metric("Post-tax NPV", _fmt(econ.get("post_tax_npv")))
    c3.metric("IRR", _fmt(econ.get("irr")))
    c4, c5, c6 = st.columns(3)
    c4.metric("Initial CAPEX", _fmt(econ.get("initial_capex")))
    c5.metric("Payback (yrs)", _fmt(econ.get("payback_years")))
    c6.metric("Mine life (yrs)", _fmt(econ.get("mine_life_years")))
    extra = {
        "Pre-tax NPV": econ.get("pre_tax_npv"),
        "Discount rate": econ.get("discount_rate"),
        "Sustaining CAPEX": econ.get("sustaining_capex"),
        "OPEX": econ.get("opex"),
    }
    for label, val in extra.items():
        if val:
            st.caption(f"{label}: {val}")
    if econ.get("metal_price_assumptions"):
        st.caption(f"Metal prices: {', '.join(econ['metal_price_assumptions'])}")


def _render_geology(geo: Optional[dict]) -> None:
    if not geo:
        st.caption("No geology summary extracted.")
        return
    items = {
        "Deposit type": geo.get("deposit_type"),
        "Host rock": geo.get("host_rock"),
        "Mineralization style": geo.get("mineralization_style"),
        "Structural controls": geo.get("structural_controls"),
        "Alteration": geo.get("alteration"),
    }
    for label, val in items.items():
        if val:
            st.markdown(f"**{label}:** {val}")


def _render_exploration(expl: Optional[dict]) -> None:
    if not expl:
        st.caption("No exploration summary extracted.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Drill holes", _fmt(expl.get("total_drill_holes")))
    c2.metric("Metres drilled", _fmt(expl.get("total_metres_drilled")))
    c3.metric("Last program", _fmt(expl.get("last_program_date")))
    if expl.get("drilling_types"):
        st.caption(f"Drilling types: {', '.join(expl['drilling_types'])}")
    if expl.get("sampling_methods"):
        st.caption(f"Sampling: {expl['sampling_methods']}")
    intercepts = expl.get("notable_intercepts") or []
    if intercepts:
        with st.expander(f"Notable intercepts ({len(intercepts)})"):
            for i in intercepts:
                st.markdown(f"• {i}")


def render_reports_tab(settings, vectorstore, llm) -> None:
    try:
        available_pdfs = [
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
    except FileNotFoundError:
        available_pdfs = []

    extractions = list_extractions(settings)

    # ── Extraction controls ─────────────────────────────────────────────
    with st.expander("⚙️ Run extraction", expanded=not bool(extractions)):
        col_pick, col_btn = st.columns([3, 1])
        with col_pick:
            run_target = st.selectbox(
                "Report to extract",
                options=available_pdfs or ["(no PDFs found)"],
                disabled=not available_pdfs,
                label_visibility="collapsed",
            )
        with col_btn:
            run_clicked = st.button(
                "Extract",
                disabled=not available_pdfs,
                use_container_width=True,
                type="primary",
            )
        if run_clicked:
            with st.spinner(f"Extracting structured data from **{run_target}** …"):
                report = extract_report(settings, vectorstore, llm, run_target)
                save_extraction(settings, report)
            st.success(f"✅ Extracted {run_target}.")
            st.rerun()

    # ── Report viewer ───────────────────────────────────────────────────
    if not extractions:
        st.info("No extractions yet. Use the panel above to run your first extraction.")
        return

    extracted_names = [e.get("source_file") or "report" for e in extractions]
    chosen = st.selectbox("View report", options=extracted_names)
    data = next(
        (e for e in extractions if (e.get("source_file") or "report") == chosen),
        extractions[0],
    )

    # ── Report header ───────────────────────────────────────────────────
    st.markdown(f"## {_fmt(data.get('report_title'), chosen)}")
    h1, h2, h3 = st.columns(3)
    h1.metric("Issuer", _fmt(data.get("issuer")))
    h2.metric("Report date", _fmt(data.get("report_date")))
    h3.metric("Authors", _fmt(", ".join(data.get("authors") or [])))

    qps = data.get("qualified_persons") or []
    if qps:
        with st.expander(f"Qualified Persons ({len(qps)})"):
            for qp in qps:
                st.markdown(
                    f"**{_fmt(qp.get('name'))}** — {_fmt(qp.get('credentials'))} "
                    f"*({_fmt(qp.get('responsibility'))})*"
                )

    if data.get("summary"):
        with st.expander("Executive Summary", expanded=True):
            st.write(data["summary"])

    st.divider()

    # ── Section tabs ────────────────────────────────────────────────────
    prop_tab, res_tab, econ_tab, geo_tab, expl_tab, raw_tab = st.tabs([
        "🗺️ Property",
        "📊 Resources & Reserves",
        "💰 Economics",
        "🪨 Geology",
        "🔍 Exploration",
        "{ } Raw JSON",
    ])

    with prop_tab:
        _render_property(data.get("property_info"))

    with res_tab:
        st.markdown("##### Mineral Resources")
        _render_estimate_table("Mineral Resources", data.get("mineral_resources") or [])
        st.markdown("##### Mineral Reserves")
        _render_estimate_table("Mineral Reserves", data.get("mineral_reserves") or [])

    with econ_tab:
        _render_economics(data.get("economics"))

    with geo_tab:
        _render_geology(data.get("geology"))

    with expl_tab:
        _render_exploration(data.get("exploration"))

    with raw_tab:
        raw_json = json.dumps(data, indent=2, default=str)
        st.download_button(
            "⬇️ Download JSON",
            data=raw_json,
            file_name=f"{chosen}_extracted.json",
            mime="application/json",
        )
        st.json(data)


def main() -> None:
    st.set_page_config(
        page_title="NI 43-101 Knowledge Base",
        page_icon="⛏️",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("⛏️ NI 43-101 Knowledge Base")
    st.caption(
        "Ask questions across your indexed NI 43-101 reports and extract structured "
        "project data — resources, reserves, economics, geology."
    )

    try:
        settings, vectorstore, llm = _load_runtime()
    except Exception as exc:
        st.error(f"Startup failed: {exc}")
        st.stop()

    if vectorstore._collection.count() == 0:
        st.warning(
            "Vector index is empty. Run `python rag_app.py ingest` to index your PDFs first."
        )
        st.stop()

    ask_tab, reports_tab = st.tabs(["💬 Ask", "📋 Reports"])
    with ask_tab:
        render_ask_tab(settings, vectorstore, llm)
    with reports_tab:
        render_reports_tab(settings, vectorstore, llm)


if __name__ == "__main__":
    main()
