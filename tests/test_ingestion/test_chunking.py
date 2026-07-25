from ingestion.chunking import elements_to_documents
from ingestion.models import ElementRecord, VisualAnalysis, ChartSpecification


def _el(**kwargs):
    defaults = dict(
        element_id="e1",
        source_file="report.pdf",
        category="NarrativeText",
        text="Body text about resources.",
        page_number=10,
        ni_item=14,
        section_title="Mineral Resource Estimates",
        parser="docling",
    )
    defaults.update(kwargs)
    return ElementRecord(**defaults)


def test_text_chunk_metadata():
    docs = elements_to_documents([_el()], chunk_size=1400, chunk_overlap=0)
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["source"] == "report.pdf"
    assert meta["page"] == 10
    assert meta["type"] == "text"
    assert meta["ni_item"] == 14
    assert meta["parser"] == "docling"
    assert "Item 14" in docs[0].page_content


def test_table_and_figure_chunks():
    table = _el(
        element_id="t1",
        category="Table",
        text="A | B",
        text_as_html="<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
        caption="Resource table",
    )
    figure = _el(
        element_id="f1",
        category="Image",
        text="",
        page_number=187,
        ni_item=16,
        section_title="Recovery Methods",
        image_path="artifacts/f1.png",
        caption="Figure 1",
    )
    analysis = VisualAnalysis(
        figure_type="line_chart",
        description="Recovery curve",
        confidence=0.94,
        chart=ChartSpecification(
            chart_type="line",
            series=[{"name": "rec", "points": [{"x": 1, "y": 90}]}],
        ),
    )
    docs = elements_to_documents(
        [table, figure],
        analyses={"f1": analysis},
        reconstructions={
            "f1": {"reconstruction_allowed": True, "reason": "chart"}
        },
        chunk_size=1400,
        chunk_overlap=0,
    )
    types = {d.metadata["type"] for d in docs}
    assert "table" in types
    assert "chart_data" in types
    fig = next(d for d in docs if d.metadata["type"] == "chart_data")
    assert fig.metadata["figure_type"] == "line_chart"
    assert fig.metadata["element_id"] == "f1"


def test_unvalidated_table_normalization_is_not_indexed():
    from ingestion.models import TableValidation

    table = _el(
        element_id="t1",
        category="Table",
        text="Authoritative raw value: 10",
    )
    validation = TableValidation(
        is_valid=False,
        confidence=0.99,
        normalized_markdown="Invented value: 999",
        issues=["crop_disagrees"],
    )
    docs = elements_to_documents(
        [table],
        validations={"t1": validation},
    )
    assert "Authoritative raw value: 10" in docs[0].page_content
    assert "Invented value: 999" not in docs[0].page_content
    assert docs[0].metadata["table_valid"] is False


def test_failed_visual_keeps_surrounding_context_searchable():
    figure = _el(
        element_id="f2",
        category="Image",
        text="",
        caption="Figure 2",
        preceding_text="Recovery improves with finer grinding.",
        following_text="The testwork is preliminary.",
        skip_reason="enrichment_failed",
    )
    docs = elements_to_documents([figure])
    assert "Recovery improves" in docs[0].page_content
    assert "testwork is preliminary" in docs[0].page_content
    assert docs[0].metadata["enrichment_status"] == "enrichment_failed"


def test_single_long_paragraph_is_hard_bounded():
    docs = elements_to_documents(
        [_el(text="resource " * 1000)],
        chunk_size=300,
        chunk_overlap=30,
    )

    assert len(docs) > 1
    assert max(len(doc.page_content) for doc in docs) <= 300


def test_inline_base64_picture_is_not_embedded_as_text():
    figure = _el(
        element_id="base64-picture",
        category="Image",
        text=(
            "Figure 1: Site plan\n\n"
            "![Image](data:image/png;base64,"
            + ("A" * 100_000)
            + ")"
        ),
        image_path="artifacts/base64-picture.png",
    )

    docs = elements_to_documents(
        [figure],
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(docs) == 1
    assert "base64" not in docs[0].page_content
    assert "Figure 1: Site plan" in docs[0].page_content
    assert len(docs[0].page_content) <= 500


def test_large_table_is_split_into_bounded_parts():
    table = _el(
        element_id="large-table",
        category="Table",
        text="| value | grade |\n" * 300,
    )

    docs = elements_to_documents(
        [table],
        chunk_size=400,
        chunk_overlap=40,
    )

    assert len(docs) > 1
    assert max(len(doc.page_content) for doc in docs) <= 400
    assert docs[0].metadata["parts"] == len(docs)
