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
    assert meta["parser"] == "unstructured"
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
        chunk_size=1400,
        chunk_overlap=0,
    )
    types = {d.metadata["type"] for d in docs}
    assert "table" in types
    assert "chart_data" in types
    fig = next(d for d in docs if d.metadata["type"] == "chart_data")
    assert fig.metadata["figure_type"] == "line_chart"
    assert fig.metadata["element_id"] == "f1"
