from ingestion.context import annotate_hierarchy, build_visual_context, needs_table_validation
from ingestion.models import ElementRecord


def _el(**kwargs):
    defaults = dict(
        element_id="e1",
        source_file="r.pdf",
        category="NarrativeText",
        text="",
        page_number=1,
    )
    defaults.update(kwargs)
    return ElementRecord(**defaults)


def test_ni_item_detection_from_title():
    elements = [
        _el(element_id="t1", category="Title", text="Item 14 — Mineral Resource Estimates", page_number=10),
        _el(element_id="n1", category="NarrativeText", text="Resources are reported...", page_number=11),
        _el(element_id="t2", category="Title", text="Item 16 - Recovery Methods", page_number=20),
        _el(element_id="n2", category="NarrativeText", text="Flotation recoveries...", page_number=21),
    ]
    out = annotate_hierarchy(elements)
    assert out[1].ni_item == 14
    assert "Mineral Resource" in out[1].section_title
    assert out[3].ni_item == 16


def test_caption_association():
    elements = [
        _el(element_id="img", category="Image", text="", page_number=5, image_path="x.png"),
        _el(element_id="cap", category="Caption", text="Figure 5-1 Recovery vs grind size", page_number=5),
        _el(element_id="n", category="NarrativeText", text="The figure shows...", page_number=5),
    ]
    out = annotate_hierarchy(elements)
    assert "Figure 5-1" in out[0].caption
    assert "figure shows" in out[0].following_text.lower()


def test_build_visual_context():
    el = _el(
        category="Image",
        ni_item=16,
        section_title="Recovery Methods",
        section_path=["Item 16", "Recovery Methods"],
        caption="Fig 1",
        preceding_text="Before",
        following_text="After",
        page_number=187,
    )
    ctx = build_visual_context(el)
    assert ctx.ni_item == 16
    assert ctx.page_number == 187
    assert ctx.caption == "Fig 1"


def test_visual_context_includes_leading_and_trailing_explicit_references():
    elements = [
        _el(
            element_id="leading-reference",
            text=(
                "As shown in Figure 16-27, the mine layout includes the "
                "tailings storage facility and process plant."
            ),
            page_number=313,
        ),
        *[
            _el(
                element_id=f"before-{index}",
                text=f"Unrelated preceding paragraph {index}.",
                page_number=313,
            )
            for index in range(8)
        ],
        _el(
            element_id="figure",
            category="Image",
            image_path="figure.png",
            page_number=314,
        ),
        _el(
            element_id="caption",
            category="Caption",
            text="Figure 16-27: General mine layout.",
            page_number=314,
        ),
        *[
            _el(
                element_id=f"after-{index}",
                text=f"Unrelated following paragraph {index}.",
                page_number=315,
            )
            for index in range(8)
        ],
        _el(
            element_id="trailing-reference",
            text=(
                "Figure 16 27 also identifies the ROM pad, camp, and north "
                "and east waste dumps."
            ),
            page_number=315,
        ),
    ]

    annotated = annotate_hierarchy(elements)
    context = build_visual_context(
        next(element for element in annotated if element.element_id == "figure")
    )

    assert context.caption == "Figure 16-27: General mine layout."
    assert context.figure_references == [
        (
            "As shown in Figure 16-27, the mine layout includes the tailings "
            "storage facility and process plant."
        ),
        (
            "Figure 16 27 also identifies the ROM pad, camp, and north and "
            "east waste dumps."
        ),
    ]


def test_needs_table_validation_for_resource_tables():
    el = _el(
        category="Table",
        text="Measured Indicated tonnage",
        section_title="Mineral Resource Estimates",
        text_as_html="<table><tr><td>1</td></tr></table>",
    )
    assert needs_table_validation(el) is True
    plain = _el(category="NarrativeText", text="hello")
    assert needs_table_validation(plain) is False
