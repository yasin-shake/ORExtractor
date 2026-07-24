import base64

from ingestion.normalizer import normalize_category, stable_element_id, normalize_elements


class _FakeMeta:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeEl:
    def __init__(self, category, text="", **meta):
        self.category = category
        self.text = text
        self.metadata = _FakeMeta(**meta) if meta else _FakeMeta()


def test_normalize_category_aliases():
    assert normalize_category("NarrativeText") == "NarrativeText"
    assert normalize_category("narrative_text") == "NarrativeText"
    assert normalize_category("FigureCaption") == "Caption"
    assert normalize_category("Image") == "Image"


def test_stable_element_id_is_deterministic():
    a = stable_element_id("r.pdf", 3, "Table", 1, "abc")
    b = stable_element_id("r.pdf", 3, "Table", 1, "abc")
    c = stable_element_id("r.pdf", 3, "Table", 2, "abc")
    assert a == b
    assert a != c


def test_normalize_elements_persists_table_html(tmp_path):
    el = _FakeEl("Table", text="A B", page_number=2, text_as_html="<table><tr><td>A</td></tr></table>")
    records = normalize_elements([el], "report.pdf", tmp_path)
    assert len(records) == 1
    assert records[0].category == "Table"
    assert records[0].page_number == 2
    assert (tmp_path / "tables").exists()
    html_files = list((tmp_path / "tables").glob("*.html"))
    assert len(html_files) == 1


def test_duplicate_images_are_detected_by_content(tmp_path):
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )
    encoded = base64.b64encode(png).decode("ascii")
    elements = [
        _FakeEl("Image", image_base64=encoded, page_number=1),
        _FakeEl("Image", image_base64=encoded, page_number=2),
    ]
    records = normalize_elements(elements, "report.pdf", tmp_path)
    assert records[0].is_duplicate is False
    assert records[1].is_duplicate is True
    assert records[1].skip_reason == "duplicate_image"
