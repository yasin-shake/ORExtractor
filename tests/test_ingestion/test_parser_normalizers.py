from dataclasses import dataclass

from ingestion.normalizers.docling import normalize_docling_document
from ingestion.normalizers.mineru import normalize_mineru_content


@dataclass
class _Prov:
    page_no: int = 2
    bbox: object | None = None


class _Item:
    label = "section_header"
    text = "Item 14 - Mineral Resource Estimates"
    self_ref = "#/texts/0"
    parent = None
    prov = [_Prov()]


class _Document:
    def iterate_items(self):
        yield _Item(), 1


def test_docling_normalizer_preserves_provenance(tmp_path):
    records = normalize_docling_document(
        _Document(),
        source_file="report.pdf",
        artifact_dir=tmp_path,
        parser_version="2.test",
    )
    assert len(records) == 1
    assert records[0].category == "Title"
    assert records[0].page_number == 2
    assert records[0].parser == "docling"
    assert records[0].parser_element_id == "#/texts/0"


def test_mineru_normalizer_uses_zero_based_page_index(tmp_path):
    records = normalize_mineru_content(
        [
            {
                "type": "table",
                "page_idx": 2,
                "table_body": "| Grade | Tonnes |",
                "bbox": [1, 2, 3, 4],
            }
        ],
        source_file="report.pdf",
        artifact_dir=tmp_path,
        parser_version="3.test",
    )
    assert records[0].category == "Table"
    assert records[0].page_number == 3
    assert records[0].text_as_markdown == "| Grade | Tonnes |"
    assert records[0].parser == "mineru"
