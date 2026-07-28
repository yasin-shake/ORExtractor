from PIL import Image

from ingestion.models import ElementRecord
from ingestion.normalizers.docling import normalize_docling_document
from ingestion.visual_filtering import filter_visual_artifacts


def _image_element(tmp_path, *, element_id, page, bbox, color="white"):
    image_path = tmp_path / f"{element_id}.png"
    Image.new("RGB", (320, 180), color).save(image_path)
    return ElementRecord(
        element_id=element_id,
        source_file="report.pdf",
        category="Image",
        page_number=page,
        coordinates={**bbox, "coord_origin": "BOTTOMLEFT"},
        image_path=str(image_path),
        image_width=320,
        image_height=180,
        metadata={"page_width": 600.0, "page_height": 800.0},
    )


def test_filter_visual_artifacts_discards_page_margin_images(tmp_path):
    header = _image_element(
        tmp_path,
        element_id="header",
        page=2,
        bbox={"l": 70.0, "t": 790.0, "r": 130.0, "b": 760.0},
    )
    content = _image_element(
        tmp_path,
        element_id="content",
        page=2,
        bbox={"l": 70.0, "t": 650.0, "r": 530.0, "b": 180.0},
        color="blue",
    )

    filtered = filter_visual_artifacts([header, content])

    assert [element.element_id for element in filtered] == ["content"]
    assert not (tmp_path / "header.png").exists()
    assert (tmp_path / "content.png").exists()


def test_filter_visual_artifacts_deduplicates_body_files_but_keeps_occurrences(
    tmp_path,
):
    first = _image_element(
        tmp_path,
        element_id="first",
        page=10,
        bbox={"l": 70.0, "t": 650.0, "r": 530.0, "b": 180.0},
        color="green",
    )
    duplicate = _image_element(
        tmp_path,
        element_id="duplicate",
        page=25,
        bbox={"l": 70.0, "t": 650.0, "r": 530.0, "b": 180.0},
        color="green",
    )
    duplicate.preceding_text = "This occurrence has different report context."

    filtered = filter_visual_artifacts([first, duplicate])

    assert [element.element_id for element in filtered] == [
        "first",
        "duplicate",
    ]
    assert first.is_duplicate is False
    assert duplicate.is_duplicate is True
    assert duplicate.skip_reason == "duplicate_visual"
    assert duplicate.metadata["duplicate_of_element_id"] == "first"
    assert duplicate.image_path == first.image_path
    assert (tmp_path / "first.png").exists()
    assert not (tmp_path / "duplicate.png").exists()
    assert "different report context" in duplicate.preceding_text


def test_filter_visual_artifacts_deduplicates_visually_identical_encodings(
    tmp_path,
):
    first = _image_element(
        tmp_path,
        element_id="png",
        page=10,
        bbox={"l": 70.0, "t": 650.0, "r": 530.0, "b": 180.0},
        color="purple",
    )
    bitmap_path = tmp_path / "bitmap.bmp"
    Image.new("RGB", (320, 180), "purple").save(bitmap_path)
    encoded_differently = ElementRecord(
        **{
            **first.model_dump(),
            "element_id": "bitmap",
            "page_number": 11,
            "image_path": str(bitmap_path),
        }
    )

    filtered = filter_visual_artifacts([first, encoded_differently])

    assert len(filtered) == 2
    assert encoded_differently.is_duplicate is True
    assert encoded_differently.image_path == first.image_path
    assert not bitmap_path.exists()


def test_filter_visual_artifacts_uses_docling_page_geometry(tmp_path):
    class BoundingBox:
        l = 70.0
        t = 790.0
        r = 130.0
        b = 760.0
        coord_origin = "BOTTOMLEFT"

    class Provenance:
        page_no = 2
        bbox = BoundingBox()

    class Picture:
        label = "picture"
        text = ""
        self_ref = "#/pictures/0"
        parent = None
        prov = [Provenance()]

        @staticmethod
        def get_image(_document):
            return Image.new("RGB", (60, 30), "orange")

    class Size:
        width = 600.0
        height = 800.0

    class Page:
        size = Size()

    class Document:
        pages = {2: Page()}

        @staticmethod
        def iterate_items():
            yield Picture(), 0

    normalized = normalize_docling_document(
        Document(),
        source_file="report.pdf",
        artifact_dir=tmp_path,
        parser_version="2.test",
    )

    assert filter_visual_artifacts(normalized) == []
    assert list((tmp_path / "images").glob("*.png")) == []
