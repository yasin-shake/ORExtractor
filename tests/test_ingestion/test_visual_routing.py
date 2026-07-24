from dataclasses import dataclass
from pathlib import Path

from ingestion.enrichment import should_enrich_figure
from ingestion.models import ElementRecord


@dataclass
class _Settings:
    visual_min_width: int = 250
    visual_min_height: int = 150


def test_visual_routing_skips_small_and_duplicate(tmp_path):
    img = tmp_path / "logo.png"
    # 1x1 PNG
    img.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )
    el = ElementRecord(
        element_id="f1",
        source_file="r.pdf",
        category="Image",
        page_number=1,
        image_path=str(img),
        image_width=1,
        image_height=1,
    )
    ok, reason = should_enrich_figure(el, _Settings())
    assert ok is False
    assert reason == "below_min_dimensions"

    el2 = ElementRecord(
        element_id="f2",
        source_file="r.pdf",
        category="Image",
        page_number=1,
        image_path=str(img),
        image_width=400,
        image_height=300,
        is_duplicate=True,
        skip_reason="duplicate_header_footer_or_logo",
    )
    ok2, reason2 = should_enrich_figure(el2, _Settings())
    assert ok2 is False
    assert "duplicate" in reason2
