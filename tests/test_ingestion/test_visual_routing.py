from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ingestion.cache import EnrichmentCache
from ingestion.enrichment import _image_payload, enrich_elements, should_enrich_figure
from ingestion.models import ElementRecord, VisualAnalysis


@dataclass
class _Settings:
    visual_min_width: int = 250
    visual_min_height: int = 150
    visual_max_width: int = 4096
    visual_max_height: int = 4096
    visual_max_calls_per_report: int = 100
    visual_token_budget_per_report: int = 350000
    bedrock_visual_max_tokens: int = 3500
    bedrock_visual_concurrency: int = 2
    bedrock_visual_model_id: str = "haiku-test"
    visual_enrichment_enabled: bool = True


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


def test_reprocess_visuals_bypasses_successful_cache(tmp_path):
    from PIL import Image

    image_path = tmp_path / "figure.png"
    Image.new("RGB", (400, 300), "white").save(image_path)
    element = ElementRecord(
        element_id="f1",
        source_file="r.pdf",
        category="Image",
        image_path=str(image_path),
        image_width=400,
        image_height=300,
    )
    parsed = VisualAnalysis(
        figure_type="photo",
        description="Source figure",
        confidence=0.95,
    )
    raw = SimpleNamespace(
        usage_metadata={"input_tokens": 20, "output_tokens": 10},
        response_metadata={},
    )
    model = MagicMock()
    model.invoke.return_value = {
        "parsed": parsed,
        "raw": raw,
        "parsing_error": None,
    }
    cache = EnrichmentCache(tmp_path / "cache")

    with patch(
        "ingestion.enrichment.get_visual_analysis_model",
        return_value=model,
    ):
        _, _, _, first = enrich_elements([element], _Settings(), cache=cache)
        _, _, _, second = enrich_elements([element], _Settings(), cache=cache)
        _, _, _, forced = enrich_elements(
            [element],
            _Settings(),
            cache=cache,
            bypass_cache=True,
        )

    assert first["bedrock_calls"] == 1
    assert first["input_tokens"] == 20
    assert second["cache_hits"] == 1
    assert forced["bedrock_calls"] == 1
    assert model.invoke.call_count == 2


def test_per_report_call_limit_skips_extra_visuals(tmp_path):
    from PIL import Image

    settings = _Settings(visual_max_calls_per_report=0)
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (400, 300), "white").save(image_path)
    element = ElementRecord(
        element_id="f1",
        source_file="r.pdf",
        category="Image",
        image_path=str(image_path),
        image_width=400,
        image_height=300,
    )
    with patch("ingestion.enrichment.get_visual_analysis_model") as factory:
        _, _, _, stats = enrich_elements([element], settings)
    assert stats["budget_skips"] == 1
    assert element.skip_reason == "visual_budget_limit"
    factory.assert_not_called()


def test_bedrock_image_payload_is_downscaled_without_touching_source(tmp_path):
    import base64
    from io import BytesIO

    from PIL import Image

    image_path = tmp_path / "large.png"
    Image.new("RGB", (800, 400), "white").save(image_path)
    original_size = image_path.stat().st_size
    settings = _Settings(visual_max_width=100, visual_max_height=100)
    encoded, media_type = _image_payload(str(image_path), settings)
    with Image.open(BytesIO(base64.b64decode(encoded))) as payload:
        assert payload.width <= 100
        assert payload.height <= 100
    assert media_type == "image/png"
    assert image_path.stat().st_size == original_size
