from dataclasses import dataclass
from pathlib import Path

from ingestion.cache import EnrichmentCache
from ingestion.enrichment import _image_payload, enrich_elements, should_enrich_figure
from ingestion.models import ElementRecord, VisualAnalysis
from ingestion.visual_model import VisualResponse


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
    class LocalVisualModel:
        provider = "ollama"
        model_id = "qwen3-vl:test"
        cache_id = "ollama:qwen3-vl:test"

        def analyze(self, _request):
            return VisualResponse(
                value=parsed,
                input_tokens=20,
                output_tokens=10,
            )

    model = LocalVisualModel()
    cache = EnrichmentCache(tmp_path / "cache")

    _, _, _, first = enrich_elements(
        [element], _Settings(), cache=cache, visual_model=model
    )
    _, _, _, second = enrich_elements(
        [element], _Settings(), cache=cache, visual_model=model
    )
    _, _, _, forced = enrich_elements(
        [element],
        _Settings(),
        cache=cache,
        bypass_cache=True,
        visual_model=model,
    )

    assert first["visual_model_calls"] == 1
    assert first["input_tokens"] == 20
    assert second["cache_hits"] == 1
    assert forced["visual_model_calls"] == 1


def test_cached_visuals_do_not_consume_resume_call_allowance(tmp_path):
    from PIL import Image

    settings = _Settings(visual_max_calls_per_report=1)
    cache = EnrichmentCache(tmp_path / "cache")
    elements = []
    for index, color in enumerate(("white", "black")):
        image_path = tmp_path / f"figure-{index}.png"
        Image.new("RGB", (400, 300), color).save(image_path)
        elements.append(
            ElementRecord(
                element_id=f"f{index}",
                source_file="r.pdf",
                category="Image",
                image_path=str(image_path),
                image_width=400,
                image_height=300,
            )
        )

    class CountingVisualModel:
        provider = "ollama"
        model_id = "qwen3-vl:test"
        cache_id = "ollama:qwen3-vl:test"

        def __init__(self):
            self.calls = []

        def analyze(self, request):
            self.calls.append(request.task)
            return VisualResponse(
                value=VisualAnalysis(
                    figure_type="photo",
                    description="resumed",
                    confidence=0.95,
                )
            )

    model = CountingVisualModel()
    _, _, _, first = enrich_elements(
        elements,
        settings,
        cache=cache,
        visual_model=model,
    )
    _, _, _, second = enrich_elements(
        elements,
        settings,
        cache=cache,
        visual_model=model,
    )

    assert first["visual_model_calls"] == 1
    assert first["deferred_element_ids"] == ["f1"]
    assert second["cache_hits"] == 1
    assert second["visual_model_calls"] == 1
    assert second["deferred_element_ids"] == []
    assert len(model.calls) == 2


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
    class NeverInvokedModel:
        provider = "ollama"
        model_id = "never"
        cache_id = "ollama:never"

        def analyze(self, _request):
            raise AssertionError("budget-skipped visual must not invoke a model")

    _, _, _, stats = enrich_elements(
        [element],
        settings,
        visual_model=NeverInvokedModel(),
    )
    assert stats["budget_skips"] == 1
    assert element.skip_reason == "visual_budget_limit"


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


def test_local_visual_model_enriches_through_the_provider_seam(tmp_path):
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

    class LocalVisualModel:
        provider = "ollama"
        model_id = "qwen3-vl:test"
        cache_id = "ollama:qwen3-vl:test"

        def analyze(self, _request):
            return VisualResponse(
                value=VisualAnalysis(
                    figure_type="photo",
                    description="A local-model description.",
                    confidence=0.91,
                ),
                input_tokens=31,
                output_tokens=17,
                latency_ms=12.5,
            )

    analyses, _, errors, stats = enrich_elements(
        [element],
        _Settings(),
        visual_model=LocalVisualModel(),
    )

    assert errors == []
    assert analyses["f1"].description == "A local-model description."
    assert stats["visual_model_calls"] == 1
    assert stats["bedrock_calls"] == 0
    assert stats["input_tokens"] == 31
    assert stats["output_tokens"] == 17


def test_large_table_prompt_forbids_partial_normalized_markdown():
    element = ElementRecord(
        element_id="t1",
        source_file="r.pdf",
        category="Table",
        text="Mineral resource table",
        text_as_html="<table>" + ("<tr><td>1.20</td></tr>" * 225) + "</table>",
    )

    class NeverInvokedModel:
        provider = "ollama"
        model_id = "qwen3-vl:test"
        cache_id = "ollama:qwen3-vl:test"

        def analyze(self, _request):
            raise AssertionError(
                "text-only truncated tables must not invoke a model"
            )

    _, validations, errors, stats = enrich_elements(
        [element],
        _Settings(),
        visual_model=NeverInvokedModel(),
    )

    assert errors == []
    assert validations["t1"].issues == ["input_truncated"]
    assert validations["t1"].normalized_markdown == ""
    assert "original parser output" in validations["t1"].description
    assert stats["visual_model_calls"] == 0
