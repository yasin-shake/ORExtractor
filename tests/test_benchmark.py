"""Unit tests for peer benchmarking helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import benchmark_field, find_peer_reports, infer_benchmark_field


def test_infer_benchmark_field():
    assert infer_benchmark_field("Is the cut-off grade reasonable?") == "cutoff_grade"
    assert infer_benchmark_field("What is the post-tax NPV?") == "post_tax_npv"


def test_benchmark_field_aggregation():
    peers = [
        {
            "source_file": "a.pdf",
            "mineral_resources": [{"cutoff_grade": 0.2, "cutoff_unit": "%"}],
        },
        {
            "source_file": "b.pdf",
            "mineral_resources": [{"cutoff_grade": 0.3, "cutoff_unit": "%"}],
        },
    ]
    target = {
        "source_file": "target.pdf",
        "mineral_resources": [{"cutoff_grade": 0.5, "cutoff_unit": "%"}],
    }
    result = benchmark_field("cutoff_grade", peers, target)
    assert result["count"] >= 2
    assert result["min"] == 0.2
    assert result["max"] == 0.3
    assert len(result.get("outliers", [])) >= 1


def test_find_peer_reports_filters():
    from rag_app import Settings
    from pathlib import Path as P

    settings = Settings(
        openai_api_key="x",
        openai_base_url=None,
        embed_model="text-embedding-3-small",
        aws_region="us-east-1",
        bedrock_model_id="test",
        knowledge_dir=P("knowledge"),
        extra_pdf_dirs=[],
        chroma_dir=P(".chroma_db"),
        collection_name="test",
        chunk_size=1400,
        chunk_overlap=150,
        embed_batch_size=64,
        upsert_batch_size=24,
        top_k=8,
        extracted_dir=P("extracted_data"),
        extract_top_k=12,
    )
    peers = find_peer_reports(settings, commodity="Cu", limit=5)
    assert isinstance(peers, list)
