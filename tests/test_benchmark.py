"""Unit tests for peer benchmarking helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import benchmark_field, find_peer_reports, infer_benchmark_field
from schemas import (
    EconomicParameters,
    MineralResource,
    MiningMethod,
    NI43101Report,
)


def test_infer_benchmark_field():
    assert infer_benchmark_field("Is the cut-off grade reasonable?") == "cutoff_grade"
    assert infer_benchmark_field("What is the post-tax NPV?") == "post_tax_npv"
    assert infer_benchmark_field("Compare mining recovery") == "mining_recovery"


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


def test_benchmark_field_consumes_current_report_schema():
    peers = [
        NI43101Report(
            source_file="a.pdf",
            mineral_resources=[
                MineralResource(
                    commodity="Cu",
                    cut_off_grade="0.20% Cu",
                )
            ],
            economics=EconomicParameters(
                post_tax_npv="US$500 million",
                irr="18%",
            ),
            mining_method=MiningMethod(dilution="5%"),
        ).model_dump(),
        NI43101Report(
            source_file="b.pdf",
            mineral_resources=[
                MineralResource(
                    commodity="Cu",
                    cut_off_grade="0.30% Cu",
                )
            ],
            economics=EconomicParameters(
                post_tax_npv="US$1.2 billion",
                irr="22%",
            ),
            mining_method=MiningMethod(dilution="7%"),
        ).model_dump(),
    ]

    cutoff = benchmark_field("cutoff_grade", peers)
    assert cutoff["count"] == 2
    assert cutoff["min"] == 0.2
    assert cutoff["max"] == 0.3
    assert cutoff["comparison_key"] == {
        "unit": "%",
        "currency": None,
        "commodity": "CU",
    }

    npv = benchmark_field("post_tax_npv", peers)
    assert npv["count"] == 2
    assert npv["min"] == 500_000_000
    assert npv["max"] == 1_200_000_000
    assert npv["comparison_key"]["currency"] == "USD"

    dilution = benchmark_field("dilution", peers)
    assert dilution["count"] == 2
    assert dilution["median"] == 6.0


def test_benchmark_field_does_not_mix_incompatible_units_or_currencies():
    reports = [
        NI43101Report(
            source_file="cu.pdf",
            mineral_resources=[
                MineralResource(commodity="Cu", cut_off_grade="0.25% Cu")
            ],
            economics=EconomicParameters(post_tax_npv="US$500 million"),
        ).model_dump(),
        NI43101Report(
            source_file="au.pdf",
            mineral_resources=[
                MineralResource(commodity="Au", cut_off_grade="0.50 g/t Au")
            ],
            economics=EconomicParameters(post_tax_npv="C$700 million"),
        ).model_dump(),
    ]

    cutoff = benchmark_field("cutoff_grade", reports)
    assert cutoff["count"] == 1
    assert cutoff["excluded_incomparable"] == 1

    npv = benchmark_field("post_tax_npv", reports)
    assert npv["count"] == 1
    assert npv["excluded_incomparable"] == 1


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
