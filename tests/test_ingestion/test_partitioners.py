import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

from ingestion.partitioners import UnstructuredApiPartitioner, get_partitioner


def test_hosted_api_partitioner_normalizes_and_persists_raw_output(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")
    raw = [
        {
            "type": "Table",
            "text": "A B",
            "metadata": {
                "page_number": 7,
                "text_as_html": "<table><tr><td>A</td></tr></table>",
            },
        }
    ]
    partitioner = UnstructuredApiPartitioner(
        api_url="https://example.test/general/v0/general",
        api_key="secret",
        strategy="hi_res",
        artifact_dir=tmp_path / "artifacts",
    )
    partition_api = MagicMock(return_value=raw)
    unstructured_module = ModuleType("unstructured")
    partition_module = ModuleType("unstructured.partition")
    api_module = ModuleType("unstructured.partition.api")
    api_module.partition_via_api = partition_api
    with patch.dict(
        sys.modules,
        {
            "unstructured": unstructured_module,
            "unstructured.partition": partition_module,
            "unstructured.partition.api": api_module,
        },
    ):
        elements = partitioner.partition(pdf)

    assert elements[0].category == "Table"
    assert elements[0].page_number == 7
    assert elements[0].text_as_html.startswith("<table>")
    assert (tmp_path / "artifacts" / "report" / "partition_raw.json").exists()
    kwargs = partition_api.call_args.kwargs
    assert kwargs["strategy"] == "hi_res"
    assert kwargs["api_url"].startswith("https://")


def test_unknown_provider_is_rejected(tmp_path):
    class _Settings:
        unstructured_provider = "mystery"
        artifact_dir = tmp_path

    try:
        get_partitioner(_Settings())
    except ValueError as exc:
        assert "UNSTRUCTURED_PROVIDER" in str(exc)
    else:
        raise AssertionError("unknown provider should fail fast")
