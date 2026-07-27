from pathlib import Path
from queue import Empty

import pytest

from ingestion.docling_worker import (
    DoclingHardTimeoutError,
    DoclingWorkerManager,
)
from ingestion.models import ElementRecord, ParserQualityReport, ParserResult


def _successful_worker(_settings, requests, responses):
    request = requests.get()
    result = ParserResult(
        source_file=request["source_file"],
        parser="docling",
        parser_version="test",
        elements=[
            ElementRecord(
                element_id="n1",
                source_file=request["source_file"],
                category="NarrativeText",
                text="worker result",
            )
        ],
        page_count=1,
        quality=ParserQualityReport(
            score=1.0,
            text_coverage=1.0,
        ),
    )
    result_path = Path(request["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(result.model_dump_json(), encoding="utf-8")
    responses.put(
        {
            "request_id": request["request_id"],
            "status": "success",
            "result_path": str(result_path),
        }
    )


class _TimeoutQueue:
    def __init__(self):
        self.items = []
        self.closed = False

    def put(self, value):
        self.items.append(value)

    def get(self, timeout=None):
        raise Empty

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        return None


class _FakeProcess:
    def __init__(self):
        self.started = False
        self.terminated = False
        self.killed = False
        self.joined = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and not (self.terminated or self.killed)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        self.joined = True


class _FakeContext:
    def __init__(self):
        self.queues = []
        self.process = _FakeProcess()

    def Queue(self):
        queue = _TimeoutQueue()
        self.queues.append(queue)
        return queue

    def Process(self, **_kwargs):
        return self.process


def test_hard_timeout_terminates_docling_worker(tmp_path):
    context = _FakeContext()
    manager = DoclingWorkerManager(
        settings={},
        timeout_seconds=0.01,
        context=context,
    )

    with pytest.raises(DoclingHardTimeoutError, match="hard timeout"):
        manager.parse(
            tmp_path / "input.pdf",
            source_file="report.pdf",
            artifact_dir=tmp_path / "artifacts",
        )

    assert context.process.terminated is True
    assert context.process.joined is True
    assert manager.is_running is False


def test_spawned_worker_roundtrips_parser_result(tmp_path):
    manager = DoclingWorkerManager(
        settings={},
        timeout_seconds=10,
        worker_target=_successful_worker,
    )
    try:
        result = manager.parse(
            tmp_path / "input.pdf",
            source_file="nested/report.pdf",
            artifact_dir=tmp_path / "artifacts",
        )
    finally:
        manager.close()

    assert result.source_file == "nested/report.pdf"
    assert result.elements[0].text == "worker result"
