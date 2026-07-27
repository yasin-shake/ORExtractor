"""Persistent, killable process seam around Docling's native pipeline."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from queue import Empty
import time
from typing import Any
from uuid import uuid4

from ingestion.models import ParserResult


class DoclingHardTimeoutError(TimeoutError):
    """Raised after the worker process is terminated at the hard deadline."""


def _worker_main(settings: dict[str, Any], requests, responses) -> None:
    from types import SimpleNamespace

    from ingestion.parsers.docling_parser import DoclingParser

    parser = DoclingParser(
        SimpleNamespace(**settings),
        process_isolation=False,
    )
    while True:
        request = requests.get()
        if request is None:
            break
        request_id = request["request_id"]
        result_path = Path(request["result_path"])
        try:
            result = parser._parse_document(
                Path(request["input_path"]),
                source_file=request["source_file"],
                artifact_dir=Path(request["artifact_dir"]),
                page_range=(
                    tuple(request["page_range"])
                    if request.get("page_range")
                    else None
                ),
                preflight=request.get("preflight"),
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                result.model_dump_json(),
                encoding="utf-8",
            )
            responses.put(
                {
                    "request_id": request_id,
                    "status": "success",
                    "result_path": str(result_path),
                }
            )
        except BaseException as exc:
            responses.put(
                {
                    "request_id": request_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    parser.close()


class DoclingWorkerManager:
    """Hide worker lifecycle, timeout enforcement, and result transport."""

    def __init__(
        self,
        *,
        settings: dict[str, Any],
        timeout_seconds: float,
        context=None,
        worker_target=None,
    ):
        self.settings = dict(settings)
        self.timeout_seconds = max(0.001, float(timeout_seconds))
        self._context = context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target or _worker_main
        self._requests = None
        self._responses = None
        self._process = None

    @property
    def is_running(self) -> bool:
        return bool(
            self._process is not None
            and self._process.is_alive()
        )

    def _start(self) -> None:
        if self.is_running:
            return
        self._requests = self._context.Queue()
        self._responses = self._context.Queue()
        self._process = self._context.Process(
            target=self._worker_target,
            args=(self.settings, self._requests, self._responses),
            name="orextractor-docling",
            daemon=True,
        )
        self._process.start()

    @staticmethod
    def _close_queue(queue) -> None:
        if queue is None:
            return
        try:
            queue.close()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        elif process is not None:
            process.join(timeout=1)
        self._close_queue(self._requests)
        self._close_queue(self._responses)
        self._process = None
        self._requests = None
        self._responses = None

    def parse(
        self,
        input_path: Path,
        *,
        source_file: str,
        artifact_dir: Path,
        page_range: tuple[int, int] | None = None,
        preflight: dict[str, Any] | None = None,
    ) -> ParserResult:
        self._start()
        request_id = uuid4().hex
        transport_dir = Path(artifact_dir) / ".worker"
        result_path = transport_dir / f"{request_id}.json"
        request = {
            "request_id": request_id,
            "input_path": str(input_path),
            "source_file": source_file,
            "artifact_dir": str(artifact_dir),
            "page_range": list(page_range) if page_range else None,
            "preflight": preflight,
            "result_path": str(result_path),
        }
        self._requests.put(request)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise DoclingHardTimeoutError(
                    f"Docling hard timeout after {self.timeout_seconds:.1f}s "
                    f"for {source_file}"
                )
            try:
                response = self._responses.get(timeout=remaining)
            except Empty as exc:
                self._terminate()
                raise DoclingHardTimeoutError(
                    f"Docling hard timeout after {self.timeout_seconds:.1f}s "
                    f"for {source_file}"
                ) from exc
            if response.get("request_id") != request_id:
                continue
            if response.get("status") != "success":
                raise RuntimeError(
                    f"Docling worker failed for {source_file}: "
                    f"{response.get('error', 'unknown worker error')}"
                )
            try:
                return ParserResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
            finally:
                try:
                    result_path.unlink()
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        if self.is_running and self._requests is not None:
            try:
                self._requests.put(None)
                self._process.join(timeout=10)
            except (OSError, ValueError):
                pass
        self._terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

