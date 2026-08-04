# Ingestion technology evaluation for ORExtractor

Date: 2026-07-28

## Decision

Two ideas are genuinely worth testing, but they should not be adopted as one
large rewrite:

1. **MinerU Hybrid `medium` as a second MinerU benchmark lane.** ORExtractor
   already has a MinerU adapter and pins MinerU 3.4.4, so this has the lowest
   integration risk. The vendor's speed gains compare `medium` with MinerU
   Hybrid `high`; they do not prove that Hybrid is faster than ORExtractor's
   current MinerU `pipeline` backend or Docling.
2. **A persistent, corpus-wide visual work queue backed by a batched model
   server.** vLLM is worth benchmarking against Ollama on Linux/AWS. Ray Data
   is justified only after there is more than one GPU or operational evidence
   that its checkpointing/distributed scheduling is needed. Ray itself is not
   the speedup.

Do not start with NeMo Retriever, Dolphin, DeepSeek-OCR, or a wholesale Marker
replacement. LiteParse or PyMuPDF4LLM may later be useful as tightly gated
native-text fast lanes, but they cannot safely replace layout parsing for
tables, captions, figures, headings, or cross-page context.

## Why this ordering fits ORExtractor

The measured pilot spent 484.9 of 780.97 seconds in visual enrichment, 248.0
seconds in parsing, and 47.8 seconds in embedding/indexing. Even eliminating
parsing entirely would reduce the pilot only to 532.97 seconds, a maximum
**1.47x** end-to-end speedup. Reducing the visual stage from 484.9 to the
proposal's idealized 61 seconds would yield about 357 seconds, or **2.19x**,
before parser changes.

The repository already has adaptive native-text OCR bypass, a persistent
Docling process, report-level parse/enrich/index overlap, one Chroma writer,
selective visual routing, header/footer filtering, crop deduplication, and
resumable text-first/visual-backfill passes (`README.md`, ingestion section).
The current implementation has one report-level enrichment worker, but each
report already uses the configured visual-model concurrency
(`ingestion/pipeline.py`, `ingestion/enrichment.py`). Therefore, dividing one
report's measured visual duration by eight is not a valid prediction unless
the baseline actually ran at concurrency one and the target provider sustains
eight useful concurrent requests.

The current local target is an RTX 5070 Ti with 16 GB VRAM; the AWS proposal is
one `g6.4xlarge` L4 with 16 vCPU and 64 GiB RAM
(`docs/local-vlm-visual-enrichment.md` and
`docs/aws-bedrock-visual-enrichment-approval.md`). Published B200 or A100
figures are not portable to either environment.

## Candidate findings

| Candidate | Verified capability and maturity | Licensing | Hardware and throughput portability | ORExtractor verdict |
|---|---|---|---|---|
| **MinerU 3.x** | MinerU 3.3 added Hybrid `medium`/`high`; the project reports `medium` as 0.13 OmniDocBench points below `high` and 35–220% faster depending on OS/input. It also says `medium` does not support image analysis. MinerU provides CLI/API/router, async tasks, multi-service/multi-GPU routing, sliding-window processing, streaming writes, and concurrent inference. Current MinerU 3.4 goes beyond the proposal's 3.3 target. [Official repository and release notes](https://github.com/opendatalab/MinerU) | Custom MinerU Open Source License: Apache-2.0 plus commercial thresholds (100M MAU or USD 20M monthly revenue) and attribution for third-party online services. It is not plain Apache-2.0. [Official license](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md) | Official minimums are 8 GB VRAM for Hybrid/VLM engine, 16 GB RAM, and 20 GB disk; `pipeline` can run on CPU. The percentage gains are within Hybrid modes, not against Docling or the pipeline backend. | **Lowest-risk benchmark.** ORExtractor already pins `mineru[pipeline]==3.4.4`, but defaults `MINERU_BACKEND=pipeline` and passes only `backend` through its adapter. Benchmark current pipeline versus Hybrid `medium` before changing routing. |
| **Marker** | Current Marker uses selective native text, a small layout model, and a shared Surya VLM server. Its current B200 benchmark reports 2.9 pages/s balanced, 7.4 pages/s fast, and 23.7 pages/s no-OCR; it also reports materially lower quality for fast/no-OCR, especially scans and math. The repository describes its bundled API server as small-scale rather than robust. [Official README and benchmark](https://github.com/datalab-to/marker/blob/master/README.md) | The proposal is outdated: current code is Apache-2.0, not GPL. Current model weights remain restricted under a modified AI Pubs Open Rail-M license, free for research/personal use and startups below USD 5M funding/revenue; broader commercial use requires a licence. [Commercial-use section](https://github.com/datalab-to/marker/blob/master/README.md#commercial-usage), [code license](https://github.com/datalab-to/marker/blob/master/LICENSE) | The quoted sustained rates are from one B200 host over a 1,403-page benchmark with native concurrency. They cannot be projected to a 16 GB consumer GPU or one L4. | **Useful benchmark and design reference, conditional on model licensing.** Do not use its B200 rates in the corpus forecast. |
| **LiteParse** | LiteParse is a local Rust/PDFium spatial-text extractor with bounding boxes, JSON/text output, page screenshots, bundled Tesseract, and optional HTTP OCR. Its own README directs dense tables, multi-column layouts, charts, handwriting, and scans to the vendor's more capable cloud parser. [Official repository](https://github.com/run-llama/liteparse) | Apache-2.0. [Official license](https://github.com/run-llama/liteparse/blob/main/LICENSE) | It is CPU-oriented and portable across Windows/Linux/macOS. Marker reports 8.9 pages/s with OCR and 1,721 pages/s without OCR on its B200 host, but also very low document-quality scores; that host's CPU specification is not given. [Marker competitor benchmark](https://github.com/datalab-to/marker/blob/master/README.md#overall-pdf-conversion) | **Possible later fast lane only.** It needs an acceptance gate proving reading order, headings, captions, tables, page provenance, and merge compatibility. It is not a complex-page parser. |
| **PyMuPDF4LLM** | Produces Markdown/JSON/text with page chunks, bounding boxes, tables, images, multi-column reading order, and selective region OCR. The maintainers claim 10x speed versus vision extraction and about 50% OCR-time reduction, but publish no directly portable hardware/corpus methodology with those headline figures. [Official repository](https://github.com/pymupdf/pymupdf4llm) | AGPL-3.0 or a separate commercial licence. ORExtractor already depends on PyMuPDF, so the project must confirm whether its existing use is covered by AGPL compliance or a commercial Artifex licence before expanding it. [Official licensing section](https://github.com/pymupdf/pymupdf4llm#licensing) | CPU-only and operationally easy on the current setup. The 10x headline is not an ORExtractor forecast. | **Technically plausible fast lane, legal review first.** Benchmark against LiteParse only if the existing licence position is acceptable. |
| **Ray Data + vLLM/SGLang** | Ray Data supports streaming CPU/GPU stages, persistent actors, GPU `map_batches`, checkpointing, and multimodal batch inference. Current Ray Data LLM supports vLLM and SGLang engines; its multimodal example uses Qwen2.5-VL-3B, batch size 16, L4 workers, and explicitly warns that undersized batches leave GPUs idle. vLLM supports Qwen3-VL image inference and multimodal input caching. [Ray batch inference](https://docs.ray.io/en/latest/data/batch_inference.html), [Ray multimodal example](https://docs.ray.io/en/master/_collections/data/examples/llm_batch_inference_vision/README.html), [vLLM Qwen3-VL example](https://docs.vllm.ai/en/stable/examples/generate/multimodal/) | Ray, vLLM, and SGLang are Apache-2.0 projects. [Ray repository](https://github.com/ray-project/ray), [vLLM repository](https://github.com/vllm-project/vllm), [SGLang repository](https://github.com/sgl-project/sglang) | A Ray example's batch size and four L4 workers do not imply that one L4 can host four replicas or batches of 8–32 ORExtractor images. The current Ollama Q8 model format/runtime also cannot be assumed to have the same memory footprint in vLLM. | **Batched serving: yes. Ray now: probably no.** First benchmark one persistent vLLM server and a simple durable queue on AWS Linux. Add Ray when scaling to multiple GPUs/nodes or when its recovery/observability offsets the complexity. |
| **NeMo Retriever** | A mature NVIDIA extraction stack for text, tables, charts, OCR, embeddings, and indexing, with library and Helm deployments. The current core requires one A10G-or-better supported GPU and about 150 GB disk; advanced parsing, captioning, and reranking can require additional dedicated GPUs. [Official support matrix](https://docs.nvidia.com/nemo/retriever/latest/extraction/support-matrix/index.html), [deployment options](https://docs.nvidia.com/nemo/retriever/latest/extraction/quickstart-library-mode/) | Framework code is Apache-2.0; NIMs and model artifacts have separate NVIDIA access/deployment terms and may require NGC/API credentials. [Official repository](https://github.com/NVIDIA/NeMo-Retriever), [getting started](https://docs.nvidia.com/nemo/retriever/latest/extraction/getting-started-about/) | NVIDIA's supported list includes A10G, L40S, A100, H100/H200, B200, and RTX Pro 6000, but not the proposed L4 or local RTX 5070 Ti. It would also replace substantial custom orchestration and likely the Chroma/Qwen embedding path. | **Not beneficial for the present one-GPU deployment.** Revisit only for a permanent multi-GPU NVIDIA platform. |
| **Dolphin v2** | A 3B, two-stage document parser that classifies digital versus photographed pages, predicts layout/reading order, then parses digital elements in parallel or photographed pages holistically. It has Transformers, vLLM, and TensorRT-LLM paths, but only 35 commits and no production service/orchestration story comparable to MinerU. [Official repository](https://github.com/ByteDance/Dolphin) | The repository/model uses the Qwen Research License, which permits non-commercial research/evaluation and requires a separate licence for commercial use. [Official license](https://github.com/ByteDance/Dolphin/blob/master/LICENSE) | A 3B model is locally testable, but the project publishes benchmark accuracy rather than a portable corpus throughput result. | **Research-only benchmark, not a production candidate under the current licence.** |
| **DeepSeek-OCR** | A 3B OCR/document-to-Markdown model with Transformers and vLLM examples. The official repository reports about 2,500 output tokens/s for concurrent PDF processing on an A100 40 GB. [Official repository](https://github.com/deepseek-ai/DeepSeek-OCR), [official model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | MIT for the published code/model repository. [Official license](https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/LICENSE) | Tokens/s is not pages/s and says nothing about table fidelity, coordinates, captions, hallucinated values, or page provenance. The A100 result is not portable to a 16 GB consumer GPU or L4. | **Optional experimental comparator only.** Its integration maturity and NI 43-101 fidelity remain unverified. |

## Claims from the proposal that should not drive a decision

- A **3–6x end-to-end gain** is an aspiration, not a source-supported forecast.
  Parser-only changes cannot reach it on the measured timing breakdown.
- The **eight-way visual estimate** assumes perfect scaling from a
  single-request baseline. Existing per-report enrichment already has a
  concurrency control, and neither provider quota nor GPU memory was included.
- Marker B200 and DeepSeek A100 measurements are **not portable** to the local
  RTX 5070 Ti or one AWS L4.
- MinerU's 0.13-point delta is a vendor benchmark result, not evidence for
  numeric fidelity, page citations, NI Item tagging, or geological-figure
  handling.
- NeMo Retriever is not merely an extraction library swap; adopting it would
  replace a large part of ORExtractor's parser-neutral model, embedding,
  orchestration, and storage path.

## Recommended bounded experiment

Use the same 20-report stratified set and frozen human-reviewed page windows
for every lane:

1. Current Docling + current MinerU pipeline + current visual provider.
2. Current Docling + MinerU Hybrid `medium` for quality-gated fallback.
3. Current parser output + persistent vLLM Qwen3-VL server, compared with the
   current Ollama structured-output path at concurrency 1, 2, 4, and 8 where
   memory permits.

Do not write production Chroma during the benchmark. Require exact source/page
provenance, canonical-element equivalence, numeric-table cell accuracy,
caption/reference linkage, header/footer rejection, hallucinated-number rate,
schema-valid enrichment, peak VRAM/RAM, retries, and cost per 1,000 pages.

The first implementation decision should follow from those measurements. The
most likely useful performance subsystem is **persistent batched visual serving
without Ray**.
