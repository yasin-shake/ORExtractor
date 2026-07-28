# Local VLM feasibility for ORExtractor visual enrichment

Date: 2026-07-27

## Conclusion

A local vision-language model is feasible on this workstation's NVIDIA
GeForce RTX 5070 Ti (16 GB VRAM) and 32 GB system RAM. Qwen3-VL-8B-Instruct
Q8 is now installed, integrated through Ollama, and configured as the local
visual-enrichment provider. Gemma 4 12B Q4 remains an untested alternative.

No primary source provides an apples-to-apples evaluation of these models
against Claude 3.5 Haiku on ORExtractor's exact tasks and Pydantic schemas.
Quality parity with a supported remote vision model remains unverified because
the configured Claude 3.5 Haiku model is not a valid image-input baseline.

## Important baseline correction

ORExtractor currently configures
`us.anthropic.claude-3-5-haiku-20241022-v1:0` as its visual model. AWS's
current model card marks image input as unsupported, identifies the model as
Legacy, and gives an end-of-life date of June 19, 2026. The current date is
after that EOL. This means the configured model is not a valid supported
vision baseline, even though ORExtractor sends image content blocks to it.

Source: [Amazon Bedrock Claude 3.5 Haiku model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-3-5-haiku.html)

## Candidate 1: Qwen3-VL-8B-Instruct

Qwen describes Qwen3-VL as its strongest vision-language generation and
highlights expanded multilingual OCR and improved long-document structure
parsing. The official model card provides Transformers and vLLM examples. The
official GGUF repository documents Windows installation through WinGet and a
Q4_K_M llama.cpp server. Its published files are approximately 5.03 GB for
Q4_K_M and 8.71 GB for Q8_0 before runtime/KV-cache overhead.

Ollama publishes an 8B Instruct build with text and image input. The default
Q4_K_M package is approximately 6.1 GB, while the Q8_0 package is
approximately 9.8 GB. Both leave practical VRAM headroom on a 16 GB card for
a bounded image request, although actual peak usage must be measured.

Primary sources:

- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Official Qwen3-VL GGUF repository](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Official Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL)
- [Ollama Qwen3-VL library entry](https://ollama.com/library/qwen3-vl)

## Candidate 2: Gemma 4 12B

Google's Gemma 4 model card explicitly lists document/PDF parsing, chart
comprehension, OCR, and image understanding. The 12B model reports 69.1% on
MMMU Pro, 0.164 average edit distance on OmniDocBench 1.5 (lower is better),
and 79.7% on MATH-Vision. Google estimates approximately 6.7 GB to load the
Q4_0 model, excluding context/KV-cache memory, which makes it feasible on this
GPU. Google recommends a high visual-token budget for OCR, document parsing,
and small text.

Gemma 4 is very new, so runtime maturity and ORExtractor schema reliability
need extra validation despite the stronger published benchmark profile.

Primary sources:

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 model overview and memory requirements](https://ai.google.dev/gemma/docs/core)
- [Ollama Gemma 4 library entry](https://ollama.com/library/gemma4)

## Runtime and integration fit

Ollama is already installed on this workstation. Its official documentation
supports image inputs and schema-constrained structured outputs, including a
vision example that passes a Pydantic JSON schema and validates the returned
JSON. This maps well to ORExtractor's `VisualAnalysis` and `TableValidation`
models.

Primary sources:

- [Ollama vision documentation](https://docs.ollama.com/capabilities/vision)
- [Ollama structured outputs documentation](https://docs.ollama.com/capabilities/structured-outputs)

ORExtractor now has a provider-neutral `VisualModel.analyze()` seam, Ollama and
Bedrock adapters, JSON-schema constrained local output with Pydantic validation,
provider/model-aware cache invalidation, a one-request local concurrency
default, and configurable request timeouts. The public `enrich_elements()` seam
accepts an injected provider, and `benchmark-visuals` exercises the same
structured-output contract without opening Chroma.

Docling OCR, Qwen embeddings, and visual enrichment are currently configured
for the same CUDA device. The existing pipeline overlaps parsing, enrichment,
and indexing. A local VLM should therefore run in a separate bounded
enrichment phase, or other GPU models should be unloaded before it starts.
Otherwise the prior CUDA/native-memory exhaustion pattern can recur.

## Measured acceptance result

The confirmed Qwen3-VL Q8 run is stored under
`benchmark_results/qwen3-vl-8b-q8-confirmed`:

- 28/28 schema-valid responses;
- 8/8 independent gold cases passed;
- 100% classification, required-text recall, and numeric recall on gold;
- 5.63 seconds mean latency, 5.40 seconds median, and 12.26 seconds p95;
- 40,605 input tokens and 11,789 output tokens;
- approximately 12.7 GB peak observed VRAM at an 8,192-token context;
- live enrichment, chunking, temporary Chroma indexing, and similarity search
  passed end to end without opening production Chroma.

The gold set contains two tables and six figures: bar, line, and scatter
charts, a process flow, a geological cross-section, and a mine plan. The
twenty retained real samples were all tables and remain unverified rather than
being treated as gold. No eligible retained real figure crops were available,
so real-report figure quality is still an explicit unknown.

Text-only tables above the safe local-model input size are not partially
normalized. The pipeline deterministically preserves the parser output and
marks validation `input_truncated`; large tables with a retained image remain
eligible for visual validation.

## Remaining acceptance work

Before claiming parity with a remote vision model, extend the fixed, versioned
set of human-reviewed real artifacts:

- numeric resource/reserve and economic tables;
- dense charts with small labels and units;
- flow/process diagrams;
- geological maps and cross-sections that must be rejected for reconstruction;
- malformed parser tables that require correction without invented values.

Measure schema-valid response rate, numeric transcription accuracy,
hallucinated-number rate, reconstruction allow/deny precision, latency, peak
VRAM/RAM, and retry rate. A human-reviewed gold set is required because public
benchmarks do not test ORExtractor's domain rules.
