# Approval request: AWS visual enrichment of the ORExtractor corpus

**Prepared:** 2026-07-28  
**Requested authorization:** **USD 250 working budget, with an absolute
USD 300 ceiling**  
**Recommended configuration:** Amazon EC2 `g6.4xlarge` plus Amazon Bedrock
Qwen3-VL 235B A22B in `us-east-1`

## Decision requested

Approve a controlled AWS run to complete visual enrichment of the 598-report
ORExtractor corpus. The requested USD 250 authorizes a representative pilot
and, only if the pilot forecast remains within the approved limit, the full
corpus run. The run must not exceed USD 300 without separate approval.

The recommended AWS-managed configuration has an estimated total cost of
**USD 70–180**, including a 10% infrastructure contingency but excluding tax,
AWS Support, and unusual internet-transfer charges. The estimate is not a
fixed AWS quotation: Bedrock is metered by actual tokens and EC2 by actual
runtime.

## Purpose and business outcome

ORExtractor converts NI 43-101 technical-report PDFs into searchable chunks.
Many material facts are contained in figures, maps, cross-sections, flow
diagrams, and tables rather than ordinary PDF text. Visual enrichment adds a
structured explanation of those assets, associates nearby captions and
references with them, and makes the information available to retrieval and
downstream extraction.

This run is intended to:

1. replace the workstation-hosted visual-model bottleneck with a scalable
   Bedrock vision model;
2. reprocess retained, useful page-content visuals across the completed
   ingestion corpus;
3. preserve parser, embedding, and Chroma indexing compatibility; and
4. complete the work quickly while keeping total incremental spend below
   USD 300.

AWS describes Qwen3-VL as a model that “processes text and images for visual
reasoning and document understanding.” [R1] That is directly aligned with this
workload.

## Measured workload baseline

The following are local measurements, not AWS estimates:

| Item | Measured value |
|---|---:|
| PDF reports | 598 |
| PDF pages | 124,622 |
| Source PDF size | 9.58 GiB |
| Existing ingestion artifacts | 34.68 GiB |
| Existing Chroma database | 10.59 GiB |
| Current data footprint | 54.86 GiB |
| Configured maximum visual calls per report | 30 |
| Absolute visual-request ceiling | 17,940 |

The request ceiling is `598 reports × 30 calls/report`. The code’s public
configuration also documents the 30-call limit and the selected-provider
concurrency setting. [L1]

A completed 513-page pilot report produced 2,983 chunks and made 30 visual
calls. Its measured timings were:

| Stage | Time |
|---|---:|
| Parse | 248.0 seconds |
| Visual enrichment | 484.9 seconds |
| Embedding | 47.8 seconds |
| End-to-end | 780.97 seconds |

The end-to-end pilot rate was approximately 1.52 seconds/page. A purely serial
extrapolation is about 52.7 hours. Bedrock concurrency should reduce the
visual-enrichment bottleneck, but mixed document complexity, retries, I/O, and
rate limiting prevent treating the theoretical speedup as guaranteed.

The confirmed local Qwen3-VL 8B benchmark produced 28/28 schema-valid responses,
8/8 passing gold cases, 5.63-second mean latency, and 12.26-second p95 latency.
Its 28 requests used 40,605 input and 11,789 output tokens. These results
establish the application contract, but Bedrock can tokenize images and schema
instructions differently. [L2]

## Recommended environment

| Component | Recommendation | Rationale |
|---|---|---|
| Region | `us-east-1` | Qwen3-VL supports in-Region inference there; the rates below are quoted for this Region. |
| Compute | EC2 `g6.4xlarge`, Linux On-Demand | One NVIDIA L4 GPU, 16 vCPU, and 64 GiB RAM provide GPU parsing/embedding plus CPU and memory headroom. [R2] |
| Image | AWS Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04 | Maintained AWS base with the required NVIDIA driver stack. [R3] |
| Persistent disk | 300 GB `gp3` EBS | Covers the measured 55 GiB footprint, working files, caches, logs, and growth. |
| Visual model | Bedrock `qwen.qwen3-vl-235b-a22b` | Active text-and-image model; Converse and structured outputs are supported by the Bedrock Runtime endpoint. [R1] |
| Visual concurrency | Start at 8 | Removes the current single-request bottleneck while remaining a pilot-controlled application setting. |
| Purchase model | On-Demand | No commitment; stop or terminate immediately after artifacts are secured. |

AWS states that On-Demand EC2 lets customers “pay for compute capacity by the
hour or second ... with no long-term commitments.” [R4]

The current configured Claude 3.5 Haiku model must **not** be used for this
purpose. Its AWS model card marks image input unsupported, lifecycle `Legacy`,
and end of life as 2026-06-19. [R5]

## Official unit-price quotations

All rates are public AWS prices accessed on 2026-07-28. The EC2 price list was
published 2026-07-24 and is effective 2026-07-01. Prices can change before the
run, so the operator must recheck the AWS Pricing Calculator or price lists at
launch.

### Compute and storage

| Resource | Official public rate | Source |
|---|---:|---|
| `g6.4xlarge`, Linux On-Demand | USD 1.3232/hour | AWS EC2 Price List: “$1.3232 per On Demand Linux g6.4xlarge Instance Hour.” [R6] |
| `g6.2xlarge`, Linux On-Demand | USD 0.9776/hour | AWS EC2 Price List [R6] |
| `g4dn.2xlarge`, Linux On-Demand | USD 0.7520/hour | AWS EC2 Price List [R6] |
| `c7i.4xlarge`, Linux On-Demand | USD 0.7140/hour | AWS EC2 Price List [R6] |
| `gp3` EBS | USD 0.08/GB-month | Includes baseline 3,000 IOPS and 125 MB/s. [R7] |
| In-use public IPv4 | USD 0.005/hour | AWS VPC public IPv4 pricing [R8] |

For 300 GB, `gp3` is USD 24 for a full month or approximately USD 0.0333 per
allocated hour when prorated. EBS remains billable while allocated even if the
instance is stopped.

### Bedrock visual-model rates

| Model | Input / 1M tokens | Output / 1M tokens | Relevant constraint |
|---|---:|---:|---|
| Amazon Nova 2 Lite | USD 0.30 | USD 2.50 | Lowest quoted cost; current model card does not list native structured outputs, so compatibility must be proven in the pilot. [R9] |
| Qwen3-VL 235B A22B | USD 0.53 | USD 2.66 | Recommended direct Qwen replacement; in-Region endpoint and structured outputs supported. [R1], [R10] |
| Claude Haiku 4.5, Global | USD 1.00 | USD 5.00 | Supported vision alternative, but Global routing may process outside the selected Region. [R11] |
| Claude Sonnet 5 | USD 2.00 | USD 10.00 | Promotional rate only through 2026-08-31; high case breaches the budget. [R10] |

AWS says supported batch inference is offered at a “50% lower price compared
to on-demand inference pricing.” [R10] Batch/Flex is not included in the
recommendation because the current enrichment path is synchronous and speed
is the primary objective. Using it safely would require a separate application
change and benchmark.

## Bedrock cost calculation

Bedrock cost is:

```text
(input tokens ÷ 1,000,000 × input rate)
+ (output tokens ÷ 1,000,000 × output rate)
```

Actual Bedrock image-token accounting is not yet measured. The approval range
therefore uses three transparent planning cases:

| Case | Requests | Input/request | Output/request | Total input | Total output |
|---|---:|---:|---:|---:|---:|
| Low | 10,000 | 2,000 | 500 | 20.00M | 5.00M |
| Planning | 15,000 | 3,000 | 750 | 45.00M | 11.25M |
| High | 17,940 | 4,500 | 1,250 | 80.73M | 22.43M |

The low case assumes visual filtering and deduplication avoid the full request
ceiling. The high case assumes every report reaches the 30-call cap and each
call is materially larger than the local benchmark. These are estimates, not
observed Bedrock usage.

### Model-only totals

| Model | Low | Planning | High |
|---|---:|---:|---:|
| Nova 2 Lite | USD 18.50 | USD 41.63 | USD 80.28 |
| **Qwen3-VL 235B** | **USD 23.90** | **USD 53.78** | **USD 102.44** |
| Claude Haiku 4.5, Global | USD 45.00 | USD 101.25 | USD 192.86 |
| Claude Sonnet 5, promotional | USD 90.00 | USD 202.50 | USD 385.71 |

Nova 2 Lite is a cost-optimization candidate, not the initial recommendation:
AWS quotes a fixed 230 input tokens for a typical image and 2–5-second typical
latency, but ORExtractor also sends prompt, context, and JSON schema content.
[R9] Its schema reliability and output quality must be compared on the same
gold set before substitution.

## Recommended configuration cost breakdown

Planning runtime for `g6.4xlarge` is 30–50 hours.

| Cost item | 30-hour case | 50-hour case | Calculation |
|---|---:|---:|---|
| EC2 `g6.4xlarge` | USD 39.70 | USD 66.16 | hours × USD 1.3232 |
| 300 GB `gp3` | USD 1.00 | USD 1.67 | hours × USD 24/720 |
| Public IPv4 | USD 0.15 | USD 0.25 | hours × USD 0.005 |
| Infrastructure base | USD 40.85 | USD 68.08 | sum above |
| 10% infrastructure contingency | USD 4.09 | USD 6.81 | runtime/logging allowance |
| **Infrastructure subtotal** | **USD 44.94** | **USD 74.89** | |
| Bedrock Qwen3-VL | USD 23.90 | USD 102.44 | low/high token cases |
| **Estimated project total** | **USD 68.84** | **USD 177.33** | |

For approval, this is rounded to **USD 70–180 expected**, with a **USD 250
working authorization** to cover tokenization uncertainty, retries, staging,
and short-lived retained storage. The **USD 300 ceiling is not permission to
continue an over-budget run**; it is the maximum exposure before separate
approval.

Not included:

- tax and AWS Support plan charges;
- unusual internet egress or cross-Region transfer;
- long-term artifact retention after the run;
- engineering labor; and
- any re-run caused by an unapproved code or configuration change.

If the 300 GB EBS volume is retained for a complete month, storage costs USD 24
rather than the prorated USD 1.00–1.67 shown above.

## Configuration alternatives

The following totals pair each environment with Bedrock Qwen3-VL. Runtime
ranges are engineering estimates derived from the measured 513-page pilot and
hardware differences; they must be validated by the representative pilot.

| Configuration | Runtime estimate | Infrastructure incl. 10% contingency | Model | Estimated total | Assessment |
|---|---:|---:|---:|---:|---|
| Existing RTX 5070 Ti workstation + Bedrock | 25–45 h | USD 0* | USD 24–103 | **USD 24–103*** | Fastest and cheapest if local uptime, upload bandwidth, and operational support are acceptable. |
| **EC2 `g6.4xlarge` + Bedrock** | **30–50 h** | **USD 45–75** | **USD 24–103** | **USD 69–177** | **Recommended AWS option: best CPU/RAM headroom and managed-run efficiency.** |
| EC2 `g6.2xlarge` + Bedrock | 40–65 h | USD 45–73 | USD 24–103 | USD 69–175 | Similar dollar range, slower; useful if `g6.4xlarge` quota is unavailable. |
| EC2 `g4dn.2xlarge` + Bedrock | 55–85 h | USD 48–74 | USD 24–103 | USD 72–177 | Older T4 GPU; acceptable fallback, not speed-optimal. |
| EC2 `c7i.4xlarge` CPU + Bedrock | 75–120 h | USD 62–99 | USD 24–103 | USD 86–202 | No GPU; Docling and local Qwen embeddings lose CUDA acceleration. |
| EC2 `g6.2xlarge` + local Qwen3-VL 8B | 90–130 h | USD 101–145 | USD 0 | USD 101–145 | Predictable API cost but much slower because parsing, embedding, and vision share one GPU. |

\* Local electricity, workstation depreciation, operator availability, and
network costs are excluded.

Although Nova 2 Lite produces a lower initial estimate, the current
provider-neutral visual path relies on schema-constrained output. Qwen3-VL is
the only costed candidate above whose current Bedrock model card explicitly
lists structured outputs. The approval therefore prioritizes lower execution
risk over a possible USD 5–25 model saving.

Spot Instances are excluded from the first corpus run. AWS provides only a
two-minute interruption notice, and safe restart behavior around live Chroma
read/write coordination has not yet been fully audited. [R12] Serverless
Lambda and Fargate are also unsuitable for the current long-running,
CUDA-dependent ingestion process.

## Pilot gate and spending controls

Approval should be conditional on all of the following:

1. **Quota preflight.** Confirm at least 16 On-Demand G-family vCPUs in
   `us-east-1`; the default quota can be zero in a new account. [R13]
2. **Representative pilot.** Process 10–20 reports spanning small and large
   files, tables, maps, figures, and difficult scans.
3. **Functional acceptance.** Record schema-valid response rate, gold-set
   quality, retries, latency, filtered/duplicate visuals, and successful
   Chroma retrieval. No full run if quality regresses materially from the
   confirmed local benchmark.
4. **Metering.** Persist actual Bedrock input/output token counts, request
   counts, EC2 runtime, and calculated cost per report.
5. **Forecast gate.** Extrapolate pilot consumption to 598 reports. Start the
   full run only when the forecast is at or below **USD 225**, leaving at least
   USD 25 operational margin inside the USD 250 authorization.
6. **Application stop guard.** Stop issuing new model requests when calculated
   spend reaches USD 225 or the forecast exceeds USD 250.
7. **AWS Budgets.** Create alerts at USD 100, 175, 225, and 275, with an
   action-enabled budget able to stop the EC2 instance. AWS Budget data can
   update only several times per day, so it is not a real-time Bedrock hard
   cap. Charges may pass a threshold before an alert or action occurs. [R14]
8. **Artifact checkpointing.** Keep the production Chroma writer single,
   checkpoint per report, and copy final Chroma, manifests, benchmark results,
   and logs to persistent storage before termination.
9. **Shutdown.** Stop or terminate compute immediately after validation and
   delete unneeded EBS volumes and public IPv4 resources.

AWS Budgets monitoring and notifications are free; the first two
action-enabled budgets are also free. [R15] Those controls complement, but do
not replace, application-side token accounting.

Controls 4 and 6 are **pre-launch implementation requirements**. This document
does not claim that the current application already persists Bedrock usage or
enforces a dollar-denominated stop. Their presence and tests must be verified
before the full-corpus gate is opened.

## Security and data handling

- Use an EC2 instance profile with temporary credentials; do not place access
  keys in `.env`, source control, shell history, or the AMI.
- Grant only the required S3/logging permissions and
  `bedrock:InvokeModel`/`bedrock:InvokeModelWithResponseStream` for the selected
  model resource, following AWS least-privilege guidance. [R16]
- Use Qwen3-VL’s in-Region `us-east-1` endpoint. AWS documents that in-Region
  inference does not leave the selected Region. [R17]
- Encrypt EBS and S3, restrict inbound security-group access, and prefer
  Session Manager rather than an internet-exposed SSH port.
- Confirm that organizational policy permits the reports and extracted
  images to be processed in the United States before launch.

AWS states that Bedrock “never shares your data with model providers or uses it
to train foundation models.” [R18]

## Proposed run configuration

After the pilot has passed the approval gates:

```bash
export AWS_REGION=us-east-1
export VISUAL_MODEL_PROVIDER=bedrock
export BEDROCK_VISUAL_MODEL_ID=qwen.qwen3-vl-235b-a22b
export VISUAL_MODEL_CONCURRENCY=8
export DOCLING_DEVICE=cuda
export DOCLING_NUM_THREADS=8
export MINERU_COMMAND="$PWD/.venv/bin/mineru"

./.venv/bin/python -u rag_app.py ingest \
  --parser docling \
  --fallback mineru \
  --reprocess-visuals
```

Do not add `--rebuild` unless a separately approved rebuild is required. Do
not increase visual concurrency above 8 until pilot latency, throttling, memory
use, and schema-valid output have been reviewed.

## Known uncertainties

| Uncertainty | Effect | Resolution |
|---|---|---|
| Bedrock image/schema token counts have not been measured in this application. | Largest cost variance. | Capture usage fields during the 10–20 report pilot. |
| Runtime extrapolation uses one 513-page report. | Corpus mix may make 30–50 hours optimistic or conservative. | Measure reports/pages per hour across the representative pilot. |
| Qwen3-VL 235B has not yet been run against the local gold set. | Quality may differ from local Qwen3-VL 8B Q8. | Require the same benchmark and schema-validation criteria. |
| Filtering may reduce requests below 17,940. | Likely lowers model cost, but the saving is unverified. | Log retained, rejected, duplicate, and enriched visual counts. |
| Safe multi-process coordination around live Chroma reads/writes remains under audit. | Parallel writers could risk data integrity. | Retain one production writer and checkpoint until the audit is complete. |
| EC2 G-family quota/capacity may be unavailable. | Delays start or requires fallback. | Request quota early; use `g6.2xlarge` only after recalculating runtime. |
| AWS prices can change. | Approval estimate may become stale. | Recheck official price lists immediately before launch. |

## Approval statement

Approval authorizes:

- a 10–20 report pilot;
- a full 598-report run only after the pilot forecast is no more than USD 225;
- total working spend up to USD 250; and
- emergency/variance headroom only up to the absolute USD 300 ceiling.

Approval does **not** authorize an unlimited run, use of Claude Sonnet 5,
Provisioned Throughput, multi-process writes to production Chroma, a full
corpus rebuild, or continued processing after a stop criterion is met.

## References

All web references were accessed 2026-07-28. AWS sources are primary sources.

- **[R1]** AWS, [Qwen3 VL 235B A22B model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-vl-235b-a22b.html).
- **[R2]** AWS, [Amazon EC2 G6 instances](https://aws.amazon.com/ec2/instance-types/g6/).
- **[R3]** AWS, [Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-22-04.html).
- **[R4]** AWS, [Amazon EC2 On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/).
- **[R5]** AWS, [Claude 3.5 Haiku model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-3-5-haiku.html).
- **[R6]** AWS Price List, [Amazon EC2 `us-east-1` CSV](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.csv).
- **[R7]** AWS, [Amazon EBS General Purpose SSD pricing](https://aws.amazon.com/ebs/general-purpose/).
- **[R8]** AWS, [Amazon VPC public IPv4 pricing](https://aws.amazon.com/vpc/pricing/).
- **[R9]** AWS Machine Learning Blog, [Pair Nova 2 Lite with Claude for cost-optimized document processing](https://aws.amazon.com/blogs/machine-learning/pair-nova-2-lite-with-claude-for-cost-optimized-document-processing/).
- **[R10]** AWS, [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/).
- **[R11]** AWS Price List, [Amazon Bedrock Foundation Models `us-east-1` CSV](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.csv), and [Claude Haiku 4.5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html).
- **[R12]** AWS, [Spot Instance interruption notices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html).
- **[R13]** AWS, [Amazon EC2 On-Demand instance quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html).
- **[R14]** AWS, [Managing costs with AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html).
- **[R15]** AWS, [AWS Budgets pricing](https://aws.amazon.com/aws-cost-management/aws-budgets/pricing/).
- **[R16]** AWS, [Identity-based policy examples for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html).
- **[R17]** AWS, [Model region compatibility](https://docs.aws.amazon.com/bedrock/latest/userguide/models-region-compatibility.html).
- **[R18]** AWS, [Amazon Bedrock security and privacy](https://aws.amazon.com/bedrock/security-privacy-responsible-ai/).
- **[L1]** ORExtractor, [`README.md`](../README.md) configuration table and
  [`rag_app.py`](../rag_app.py) settings loader.
- **[L2]** ORExtractor, [`local-vlm-visual-enrichment.md`](local-vlm-visual-enrichment.md)
  and `benchmark_results/qwen3-vl-8b-q8-confirmed`.
