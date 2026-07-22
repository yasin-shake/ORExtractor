# WSL 2 + RTX 5070 Ti olmOCR ingestion API

This setup runs two local processes inside the same Ubuntu WSL 2 distribution:

1. A vLLM OpenAI-compatible server serving `allenai/olmOCR-2-7B-1025-FP8` on port `8001`.
2. ORExtractor FastAPI on port `8000`, including upload, ingestion, chat, and downloadable ChromaDB export endpoints.

The API and inference server use separate Python environments to avoid dependency conflicts.

## 1. Prepare Windows and WSL

In an elevated Windows PowerShell terminal:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --shutdown
```

Install the latest NVIDIA Windows driver that supports the RTX 5070 Ti and WSL CUDA. Do not install a Linux NVIDIA display driver inside WSL.

Start Ubuntu and verify GPU passthrough:

```bash
nvidia-smi
```

The RTX 5070 Ti should be listed. If `nvidia-smi` fails, fix the Windows driver/WSL integration before continuing.

## 2. Install system packages

```bash
sudo apt update
sudo apt install -y \
  git curl build-essential python3.11 python3.11-venv python3-pip \
  libgl1 libglib2.0-0
```

Keep the repository in the Linux filesystem, not under `/mnt/c`, because Chroma and model-cache workloads perform substantially better there.

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/yasin-shake/ORExtractor.git
cd ORExtractor
git switch agent/wsl-olmocr-chromadb-export
```

## 3. Create the ORExtractor API environment

```bash
python3.11 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.wsl.example .env
```

Edit `.env`:

```bash
nano .env
```

At minimum, set:

- `OPENAI_API_KEY` for embeddings.
- `API_KEY` to a long random secret.
- AWS/Bedrock values only if chat and structured extraction will be used.
- `OLMOCR_MODEL=allenai/olmOCR-2-7B-1025-FP8`.
- `OLMOCR_WORKERS=1` initially for the 16 GB RTX 5070 Ti.

Create runtime directories:

```bash
mkdir -p knowledge extracted_data spatial_data .chroma_db
```

## 4. Create the GPU inference environment

Open a second WSL terminal:

```bash
cd ~/src/ORExtractor
python3.11 -m venv .venv-olmocr
source .venv-olmocr/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install vllm
```

Verify that PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## 5. Start olmOCR through vLLM

In the GPU environment:

```bash
source ~/src/ORExtractor/.venv-olmocr/bin/activate
export HF_HOME="$HOME/.cache/huggingface"

vllm serve allenai/olmOCR-2-7B-1025-FP8 \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --served-model-name allenai/olmOCR-2-7B-1025-FP8
```

The first launch downloads roughly 10 GB of model files.

If startup fails with CUDA out-of-memory, retry in this order:

```bash
# First reduce model context
--max-model-len 12288

# If still necessary
--max-model-len 8192

# Last resort: reduce memory reservation slightly
--gpu-memory-utilization 0.88
```

Do not increase `OLMOCR_WORKERS` until a representative long report completes successfully. One worker is slower but avoids concurrent page requests competing for limited VRAM.

Check the server:

```bash
curl http://127.0.0.1:8001/v1/models
```

## 6. Start the ORExtractor API

In the API terminal:

```bash
cd ~/src/ORExtractor
source .venv-api/bin/activate
uvicorn chroma_export_api:app --host 0.0.0.0 --port 8000
```

Open Swagger locally:

```text
http://localhost:8000/docs
```

Check health through an authenticated index-info request:

```bash
curl -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/api/chroma/info
```

If `$API_KEY` is not exported in the shell, substitute its actual value.

## 7. Upload documents and receive ChromaDB

### One request: upload, ingest, and download ChromaDB

```bash
curl --fail-with-body \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "rebuild=false" \
  -F "files=@/path/to/report-one.pdf;type=application/pdf" \
  -F "files=@/path/to/report-two.pdf;type=application/pdf" \
  -o orextractor-chromadb.zip \
  http://REMOTE_PC_IP:8000/api/ingest/export
```

Set `rebuild=true` only when the entire existing collection should be deleted and rebuilt from every PDF currently in `knowledge/`.

### Upload and retain the database on the server

The existing endpoint remains available:

```bash
curl --fail-with-body \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "files=@/path/to/report.pdf;type=application/pdf" \
  http://REMOTE_PC_IP:8000/api/ingest
```

### Download the current ChromaDB later

```bash
curl --fail-with-body \
  -H "X-API-Key: YOUR_API_KEY" \
  -o orextractor-chromadb.zip \
  http://REMOTE_PC_IP:8000/api/chroma/export
```

The ZIP contains:

```text
chroma_db/             # persistent Chroma files
export_manifest.json   # collection, embedding model, chunk settings, files, vector count
```

## 8. Restore and open the exported database

Extract it:

```bash
unzip orextractor-chromadb.zip -d orextractor-export
```

Use the same collection name and embedding model used during ingestion:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = Chroma(
    collection_name="ni43101_knowledge",
    persist_directory="orextractor-export/chroma_db",
    embedding_function=embeddings,
)

print(store._collection.count())
results = store.similarity_search("What is the post-tax NPV?", k=5)
for result in results:
    print(result.metadata)
    print(result.page_content[:500])
```

The embedding function must match the original index for new queries and future inserts to remain in the same vector space.

## 9. Remote access and security

Do not expose port `8000` directly to the public internet. Preferred options are:

- Tailscale between the sending machine and the remote PC.
- A corporate VPN.
- An SSH tunnel.
- A reverse proxy with TLS, request-size limits, authentication, and an IP allowlist.

Example SSH tunnel from the client:

```bash
ssh -L 8000:127.0.0.1:8000 user@REMOTE_PC_IP
```

Then send requests to `http://127.0.0.1:8000` on the client.

The API serializes ingestion/export operations with an in-process lock. Run one Uvicorn worker only; multiple workers would each have a separate lock and could write to the same Chroma directory concurrently.

## 10. Operational recommendations

- Back up `.chroma_db`, `knowledge`, `extracted_data`, and `.env` separately.
- Keep `OPENAI_EMBED_MODEL`, chunk size, and chunk overlap fixed for a collection.
- Use the existing content-fingerprint manifest to skip unchanged PDFs.
- Start with one representative 300-800 page report before uploading the full corpus.
- Watch VRAM with `watch -n 1 nvidia-smi`.
- Keep WSL virtual-disk free space above the model cache plus PDFs, rendered pages, Chroma, and temporary ZIP size.
- Pin package versions after the pilot succeeds: `pip freeze > requirements-lock-wsl.txt` in each environment.
