"""ORExtractor REST API — thin app shell that mounts routers."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

_HERE = Path(__file__).parent

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader

import api_state
from rag_app import get_chat_model, get_embedder, get_vectorstore, load_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    expected = os.getenv("API_KEY", "").strip()
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_state.settings = load_settings()
    api_state.embedder = get_embedder(api_state.settings)
    api_state.vectorstore = get_vectorstore(api_state.settings, api_state.embedder)
    api_state.llm = get_chat_model(api_state.settings)
    yield


app = FastAPI(
    title="NI 43-101 RAG & Extraction API",
    version="1.0.0",
    description=(
        "REST API for ingesting NI 43-101 mineral project reports, answering questions "
        "via RAG, and extracting structured project data."
    ),
    lifespan=lifespan,
    dependencies=[Depends(_verify_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/dashboard", include_in_schema=False, dependencies=[])
def serve_dashboard():
    return FileResponse(_HERE / "dashboard.html")


from api_routers import chat, chroma, documents, ingestion, reports  # noqa: E402

app.include_router(documents.router)
app.include_router(ingestion.router)
app.include_router(chroma.router)
app.include_router(reports.router)
app.include_router(chat.router)
