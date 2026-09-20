from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .models import AccessContext, Document
from .service import SecureRAGService


class IngestRequest(BaseModel):
    id: str
    tenant_id: str
    title: str
    text: str
    allowed_users: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    version: int = 1


class QueryRequest(BaseModel):
    query: str
    user_id: str
    tenant_id: str
    groups: list[str] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)


def create_app(service: SecureRAGService | None = None) -> FastAPI:
    service = service or SecureRAGService()
    app = FastAPI(title="SecureRAG", version="0.1.0")

    @app.post("/v1/documents")
    def ingest(request: IngestRequest) -> dict[str, str]:
        service.ingest(
            Document(
                id=request.id,
                tenant_id=request.tenant_id,
                title=request.title,
                text=request.text,
                allowed_users=frozenset(request.allowed_users),
                allowed_groups=frozenset(request.allowed_groups),
                version=request.version,
            )
        )
        return {"status": "indexed", "document_id": request.id}

    @app.post("/v1/query")
    def query(request: QueryRequest) -> dict[str, object]:
        result = service.query(
            request.query,
            AccessContext(
                user_id=request.user_id,
                tenant_id=request.tenant_id,
                groups=frozenset(request.groups),
            ),
            top_k=request.top_k,
        )
        return {
            "trace_id": result.trace_id,
            "answer": result.answer,
            "citations": [
                {
                    "document_id": hit.chunk.document_id,
                    "chunk_id": hit.chunk.id,
                    "score": round(hit.score, 4),
                }
                for hit in result.hits
            ],
        }

    @app.get("/v1/audit")
    def audit() -> list[dict[str, object]]:
        return [event.__dict__ for event in service.store.audit_events()]

    return app


app = create_app()

