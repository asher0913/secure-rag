"""HTTP API. Identity comes from a verified bearer token; groups come from the directory.

- ``POST /v1/query`` (role ``reader``): the body carries the query only. User and tenant come
  from the token, groups from the service's directory. A body that tries to name a user, tenant
  or groups is rejected with 422.
- ``POST /v1/documents`` (role ``writer``): documents are written into the caller's own tenant.
  Overwriting a document id owned by another tenant is refused, and the server assigns versions.
- ``GET /v1/audit`` (role ``auditor``): only the caller's tenant's events.

Run it with a secret: ``SECURE_RAG_TOKEN_SECRET=... uvicorn secure_rag.api:create_app --factory``,
and mint demo tokens with ``secure-rag token``.
"""

# No ``from __future__ import annotations`` here: FastAPI must resolve the Annotated role
# dependencies defined inside create_app when the routes are declared.
import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .auth import AUDITOR, READER, WRITER, Principal, TokenError, TokenSigner
from .models import AccessContext, Document
from .service import SecureRAGService

SECRET_ENV = "SECURE_RAG_TOKEN_SECRET"


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    title: str
    text: str
    allowed_users: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # user_id, tenant_id and groups are not the client's to set

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)


def create_app(service: SecureRAGService | None = None, signer: TokenSigner | None = None) -> FastAPI:
    if signer is None:
        secret = os.environ.get(SECRET_ENV)
        if not secret:
            raise RuntimeError(f"set {SECRET_ENV} to the token signing secret (at least 16 bytes)")
        signer = TokenSigner(secret)
    service = service or SecureRAGService()
    app = FastAPI(title="SecureRAG", version="1.1.0")

    def principal(authorization: str | None = Header(default=None)) -> Principal:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        try:
            return signer.verify(token.strip())
        except TokenError as error:
            raise HTTPException(401, str(error), headers={"WWW-Authenticate": "Bearer"}) from error

    def require(role: str):
        def check(caller: Annotated[Principal, Depends(principal)]) -> Principal:
            if role not in caller.roles:
                raise HTTPException(403, f"the {role!r} role is required")
            return caller

        return Annotated[Principal, Depends(check)]

    Reader, Writer, Auditor = require(READER), require(WRITER), require(AUDITOR)

    @app.post("/v1/documents")
    def ingest(request: IngestRequest, caller: Writer) -> dict[str, object]:
        existing = service.store.get_document(request.id)
        if existing is not None and existing.tenant_id != caller.tenant_id:
            raise HTTPException(403, "document id belongs to another tenant")
        version = existing.version + 1 if existing else 1
        service.ingest(
            Document(
                id=request.id,
                tenant_id=caller.tenant_id,
                title=request.title,
                text=request.text,
                allowed_users=frozenset(request.allowed_users),
                allowed_groups=frozenset(request.allowed_groups),
                version=version,
            )
        )
        return {"status": "indexed", "document_id": request.id, "version": version}

    @app.post("/v1/query")
    def query(request: QueryRequest, caller: Reader) -> dict[str, object]:
        try:
            groups = service.authority.groups(caller.tenant_id, caller.user_id)
        except Exception as error:  # the directory is down: answer nothing rather than guess
            raise HTTPException(503, "directory unavailable") from error
        context = AccessContext(caller.user_id, caller.tenant_id, groups or frozenset())
        result = service.query(request.query, context, top_k=request.top_k)
        return {
            "trace_id": result.trace_id,
            "answer": result.answer,
            "citations": [
                {"document_id": hit.chunk.document_id, "chunk_id": hit.chunk.id, "score": round(hit.score, 4)}
                for hit in result.hits
            ],
        }

    @app.get("/v1/audit")
    def audit(caller: Auditor) -> list[dict[str, object]]:
        return [event.__dict__ for event in service.store.audit_events() if event.tenant_id == caller.tenant_id]

    return app
